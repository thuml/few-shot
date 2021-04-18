import json
import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from few_shot import backbones, models
from few_shot.config import cfg
from few_shot.dataset import get_loader
from few_shot.logger import setup_logger
from few_shot.utils import AverageMeter, GradualWarmupScheduler
from few_shot.utils import accuracy, get_assigned_file, get_resume_file, get_best_file, get_few_shot_label


def train(model, train_loader, optimizer, summary_writer, epoch, scheduler=None):
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        scores, y = model(x, y)
        loss = model.calculate_loss(scores, y)
        acc = accuracy(scores, y) * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step = epoch * len(train_loader) + i
        if scheduler is not None:
            scheduler.step(step)

        train_loss.update(loss.item(), x.shape[0])
        top1.update(acc, x.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        summary_writer.add_scalar('loss', loss.item(), step)
        summary_writer.add_scalar('train_acc', acc, step)
        if i % cfg.train.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f'Train: [{epoch}][{i}/{len(train_loader)}] '
                        f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        f'Lr: {lr:.5f} '
                        f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                        f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})')


def extract_feature(model, loader):
    backbone = nn.DataParallel(model.module.backbone) if cfg.method.backbone == 'wrn_28_10' else model.backbone
    backbone.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in tqdm(loader, desc='extracting feature', ncols=80):
            feature = backbone(x.cuda(non_blocking=True)).cpu().numpy()
            all_feats.append(feature)
            all_labels.append(y)

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    label2feats = defaultdict(list)
    for feat, label in zip(all_feats, all_labels):
        label2feats[label].append(feat)

    return label2feats


def test_all(model, base_loader, base_val_loader, val_loader, test_loader):
    logger.info('=> testing supervised accuracy for base train data')
    acc_base = test_supervised(model, base_loader)
    md_header = "| train acc |"
    md_middle = "| ---- |"
    md_content = f"| {acc_base:.2%} |"

    if base_val_loader is not None:
        logger.info('=> testing supervised accuracy for base val data')
        acc_base_val = test_supervised(model, base_val_loader)
        md_header += " val acc |"
        md_middle += " ---- |"
        md_content += f" {acc_base_val:.2%} |"

    for loader, split in ((val_loader, 'val'), (test_loader, 'test')):
        logger.info(f'=> testing few shot accuracy for {split} set')
        results = test_few_shot(model, loader, cfg.test.num_episode, cfg.test.n_support)
        for n_support, (acc_mean, confidence_interval) in zip(cfg.test.n_support, results):
            md_header += f" {n_support} shot {split} |"
            md_middle += " ---- |"
            md_content += f" {acc_mean:4.2%} ± {confidence_interval:4.2%} |"

    md_str = '\n'.join([md_header, md_middle, md_content])
    logger.info('-' * 80)
    logger.info(md_str)
    logger.info('-' * 80)


def test_supervised(model, loader):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        t = tqdm(loader, ncols=80)
        for (x, y) in t:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            scores, y = model.forward(x, y)
            pred = scores.argmax(dim=-1)
            correct += (pred == y).float().sum().item()
            total += y.shape[0]

            acc = correct / total
            t.set_postfix(acc=f'{acc:4.2%}')

    logger.info(f'supervised accuracy: {acc:4.2%}')
    return correct / total


# random select data to test
def sample_task(label2feats, n_way, n_support, n_query):
    class_list = label2feats.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = label2feats[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]])
                      for i in range(n_support + n_query)])  # stack each batch
    z_all = torch.from_numpy(np.array(z_all))
    return z_all


def test_few_shot(model, loader, num_episode, nums_support):
    label2feats = extract_feature(model, loader)

    results = []
    for n_support in nums_support:
        logger.info(f"=> test {n_support} shot accuracy")
        if not is_meta_methods():
            # need to rebuild model for baseline methods
            model = build_model(n_support, training=False)
        else:
            # for meta methods, only need to change the n_support parameter
            model.n_support = n_support
        model.eval()

        t = trange(num_episode, desc='testing', ncols=80)
        acc_all = []

        for _ in t:
            z_all = sample_task(label2feats, cfg.test.n_way, n_support, cfg.test.n_query).cuda()
            y = get_few_shot_label(cfg.test.n_way, cfg.test.n_query).cuda()
            scores, y = model(z_all, y)
            acc = accuracy(scores, y).item()
            acc_all.append(acc)

            t.set_postfix(acc=np.mean(acc_all))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        confidence_interval = 1.96 * acc_std / np.sqrt(num_episode)
        logger.info(f'{n_support} shot accuracy: {acc_mean:4.2%}±{confidence_interval:4.2%}')

        results.append((acc_mean, confidence_interval))

    return results


def load_checkpoint(model, optimizer, resume):
    # determin resume file
    resume_file = None
    if os.path.isfile(resume):
        resume_file = resume
    elif resume.startswith('epoch_'):
        resume_file = get_assigned_file(
            cfg.misc.checkpoint_dir, resume.split('_')[1])
    elif resume == "last":
        resume_file = get_resume_file(cfg.misc.checkpoint_dir)
    elif resume == "best":
        resume_file = get_best_file(cfg.misc.checkpoint_dir)

    assert resume == "" or resume_file is not None, f"resume `{resume}` is not valid"

    resume_epoch, best_acc, acc = -1, 0, 0
    if resume_file is not None:
        logger.info(f"=> loading checkpoint from: {resume_file}")
        ckpt = torch.load(resume_file)
        resume_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        acc = ckpt.get('acc', 0)
        best_acc = ckpt.get('best_acc', 0)

    return model, optimizer, resume_epoch, best_acc, acc


def get_scheduler(optimizer, n_iter_per_epoch):
    if cfg.train.lr_scheduler == "warmup_cosine":
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer, eta_min=0.000001,
            T_max=(cfg.train.stop_epoch - cfg.train.warmup_params.epoch) * n_iter_per_epoch)
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.train.warmup_params.multiplier,
            warmup_epoch=cfg.train.warmup_params.epoch * n_iter_per_epoch,
            after_scheduler=cosine_scheduler)
    else:
        scheduler = None
    return scheduler


def get_optimizer(model):
    # optimizer
    if cfg.train.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **cfg.train.adam_params)
    elif cfg.train.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **cfg.train.adam_params)
    elif cfg.train.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **cfg.train.sgd_params)
    else:
        raise ValueError(f'Unsupported optimization: {cfg.train.optimization}')
    return optimizer


def is_meta_methods():
    return cfg.method.metric in models.meta_metrics.__all__

def build_model(n_support, training=True):
    flatten = cfg.method.metric != 'RelationNet'
    backbone = backbones.__dict__[cfg.method.backbone](flatten=flatten)
    if is_meta_methods():
        model = models.Meta(backbone=backbone,
                            n_support=n_support,
                            metric_type=cfg.method.metric,
                            metric_params=cfg.method.metric_params)
        if cfg.method.backbone == 'wrn_28_10':
            raise ValueError(f"training meta methods on wrn_28_10 is not supported now")
    elif training:
        model = models.BaselineTrain(backbone=backbone,
                                     num_class=cfg.dataset.num_class,
                                     metric_type=cfg.method.metric,
                                     metric_params=cfg.method.metric_params)
        if cfg.method.backbone == 'wrn_28_10':
            model = nn.DataParallel(model)
            model.calculate_loss = model.module.calculate_loss
    else:
        model = models.BaselineFinetune(n_way=cfg.test.n_way,
                                        n_support=n_support,
                                        metric_type=cfg.method.metric,
                                        metric_params=cfg.method.metric_params_test,
                                        finetune_params=cfg.test.finetune_params)

    return model.cuda()


def main():
    # build model
    model = build_model(n_support=cfg.train.meta.n_support, training=True)
    optimizer = get_optimizer(model)

    # load checkpoint
    model, optimizer, resume_epoch, best_acc, _ = load_checkpoint(model, optimizer, cfg.misc.resume)

    # data loader
    set_dataset = is_meta_methods()
    base_loader = get_loader(cfg.dataset.base_file, cfg.train.batch_size, train=True, set_dataset=set_dataset)
    if cfg.dataset.base_val_file != "":
        base_val_loader = get_loader(cfg.dataset.base_val_file, cfg.test.batch_size,
                                     train=False, set_dataset=set_dataset)
    else:
        base_val_loader = None
    val_loader = get_loader(cfg.dataset.val_file, cfg.test.batch_size, train=False)
    test_loader = get_loader(cfg.dataset.novel_file, cfg.test.batch_size, train=False)

    # test only
    if cfg.misc.evaluate:
        return test_few_shot(model, test_loader, cfg.test.num_episode, nums_support=cfg.test.n_support)

    scheduler = get_scheduler(optimizer, len(base_loader))

    summary_writer = SummaryWriter(cfg.misc.log_dir)
    for epoch in range(resume_epoch + 1, cfg.train.stop_epoch):
        train(model, base_loader, optimizer, summary_writer, epoch, scheduler)

        # validate and save checkpoint
        if (epoch + 1) % cfg.val.freq == 0 or ((epoch + 1) == cfg.train.stop_epoch):
            results = test_few_shot(model, val_loader, cfg.val.num_episode, nums_support=(cfg.val.n_support,))
            acc = results[0][0]
            summary_writer.add_scalar('val_acc_epoch', acc, epoch)
            summary_writer.add_scalar('val_acc', acc, epoch * len(base_loader))

            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            state = {
                'epoch': epoch,
                'state': model.state_dict(),
                'acc': acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join(cfg.misc.checkpoint_dir, f'{epoch}.tar')
            logger.info(f'=> saving checkpoint to {filename}')
            torch.save(state, filename)
            if is_best:
                best_file = os.path.join(cfg.misc.checkpoint_dir, 'best_model.tar')
                logger.info(f'=> best accuracy, saving to {best_file}')
                shutil.copyfile(filename, best_file)

    logger.info('=> testing accuracy of last model')
    test_all(model, base_loader, base_val_loader, val_loader, test_loader)

    if os.path.isfile(os.path.join(cfg.misc.checkpoint_dir, 'best_model.tar')):
        # release GPU memory used by benchmark to avoid OOM
        torch.cuda.empty_cache()
        model, _, resume_epoch, best_acc, _ = load_checkpoint(model, optimizer, 'best')
        logger.info(f'=> testing accuracy of best model in {resume_epoch} epoch with best validate accuracy {best_acc}')
        test_all(model, base_loader, base_val_loader, val_loader, test_loader)


if __name__ == '__main__':
    # setup logger
    logger = setup_logger(output=cfg.misc.output_dir, name="few_shot")
    path = os.path.join(cfg.misc.output_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(cfg)

    # for reproducibility
    np.random.seed(cfg.misc.rng_seed)
    torch.manual_seed(cfg.misc.rng_seed)
    torch.backends.cudnn.deterministic = True
    # for efficient
    torch.backends.cudnn.benchmark = True

    main()
