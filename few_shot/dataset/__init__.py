import torch
from torchvision import transforms

from few_shot.config import cfg
from .simple_dataset import SimpleDataset
from .set_dataset import SetDataset, EpisodicBatchSampler


def get_loader(json_file, batch_size, train=True, set_dataset=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.257, 0.276])
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.method.image_size),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        size = int(cfg.method.image_size * 1.15)
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(cfg.method.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    if set_dataset:
        dataset = SetDataset(json_file, cfg.train.meta.n_support + cfg.train.meta.n_query, transform)
        sampler = EpisodicBatchSampler(len(dataset), cfg.test.n_way, cfg.train.meta.num_episode)
        loader_params = dict(shuffle=False, batch_sampler=sampler, batch_size=1)
    else:
        dataset = SimpleDataset(json_file, transform)
        loader_params = dict(shuffle=train, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(dataset, num_workers=cfg.misc.num_workers, pin_memory=True, **loader_params)
    return loader
