# Few-shot

`Few-shot` is a lightweight library that implements state-of-the-art few-shot learning algorithms. 

 In the current version, the following algorithms are included. We welcome other researchers to contribute to this framework.

* Neg-Cosine/Neg-Softmax: [Negative Margin Matters: Understanding Margin in Few-shot Classification](https://arxiv.org/abs/2003.12060).
* MatchingNet: [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080).
* ProtoNet: [Prototypical networks for few-shot learning](https://arxiv.org/abs/1703.05175).
* RelationNet: [Learning to compare: Relation network for few-shot learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html).
* Baseline/Baseline++: [A closer look at few-shot classification](https://arxiv.org/abs/1904.04232).

## Main Results

The few-shot classification accuracy on the novel classes with ResNet-18 as the backbone is listed bellowing:

| Method     | Mini-ImageNet<br/>1 - shot | Mini-ImageNet<br/>5 - shot | CUB<br/>1 - shot | CUB<br/>5 - shot | Mini -> CUB<br/>5-shot |
| :-------: | :-------: | :-------: | :----------: | :----------: | :-------------: |
| MatchingNet        | 52.91+-0.88 | 68.88+-0.69 | 72.36+-0.90 | 83.64+-0.60 | 53.07+-0.74 |
| MatchingNet (Ours) | 58.84+-0.80 | 72.14+-0.67 | 73.38+-0.89 | 85.10+-0.52 | 59.18+-0.69 |
| ProtoNet           | 54.16+-0.82 | 73.68+-0.65 | 71.88+-0.91 | 87.42+-0.48 | 62.02+-0.70 |
| ProtoNet (Ours)    | 57.35+-0.81 | 75.80+-0.64 | 73.44+-0.84 | 88.29+-0.48 | 61.43+-0.78 |
| RelationNet        | 52.48+-0.86 | 69.83+-0.68 | 67.59+-1.02 | 82.75+-0.58 | 57.71+-0.73 |
| RelationNet (Ours) | 54.11+-0.81 | 70.13+-0.70 | 70.56+-0.94 | 84.61+-0.56 | 54.16+-0.70 |
| Baseline           | 51.75+-0.80 | 74.27+-0.63 | 65.51+-0.87 | 82.85+-0.55 | 65.57+-0.70 |
| Baseline (Ours)    | 52.41+-0.77 | 74.82+-0.68 | 63.89+-0.87 | 82.69+-0.55 | 64.56+-0.73 |
| Baseline++         | 51.87+-0.77 | 75.68+-0.63 | 67.02+-0.90 | 83.58+-0.54 | 62.04+-0.76 |
| Baseline++ (Ours)  | 55.89+-0.82 | 76.99+-0.61 | 70.93+-0.89 | 88.64+-0.47 | 66.08+-0.69 |
| Neg-Sofmax (Ours)  | 59.02+-0.81 | 78.80+-0.61 | 71.48+-0.83 | 87.30+-0.48 | 69.30+-0.73 |
| Neg-Cosine (Ours)  | 62.33+-0.82 | 80.94+-0.59 | 72.66+-0.85 | 89.40+-0.43 | 67.03+-0.76 |


> Notes: 
> * The results without `Ours` in the above comes from [A closer look at few-shot classification](https://arxiv.org/abs/1904.04232). 
> * Mini -> CUB stands for mini-ImageNet -> CUB

The above results shows that our implementation gets much higher few-shot accuracies than orginal implementation in almost all algorithms and datasets.

You can download some pre-trained model checkpoints of `resnet-18` from [OneDrive](https://1drv.ms/u/s!AsaPPmtCAq08pRM54_CuGPFbfgUz?e=ydjBfW).

## Getting started

### Environment

 - `Anaconda` with `python >= 3.6`
 - `pytorch=1.2.0, torchvison, cuda=9.2`
 - others: `pip install yacs termcolor`

### Datasets

#### CUB

* Change directory to `./data/CUB`
* run `bash ./download_CUB.sh`

#### mini-ImageNet

* Change directory to `./data/miniImagenet`
* run `bash ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out the corresponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

#### mini-ImageNet->CUB

* Finish preparation for CUB and mini-ImageNet, and you are done!

## Train and eval

Run the following commands to train and evaluate:

```bash
python main.py --config [CONFIGFILENAME] \
    method.backbone [BACKBONE] \
    method.image_size [IMAGESIZE] \
    method.metric [Metric] \
    [OPTIONARG]
```

 For additional options, please refer to `./few-shot/config.py`.

We have also provided some scripts to reproduce the results in the table. Please check `./scripts` for details.

## References
This algorithm library is extended from https://github.com/bl0/negative-margin.few-shot, which builds upon several existing publicly available code:

* Framework, Backbone, Method: Closer Look at Few Shot
https://github.com/wyharveychen/CloserLookFewShot
* Backbone(resnet12): MetaOpt
https://github.com/kjunelee/MetaOptNet
* Backbone(wrn_28_10): S2M2_R
https://github.com/nupurkmr9/S2M2_fewshot
