#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet.yml \
    --supp 5-shot_resnet12 \
    method.metric neg-softmax \
    train.adam_params.weight_decay 1e-5 \
    method.metric_params.margin -0.1 \
    method.metric_params.scale_factor 10.0 \
    train.drop_rate 0.1 \
    train.dropblock_size 5 \
    method.backbone resnet12 \
    method.image_size 84 \
