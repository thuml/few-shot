#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet.yml \
    --supp wrn-28-10 \
    train.adam_params.weight_decay 0. \
    method.metric_params.margin -0.2 \
    method.backbone wrn_28_10 \
    method.image_size 80 \
    train.batch_size 128