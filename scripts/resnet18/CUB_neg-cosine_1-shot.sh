#!/bin/bash

set -x
set -e

python main.py --config configs/CUB.yml \
    --supp 1-shot \
    val.n_support 1 \
    train.adam_params.weight_decay 5e-4 \
    method.metric_params.margin -0.01 \
    train.drop_rate 0.1 \
    train.dropblock_size 5
