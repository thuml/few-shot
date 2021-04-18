#!/bin/bash

set -x
set -e

python main.py --config configs/mini2CUB.yml \
    --supp 5-shot \
    method.metric neg-softmax \
    train.adam_params.weight_decay 1e-6 \
    method.metric_params.margin -0.005 \
    method.metric_params.scale_factor 5.0
