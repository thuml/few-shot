#!/bin/bash

set -x
set -e

python main.py --config configs/CUB.yml \
    --supp 5-shot \
    method.metric neg-softmax \
    method.metric_params.margin -0.005 \
    method.metric_params.scale_factor 5.0 \
    train.adam_params.weight_decay 3e-3
