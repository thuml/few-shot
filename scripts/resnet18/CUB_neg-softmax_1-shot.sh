#!/bin/bash

set -x
set -e

python main.py --config configs/CUB.yml \
    --supp 1-shot \
    method.metric neg-softmax \
    method.metric_params.margin -0.05 \
    method.metric_params.scale_factor 5.0 \
    val.n_support 1 \
    train.adam_params.weight_decay 5e-3
