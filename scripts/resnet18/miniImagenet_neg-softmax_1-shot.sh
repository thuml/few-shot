#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet.yml \
    --supp 1-shot \
    method.metric neg-softmax \
    val.n_support 1 \
    train.adam_params.weight_decay 1e-4 \
    method.metric_params.margin -0.3 \
    method.metric_params.scale_factor 5.0
