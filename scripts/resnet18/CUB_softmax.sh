#!/bin/bash

set -x
set -e

python main.py --config configs/CUB.yml \
    method.metric softmax \
    train.adam_params.weight_decay 3e-3