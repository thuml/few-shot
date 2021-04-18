#!/bin/bash

set -x
set -e

python main.py --config configs/CUB.yml \
    --supp conv4 \
    method.backbone conv4 \
    method.image_size 80 \
    test.batch_size 256 \
    method.metric RelationNet