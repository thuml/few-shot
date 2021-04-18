#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet.yml \
    method.metric RelationNet
