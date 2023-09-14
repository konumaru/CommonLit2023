#!/bin/bash

python src/feature.py && \
python src/train.py --multirun model=xgb,lgbm && \
python src/stacking.py
