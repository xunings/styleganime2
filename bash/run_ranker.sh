#!/usr/bin/env bash

usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
export SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/results/generate_images/kagura_psi0d8_1001to1100_20200227
export SCRIPT=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/misc/ranker.py
export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl

find $SRC_DIR -type f -name "*.png" | xargs python3 $SCRIPT --model $MODEL --images | tee rank.txt

cat rank.txt | sort --field-separator ' ' --key 2 --numeric-sort | head -25 | cut -d ' ' -f1 | xargs -n1 -I{} cp {} rank/worst/

cat rank.txt | sort --field-separator ' ' --key 2 --numeric-sort | head -25 | cut -d ' ' -f1 | xargs -n1 -I{} cp {} rank/worst/

