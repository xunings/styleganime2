#!/usr/bin/env bash

usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
# export SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/results/generate_images/kagura-010060-20200227/kagura_psi0d8_1001to1100_20200227
export SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/results/generate_images/kagura-010060-20200227/kagura_psi0d8_20200227
export SCRIPT=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/misc/ranker.py
export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl
# export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/kagura-010060-20200227.pkl
export OUTFILE=rank/rank.txt
export OUTFILE_SORTED=rank/rank_sorted.txt
export WORSTDIR=rank/worst/
export BESTDIR=rank/best/
export SELECT_NUM=100

find "$SRC_DIR" -type f -name "*.png" | xargs python3 "$SCRIPT" --model "$MODEL" --output "$OUTFILE" --images

cat "$OUTFILE" | sort --field-separator ' ' --key 2 --numeric-sort > ${OUTFILE_SORTED}

mkdir "$WORSTDIR"
mkdir "$BESTDIR"

echo 'Finding the worst...'
cat ${OUTFILE_SORTED} | head -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$WORSTDIR"

echo 'Finding the best...'
cat ${OUTFILE_SORTED} | tail -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$BESTDIR"

