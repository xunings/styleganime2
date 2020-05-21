#!/usr/bin/env bash

usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
export SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/results/generate_images/kagura-010060-20200227/kagura_psi0d8_20200227
export SCRIPT=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/misc/ranker.py
export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl
# export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/kagura-010060-20200227.pkl
export RESULTDIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/bash/rank/kagura/shuffle5_skylionmodel
export N_SHUFFLES=5
export SELECT_NUMS="25 100"

# Discriminator ranking
export OUTFILE="$RESULTDIR"/rank.txt
export OUTFILE_SORTED="$RESULTDIR"/rank_sorted.txt
# Process 500 images each time.
# Note: xargs will auto split the input if it is too long, leading to unexpected behavior.
find "$SRC_DIR" -type f -name "*.png" | sort | \
    xargs -n500 python3 "$SCRIPT" --model "$MODEL" --num_shuffles "$N_SHUFFLES" --output "$OUTFILE" --images
cat "$OUTFILE" | sort --field-separator ' ' --key 2 --numeric-sort > ${OUTFILE_SORTED}

# Find the best and worst images
for SELECT_NUM in ${SELECT_NUMS}; do
    export WORSTDIR="$RESULTDIR"/worst_"$SELECT_NUM"
    export BESTDIR="$RESULTDIR"/best_"$SELECT_NUM"
    mkdir "$WORSTDIR" "$BESTDIR"
    echo 'Finding the worst' "$SELECT_NUM"
    cat ${OUTFILE_SORTED} | head -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$WORSTDIR"
    echo 'Finding the best' "$SELECT_NUM"
    cat ${OUTFILE_SORTED} | tail -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$BESTDIR"
done
