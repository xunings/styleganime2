#!/usr/bin/env bash

usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
export SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/results/generate_images/kagura-010060-20200227/kagura_psi0d8_20200227
export SCRIPT=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/misc/ranker.py
export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl
# export MODEL=/storage04/users/shaoxuning/projects/acgface/try/stylegan2/models/kagura-010060-20200227.pkl
export OUTFILE=rank/rank.txt
export OUTFILE_SORTED=rank/rank_sorted.txt
export WORSTDIR=rank/worst
export BESTDIR=rank/best
export N_RUNS_PER_IMG=5
export SELECT_NUMS="25 100"

rm $OUTFILE $OUTFILE_SORTED

find "$SRC_DIR" -type f -name "*.png" | \
xargs python3 "$SCRIPT" --model "$MODEL" --num_runs_per_img "$N_RUNS_PER_IMG" --output "$OUTFILE" --images

cat "$OUTFILE" | sort --field-separator ' ' --key 2 --numeric-sort > ${OUTFILE_SORTED}

for SELECT_NUM in ${SELECT_NUMS}; do
    echo 'Finding the worst' "$SELECT_NUM"
    mkdir "$WORSTDIR"_"$SELECT_NUM"
    cat ${OUTFILE_SORTED} | head -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$WORSTDIR"_"$SELECT_NUM"
    echo 'Finding the best' "$SELECT_NUM"
    mkdir "$BESTDIR"_"$SELECT_NUM"
    cat ${OUTFILE_SORTED} | tail -"$SELECT_NUM" | cut -d ' ' -f1 | xargs -n1 -I{} cp {} "$BESTDIR"_"$SELECT_NUM"
done
