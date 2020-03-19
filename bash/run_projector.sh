#!/usr/bin/env bash
usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1

SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
MODEL=2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl
INPUT_DIR=/storage04/users/shaoxuning/pictures/face/selfie2anime_test/crop_512_margin0d5/
# MODEL=stylegan2-ffhq-config-f.pkl

find "$INPUT_DIR" -type f | xargs -n100 \
python "$SRC_DIR"/run_projector.py project-real-images --network="$SRC_DIR"/models/"$MODEL" \
--dataset=dummy --data-dir=dummy \
--result-dir "$SRC_DIR"/results/projector \
--num-snapshots=100 \
--input_images

# --dataset=selfie2anime_test --data-dir="$SRC_DIR"/datasets \
