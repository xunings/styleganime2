#!/usr/bin/env bash
usecuda100
export CUDA_VISIBLE_DEVICES=1
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
SRC_DIR=/storage04/users/shaoxuning/projects/acgface/try/stylegan2
MODEL=kagura-010060-20200227.pkl

python3 "$SRC_DIR"/run_generator.py generate-images --seeds=1-1000 --truncation-psi=0.6 \
    --network="$SRC_DIR"/models/"$MODEL" --result-dir="$SRC_DIR"/results/generate_images
python3 "$SRC_DIR"/run_generator.py generate-images --seeds=1001-2000 --truncation-psi=0.8 \
    --network="$SRC_DIR"/models/"$MODEL" --result-dir="$SRC_DIR"/results/generate_images
python3 "$SRC_DIR"/run_generator.py generate-images --seeds=2001-3000 --truncation-psi=1.1 \
    --network="$SRC_DIR"/models/"$MODEL" --result-dir="$SRC_DIR"/results/generate_images

