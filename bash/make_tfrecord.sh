#!/usr/bin/env bash
usecuda100
source /storage04/users/shaoxuning/.env/venv_py3.6_tf1.14gpu/bin/activate
IMG_DIR=/storage04/users/shaoxuning/datasets/acg-icrawler/kagura_crop_waifu4x_aug_20200227

python ../dataset_tool.py create_from_images ../datasets/kagura_waifu4x_20200227 $IMG_DIR

