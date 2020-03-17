# Ref: https://www.gwern.net/Faces#discriminator-ranking

# import os
import pickle
import numpy as np
import cv2
# import dnnlib
import dnnlib.tflib as tflib
# import sys
import argparse
import PIL.Image


def main(args):
    tflib.init_tf()
    _G, D, _Gs = pickle.load(open(args.model, "rb"))

    for file_path in args.images:
        # print(file_path)
        img = np.asarray(PIL.Image.open(file_path))
        img = img.reshape(1, 3, 512, 512)

        # img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # See dataset_tool.display, the image input must be RGB and NCHW
        # But the img read by cv2 is BGR and NHWC
        # img = img.transpose((2, 0, 1))  # HWC -> CHW
        # img = img[::-1, :, :]  # BGR -> RGB
        # img = np.expand_dims(img, axis=0)
        # img = img.reshape(1, 3, 512, 512)
        score = D.run(img, None)
        print(file_path, score[0][0])
        with open(args.output, 'a') as fout:
            fout.write(file_path + ' ' + str(score[0][0]) + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='.pkl model')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--output', type=str, default='rank.txt')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())