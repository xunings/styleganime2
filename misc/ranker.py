# Ref: https://www.gwern.net/Faces#discriminator-ranking

import os
import pickle
import numpy as np
import cv2
import dnnlib.tflib as tflib
# import sys
import argparse
import PIL.Image
from training.misc import adjust_dynamic_range


def preprocess(file_path):
    # print(file_path)
    img = np.asarray(PIL.Image.open(file_path))
    img = img.transpose([2, 0, 1])  # HWC => CHW
    img = img.reshape((1, 3, 512, 512))
    # img = np.expand_dims(img, axis=0)
    img = adjust_dynamic_range(data=img, drange_in=[0, 255], drange_out=[-1.0, 1.0])
    return img


def main(args):

    minibatch_size = 4
    input_shape = (minibatch_size, 3, 512, 512)
    # print(args.images)
    images = args.images
    images.sort()
    # Number of minibatches per image
    # Each minibatch contains the target image plus (minibatch_size - 1) other images.
    n_minibatches = min(args.num_runs_per_img,
                        len(images) - 1 // (minibatch_size - 1))

    tflib.init_tf()
    _G, D, _Gs = pickle.load(open(args.model, "rb"))
    # D.print_layers()

    for i in range(len(images)):
        print('Processing: {}'.format(images[i]))
        img = preprocess(images[i])
        idx_another_img = i
        scores_this_img = []
        for _ in range(n_minibatches):
            img_minibatch = np.zeros(input_shape)
            img_minibatch[0,:] = img
            for j in range(minibatch_size - 1):
                idx_another_img = idx_another_img+1 if idx_another_img+1 < len(images) else 0
                print('Companion: {}'.format(images[idx_another_img]))
                another_img = preprocess(images[idx_another_img])
                img_minibatch[j+1, :] = another_img
            # img_minibatch = adjust_dynamic_range(data=img_minibatch, drange_in=[0, 255], drange_out=[-1.0, 1.0])
            output = D.run(img_minibatch, None, resolution=512)
            print('output: {}'.format(output))
            scores_this_img.append(output[0][0])
        score_this_img = sum(scores_this_img) / len(scores_this_img)
        print(images[i], score_this_img)

        with open(args.output, 'a') as fout:
            fout.write(images[i] + ' ' + str(score_this_img) + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='.pkl model')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--output', type=str, default='rank.txt')
    parser.add_argument('--num_runs_per_img', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())