# Ref: https://www.gwern.net/Faces#discriminator-ranking

import pickle
import numpy as np
import cv2
import dnnlib.tflib as tflib
import random
import argparse
import PIL.Image
from training.misc import adjust_dynamic_range


def preprocess(file_path):
    # print(file_path)
    img = np.asarray(PIL.Image.open(file_path))

    # Preprocessing from dataset_tool.create_from_images
    img = img.transpose([2, 0, 1])  # HWC => CHW
    # img = np.expand_dims(img, axis=0)
    img = img.reshape((1, 3, 512, 512))

    # Preprocessing from training_loop.process_reals
    img = adjust_dynamic_range(data=img, drange_in=[0, 255], drange_out=[-1.0, 1.0])
    return img


def main(args):
    random.seed(args.random_seed)
    minibatch_size = args.minibatch_size
    input_shape = (minibatch_size, 3, 512, 512)
    # print(args.images)
    images = args.images
    images.sort()

    tflib.init_tf()
    _G, D, _Gs = pickle.load(open(args.model, "rb"))
    # D.print_layers()

    image_score_all = [(image, []) for image in images]

    # Shuffle the images and process each image in multiple minibatches.
    # Note: networks.stylegan2.minibatch_stddev_layer
    # calculates the standard deviation of a minibatch group as a feature channel,
    # which means that the output of the discriminator actually depends
    # on the companion images in the same minibatch.
    for i_shuffle in range(args.num_shuffles):
        # print('shuffle: {}'.format(i_shuffle))
        random.shuffle(image_score_all)
        for idx_1st_img in range(0, len(image_score_all), minibatch_size):
            idx_img_minibatch = []
            images_minibatch = []
            input_minibatch = np.zeros(input_shape)
            for i in range(minibatch_size):
                idx_img = (idx_1st_img + i) % len(image_score_all)
                idx_img_minibatch.append(idx_img)
                image = image_score_all[idx_img][0]
                images_minibatch.append(image)
                img = preprocess(image)
                input_minibatch[i, :] = img
            output = D.run(input_minibatch, None, resolution=512)
            print('shuffle: {}, indices: {}, images: {}'
                  .format(i_shuffle, idx_img_minibatch, images_minibatch))
            print('Output: {}'.format(output))
            for i in range(minibatch_size):
                idx_img = idx_img_minibatch[i]
                image_score_all[idx_img][1].append(output[i][0])

    with open(args.output, 'a') as fout:
        for image, score_list in image_score_all:
            print('Image: {}, score_list: {}'.format(image, score_list))
            avg_score = sum(score_list)/len(score_list)
            fout.write(image + ' ' + str(avg_score) + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='.pkl model')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--output', type=str, default='rank.txt')
    parser.add_argument('--minibatch_size', type=int, default=4)
    parser.add_argument('--num_shuffles', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())