# Ref: https://www.gwern.net/Faces#discriminator-ranking

import os
import pickle
import numpy as np
import cv2
# import dnnlib
import dnnlib.tflib as tflib
# import sys
import argparse
import PIL.Image
import tensorflow as tf
from training.training_loop import process_reals
from training.misc import adjust_dynamic_range


def preprocess(file_path):
    # print(file_path)
    img = np.asarray(PIL.Image.open(file_path))
    img = img.transpose([2, 0, 1])  # HWC => CHW
    img = img.reshape((1, 3, 512, 512))
    # img = np.expand_dims(img, axis=0)
    return img


def main(args):
    tflib.init_tf()
    input_shape = (1, 3, 512, 512)
    _G, D, _Gs = pickle.load(open(args.model, "rb"))
    D.print_layers()

    '''
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')
        img_processed, _ = process_reals(input_placeholder,
                                         labels=None,
                                         lod=0.0,
                                         mirror_augment=False,
                                         drange_data=[0, 255],
                                         drange_net=[-1.0, 1.0])
        # with tf.device('/gpu:0'):
        #     scores = D.get_output_for(img_processed, None, is_training=False)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    '''

    for file_path in args.images:
        print('Processing: {}'.format(file_path))
        dir_path = os.path.dirname(file_path)
        ext = os.path.splitext(file_path)
        other_images = [os.path.join(dir_path, file) for file in os.listdir(dir_path)
                        if file.endswith(ext) and os.path.join(dir_path, file) != file_path]
        other_images.sort()
        tries = min(5, len(other_images) // 3)

        img = preprocess(file_path)
        score_tries = []

        for i in range(tries):
            img_minibatch = img
            for j in range(3):
                another_img_path = other_images[i*3+j]
                print('Companion image: {}'.format(another_img_path))
                another_img = preprocess(another_img_path)
                img_minibatch = np.concatenate((img_minibatch, another_img), axis=0)
            # img_minibatch = np.random.randint(low=0, high=255, size=(4, 3, 512, 512))
            img_minibatch = adjust_dynamic_range(data=img_minibatch, drange_in=[0, 255], drange_out=[-1.0, 1.0])
            score = D.run(img_minibatch, None, resolution=512)
            print('score: {}'.format(score))
            score_tries.append(score[0][0])

        score_this_img = sum(score_tries)/len(score_tries)
        print(file_path, score_this_img)


        # img = sess.run(img_processed, feed_dict={input_placeholder: img})
        # img = np.expand_dims(img, axis=0)
        # img_dummy = np.zeros((3, 3, 512, 512))
        # img_dummy = np.random.randint(low=0, high=255, size=(3, 3, 512, 512))
        # img = np.concatenate((img, img_dummy), axis=0)
        # img = adjust_dynamic_range(data=img, drange_in=[0, 255], drange_out=[-1.0, 1.0])

        # img = np.tile(img, (4, 1, 1, 1))

        # Not in twdne, but I think it is necessary to adjust the range here.
        # See training_loop.process_reals
        # img = img.astype('float32')
        # scale = 2.0 / 255.0
        # bias = -1.0
        # [0, 255] to [-1, 1]
        # img = img * scale + bias

        # img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # See dataset_tool.display, the image input must be RGB and NCHW
        # But the img read by cv2 is BGR and NHWC
        # img = img.transpose((2, 0, 1))  # HWC -> CHW
        # img = img[::-1, :, :]  # BGR -> RGB
        # img = np.expand_dims(img, axis=0)
        # img = img.reshape(1, 3, 512, 512)
        # score = D.run(img, None, resolution=512, mbstd_group_size = 0)
        # score = D.run(img, None, resolution=512)
        # score = sess.run(scores, feed_dict={input_placeholder: img})
        # print(file_path, score[0][0])
        with open(args.output, 'a') as fout:
            fout.write(file_path + ' ' + str(score_this_img) + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='.pkl model')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--output', type=str, default='rank.txt')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())