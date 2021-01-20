from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, itertools, pathlib
import numpy as np
# from PIL import ImageFile
import tensorflow as tf
# from keras_preprocessing.image import ImageDataGenerator
from data_generator.image import ImageDataGenerator

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
# BATCH_SIZE = 32
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
#
# data_dir = pathlib.Path('./data/segmentation')
# class_name_ignore = ['.DS_Store']
# CLASS_NAMES = np.array([item.name for item in data_dir.glob('train/img/*') if item.name not in class_name_ignore])
#
#
# def random_flip(x: tf.Tensor) -> tf.Tensor:
#     x = tf.image.random_flip_left_right(x)
#
#     return x
#
#
# def random_zoom(x: tf.Tensor, zoom_range: tuple) -> tf.Tensor:
#     x = tf.keras.preprocessing.image.random_zoom(
#         x,
#         zoom_range,
#         row_axis=1,
#         col_axis=0,
#         channel_axis=2,
#         fill_mode='reflect',
#         interpolation_order=1
#     )
#
#     return x

# def preprocess(img, mean, std, label, normalize_label=False):
#     out_img = img / img.max() # scale to [0,1]
#     out_img = (out_img - np.array(mean).reshape(1,1,3)) / np.array(std).reshape(1,1,3)
#
#     if normalize_label:
#         if np.unique(label).size > 2:
#             print ('WRANING: the label has more than 2 classes. Set normalize_label to False')
#         label = label / label.max() # if the loaded label is binary has only [0,255], then we normalize it
#     return out_img, label.astype(np.int32)
#
#
# def deprocess(img, mean, std, label):
#     out_img = img / img.max() # scale to [0,1]
#     out_img = (out_img * np.array(std).reshape(1,1,3)) + np.array(std).reshape(1,1,3)
#     out_img = out_img * 255.0
#
#     return out_img.astype(np.uint8), label.astype(np.uint8)
#

"""
data_weigthed_loader consider the label for specific treatmentss
"""


def pre_process_image(image):
    image /= 255.
    image -= 0.5
    image *= 2.

    return image


def data_loader(path, batch_size, imSize,
                ignore_val=44, pos_val=255, neg_val=155, pos_class=[0,1], neg_class=[2]):
    # pos_class and neg_class in the folder name for keras ImageDataGenerator input
    # 0,1,2 are low, high, normal

    def imerge(img_gen, mask_gen):
        for (imgs, img_labels), (mask, mask_labels) in itertools.zip_longest(img_gen, mask_gen):
            # compute weight to ignore particular pixels
            # mask = np.expand_dims(mask[:,:,:,0], axis=3)
            mask = mask[:, :, :, 0]
            weight = np.ones(mask.shape, np.float32)
            weight[mask == ignore_val] = 0.5 # this is set by experience

            # In mask, ignored pixel has value ignore_val.
            # The weight of these pixel is set to zero, so they do not contribute to loss
            # The returned mask is still binary.
            # compute per sample
            for c, mask_label in enumerate(mask_labels):
                assert(mask_labels[c] == img_labels[c])
                mask_pointer = mask[c]
                if mask_label in pos_class:
                    mask_pointer[mask_pointer < pos_val] = 0
                    assert(np.where(mask_pointer == neg_val)[0].size == 0)
                    mask_pointer[mask_pointer==pos_val] = 1
                elif mask_label in neg_class:
                    mask_pointer[mask_pointer < neg_val] = 0
                    assert(np.where(mask_pointer == pos_val)[0].size == 0)
                    mask_pointer[mask_pointer==neg_val] = 0
                else:
                    print ('WARNING: mask beyond the expected class range')
                    mask_pointer /= 255.0

                mask_pointer[mask_pointer==ignore_val] = 0

            '''
            for c, mask_label in enumerate(mask_labels):
                assert(mask_labels[c] == img_labels[c])
                mask_pointer = mask[c]
                if mask_label in pos_class:
                    mask_pointer[mask_pointer != pos_val] = 0
                    assert(np.where(mask_pointer == neg_val)[0].size == 0)
                    mask_pointer[mask_pointer == pos_val] = 1
                elif mask_label in neg_class:
                    mask_pointer[:, :] = 0
                    assert(np.where(mask_pointer == pos_val)[0].size == 0)
                    # mask_pointer[mask_pointer == neg_val] = 0
                else:
                    print ('WARNING: mask beyond the expected class range')
                    mask_pointer /= 255.0

                mask_pointer[mask_pointer == ignore_val] = 0
            '''
            assert set(np.unique(mask)).issubset([0, 1])
            # assert set(np.unique(weight)).issubset([0, 1])

            # img, mask = preprocess(imgs, mean, std, mask)

            # yield imgs, mask, weight, img_labels
            imgs = pre_process_image(imgs)

            # imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
            # mask = tf.convert_to_tensor(mask, dtype=tf.int32)
            # weight = tf.convert_to_tensor(weight, dtype=tf.float32)

            mask = mask.astype(np.int32)

            yield imgs, mask, weight
            # yield tf.data.Dataset.from_tensor_slices((imgs, mask, weight))

    train_data_gen_args = dict(
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='reflect')

    seed = 1234
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/img',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                seed=seed)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/groundTruth',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed)

    test_image_datagen = ImageDataGenerator().flow_from_directory(
                                path+'test/img',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                seed=seed)
    test_mask_datagen = ImageDataGenerator().flow_from_directory(
                                path+'test/groundTruth',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed)

    # train_dataset = tf.data.Dataset.from_generator(
    #     generator=(lambda: imerge(train_image_datagen, train_mask_datagen)),
    #     output_types=(tf.float32, tf.int32, tf.float32)
    # )
    #
    # test_data_set = tf.data.Dataset.from_generator(
    #     generator=(lambda: imerge(test_image_datagen, test_mask_datagen)),
    #     output_types=(tf.float32, tf.int32, tf.float32)
    # )

    train_data_set = imerge(train_image_datagen, train_mask_datagen)
    test_data_set = imerge(test_image_datagen, test_mask_datagen)

    return train_data_set, test_data_set, train_image_datagen.samples, test_image_datagen.samples


