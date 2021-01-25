'''
The value of each pixels in the label decides a specific label.
255: positive
155: negative
44: ignore
'''

# ref: https://medium.com/swlh/dump-keras-imagedatagenerator-start-using-tensorflow-tf-data-part-1-a30330bdbca9

import pathlib
import glob
import tensorflow as tf

from opts import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

ground_truth_relative_path = ''
if os.name == 'nt':
    ground_truth_relative_path += os.sep

ground_truth_relative_path += os.sep + 'groundTruth'


# ref: https://stackoverflow.com/questions/38376478/changing-the-scale-of-a-tensor-in-tensorflow
def normalize_tensor(tensor):
    return tf.divide(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )


def pre_process_image(image):
    image = normalize_tensor(image)
    image = tf.subtract(image, [0.5])
    image = tf.multiply(image, [2.])

    return image


def pre_process_ground_truth(ground_truth, positive_value=255, ignore_value=44):
    ground_truth = tf.cast(ground_truth, dtype=tf.int32)[..., 0]

    label = tf.where(ground_truth == positive_value, 1, 0)
    weight = tf.where(ground_truth == ignore_value, 0.5, 1)

    return label, weight


def load_image(image_path, image_size):
    ground_truth_path = tf.strings.regex_replace(image_path, "[\\\/](img)", ground_truth_relative_path)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])

    ground_truth = tf.io.read_file(ground_truth_path)
    ground_truth = tf.image.decode_png(ground_truth, channels=1)
    ground_truth = tf.image.resize(ground_truth, [image_size, image_size])

    return image, ground_truth


def load_train_data(image_path, image_size, image_zoom_range, is_flip_horizontal=False,
                    positive_value=255, negative_value=155, ignore_value=44,
                    image_class_positive=tuple([0, 1]), image_class_negative=tuple([2])):

    # wraps with tf.function to crop on the same region in the each of images
    # ref: https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=nightly
    def _image_random_zoom(image, ground_truth, image_original_size, image_zoom_range: float):
        image_crop_coefficient = tf.random.uniform([1], minval=1.0 - image_zoom_range, maxval=1)
        image_crop_size = tf.multiply(tf.cast(image_original_size, dtype=tf.float32), image_crop_coefficient)
        image_crop_size = tf.cast(image_crop_size, dtype=tf.int32)

        original_image_crop_size = tf.concat([image_crop_size, image_crop_size, [3]], axis=0)
        label_image_crop_size = tf.concat([image_crop_size, image_crop_size, [1]], axis=0)

        image = tf.image.random_crop(image, size=original_image_crop_size, seed=42)
        image = tf.image.resize(image,
                                size=(image_original_size, image_original_size),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        ground_truth = tf.image.random_crop(ground_truth, size=label_image_crop_size, seed=42)
        ground_truth = tf.image.resize(ground_truth,
                                       size=(image_original_size, image_original_size),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, ground_truth

    image, ground_truth_image = load_image(image_path, image_size=image_size)

    image, label = _image_random_zoom(image, ground_truth_image,
                                      image_original_size=image_size,
                                      image_zoom_range=image_zoom_range)

    # image random flip
    if is_flip_horizontal and tf.random.uniform([1]) < 0.5:
        image = tf.image.flip_left_right(image)
        ground_truth_image = tf.image.flip_left_right(ground_truth_image)

    # pre process
    image = pre_process_image(image)
    label, weight = pre_process_ground_truth(ground_truth_image,
                                             positive_value=positive_value,
                                             ignore_value=ignore_value)

    return image, label, weight


def load_test_data(image_path, image_size, positive_value=255, ignore_value=44):
    image, label = load_image(image_path, image_size=image_size)

    image = pre_process_image(image)
    label, weight = pre_process_ground_truth(label,
                                             positive_value=positive_value,
                                             ignore_value=ignore_value)

    return image, label, weight


def prepare_for_training(data_set: tf.data.Dataset, batch_size, cache_path=None, shuffle_buffer_size=1000):
    if cache_path != '':
        cache_filename = 'dataset_train.tfcache'
        data_set = data_set.cache(''.join([cache_path, '/', cache_filename]))

    data_set = data_set.shuffle(buffer_size=shuffle_buffer_size)
    # repeat forever
    data_set = data_set.repeat()
    data_set = data_set.batch(batch_size=batch_size)
    # `prefetch` lets the dataset fetch batches in the background
    # while the model is training.
    data_set = data_set.prefetch(buffer_size=AUTOTUNE)

    return data_set


def prepare_for_testing(data_set: tf.data.Dataset, batch_size, cache_path=''):
    if cache_path != '':
        cache_filename = 'dataset_test.tfcache'
        data_set = data_set.cache(''.join([cache_path, '/', cache_filename]))

    data_set = data_set.repeat()
    data_set = data_set.batch(batch_size=batch_size)

    return data_set


def load_data_set(data_root_path, batch_size, image_size, train_iter_epoch_ratio, is_use_cache, cache_path):
    if not is_use_cache:
        cache_path = ''

    data_root_path = pathlib.Path(data_root_path)
    image_dir = 'img/*/*.png'
    train_image_path_str = str(data_root_path / str('train/'+image_dir))
    train_data_len = len(glob.glob(train_image_path_str))
    tf.print(f'[INFO] train data: #{train_data_len}')
    train_batch_per_epoch_num = int(train_data_len / batch_size * train_iter_epoch_ratio)

    train_image_path_list = tf.data.Dataset.list_files(train_image_path_str)

    # map with additional params
    # ref: https://stackoverflow.com/questions/46263963/how-to-map-a-function-with-additional-parameter-using-the-new-dataset-api-in-tf1
    train_data_set = train_image_path_list.map(lambda image_path: load_train_data(image_path,
                                                                                  image_size=image_size,
                                                                                  image_zoom_range=0.2,
                                                                                  is_flip_horizontal=True),
                                               num_parallel_calls=AUTOTUNE)

    train_data_set = prepare_for_training(train_data_set,
                                          batch_size=batch_size,
                                          cache_path=cache_path,
                                          shuffle_buffer_size=1000)

    test_image_path_str = str(data_root_path / str('test/'+image_dir))
    test_data_len = len(glob.glob(test_image_path_str))
    tf.print(f'[INFO] test data: #{test_data_len}')
    test_batch_per_epoch_num = int(test_data_len / batch_size)

    test_image_path_list = tf.data.Dataset.list_files(test_image_path_str)
    test_data_set = test_image_path_list.map(lambda image_path: load_test_data(image_path,
                                                                               image_size=image_size))

    test_data_set = prepare_for_testing(test_data_set,
                                        cache_path=cache_path,
                                        batch_size=batch_size)

    return train_data_set, test_data_set, train_batch_per_epoch_num, test_batch_per_epoch_num
