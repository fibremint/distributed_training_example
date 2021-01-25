import tensorflow as tf
import datetime

from opts import *
from dataset_generator import load_data_set
from model import UNet
from utility import print_status, clear_status, ProcessTimeStopwatch, stringify_arguments, ProcessRunElapsed


tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

train_data_set, test_data_set, train_batch_per_epoch_num, test_batch_per_epoch_num \
    = load_data_set(data_root_path=opt.data_path,
                    batch_size=opt.batch_size,
                    image_size=opt.image_size,
                    train_iter_epoch_ratio=opt.train_iter_epoch_ratio,
                    is_use_cache=opt.is_use_dataset_cache,
                    cache_path=opt.dataset_cache_path)

optimizer = None
if opt.is_use_lr_decay:
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        opt.learning_rate,
        decay_steps=train_batch_per_epoch_num // opt.lr_decay_epoch + 1,
        decay_rate=opt.lr_decay_rate,
        staircase=True)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
else:
    optimizer = tf.optimizers.Adam(learning_rate=opt.learning_rate)

model = UNet().create_model(img_shape=[opt.image_size, opt.image_size, 3], num_class=opt.num_class,
                            rate=opt.drop_rate)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_mean_iou = tf.keras.metrics.MeanIoU(num_classes=opt.num_class)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_mean_iou = tf.keras.metrics.MeanIoU(num_classes=opt.num_class)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = ''.join([opt.log_path, '/', current_time, '/', 'train'])
test_log_dir = ''.join([opt.log_path, '/', current_time, '/', 'test'])
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def loss_fn(labels, label_weights, predictions):
    cross_entropy_loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
    cross_entropy_loss_pixel = tf.multiply(cross_entropy_loss_pixel, label_weights)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_pixel) / (tf.reduce_sum(label_weights) + 0.00001)

    if opt.weight_decay > 0:
        cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
             if 'batch_normalization' not in v.name])

    return cross_entropy_loss


@tf.function
def train_step(inputs, labels, weights):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        pred_loss = loss_fn(labels=labels, label_weights=weights, predictions=predictions)

    gradients = tape.gradient(pred_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    predictions = tf.argmax(predictions, axis=-1)

    train_loss.update_state(pred_loss)
    train_mean_iou.update_state(labels, predictions)


@tf.function
def test_step(inputs, labels, weights):
    predictions = model(inputs, training=False)
    pred_loss = loss_fn(labels=labels, label_weights=weights, predictions=predictions)

    predictions = tf.nn.softmax(predictions, axis=-1)
    predictions = predictions[..., 1]
    predictions = tf.where(predictions > 0.5, 1, 0)

    test_loss.update_state(pred_loss)
    test_mean_iou.update_state(labels, predictions)


if __name__ == '__main__':
    stopwatch = ProcessTimeStopwatch()

    train_with_elapsed = ProcessRunElapsed(train_step)
    test_with_elapsed = ProcessRunElapsed(test_step)

    print(f'[INFO] training started')
    for idx_epoch in range(opt.epoch):
        status = ''

        stopwatch.start()

        for idx_step, (images, labels, weights) in enumerate(train_data_set.take(train_batch_per_epoch_num)):
            elapsed_time = train_with_elapsed(inputs=images, labels=labels, weights=weights)

            step_progress_percentage = (idx_step+1) / train_batch_per_epoch_num * 100
            # get learning rate when learning rate decay is applied
            # ref: https://stackoverflow.com/questions/36990476/getting-the-current-learning-rate-from-a-tf-train-adamoptimizer
            learning_rate = optimizer._decayed_lr(tf.float32)

            status = stringify_arguments('[INFO] train ',
                                         epoch=f'{idx_epoch + 1}/{opt.epoch}',
                                         step=f'{idx_step+1}/{train_batch_per_epoch_num} '
                                              f'({format((idx_step+1) / train_batch_per_epoch_num * 100, ".1f")}%)',
                                         learning_rate=f'{format(learning_rate, ".7f")}',
                                         loss=f'{format(train_loss.result(), ".7f")}',
                                         mean_iou=f'{format(train_mean_iou.result(), ".7f")}')
            print_status(f'{status} ({format(elapsed_time, ".1f")}s)')

            curr_step = idx_epoch * train_batch_per_epoch_num + idx_step
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=curr_step)
                tf.summary.scalar('mean_iou', train_mean_iou.result(), step=curr_step)
                tf.summary.scalar('learning_rate', learning_rate, step=curr_step)

        clear_status()
        print(f'{status} ({format(stopwatch.stop(), ".1f")}s)')

        stopwatch.start()
        batch_num = test_batch_per_epoch_num
        for idx_step, (images, labels, weights) in enumerate(test_data_set.take(test_batch_per_epoch_num)):
            elapsed_time = test_with_elapsed(inputs=images, labels=labels, weights=weights)

            status = stringify_arguments('[INFO] test ',
                                         epoch=f'{idx_epoch + 1}/{opt.epoch}',
                                         step=f'{idx_step+1}/{test_batch_per_epoch_num} '
                                              f'({format((idx_step+1) / test_batch_per_epoch_num * 100, ".1f")}%)',
                                         loss=f'{format(test_loss.result(), ".7f")}',
                                         mean_iou=f'{format(test_mean_iou.result(), ".7f")}')
            print_status(f'{status} ({format(elapsed_time, ".1f")}s)')

            curr_step = idx_epoch * test_batch_per_epoch_num + idx_step
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=curr_step)
                tf.summary.scalar('mean_iou', test_mean_iou.result(), step=curr_step)

        clear_status()
        print(f'{status} ({format(stopwatch.stop(), ".1f")}s)')

        # save weights
        if opt.is_save_checkpoint:
            checkpoint_path = ''.join([opt.checkpoint_path, '/', current_time,
                                       '/', 'epoch', str(idx_epoch+1)])
            if opt.is_use_lr_decay:
                checkpoint_path = ''.join([checkpoint_path, '_lr_decay'])

            model.save_weights(checkpoint_path)
            print(f'[INFO] checkpoint saved ({checkpoint_path})')

        # reset metrics
        train_loss.reset_states()
        train_mean_iou.reset_states()
        test_loss.reset_states()
        test_mean_iou.reset_states()

print(f'[INFO] done!')
