import tensorflow as tf
import horovod.tensorflow as hvd

from opts import *
from data_gen import data_loader
from model import UNet

tf.random.set_seed(0)

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

train_generator, _, train_num, _ =\
        data_loader(opt.data_path, opt.batch_size, imSize=opt.imSize)

num_train_batch_per_epoch = int(train_num / opt.batch_size * opt.iter_epoch_ratio)
num_train_batch_per_epoch = int(num_train_batch_per_epoch // hvd.size())

def loss_fn(labels, label_weights, predictions):
    cross_entropy_loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
    cross_entropy_loss_pixel = tf.multiply(cross_entropy_loss_pixel, label_weights)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_pixel) / (tf.reduce_sum(label_weights) + 0.00001)

    # if opt.weight_decay > 0:
    #     cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
    #         [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
    #          if 'batch_normalization' not in v.name])

    return cross_entropy_loss

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    opt.learning_rate * hvd.size(),
    decay_steps=num_train_batch_per_epoch // opt.lr_decay_epoch,
    decay_rate=opt.lr_decay,
    staircase=True)

optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
optimizer = hvd.DistributedOptimizer(optimizer)

model = UNet().create_model(img_shape=[opt.imSize, opt.imSize, 3], num_class=opt.num_class, rate=opt.drop_rate)

@tf.function
def train_step(inputs, labels, weights, first_batch):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        pred_loss = loss_fn(labels=labels, label_weights=weights, predictions=predictions)
                
    tape = hvd.DistributedGradientTape(tape)
                
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)
                      
    # predicted_results = tf.where(predictions > opt.threshold, 1, 0)
                      
    # mean_iou_metric.update_state(
    #     labels,
    #     tf.math.argmax(predictions, axis=3)
    # )

    return predictions, pred_loss




for idx_epoch in range(opt.epoch):
    # if hvd.rank() == 0:
    # print(f'TRAIN EPOCH ({idx_epoch + 1}/{opt.epoch})')

    # for idx_step, (images, labels, weights) in enumerate(tr.take(num_train_batch_per_epoch)):
    for idx_step in range(num_train_batch_per_epoch):
        images, labels, weights = next(train_generator)

    # for idx_step, (images, labels, weights) in enumerate(tr.take(num_train_batch_per_epoch)):

        # predicted_results, loss_value = train_step(model, images, labels, weights, idx_step == 0)
        #loss_value, grads = grad(model, images, labels, weights)
        #optimizer.apply_gradients(zip(grads, model.trainable_variables))
        predicted_results, loss_value = train_step(images, labels, weights, idx_step == 0)

        if hvd.local_rank() == 0 and (idx_epoch * opt.epoch + idx_step) % 10 == 0:
        #if (idx_epoch * opt.epoch + idx_step) % 10 == 0:
            # mean_iou = get_mean_iou(predicted_results, labels)
            # get learning rate
            # ref: https://stackoverflow.com/questions/36990476/getting-the-current-learning-rate-from-a-tf-train-adamoptimizer
            # print(f'[INFO] train epoch: ({idx_epoch + 1}/{opt.epoch}),'
            #       f' step: ({idx_step}/{num_train_batch_per_epoch}),'
            #       f' loss: {format(loss_value, ".3f")}')
            #       # f' meanIoU: {format(mean_iou, ".3f")}')
            step_progress_percentage = idx_step / num_train_batch_per_epoch * 100

            print(f'[INFO] train epoch: ({idx_epoch + 1}/{opt.epoch}),'
                  f' step: {format(step_progress_percentage, ".1f")}%'
                  f' ({idx_step}/{num_train_batch_per_epoch}),'
                  f' learning_rate: {format(optimizer._decayed_lr(tf.float32), ".7f")},'
                  f' loss: {format(loss_value, ".7f")}')

print(f'[INFO] done!')

#
#
# @tf.function
# def train_step(model, inputs, labels, weights, first_batch):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         pred_loss = loss_fn(labels=labels, label_weights=weights, predictions=predictions)
#
#         gradients = tape.gradient(pred_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     # gradients = tape.gradient(pred_loss, model.trainable_variables)
#     # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#
#     # predicted_results = tf.where(predictions > opt.threshold, 1, 0)
#
#     # mean_iou_metric.update_state(
#     #     labels,
#     #     tf.math.argmax(predictions, axis=3)
#     # )
#
#     return predictions, pred_loss
#
#
# def get_mean_iou(predicted_results, labels):
#     predicted_results = np.argmax(predicted_results[0], axis=2)
#     mean_iou, _ = mean_IU(predicted_results, labels[0])
#
#     return mean_iou
#
#
# if __name__ == '__main__':
#     model = create_model(img_shape=[opt.imSize, opt.imSize, 3], num_class=opt.num_class, rate=opt.drop_rate)
#
#     for idx_epoch in range(opt.epoch):
#         # if hvd.rank() == 0:
#         # print(f'TRAIN EPOCH ({idx_epoch + 1}/{opt.epoch})')
#
#         for idx_step in range(num_train_batch_per_epoch):
#             images, labels, weights = next(train_generator)
#         # for idx_step, (images, labels, weights) in enumerate(tr.take(num_train_batch_per_epoch)):
#
#             predicted_results, loss_value = train_step(model, images, labels, weights, idx_step == 0)
#
#             if (idx_epoch * opt.epoch + idx_step) % 10 == 0:
#                 mean_iou = get_mean_iou(predicted_results, labels)
#                 # get learning rate
#                 # ref: https://stackoverflow.com/questions/36990476/getting-the-current-learning-rate-from-a-tf-train-adamoptimizer
#                 print(f'[INFO] train epoch: ({idx_epoch + 1}/{opt.epoch}),'
#                       f' step: ({idx_step}/{num_train_batch_per_epoch}),'
#                       f' learning_rate: {format(optimizer._decayed_lr(tf.float32), ".7f")},'
#                       f' loss: {format(loss_value, ".3f")},'
#                       f' meanIoU: {format(mean_iou, ".3f")}')
#
#     print(f'[INFO] done!')
