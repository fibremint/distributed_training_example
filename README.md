# Deep Learning training example based on Tensorflow 2

The training codes are migrated from [this repository](https://github.com/zizhaozhang/nmi-wsi-diagnosis/tree/master/segmentation) which written in Tensorflow 1 to Tensorflow 2.

## Features
* Deep Learning training loop with Tensorflow 2
* Training data preparation (load, augmentation, pre-processing) implemented in Tensorflow API, served as Tensorflow Dataset
* Recorded training history (loss, accuracy, learning rate) is visible on the Tensorboard

## Training argument example

```
python train.py --batch-size 16 \
                --epoch 20 \
                --train-iter-epoch-ratio 0.025 \
                --learning-rate 0.001 \
                --lr-decay \
                --data-path ../train_data \
                --dataset-cache \
                --dataset-cache-path ./cache \
                --log-path ./log \
                --save-checkpoint \
                --checkpoint-path ./checkpoint
```

## Reference
* **Training**: [zizhaozhang/nmi-wsi-diagnosis](https://github.com/zizhaozhang/nmi-wsi-diagnosis)
