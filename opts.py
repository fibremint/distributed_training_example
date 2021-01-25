'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:01
 * @modify date 2017-05-25 02:20:01
 * @desc [description]
'''

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=6, help='# of epochs')
parser.add_argument('--batch-size', type=int, default=16, help='input batch size')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')

parser.add_argument('--lr-decay', dest='is_use_lr_decay', action='store_true')
parser.add_argument('--no-lr-decay', dest='is_use_lr_decay', action='store_false')
parser.set_defaults(is_use_lr_decay=False)
parser.add_argument('--lr-decay-rate', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--lr-decay-epoch', type=int, default=1, help='how many epoch to decay learning rate')

parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--drop-rate', type=float, default=0.0, help='drop rate of unet')

parser.add_argument('--dataset-cache', dest='is_use_dataset_cache', action='store_true')
parser.add_argument('--no-dataset-cache', dest='is_use_dataset_cache', action='store_false')
parser.set_defaults(is_use_dataset_cache=False)
parser.add_argument('--dataset-cache-path', type=str, default='.', help='path where dataset cache is stored')

parser.add_argument('--image_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--train-iter-epoch-ratio', type=float, default=0.3, help='# of ratio of total images as an epoch')
parser.add_argument('--num-class', type=int, default=2, help='# of classes')
parser.add_argument('--data-path', type=str, default='', help='where dataset saved. See loader.py '
                                                              'to know how to organize the dataset folder')
parser.add_argument('--log-path', type=str, default='./log', help='where tensorflow summary is saved')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoint', help='where checkpoint saved')
parser.add_argument('--save-checkpoint', dest='is_save_checkpoint', action='store_true')
parser.add_argument('--no-save-checkpoint', dest='is_save_checkpoint', action='store_false')
parser.set_defaults(is_save_checkpoint=True)

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# hardcode here
# dataset_mean = [0.5,0.5,0.5]
# dataset_std = [0.5,0.5,0.5]

# training data directory
if opt.data_path == '':
    opt.data_path = '../train-data/'

# opt.data_path='../data/segmentation/'
# opt.checkpoint_path = '../checkpoint/'
opt.model_weight_save_path = os.path.join(opt.checkpoint_path, 'wsi-seg-weight.h5')
# training model directory
# checkpoint_root = '../checkpoints/segmentation/'
# opt.checkpoint_path = os.path.join(checkpoint_root, opt.checkpoint_path)
# if not os.path.isdir(opt.checkpoint_path):
#     os.mkdir(opt.checkpoint_path)
# if not os.path.isdir(os.path.join(opt.checkpoint_path,'img')):
#     os.mkdir(os.path.join(opt.checkpoint_path,'img'))
