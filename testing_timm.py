import yaml
import argparse
import numpy as np
import torch
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from timm.models import create_model
import metaformer_baselines
from timm.data import create_dataset, create_loader, resolve_data_config, transforms_factory
from timm.data.loader import _worker_init, PrefetchLoader
from functools import partial
from itertools import repeat

def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False

def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
# parser.add_argument('data_dir', metavar='DIR', help='path to dataset',
#                     default='/home/yous/Desktop/cerrion/datasets/retest')
group.add_argument('--dataset', '-d', metavar='NAME', default='ImageFolder',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
group.add_argument('--train-interpolation', type=str, default='bilinear',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('-j', '--workers', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 8)')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--eval-metric', default='f1_score', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "f1_score"')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    use_prefetcher = True
    args.data_dir = '../ba_project/datasets/retest'
    modargs = {'model_name': 'convformer_s18_384', 'pretrained': False, 'num_classes': None,
               'drop_rate': 0.0, 'drop_connect_rate': None, 'drop_path_rate': 0.3,
               'drop_block_rate': None, 'global_pool': None, 'bn_momentum': None,
               'bn_eps': None, 'scriptable': False,
               'checkpoint_path': './convformer_s18_384.pth',
               'head_dropout': 0.4}
    a = create_model(**modargs)
    data_config = resolve_data_config(vars(args), model=a, verbose=True)
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map, download=False, batch_size=1, repeats=0)
    impath, im_label = dataset_train.parser.samples[1]
    im = Image.open(impath).convert('RGB')

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=1,
        is_training=True,
        use_prefetcher=use_prefetcher,
        no_aug=True,
        re_prob=args.reprob,
        re_mode='pixel',
        re_count=1,
        re_split=False,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=0,
        interpolation='random',
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        collate_fn=fast_collate,
        pin_memory=False,
        use_multi_epochs_loader=False,
        worker_seeding='all',
        tf_preprocessing=False
    )
    # loader_train.dataset.transform = transf_torch
    # trans_tiff = loader_train.dataset.transform
    # trans_tiff = transforms_factory.transforms_noaug_train(img_size=(384,384),
    #                                                        use_prefetcher=use_prefetcher,
    #                                                        mean=data_config['mean'],
    #                                                        std=data_config['std'])
    trans_tiff = transforms_factory.transforms_imagenet_train(
        (384,384),
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation='random',
        use_prefetcher=use_prefetcher,
        mean=data_config['mean'],
        std=data_config['std'],
        re_prob=args.reprob,
        re_mode='pixel',
        re_count=1,
        re_num_splits=0,
        separate=False)
    print(trans_tiff)
    print(loader_train.dataset.transform)
    exit()
    im_tiff = trans_tiff(im)
    mean = expand_to_chs(data_config['mean'], 3)
    std = expand_to_chs(data_config['std'], 3)
    normalization_shape = (1, 3, 1, 1)
    mean = torch.tensor([x * 255 for x in mean]).view(normalization_shape)
    std = torch.tensor([x * 255 for x in std]).view(normalization_shape)
    im_tiff, im_label = fast_collate([(im_tiff, im_label)])
    im_tiff = im_tiff.float().sub_(mean).div_(std)

    samples = []
    for x in loader_train:
        samples.append(x[0])

    print(torch.equal(im_tiff.cpu(), samples[0].cpu()))

if __name__ == '__main__':
    main()
