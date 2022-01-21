#!/usr/bin/env python
import pickle
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import scipy.stats as stats
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from external.nas_parser import *
from nas.nas_utils.general_purpose import extract_list_alphas, update_alpha_beta_tensorboard
from nas.nas_utils.predictor import Bilinear
from nas.src.optim.block_frank_wolfe import flatten_attention_latency_grad_alpha_beta_blocks
from nas.src.optim.utils import flatten_expected_accuracy
from timm import create_model
from timm.data import Dataset, CsvDataset, create_loader, FastCollateMixup, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.models import resume_checkpoint, convert_splitbn_model
from timm.optim import create_optimizer_alpha
from timm.utils import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=6)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--csv-file', default='data.csv',
                    help='file name for csv. Expected to be in data folder')
parser.add_argument('--model', default='mobilenasnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "mobilenasnet"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--min-crop-factor', type=float, default=0.08,
                    help='minimum size of crop for image transformation in training')
parser.add_argument('--squish', action='store_true', default=False,
                    help='use squish for resize input image')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0.2, metavar='PCT',
                    help='Random erase prob (default: 0.2)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=True,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='./outputs', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--nonstrict_checkpoint', type=str2bool, nargs='?', const=True, default=True,
                    help='Ignore missmatch in size when loading model weights. Used for transfer learning')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Write to TensorboardX')
parser.add_argument("--single-view", action='store_true', default=False,
                    help="train only the fc layer")
parser.add_argument("--debug", action='store_true', default=False,
                    help="logging is set to debug")
parser.add_argument("--train_percent", type=int, default=100,
                    help="what percent of data to use for train (don't forget to leave out val")
parser.add_argument('--resnet_structure', type=int, nargs='+', default=[3, 4, 6, 3], metavar='resnetstruct',
                    help='custom resnet structure')
parser.add_argument('--resnet_block', default='Bottleneck', type=str, metavar='block',
                    help='custom resnet block')

parser.add_argument("--ema_KD", action='store_true', default=False, help="use KD from EMA")
parser.add_argument('--temperature_T', type=float, default=1,
                    help='factor for temperature of the teacher')
parser.add_argument('--temperature_S', type=float, default=1,
                    help='factor for temperature of the student')
parser.add_argument('--keep_only_correct', action='store_true', default=False,
                    help='Hard threshold for training from example')
parser.add_argument('--only_kd', action='store_true', default=False,
                    help='Hard threshold for training from example')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Verbose mode')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')

parser.add_argument('--accuracy_lut_filename', default='accuracy_lut.pkl',
                    help='The filename of the expected accuracy LUT (default: accuracy_lut.pkl)')
parser.add_argument('--test_accuracy_lut_filename', default=None,
                    help='The filename of the measured accuracy LUT for test architectures (default: None)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to evaluate the contribution over (default: 10)')
parser.add_argument('--num_iter', type=int, default=-1,
                    help='Maximal number of validation batches for configuration (default: -1, full validation)')
parser.add_argument('--accuracy_vectors_filename', default=None,
                    help='The filename of the processed accuracy vectors (default: None)')
parser.add_argument('--estimator_ckpt_filename', default=None,
                    help='The output filename of the bilinear estimator checkpoint (default: None)')
parser.add_argument('--test_figure_filename', default=None,
                    help='The output filename for the output figures (default: None)')

add_nas_to_parser(parser)

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


def get_train_val_dir(basedir):
    train_dir = val_dir = None
    for reg in 'train train_set'.split():
        if os.path.exists(os.path.join(basedir, reg)):
            train_dir = os.path.join(basedir, reg)
            break

    if train_dir is None:
        logging.error('Training folder does not exist at: {}'.format(basedir))
        exit(1)

    for reg in 'val validation val_set test'.split():
        if os.path.exists(os.path.join(basedir, reg)):
            val_dir = os.path.join(basedir, reg)
            break

    if val_dir is None:
        logging.error('Validation folder does not exist at: {}'.format(basedir))
        exit(1)

    return train_dir, val_dir


def main():
    args, args_text = _parse_args()
    default_level = logging.INFO
    if args.debug:
        default_level = logging.DEBUG

    setup_default_logging(default_level=default_level)
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    writer = None
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    assert args.rank >= 0
    DistributedManager.set_args(args)
    sys.stdout = FilteredPrinter(filtered_print, sys.stdout, args.rank == 0)
    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))

    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    if args.tensorboard and DistributedManager.is_master():
        writer = SummaryWriter('outputs')

    if os.path.exists(os.path.join(args.data, args.csv_file)):
        dataset_train = CsvDataset(os.path.join(args.data, args.csv_file),
                                   single_view=args.single_view, data_percent=args.train_percent)
        dataset_eval = CsvDataset(os.path.join(args.data, args.csv_file),
                                  single_view=True, data_percent=10, reverse_order=True)
    else:
        train_dir, eval_dir = get_train_val_dir(args.data)
        dataset_train = Dataset(train_dir)
        if args.train_percent < 100:
            dataset_train, dataset_valid = dataset_train.split_dataset(1.0 * args.train_percent / 100.0)

        dataset_eval = Dataset(eval_dir)

    logging.info(f'Training data has {len(dataset_train)} images')
    args.num_classes = len(dataset_train.class_to_idx)
    logging.info(f'setting num classes to {args.num_classes}')

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
        strict=not args.nonstrict_checkpoint,
        resnet_structure=args.resnet_structure,
        resnet_block=args.resnet_block,
        heaviest_network=args.heaviest_network,
        use_kernel_3=args.use_kernel_3,
        exp_r=args.exp_r,
        depth=args.depth,
        reduced_exp_ratio=args.reduced_exp_ratio,
        use_dedicated_pwl_se=args.use_dedicated_pwl_se,
        force_sync_gpu=args.force_sync_gpu,
        multipath_sampling=args.multipath_sampling,
        use_softmax=args.use_softmax,
        detach_gs=args.detach_gs,
        no_swish=args.no_swish,
        search_mode=True
    )
    if args.force_se and 'mobilenasnet' in args.model:
        model.set_force_se(True)

    list_alphas = None

    if args.qc_init:
        if args.init_to_biggest_alpha:
            model.set_all_alpha(er=6, k=5, se=0.25 if args.force_se else 0, use_only=False)
        else:
            model.set_all_alpha(er=3, k=3, se=0.25 if args.force_se else 0, use_only=False)

        model.set_all_beta(2, use_only=False)

    elif args.init_to_smallest:
        model.set_all_alpha(er=3, k=3, se=0.25 if args.force_se else 0, use_only=False)
        model.set_all_beta(2, use_only=False)

    elif args.init_to_biggest:
        model.set_last_alpha(use_only=False)
        model.set_last_beta(use_only=False)

    elif args.init_to_biggest_alpha:
        model.set_all_alpha(er=6, k=5, se=0.25 if args.force_se else 0, use_only=False)
        model.set_all_beta(2, use_only=False)

    else:
        model.set_uniform_alpha()
        model.set_uniform_beta(stage=1)

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    model.eval()

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # For backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True

        elif has_native_amp:
            args.native_amp = True

    if args.apex_amp and has_apex:
        use_amp = 'apex'

    elif args.native_amp and has_native_amp:
        use_amp = 'native'

    elif args.apex_amp or args.native_amp:
        logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            logging.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None

        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()

        assert not args.channels_last, "Channels last not supported with DP, use DDP."

    else:
        model.cuda()
        model.train()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    # optim2 = None
    # if args.train_nas or args.search_elastic_model and not args.fixed_alpha:
    #     optim = create_optimizer_alpha(args, list_alphas, args.lr_alphas)
    #     if hasattr(optim, 'fixed_latency'):
    #         optim.fixed_latency = fixed_latency
    #
    #     if args.nas_optimizer.lower() == 'sgd':
    #         args2 = deepcopy(args)
    #         args2.nas_optimizer = 'block_frank_wolfe'
    #         optim2 = create_optimizer_alpha(args2, list_alphas, args.lr_alphas)
    #         optim2.fixed_latency = fixed_latency
    #
    # amp_autocast = suppress  # do nothing
    # loss_scaler = None
    # if use_amp == 'apex':
    #     if optim is not None:
    #         model, optim = amp.initialize(model, optim, opt_level='O1')
    #
    #     loss_scaler = ApexScaler()
    #     if args.local_rank == 0:
    #         logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    #
    # elif use_amp == 'native':
    #     amp_autocast = torch.cuda.amp.autocast
    #     loss_scaler = NativeScaler()
    #     if args.local_rank == 0:
    #         logging.info('Using native Torch AMP. Training in mixed precision.')
    #
    # else:
    #     if args.local_rank == 0:
    #         logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)

    if resume_state and not args.no_resume_opt:
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')

            amp.load_state_dict(resume_state['amp'])

    del resume_state

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')

        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                logging.info("Using NVIDIA APEX DistributedDataParallel.")

            model = ApexDDP(model, delay_allreduce=True)

        else:
            if args.local_rank == 0:
                logging.info("Using native Torch DistributedDataParallel.")

            # NOTE: EMA model does not need to be wrapped by DDP
            model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    collate_fn = None
    if args.prefetcher and args.mixup > 0:
        assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
        collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)

    dataset_val = dataset_valid if args.train_percent < 100 else dataset_eval
    loader_valid = create_loader(
        dataset_val,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        squish=args.squish,
        infinite_loader=True,
        force_data_sampler=True
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        squish=args.squish,
    )

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])

        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # Disable weights gradients and BN statistics and enable alpha-beta gradients
    # freeze_weights_unfreeze_alphas(model, optim)

    list_alphas = extract_list_alphas(model)
    optim = create_optimizer_alpha(args, list_alphas, args.lr_alphas)
    alpha_attention_vec, _, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)

    torch.cuda.empty_cache()

    # Search Contributions
    if os.path.exists(args.accuracy_lut_filename):
        with open(args.accuracy_lut_filename, 'rb') as f:
            accuracy_dict = pickle.load(f)
            print(accuracy_dict)

    else:
        accuracy_dict = {'base': torch.tensor(0).float().cuda()}
        for _ in range(args.epochs):
            accuracy_dict['base'] += validate(model, loader_valid, validate_loss_fn, args,
                                              log_suffix=' base', num_iter=args.num_iter)['top1']

        accuracy_dict['base'] /= args.epochs

        if DistributedManager.distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            torch.distributed.all_reduce(accuracy_dict['base'], op=torch.distributed.ReduceOp.SUM, group=grp)
            accuracy_dict['base'] /= ws

        accuracy_dict['base'] = accuracy_dict['base'].item()

    if args.accuracy_vectors_filename is None or\
            not os.path.exists(args.accuracy_vectors_filename + '_alpha') or \
            not os.path.exists(args.accuracy_vectors_filename + '_beta'):
        count = 0
        for e, entry in enumerate(list_alphas):
            if 'beta' not in entry:
                continue

            for a in range(len(entry['submodules']) - len(entry['beta'])):
                # Measure prefix alpha block
                prefix_alpha_entry = entry['alpha_entries'][a]
                logits = prefix_alpha_entry['module'].alpha
                accuracy = torch.zeros_like(logits).cuda()
                logits.data = torch.zeros_like(logits.data) - 1000
                for i, submodule_name in enumerate(prefix_alpha_entry['submodules']):
                    logits.data[i] = 0
                    if submodule_name in accuracy_dict:
                        accuracy[i] = accuracy_dict[submodule_name]
                        print('Loaded expected accuracy of {}'.format(submodule_name))
                        logits.data = torch.zeros_like(logits.data) - 1000
                        continue

                    for _ in range(args.epochs):
                        accuracy[i] += validate(model, loader_valid, validate_loss_fn, args,
                                                log_suffix=' ' + submodule_name, num_iter=args.num_iter)['top1']

                    accuracy[i] /= args.epochs
                    if DistributedManager.distributed:
                        grp = DistributedManager.grp
                        ws = torch.distributed.get_world_size()
                        torch.distributed.all_reduce(accuracy, op=torch.distributed.ReduceOp.SUM, group=grp)
                        accuracy /= ws

                    accuracy_dict[submodule_name] = accuracy[i].item()
                    with open(args.accuracy_lut_filename, 'wb') as f:
                        pickle.dump(accuracy_dict, f)

                    # print_solution(list_alphas, optim, args)
                    update_alpha_beta_tensorboard(count, list_alphas, writer)
                    count += 1
                    logits.data = torch.zeros_like(logits.data) - 1000

                prefix_alpha_entry['expected_accuracy'] = accuracy

            # Measure beta accuracy
            beta_accuracy = torch.zeros_like(entry['beta']).cuda()
            entry['beta'].data = torch.zeros_like(entry['beta'].data) - 1000
            for b, submodule_name in enumerate(entry['submodules'][-len(entry['beta']):]):
                entry['beta'].data[b] = 0
                if submodule_name in accuracy_dict:
                    beta_accuracy[b] = accuracy_dict[submodule_name]
                    print('Loaded expected accuracy of {}'.format(submodule_name))

                else:
                    for _ in range(args.epochs):
                        beta_accuracy[b] += validate(model, loader_valid, validate_loss_fn, args,
                                                     log_suffix=' ' + submodule_name, num_iter=args.num_iter)['top1']

                    beta_accuracy[b] /= args.epochs
                    if DistributedManager.distributed:
                        grp = DistributedManager.grp
                        ws = torch.distributed.get_world_size()
                        torch.distributed.all_reduce(beta_accuracy, op=torch.distributed.ReduceOp.SUM, group=grp)
                        beta_accuracy /= ws

                    accuracy_dict[submodule_name] = beta_accuracy[b].item()
                    with open(args.accuracy_lut_filename, 'wb') as f:
                        pickle.dump(accuracy_dict, f)

                update_alpha_beta_tensorboard(count, list_alphas, writer)
                count += 1

                # Measure associated alpha blocks
                a = b + len(entry['submodules']) - len(entry['beta'])
                alpha_entry = entry['alpha_entries'][a]
                logits = alpha_entry['module'].alpha
                accuracy = torch.zeros_like(logits).cuda()
                logits.data = torch.zeros_like(logits.data) - 1000
                for i, submodule_name in enumerate(alpha_entry['submodules']):
                    logits.data[i] = 0
                    if submodule_name in accuracy_dict:
                        accuracy[i] = accuracy_dict[submodule_name]
                        print('Loaded expected accuracy of {}'.format(submodule_name))

                    else:
                        for _ in range(args.epochs):
                            accuracy[i] += validate(model, loader_valid, validate_loss_fn, args,
                                                    log_suffix=' ' + submodule_name, num_iter=args.num_iter)['top1']

                        accuracy[i] /= args.epochs
                        if DistributedManager.distributed:
                            grp = DistributedManager.grp
                            ws = torch.distributed.get_world_size()
                            torch.distributed.all_reduce(accuracy, op=torch.distributed.ReduceOp.SUM, group=grp)
                            accuracy /= ws

                        accuracy_dict[submodule_name] = accuracy[i].item()
                        with open(args.accuracy_lut_filename, 'wb') as f:
                            pickle.dump(accuracy_dict, f)

                    update_alpha_beta_tensorboard(count, list_alphas, writer)
                    count += 1
                    logits.data = torch.zeros_like(logits.data) - 1000

                alpha_entry['expected_accuracy'] = accuracy
                entry['beta'].data = torch.zeros_like(entry['beta'].data) - 1000

            entry['expected_accuracy'] = beta_accuracy

        alpha_expected_accuracy_vec, beta_expected_accuracy_vec = flatten_expected_accuracy(list_alphas)
        if args.accuracy_vectors_filename is not None:
            with open(args.accuracy_vectors_filename + '_alpha', 'wb') as f:
                np.save(f, alpha_expected_accuracy_vec )
                print(f'Saved processed accuracy alpha vector at {args.accuracy_vectors_filename}_alpha')

            with open(args.accuracy_vectors_filename + '_beta', 'wb') as f:
                np.save(f, beta_expected_accuracy_vec )
                print(f'Saved processed accuracy alpha vector at {args.accuracy_vectors_filename}_beta')

    if not DistributedManager.is_master():
        return

    if args.accuracy_vectors_filename is not None:
        print('Loading processed accuracy vectors')
        with open(args.accuracy_vectors_filename + '_alpha', 'rb') as f:
           alpha_expected_accuracy_vec = np.load(f)

        with open(args.accuracy_vectors_filename + '_beta', 'rb') as f:
            beta_expected_accuracy_vec = np.load(f)

    optim._accuracy_formula(alpha_blocks, beta_blocks,
                            accuracy_dict['base'],
                            alpha_expected_accuracy_vec, beta_expected_accuracy_vec,
                            alpha_attention_vec, beta_attention_vec)

    if args.estimator_ckpt_filename is not None and not os.path.exists(args.estimator_ckpt_filename):
        estimator = Bilinear(len(alpha_attention_vec), len(beta_attention_vec))
        accuracy_base = accuracy_dict['base'].to(device=estimator.linear.bias.device) \
            if isinstance(accuracy_dict['base'], torch.Tensor) else torch.tensor(accuracy_dict['base'])
        estimator.linear.bias.data = accuracy_base * torch.ones_like(estimator.linear.bias)
        estimator.linear.weight.data = torch.cat([torch.tensor(optim.p_acc_a), torch.tensor(optim.p_acc_b)]).unsqueeze(0)
        estimator.bilinear.weight.data = torch.tensor(optim.Q_acc).unsqueeze(0)
        torch.save(estimator.state_dict(), args.estimator_ckpt_filename)
        print(f'Saved {args.estimator_ckpt_filename}')

    if not os.path.exists(args.test_accuracy_lut_filename):
        return

    print('----------------------- Ranking Correlation with Supernet --------------------------------')
    with open(args.test_accuracy_lut_filename, 'rb') as f:
        test_accuracy_dict = pickle.load(f)

    predicted_list = []
    oneshot_list = []

    accuracy_base = accuracy_dict['base']
    accuracy_base = accuracy_base.cpu().numpy() if isinstance(accuracy_base, torch.Tensor) else accuracy_base
    for record in test_accuracy_dict.values():
        oneshot_list.append(record['accuracy'])
        for e, (entry, argmax) in enumerate(zip(list_alphas, record['logits'])):
            key = 'beta' if 'beta' in entry else 'alpha'
            logits = entry['module'].alpha if key == 'alpha' else entry[key]
            logits.data = torch.zeros_like(logits.data) - 10000
            logits.data[argmax] = 0

        alpha, _, _, _, beta, _, _ = flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
        predicted_list.append(optim.accuracy_formula(alpha, beta, accuracy_base).item())

    x = np.array(oneshot_list)
    y = np.array(predicted_list)

    kt_str = 'Kendall-tau: {0:0.2f}'.format(stats.kendalltau(x, y)[0])
    sp_str = 'Spearman: {0:0.2f}'.format(stats.spearmanr(x, y)[0])
    ps_str = 'Pearson: {0:0.2f}'.format(stats.pearsonr(x, y)[0])
    mse_str = 'MSE: {0:0.2f}'.format(np.mean(np.square(x - y)))
    print('{} | {} | {} | {}'.format(kt_str, sp_str, ps_str, mse_str))

    if args.test_figure_filename is not None:
        plt.figure()
        plt.scatter(x, y)
        plt.title('{} | {} | {} | {}'.format(kt_str, sp_str, ps_str, mse_str))
        plt.xlabel('Oneshot Accuracy')
        plt.ylabel('Predicted Accuracy')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
        plt.plot([xmin, xmax], [ymin, ymax], color='green', linestyle='dashed')
        plt.savefig('test_accuracy_prediction_ranking.png')


def validate(model, loader, loss_fn, args, log_suffix='', num_iter=-1, verbose=True):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if num_iter > 0 and batch_idx == num_iter:
                break

            last_batch = batch_idx == last_idx

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            k = min(5, args.num_classes)
            acc1, acc5 = accuracy(output, target, topk=(1, k))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if verbose and args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    main()
