#!/usr/bin/env python
import pickle
import sys
import time
from contextlib import suppress
from datetime import datetime

import matplotlib.pyplot as plt
import yaml
from scipy.stats import stats
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tqdm import tqdm

from accuracy_contribution import validate
from external.nas_parser import *
from nas.nas_utils.general_purpose import extract_structure_param_list, target_time_loss, \
    freeze_weights_unfreeze_alphas, get_stage_block_from_name, STAGE_BLOCK_DELIMITER, OptimLike, \
    update_alpha_beta_tensorboard
from nas.nas_utils.predictor import Quadratic, Bilinear, MLP, Predictor, construct_predictors, predict
from nas.src.optim.block_frank_wolfe import flatten_attention_latency_grad_alpha_beta_blocks
from nas.src.optim.utils import update_attentions_inplace
from timm import create_model
from timm.data import Dataset, CsvDataset, create_loader, FastCollateMixup, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.models import resume_checkpoint, convert_splitbn_model
from timm.models.mobilenasnet import transform_model_to_mobilenet, measure_time
from timm.optim import create_optimizer_alpha
from timm.utils import *
from timm.utils_new.cuda import ApexScaler, NativeScaler

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

import gc
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
parser.add_argument('data', metavar='DIR', default=None,
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

parser.add_argument('--predictor_type', default='bilinear', choices=['bilinear', 'quadratic', 'mlp'],
                    help='The type of the predictor model (default: bilinear)')
parser.add_argument('--predictor_ckpt_filename',
                    help='The filename of the predictor checkpoint')
parser.add_argument('--test_accuracy_lut_filename', default=None,
                    help='The filename of the measured accuracy LUT for test architectures (default: None)')
parser.add_argument('--test_figure_filename', default=None,
                    help='The output filename for the output figures (default: None)')
parser.add_argument('--eval_child_model', action='store_true', default=False,
                    help='Evaluate the generated child model with weights loaded from the supernetwork')
parser.add_argument('--verbose_search', action='store_true', default=False,
                    help='Verbose search mode')

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
        logging.info('Distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))

    else:
        logging.info('A single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    if args.eval_child_model:
        if os.path.exists(os.path.join(args.data, args.csv_file)):
            dataset_eval = CsvDataset(os.path.join(args.data, args.csv_file),
                                      single_view=True, data_percent=10, reverse_order=True)
        else:
            _, eval_dir = get_train_val_dir(args.data)
            dataset_eval = Dataset(eval_dir)

        logging.info(f'Evaluation data has {len(dataset_eval)} images')
        args.num_classes = len(dataset_eval.class_to_idx)
        logging.info(f'Setting num classes to {args.num_classes}')

    else:
        args.num_classes = 1000

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

    model.cuda()
    model.train()

    optim = None
    list_alphas = None
    fixed_latency = 0
    if args.search_elastic_model:
        model.set_hard_backprop(False)
        model.eval()
        with torch.no_grad():
            x = torch.rand(64, 3, 224, 224).cuda()
            out = model(x)
        del out, x
        gc.collect()
        torch.cuda.empty_cache()
        list_alphas, fixed_latency = extract_structure_param_list(model, file_name=args.lut_filename,
                                                              batch_size=args.lut_measure_batch_size,
                                                              repeat_measure=args.repeat_measure,
                                                              target_device=args.target_device)

    fixed_latency = args.latency_corrective_slope * fixed_latency + args.latency_corrective_intercept

    optim = create_optimizer_alpha(args, list_alphas, args.lr_alphas)
    if hasattr(optim, 'fixed_latency'):
        optim.fixed_latency = fixed_latency

    # Optionally resume from a checkpoint
    resume_state = {}
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


    if args.eval_child_model:
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
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.smoothing:
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

    torch.cuda.empty_cache()

    alpha_attention_vec, _, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
    alphas = len(alpha_attention_vec)
    betas = len(beta_attention_vec)
    predictor = construct_predictors(args.predictor_type, alphas, betas)
    predictor.load_state_dict(torch.load(args.predictor_ckpt_filename))
    print(f'Loaded {args.predictor_ckpt_filename}')

    alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
    _, _ = optim._latency_constraint(alpha_blocks, beta_blocks, latency_vec,
                                                    alpha_vec=alpha_attention_vec, beta_vec=beta_attention_vec)

    if args.test_accuracy_lut_filename is not None:
        print('----------------------- Ranking Correlation with Supernet --------------------------------')
        with open(args.test_accuracy_lut_filename, 'rb') as f:
            test_accuracy_dict = pickle.load(f)

        xs, ys, xs_oh = [], [], []
        for i, record in enumerate(test_accuracy_dict.values()):
            record['logits'] = record['logits']
            ys.append(torch.tensor(record['accuracy']).unsqueeze(dim=0))
            xs.append(torch.tensor(record['logits']))

            x_oh = []
            for entry, argmax in zip(list_alphas, record['logits']):
                key = 'beta' if 'beta' in entry else 'alpha'
                if key == 'beta':
                    continue
                logits = entry['module'].alpha if key == 'alpha' else entry[key]
                one_hot = [0 for _ in range(len(logits.data))]
                one_hot[argmax] = 1
                x_oh += one_hot

            for entry, argmax in zip(list_alphas, record['logits']):
                key = 'beta' if 'beta' in entry else 'alpha'
                if key == 'alpha':
                    continue
                logits = entry['module'].alpha if key == 'alpha' else entry[key]
                one_hot = [0 for _ in range(len(logits.data))]
                one_hot[argmax] = 1
                x_oh += one_hot

            xs_oh.append(torch.tensor(x_oh))

        X, Y = xs_oh, ys

        predictor.model.cuda()
        y = predict(predictor, X)
        x = np.array([y.item() for y in Y], dtype=y.dtype)

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
            plt.savefig(args.test_figure_filename)

    search(args, model, list_alphas, optim, saver, predictor, args.verbose_search)
    alpha_attention_vec, _, _, _, beta_attention_vec, _, _ = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
    x = np.concatenate((alpha_attention_vec, beta_attention_vec))
    X = [torch.tensor(x)]
    predictor.model.cuda()
    predicted_accuracy = predict(predictor, X)
    print(f'Predicted Accuracy: {predicted_accuracy}')

    if not args.eval_child_model:
        return

    # Evaluate child model
    child_model, string_model = transform_model_to_mobilenet(model)
    if args.num_gpu > 1:
        child_model = torch.nn.DataParallel(child_model, device_ids=list(range(args.num_gpu)))

    child_model.cuda()
    validate(child_model, loader_eval, validate_loss_fn, args, log_suffix=' child model')
    if saver is not None:
        saver.save_checkpoint(child_model, optim, args, epoch=2, metric=2)

    model.eval()
    child_model.eval()
    print(f"Computing latency for {string_model}")
    unwrapped_model = model if hasattr(model, 'extract_expected_latency') else model.module
    latency_predicted = unwrapped_model.extract_expected_latency(file_name=args.lut_filename,
                                                                 batch_size=args.lut_measure_batch_size,
                                                                 iterations=args.repeat_measure,
                                                                 target=args.target_device)

    latency_measured = measure_time(child_model)
    diff = latency_measured - latency_predicted
    print(f"Latency_predicted={latency_predicted}, Latency_measured={latency_measured}, Diff={diff}")


def search(args, model, list_alphas, optim, saver, predictor, verbose=False):
    print('------------------------ Search -------------------------------')
    if hasattr(optim, 'qcp'):
        alpha_attention_vec, _, _, _, beta_attention_vec, _, _ = \
            flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
        alphas = len(alpha_attention_vec)
        betas = len(beta_attention_vec)
        optim.Q_acc = np.array(predictor.model.bilinear.weight.squeeze().detach().cpu(), dtype=np.double)
        p_acc = np.array(predictor.model.linear.weight.detach().cpu()[0], dtype=np.double)
        optim.p_acc_a = p_acc[:alphas]
        optim.p_acc_b = p_acc[-betas:]
        optim.qcp(np.zeros(alphas), np.zeros(betas), linear=False)
    elif hasattr(optim, 'evo'):
        optim.set_predictor(predictor)
        optim.evo()
    else:
        # model.set_all_alpha(er=3, k=3, se=0.25 if args.force_se else 0, use_only=False)
        # model.set_all_beta(2, use_only=False)
        alpha_attn, beta_attn = optimize_diff_predictor(predictor, optim, args.bcfw_steps)
        update_attentions_inplace(list_alphas, alpha_attention_vec=alpha_attn.detach().numpy(),
                                  beta_attention_vec=beta_attn.detach().numpy())
        if verbose:
            print_solution(list_alphas, optim, args)

        # print('------------------------ Sparsify -------------------------------')
        optim.sparsify()

    if verbose:
        print_solution(list_alphas, optim, args)

    if saver is not None:
        saver.save_checkpoint(model, optim, args, epoch=0, metric=1)

    # Set alpha and beta to argmax
    model.set_argmax_alpha_beta() if hasattr(model, 'set_argmax_alpha_beta') else model.module.set_argmax_alpha_beta()

    if verbose:
        print_solution(list_alphas, optim, args)

    if saver is not None:
        saver.save_checkpoint(model, optim, args, epoch=1, metric=5)


def optimize_diff_predictor(predictor, optimizer, steps):
    predictor.model.to(device='cpu')
    predictor.eval()
    predictor.requires_grad = False
    alpha_attn, beta_attn = smallest_sol(*generate_cnames(optimizer.alpha_blocks, optimizer.beta_blocks))
    alpha_attn = torch.tensor(alpha_attn, requires_grad=True, device=next(predictor.model.parameters()).device)
    beta_attn = torch.tensor(beta_attn, requires_grad=True, device=next(predictor.model.parameters()).device)
    bar = tqdm(range(steps)) if DistributedManager.local_rank == 0 else range(steps)
    for _ in bar:
        alpha_attn.grad = torch.zeros_like(alpha_attn)
        beta_attn.grad = torch.zeros_like(beta_attn)

        x = torch.cat([alpha_attn, beta_attn])
        criterion = -predictor(x)
        criterion.backward()
        with torch.no_grad():
            bcfw_step(alpha_attn, beta_attn, optimizer)

    return alpha_attn, beta_attn


def smallest_sol(anames, bnames):
    avals = [1.0 if name[0] == 'a' and name.split('_')[-1] == '0' else 0.0 for name in anames]
    bvals = [1.0 if name[0] == 'b' and name.split('_')[-1] == '1' else 0.0 for name in bnames]

    return avals, bvals


def generate_cnames(alpha_blocks, beta_blocks):
    aname, bname = [], []
    alpha_offset = 0
    for beta_block, beta_block_size in enumerate(beta_blocks):
        aname += [f'a_{beta_block}_0_{c}' for c in range(alpha_blocks[alpha_offset])]
        alpha_offset += 1
        for b in range(1, beta_block_size + 1):
            bname.append(f'b_{beta_block}_{b}')
            aname += [f'a_{beta_block}_{b}_{c}' for c in range(alpha_blocks[alpha_offset])]
            alpha_offset += 1

    assert alpha_offset == len(alpha_blocks)

    return aname, bname


def bcfw_step(alpha_attn, beta_attn, bcfw_opt):
    prob = np.random.random()
    if prob < 0.5 and bcfw_opt.k > 1:
        alpha_attn.data = bcfw_opt.alpha_lp(alpha_attn, bcfw_opt.alpha_blocks, bcfw_opt.latency_vec,
                                            alpha_attn.grad, beta_attn, bcfw_opt.beta_blocks)
    else:
        beta_attn.data = bcfw_opt.beta_lp(alpha_attn, bcfw_opt.latency_vec, bcfw_opt.alpha_blocks,
                                          beta_attn, beta_attn.grad, bcfw_opt.beta_blocks)


def print_solution(list_alphas, optim, args, alpha_attn=None, beta_attn=None):
    alpha_attention_vec, _, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)

    alpha_attention_vec = alpha_attn if alpha_attn is not None else alpha_attention_vec
    beta_attention_vec = beta_attn if beta_attn is not None else beta_attention_vec

    if args.local_rank != 0:
        return

    print('alpha:')
    reshaped = np.reshape(alpha_attention_vec, (len(alpha_blocks), -1)).copy()
    print(reshaped)
    print('beta:')
    reshaped = np.reshape(beta_attention_vec, (len(beta_blocks), -1)).copy()
    print(reshaped)
    check_rounding_constraint(optim, alpha_attention_vec, beta_attention_vec, alpha_blocks, beta_blocks)


def check_rounding_constraint(optim, alpha, beta, alpha_blocks, beta_blocks):
    latency = optim.latency_formula(alpha, beta, optim.fixed_latency)
    print('constraint: {} <= {}'.format(latency, optim.T))
    alpha = argmax_attention(alpha, alpha_blocks)
    beta = argmax_attention(beta, beta_blocks)
    latency = optim.latency_formula(alpha, beta, optim.fixed_latency)
    print('argmax constraint: {} <= {}'.format(latency, optim.T))


def argmax_attention(attention, blocks):
    offset = 0
    argmax_attention_vec = np.zeros_like(attention)
    for block in blocks:
        argmax = np.argmax(attention[offset: offset + block])
        argmax_attention_vec[offset: offset + block] = 0
        argmax_attention_vec[offset + argmax] = 1
        offset += block

    return argmax_attention_vec


if __name__ == '__main__':
    main()


