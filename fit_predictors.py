#!/usr/bin/env python
import pickle
import sys

import matplotlib.pyplot as plt
import yaml
from scipy.stats import stats

from external.nas_parser import *
from nas.nas_utils.general_purpose import extract_structure_param_list
from nas.nas_utils.predictor import construct_predictors, \
    closed_form_solution, predict
from nas.src.optim.block_frank_wolfe import flatten_attention_latency_grad_alpha_beta_blocks
from timm import create_model
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

parser = argparse.ArgumentParser(description='PyTorch ImageMLP Training')
# Dataset / Model parameters
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Write to TensorboardX')
parser.add_argument("--debug", action='store_true', default=False,
                    help="logging is set to debug")

# NAS Predictors
parser.add_argument('--predictor_type', default='bilinear', choices=['bilinear', 'quadratic', 'mlp'],
                    help='The type of the predictor model (default: bilinear)')
parser.add_argument('--fit_accuracy_lut_filename', default=None,
                    help='The filename of the measured accuracy LUT to fit (default: None)')
parser.add_argument('--closed_form_solution', action='store_true', default=False,
                    help="Fit the predictor by a linear regression's closed form solution")
parser.add_argument('--test_accuracy_lut_filename', default=None,
                    help='The filename of the measured accuracy LUT for test architectures (default: None)')
parser.add_argument('--num_samples_to_fit', type=int, default=None,
                    help='The number of samples to fit the accuracy predictor to')
parser.add_argument('--predictor_ckpt_filename', default=None,
                    help='The output filename of the trained predictor checkpoint (default: None)')
parser.add_argument('--test_figure_filename', default=None,
                    help='The output filename for the output figures (default: None)')
parser.add_argument("--svd_truncate", type=int, default=3000,
                    help="The number of smallest singular values to truncate for regularizing the closed solution")

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

    model = create_model(
        'mobilenasnet',
        pretrained=False,
        num_classes=1,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
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

    torch.cuda.empty_cache()
    list_alphas, _ = extract_structure_param_list(model)
    if args.fit_accuracy_lut_filename is not None:
        print('------------------------ Fitting Predictors --------------------------------')
        with open(args.fit_accuracy_lut_filename, 'rb') as f:
            fit_accuracy_dict = pickle.load(f)

        xs, ys, xs_oh = [], [], []
        for i, record in enumerate(fit_accuracy_dict.values()):
            if args.num_samples_to_fit is not None and i >= args.num_samples_to_fit:
                break
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

        # Fit the predictor
        alpha_vec, _, _, _, beta_vec, _, _ = flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
        alphas = len(alpha_vec)
        betas = len(beta_vec)
        predictor = construct_predictors(args.predictor_type, alphas, betas)
        if args.closed_form_solution:
            predictor = closed_form_solution(predictor, X, Y, args.svd_truncate)
        else:
            predictor.fit(torch.stack(X).numpy(), torch.stack(Y).numpy().squeeze(), lr=1e-6, weight_decay=1e-6)

        predictor.model.eval()
        if args.predictor_ckpt_filename is not None and not os.path.exists(args.predictor_ckpt_filename):
            torch.save(predictor.state_dict(), args.predictor_ckpt_filename)
            print(f'Saved {args.predictor_ckpt_filename}')

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


if __name__ == '__main__':
    main()


