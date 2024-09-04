from argparse import Namespace
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import torch
from torch.distributed.elastic.multiprocessing.errors import record

from mega_nerf.opts import get_opts_base
from mega_nerf.runner import Runner


def _get_eval_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None

    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).eval()
    else:
        Runner(hparams).eval()


if __name__ == '__main__':
    main(_get_eval_opts())
