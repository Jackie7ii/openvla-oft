import argparse
import torch

from .detr.detr_vae import build as build_vae
from .detr.detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args):
    return build_vae(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)


def get_args_parser():
    parser = argparse.ArgumentParser('ACT model args', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--camera_names', default=[], type=list)

    # Transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=400, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--kl_weight', default=10, type=int)
    parser.add_argument('--chunk_size', default=None, type=int)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser
