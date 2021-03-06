import _init_paths
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from lib.datasets.read import LinemodDataset
from lib.noLoss import Resnet18_8s
from lib.utils import *
from trainers.generate import CoreTrainer
import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--lambda_sym_cor', type=float, default=0.1)
    parser.add_argument('--lambda_mask', type=float, default=1.0)
    parser.add_argument('--lambda_pts2d', type=float, default=10.0)
    parser.add_argument('--lambda_graph', type=float, default=0.1)
    parser.add_argument('--object_name', type=str, default='ape')
    parser.add_argument('--dataset', type=str, default='data/linemode')
    parser.add_argument('--save_dir', type=str, default='saved_weights/occlusion_linemod/ape')
    parser.add_argument('--load_dir', type=str, default='saved_weights/occlusion_linemod/ape/checkpoints/0.02/499')
    parser.add_argument('--test_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--num_keypoints', type=int, default=8)
    parser.add_argument('--use_keypoint', type=int, default=1, help='boolean flag indicating whether keypoints are used in pose regression')
    parser.add_argument('--use_edge', type=int, default=1, help='boolean flag indicating whether edge are used in pose regression')
    parser.add_argument('--use_symmetry', type=int, default=1, help='boolean flag indicating whether symmetry are used in pose regression')
    args = parser.parse_args()
    return args

def initialize(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_loaders(args):
    full_set = LinemodDataset(args.dataset, object_name=args.object_name)
    train_size = int(0.0 * len(full_set))
    test_size = len(full_set) - train_size
    train_set, test_set = torch.utils.data.random_split(full_set, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def setup_model(args):
    model = Resnet18_8s(num_keypoints=args.num_keypoints)
    if cuda:
        model = nn.DataParallel(model).cuda()
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_session(model, optimizer, args)
    else:
        start_epoch = 0
    return model, optimizer, start_epoch

# main function
if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    train_loader, test_loader = setup_loaders(args)
    model, optimizer, start_epoch = setup_model(args)
    trainer = CoreTrainer(model,
                          optimizer,
                          train_loader,
                          test_loader,
                          args)
    trainer.generate_data()
