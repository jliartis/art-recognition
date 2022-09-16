import argparse
import pickle

import torch

from src.models import RegNet, regnet_y_1_6gf
from src.data import Artists


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-fname', type=str)
parser.add_argument('--data-fn', type=str)
parser.add_argument('--features-fn', type=str)
parser.add_argument("--base-path", type=str)
parser.add_argument("--ids-fn", type=str)
parser.add_argument("--images-dir", type=str)
args = parser.parse_args()

device = torch.device('cuda')
net = RegNet(62, regnet_y_1_6gf, frozen_layers=0).to(device)

checkpoint_fn = args.checkpoint_fname
checkpoint = torch.load(checkpoint_fn)
net.load_state_dict(checkpoint['model_state'])
net.eval()

data = Artists(args.base_path, args.ids_fn, args.images_dir, train=False)
features = [(net.avgpool(net.features(x.cuda().unsqueeze(0))).flatten().cpu().numpy(), y) for x, y in data]

with open(args.features_fn, 'wb') as fp:
    pickle.dump(features, fp)
