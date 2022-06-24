import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from data import Artists
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-fname",
        type=str,
    )
    parser.add_argument(
        "--base-path",
        type=str
    )
    parser.add_argument(
        "--image-ids-fn",
        type=str
    )
    parser.add_argument(
        "--images-dir",
        type=str
    )
    parser.add_argument(
        "--features-fname",
        type=str
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true"
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu else "cuda")
    model, _, _, _ = load_checkpoint(args.checkpoint_fname)
    model.eval()
    model.to(device)
    dataset = Artists(args.base_path, args.image_ids_fn, args.images_dir, False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )

    x, y = None, None

    for imgs, img_class_ids in tqdm(loader):
        imgs = imgs.to(device)
        features = model.features(imgs)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

        if x is None:
            x = features.cpu().detach().numpy()
            y = img_class_ids.cpu().numpy()
        else:
            x = np.concatenate((x, features.cpu().detach().numpy()))
            y = np.concatenate((y, img_class_ids.cpu().numpy()))

    with open(args.features_fname, "wb") as fp:
        pickle.dump((x, y), fp)


if __name__ == "__main__":
    main()
