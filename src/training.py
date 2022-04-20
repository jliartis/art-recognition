import os
import argparse

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler

from data import Artists
from models import RegNet


def run(net, device, loader, optimizer, scheduler, split='val', epoch=0, train=False, dry_run=False,
        smoothing=0.0):
    if train:
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    loader = tqdm(
        loader,
        ncols=0,
        desc='{1} E{0:02d}'.format(epoch, 'train' if train else 'val')
    )

    running_loss = 0
    preds_all = []
    labels_all = []
    for (imgs, img_class_ids) in loader:
        imgs, img_class_ids = (
            imgs.to(device), img_class_ids.to(device).long()
        )

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(imgs)
            loss = F.cross_entropy(output, img_class_ids, label_smoothing=smoothing)

        _, preds = torch.max(output, 1)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        if dry_run:
            break

    if train:
        scheduler.step()

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    print('Epoch: {}.. '.format(epoch),
          '{} Loss: {:.3f}.. '.format(split, running_loss / len(loader)),
          '{} Accuracy: {:.3f}.. '.format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(net, base_path, train_ids_fn, val_ids_fn, images_dir, model_fname, batch_size=16, lr=1e-2,
          warmup=0, n_layers=0, epochs=10, device="cpu", smoothing=0, num_workers=6, dry_run=False):
    train_dataset = Artists(base_path, train_ids_fn, images_dir, True)
    val_dataset = Artists(base_path, val_ids_fn, images_dir, False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    cur_best_val_loss = np.inf

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(warmup):
        _ = run(net, device, train_loader, optimizer, scheduler, split='train',
                epoch=epoch, train=True, dry_run=dry_run, smoothing=smoothing)
        _ = run(net, device, val_loader, optimizer, scheduler, split='val',
                epoch=epoch, train=False, dry_run=dry_run, smoothing=smoothing)

        if dry_run:
            break

    net.finetune(n_layers)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       net.parameters()), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epochs):
        _ = run(net, device, train_loader, optimizer, scheduler, split='train',
                epoch=epoch, train=True, dry_run=dry_run, smoothing=smoothing)
        val_loss = run(net, device, val_loader, optimizer, scheduler, split='val',
                       epoch=epoch, train=False, dry_run=dry_run, smoothing=smoothing)

        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(model_fname)
            torch.save(net.state_dict(), model_fname)
            cur_best_val_loss = val_loss

        if dry_run:
            break


def main():
    num_classes = 20
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-path",
        type=str,
        help="Base data directory"
    )
    parser.add_argument(
        "--train-ids-fn",
        type=str,
        help="File containing image ids of training set"
    )
    parser.add_argument(
        "--val-ids-fn",
        type=str,
        help="File containing image ids of validation set"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Subdirectory containing the images"
    )
    parser.add_argument(
        "--checkpoint-fname",
        type=str,
        help="Filename to use for storing model checkpoints"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick run through code without looping"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for backpropagation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate for SGD optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing regularization to apply"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of training epochs without updating pretrained model weights"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of layers to keep frozen even after warmup"
    )

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = RegNet(num_classes).to(device)
    train(net, args.base_path, args.train_ids_fn, args.val_ids_fn, args.images_dir,
          args.checkpoint_fname, batch_size=args.batch_size, lr=args.learning_rate,
          warmup=args.warmup, n_layers=args.freeze, epochs=args.epochs, device=device,
          smoothing=args.label_smoothing, num_workers=args.workers,
          dry_run=args.dry_run)


if __name__ == "__main__":
    main()
