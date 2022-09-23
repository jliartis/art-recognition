import os
import argparse
import configparser

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from data import Artists
from models import RegNet, model_dict


def run(net, device, loader, optimizer, scheduler, split='val', epoch=0,
        train=False, dry_run=False,
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
    # print("run started")
    for (imgs, img_class_ids) in loader:
        # print("loading images to gpu")
        imgs, img_class_ids = (
            imgs.to(device), img_class_ids.to(device).long()
        )
        # print("loaded images to gpu")

        if train:
            # print("setting gradients to 0")
            net.zero_grad(set_to_none=True)

        # print("forward pass")
        with torch.set_grad_enabled(train):
            output = net(imgs)
            loss = F.cross_entropy(output, img_class_ids,
                                   label_smoothing=smoothing)
        _, preds = torch.max(output, 1)

        if train:
            # print("starting backward pass")
            loss.backward()
            optimizer.step()
            # print("ended backward pass")
        running_loss += loss.item()
        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        if dry_run:
            break

    if train:
        scheduler.step()

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)
    print('Epoch: {}.. '.format(epoch),
          '{} Loss: {:.3f}.. '.format(split, running_loss / len(loader)),
          '{} Accuracy: {:.3f}.. '.format(split, bal_acc)
          )

    return running_loss / len(loader), bal_acc


def train(net, base_path, train_ids_fn, val_ids_fn, images_dir,
          checkpoint_fname, config, device=torch.device('cpu'), dry_run=False,
          plot=False, checkpoint_freq=10):
    train_dataset = Artists(base_path, train_ids_fn, images_dir, True)
    val_dataset = Artists(base_path, val_ids_fn, images_dir, False)

    batch_size = int(config['MAIN']['batch_size'])
    workers = int(config['MAIN']['workers'])
    epochs = int(config['MAIN']['epochs'])
    warmup = int(config['MAIN']['warmup'])
    freeze = int(config['MAIN']['freeze'])
    label_smoothing = float(config['MAIN']['label_smoothing'])

    lr = float(config['SGD']['lr'])
    momentum = float(config['SGD']['momentum'])
    weight_decay = float(config['SGD']['weight_decay'])
    nesterov = config['SGD'].getboolean('nesterov')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    cur_best_val_loss = np.inf

    optimizer = torch.optim.SGD(
        [param for param in net.parameters() if param.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup + epochs)

    if plot:
        # plt.ion()
        # figure, ax = plt.subplots()
        train_loss_list = []
        val_loss_list = []
        train_acc_list = list()
        val_acc_list = list()
        # train_line, = ax.plot([], [])
        # val_line, = ax.plot([], [])
    for epoch in range(warmup):
        train_loss, train_acc = run(net, device, train_loader, optimizer, scheduler,
                                    split='train', epoch=epoch, train=True,
                                    dry_run=dry_run, smoothing=label_smoothing)
        val_loss, val_acc = run(net, device, val_loader, optimizer, scheduler,
                                split='val', epoch=epoch, train=False, dry_run=dry_run,
                                smoothing=label_smoothing)

        if dry_run:
            break

    net.finetune(freeze, optimizer)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
    #                                    net.parameters()),
    #                             lr=scheduler.get_last_lr()[0], momentum=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup + epochs)

    for epoch in range(epochs):
        print("epoch ", epoch, "/", epochs, ";")
        print("starting training")
        train_loss, train_acc = run(net, device, train_loader, optimizer, scheduler,
                                    split='train', epoch=epoch, train=True,
                                    dry_run=dry_run, smoothing=label_smoothing)
        print("ended training")
        print("starting validation")
        val_loss, val_acc = run(net, device, val_loader, optimizer, scheduler,
                                split='val', epoch=epoch, train=False, dry_run=dry_run,
                                smoothing=label_smoothing)
        print("ended validation")
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        }

        if plot and epoch > 3:
            # Training-validation loss
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            # train_line.set_ydata(train_loss_list)
            # train_line.set_xdata(range(warmup + epoch + 1))
            # val_line.set_ydata(val_loss_list)
            # val_line.set_xdata(range(warmup + epoch + 1))
            # ax.set_ylim([0, 1.1 * max(*train_loss_list, *val_loss_list)])
            # ax.set_xlim([0, warmup + epoch + 1])
            # figure.tight_layout()
            # figure.canvas.draw()
            # figure.canvas.flush_events()
            plt.plot(train_loss_list, label="train loss")
            plt.plot(val_loss_list, label="val loss")
            plt.legend()
            plt.savefig(checkpoint_fname + "_loss.png")
            plt.clf()

            # Training and validation accuracy

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            plt.plot(train_acc_list, label="train accuracy")
            plt.plot(val_acc_list, label="val accuracy")
            plt.legend()
            plt.savefig(checkpoint_fname + "_acc.png")
            plt.clf()

        if epoch % checkpoint_freq:
            with open(checkpoint_fname + "{:03d}.pt".format(epoch), "wb") as fp:
                torch.save(checkpoint, fp)

        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(checkpoint_fname + "_best.pt")
            with open(checkpoint_fname + "_best.pt", "wb") as fp:
                torch.save(checkpoint, fp)
            cur_best_val_loss = val_loss

        if dry_run:
            break

    # if plot:
    #     plt.plot(train_loss_list, label="train loss")
    #     plt.plot(val_loss_list, label="val loss")
    #     plt.legend
    #     plt.savefig(checkpoint_fname + ".png")


def main():
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
        "--config-fname",
        type=str,
        help="File containing training configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training and validation loss"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=62
    )

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_fname)

    use_cuda = (torch.cuda.is_available()
                and not config['MAIN'].getboolean('no_cuda'))
    device = torch.device('cuda' if use_cuda else 'cpu')

    warmup_layers = config['MAIN'].get('warmup_layers', None)
    model = config['MAIN']['model']
    net = RegNet(args.num_classes, model_dict[model],
                 None if warmup_layers is None else int(warmup_layers)
                 ).to(device)

    checkpoint_freq = int(config['MAIN'].get('checkpoint_frequency', '10'))

    train(net, args.base_path, args.train_ids_fn, args.val_ids_fn,
          args.images_dir, args.checkpoint_fname, config, device=device,
          dry_run=args.dry_run, plot=args.plot,
          checkpoint_freq=checkpoint_freq)


if __name__ == "__main__":
    main()
