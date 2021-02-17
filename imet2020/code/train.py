import os
import atexit
from argparse import ArgumentParser

import mlcrate as mlc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from fastprogress import progress_bar, master_bar
from sklearn.metrics import fbeta_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from timm.models import *

from dataset import MetDataset
from augmentations import *
from utils import find_exp_num, get_logger, remove_abnormal_exp, seed_everything, save_model

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-config", required=True)
    parser.add_argument("options", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(remove_abnormal_exp, log_path=config.log_path,
                    config_path=config.config_path)
    seed_everything(config.seed)

    exp_num = find_exp_num(log_path=config.log_path)
    exp_num = str(exp_num).zfill(3)
    config.weight_path = os.path.join(config.weight_path, f'exp_{exp_num}')
    OmegaConf.save(config, os.path.join(
        config.config_path, f'exp_{exp_num}.yaml'))
    logger, csv_logger = get_logger(config, exp_num)
    timer = mlc.time.Timer()
    logger.info(mlc.time.now())
    logger.info(f'config: {config}')

    train_df = pd.read_csv(os.path.join(config.root, 'train.csv'))
    X = train_df['id']
    X = np.array([os.path.join(config.root, 'train', f'{i}.png') for i in X])
    y = train_df['attribute_ids']
    y = np.array([list(map(int, i.split(' '))) for i in y])
    y = [np.eye(3474)[i].sum(0) for i in y]

    transform = eval(config.transform.name)(config.transform.size)
    logger.info(f'augmentation: {transform}')
    strong_transform = eval(config.strong_transform.name)
    logger.info(f'strong augmentation: {config.strong_transform.name}')

    mskf = MultilabelStratifiedKFold(
        n_splits=config.train.n_splits, shuffle=True, random_state=config.seed)
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_data = MetDataset(X_train, y_train, transform['albu_train'])
        val_data = MetDataset(X_val, y_val, transform['albu_val'])
        train_loader = DataLoader(train_data, **config.train_loader)
        val_loader = DataLoader(val_data, **config.val_loader)

        model = eval(config.model)(True)
        if 'fc.weight' in model.state_dict().keys():
            model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
        elif 'classifier.weight' in model.state_dict().keys():
            model.classifier = nn.Linear(
                model.classifier.in_features, config.train.num_labels)
        model = model.cuda()
        optimizer = eval(model.optimizer.name)(
            model.parameters(), lr=config.optimizer.lr)
        scheduler = eval(model.scheduler.name)(
            optimizer, config.epoch // config.scheduler.cycle)
        criterion = eval(config.loss)()
        scaler = GradScaler()

        best_acc = 0
        best_loss = 1e10
        mb = master_bar(range(config.epoch))
        for epoch in mb:
            timer.add('train')
            train_loss, train_acc = train(
                config, model, transform, strong_transform, train_loader, optimizer, criterion, mb, epoch, scaler)
            train_time = timer.fsince('train')

            timer.add('val')
            val_loss, val_acc = validate(
                config, model, transform, val_loader, criterion, mb, epoch)
            val_time = timer.fsince('val')

            output1 = 'epoch: {} train_time: {} validate_time: {}'.format(
                epoch, train_time, val_time)
            output2 = 'train_loss: {:.3f} train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f}'.format(
                epoch + 1, train_loss, train_acc, val_loss, val_acc)
            logger.info(output1)
            logger.info(output2)
            mb.write(output1)
            mb.write(output2)
            csv_logger.write(
                [epoch, train_loss, train_acc, val_loss, val_acc])

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = os.path.join(config.weight_path, 'best_loss.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer)
            if val_acc > best_acc:
                best_acc = val_acc
                save_name = os.path.join(config.weight_path, 'best_acc.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer)

            save_name = os.path.join(config.weight_path, 'last_epoch.pth')
            save_model(save_name, epoch, val_loss,
                       val_acc, model, optimizer)


@torch.enable_grad()
def train(config, model, transform, strong_transform, loader, optimizer, criterion, mb, epoch, scaler):
    preds = []
    gt = []
    losses = []
    scores = []

    model.train()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)
        if epoch < config.train.epoch - 5:
            images, labels_a, labels_b, lam = strong_transform(
                images, **config.strong_transform.params)
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels_a) * lam + \
                    criterion(logits, labels_b) * (1 - lam)
                loss /= config.train.accumulte
        else:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels) / config.train.accumulte

        scaler.scale(loss).backward()
        if not (it + 1) % config.train.accumulte:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        logits = (logits.sigmoid() > 0.5).detach().cpu().numpy().astype(int)
        labels = labels.detach().cpu().numpy().astype(int)
        score = fbeta_score(labels, logits, beta=2, average='samples')
        scores.append(score)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

        mb.child.comment = 'loss: {:.3f} avg_loss: {:.3f} acc: {:.3f} avg_acc: {:.3f}'.format(
            loss.item(),
            np.mean(losses),
            score,
            np.mean(scores),
        )

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = fbeta_score(gt, preds, beta=2, average='samples')
    return np.mean(losses), score


@torch.no_grad()
def validate(config, model, transform, loader, criterion, mb, device):
    preds = []
    gt = []
    losses = []

    model.eval()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)

        logits = model(images)
        loss = criterion(logits, labels) / config.train.accumulte

        logits = (logits.sigmoid() > 0.5).cpu().numpy().astype(int)
        labels = labels.cpu().numpy().astype(int)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = fbeta_score(gt, preds, beta=2, average='samples')
    return np.mean(losses), score


if __name__ == '__main__':
    main()
