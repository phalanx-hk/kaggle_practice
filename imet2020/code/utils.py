import os
import random
from glob import glob

import numpy as np
import torch
import mlcrate as mlc
import pandas as pd
from loguru import logger


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def find_exp_num(log_path: str) -> int:
    log_files = glob(os.path.join(log_path, '*.csv'))
    if not len(log_files):
        return 1
    else:
        exp_nums = [os.path.splitext(i)[0].split(
            '/')[-1].split('_')[-1] for i in log_files]
        exp_nums = list(map(int, exp_nums))
        return max(exp_nums) + 1


def remove_abnormal_exp(log_path: str, config_path: str) -> None:
    log_files = glob(os.path.join(log_path, '*.csv'))
    for log_file in log_files:
        log_df = pd.read_csv(log_file)
        if len(log_df) == 0:
            exp_num = os.path.splitext(
                log_file)[0].split('/')[-1].split('_')[-1]
            os.remove(os.path.join(log_path, f'exp_{exp_num}.log'))
            os.remove(os.path.join(log_path, f'exp_{exp_num}.csv'))
            os.remove(os.path.join(config_path, f'exp_{exp_num}.yaml'))


def get_logger(config, exp_num):
    logger.remove()
    logger.add(os.path.join(config.log_path, f'exp_{exp_num}.log'))
    header = config.header.split(' ')
    csv_log = mlc.LinewiseCSVWriter(os.path.join(
        config.log_path, f'exp_{exp_num}.csv'), header=header)
    return logger, csv_log


def save_model(save_name, epoch, loss, acc, model, optimizer):
    state = {
        'epoch': epoch,
        'loss': loss,
        'acc': acc,
        'weight': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state, save_name)
