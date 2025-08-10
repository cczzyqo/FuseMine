#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
import dgl
import logging
from einops.einops import rearrange
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)


def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def load_yaml_config(yaml_file):
    """加载YAML配置文件"""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config(base_config,dataset):
    
    # 检查基础配置文件是否存在
    if not os.path.exists(base_config):
        raise FileNotFoundError(f"基础配置文件不存在: {base_config}")
    
    dataset_conf = os.path.join('./config', f"{dataset}.yaml")
    # 构建数据集配置文件路径
    if not os.path.exists(dataset_conf):
        raise FileNotFoundError(f"数据集配置文件不存在: {dataset_conf}")
    
    # 使用OmegaConf直接加载YAML文件
    base_conf = OmegaConf.load(base_config)
    print(f"已加载基础配置: {base_config}")
    
    # 加载数据集特定配置
    dataset_conf = OmegaConf.load(dataset_conf)
    print(f"已加载数据集配置: {dataset_conf}")
    
    # 合并配置 (后面的会覆盖前面的)
    merged_conf = OmegaConf.merge(base_conf, dataset_conf)
    print(f"已合并配置文件")
    
    # # 应用命令行参数覆盖 (如果指定了)
    # cli_conf = {}
    # if args.batch_size is not None:
    #     cli_conf.setdefault("Global", {})["Batch_Size"] = args.batch_size
    # if args.lr is not None:
    #     cli_conf.setdefault("Global", {})["LR"] = args.lr
    # if args.seed is not None:
    #     cli_conf.setdefault("Global", {})["Seed"] = args.seed
    # if args.output_dir is not None:
    #     cli_conf.setdefault("Result", {})["Output_Dir"] = args.output_dir
    
    # if cli_conf:
    #     cli_conf_obj = OmegaConf.create(cli_conf)
    #     merged_conf = OmegaConf.merge(merged_conf, cli_conf_obj)
    #     print(f"已应用命令行参数覆盖")
    
    return merged_conf

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def calculate_metrics(y_true, y_pred, y_score):
    """计算分类指标"""
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix': cm,
        'specificity': specificity,
        'sensitivity': sensitivity
    }


