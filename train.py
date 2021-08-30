# -*- coding:utf-8 -*-

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃CREATE BY SNIPER┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛
"""
import os
import argparse
from importlib import import_module

import config

parser = argparse.ArgumentParser(description='model config')
parser.add_argument('-model_name', type=str, help='the model .py name', required=True)
parser.add_argument('-dpot', type=str, default='0', help='dropout rate', required=False)
parser.add_argument('-save_name', type=str, help='model save name(not path)', required=True)
parser.add_argument('-gpu', type=str, help='index of gpu', required=False)

parser.add_argument('-roberta', type=bool, default=False, help='use  roberta', required=False)
parser.add_argument('-macbert', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-nezha_base', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-nezha_large', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-roformer_char', type=bool, default=False, help='use  roformer char level', required=False)
parser.add_argument('-electra', type=bool, default=False, help='use  electra', required=False)
parser.add_argument('-wwm_bert', type=bool, default=False, help='use  electra', required=False)
parser.add_argument('-wwm_roberta', type=bool, default=False, help='use  electra', required=False)
parser.add_argument('-wwm_roberta_large', type=bool, default=False, help='use  electra', required=False)

parser.add_argument('-emoji', type=bool, default=False, help='add emoji', required=False)
parser.add_argument('-mask_rate', type=str, default='0.2', help='mask rate', required=False)
parser.add_argument('-bsz', type=str, default='64', help='batch size', required=False)
parser.add_argument('-epoch', type=str, default='10', help='epoch', required=False)
parser.add_argument('-lr', type=str, default='1e-5', help='learn ', required=False)
parser.add_argument('-data_trd', type=str, default='0.95', help='generate train data threshold ', required=False)
parser.add_argument('-random_rate', type=str, default='0', help='generate train random throw ratio ', required=False)
parser.add_argument('-zero_rate', type=str, default='0', help='cant describe ', required=False)
parser.add_argument('-grad_step', type=str, default='1', help='grad step', required=False)
parser.add_argument('-recompute', type=bool, default=False, help='use recompute', required=False)
parser.add_argument('-k', type=int, default=0, help='k-fold k', required=False)

args = parser.parse_args()
# 梯度累积
config.GRAD_STEP = int(args.grad_step)
# 重计算
if args.recompute:
    os.environ['RECOMPUTE'] = '1'

# data generate
config.TRAIN_DATA_THRESHOLD = float(args.data_trd)
if config.TRAIN_DATA_THRESHOLD == 0:
    config.WITH_MORE_DATA = False
config.RANDOM_TRAIN_DATA = float(args.random_rate)
config.ZERO_RATE = float(args.zero_rate)

# gpu
if args.gpu is not None:
    config.GPU_INDEX = args.gpu
else:
    config.GPU_INDEX = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_INDEX

# dropout rate
if args.dpot is not None:
    config.DROPOUT_RATE = float(args.dpot)

# save name
if args.save_name is None:
    config.SAVE_MODEL = False
else:
    config.MODEL_SAVE_PATH = './outputs/' + args.save_name

# model name
config.MODEL_NAME = args.model_name

# emoji
config.REMOVE_EMOJI = args.emoji

# k fold
config.K_fold = args.k

# roberta
if args.roberta is True:
    config.BERT_VOCAB_PATH = './roberta/vocab.txt'
    config.BERT_CKPT_PATH = './roberta/roberta_zh_large_model.ckpt'
    config.BERT_CONFIG_PATH = './roberta/bert_config_large.json'

# macbert base
if args.macbert is True:
    config.BERT_VOCAB_PATH = './macbert/vocab.txt'
    config.BERT_CKPT_PATH = './macbert/chinese_macbert_base.ckpt'
    config.BERT_CONFIG_PATH = './macbert/macbert_base_config.json'

# nezha base
if args.nezha_base is True:
    config.BERT_VOCAB_PATH = './nezha/vocab.txt'
    config.BERT_CKPT_PATH = './nezha/model.ckpt-691689'
    config.BERT_CONFIG_PATH = './nezha/bert_config.json'
    config.NEZHA = True

# nezha large
if args.nezha_large is True:
    config.BERT_VOCAB_PATH = './NEZHA-Large-WWM/vocab.txt'
    config.BERT_CKPT_PATH = './NEZHA-Large-WWM/model.ckpt-346400'
    config.BERT_CONFIG_PATH = './NEZHA-Large-WWM/bert_config.json'
    config.NEZHA = True

# roformer_char
if args.roformer_char is True:
    config.BERT_VOCAB_PATH = './roformer_char/vocab.txt'
    config.BERT_CKPT_PATH = './roformer_char/bert_model.ckpt'
    config.BERT_CONFIG_PATH = './roformer_char/bert_config.json'
    config.ROFOMER = True

if args.electra is True:
    config.BERT_VOCAB_PATH='/home/hadoop-aipnlp/cephfs/data/yuanxianqi/Matchv120210629/electra_180g_large/vocab.txt'
    config.BERT_CKPT_PATH = '/home/hadoop-aipnlp/cephfs/data/yuanxianqi/Matchv120210629/electra_180g_large/electra_180g_large.ckpt'
    config.BERT_CONFIG_PATH = '/home/hadoop-aipnlp/cephfs/data/yuanxianqi/Matchv120210629/electra_180g_large/large_discriminator_config.json'
    config.ELECTRA = True

if args.wwm_bert is True:
    config.BERT_VOCAB_PATH='/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
    config.BERT_CKPT_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
    config.BERT_CONFIG_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'


if args.wwm_roberta is True:
    config.BERT_VOCAB_PATH='/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
    config.BERT_CKPT_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
    config.BERT_CONFIG_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'

if args.wwm_roberta_large is True:
    config.BERT_VOCAB_PATH='/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
    config.BERT_CKPT_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
    config.BERT_CONFIG_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'

# batch size
config.BATCH_SIZE = int(args.bsz)

# epoch
config.EPOCH = int(args.epoch)

# mask rate
config.MASK_RATE = float(args.mask_rate)

# learn rate
config.LEARN_RATE = float(args.lr)

x = import_module(args.model_name)
x.train()
