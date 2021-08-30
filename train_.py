# -*- coding:utf-8 -*-

"""
使用方式：
    python train.py -model_name electra_180g_large_dpot_dense_ema_rdrop
                    -save_name electra_180g_large_dpot_dense_ema_rdrop.weights
                    -electra_180g_large True
                    -gpu 2
                    -lr 1e-5
"""

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
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from utils.common_function import ce_with_rdrop


import config

parser = argparse.ArgumentParser(description='model config')
parser.add_argument('-model_name', type=str, help='the model .py name', required=True)
parser.add_argument('-dpot', type=str, default='0', help='dropout rate', required=False)
parser.add_argument('-save_name', type=str, help='model save name(not path)', required=True)
parser.add_argument('-gpu', type=str, help='index of gpu', required=False)
parser.add_argument('-roberta', type=bool, default=False, help='use  roberta', required=False)
parser.add_argument('-emoji', type=bool, default=False, help='add emoji', required=False)
parser.add_argument('-size', type=str, default='64', help='batch_size', required=False)
parser.add_argument('-mask_rate', type=str, default='0.2', help='mask rate', required=False)
parser.add_argument('-bsz', type=str, default='64', help='batch size', required=False)
parser.add_argument('-epoch', type=str, default='10', help='epoch', required=False)
parser.add_argument('-lr', type=str, default='1e-1', help='learn ', required=False)
parser.add_argument('-rmask', type=str, default='0.3', help='random_mask_rate ', required=False)

parser.add_argument('-roberta_HIT', type=bool, default=False, help='use roberta_HIT', required=False)
parser.add_argument('-electra_180g_large', type=bool, default=False, help='use electra_180g_large', required=False)
parser.add_argument('-albert_tiny_zh_google', type=bool, default=False, help='use albert_tiny_zh_google', required=False)
parser.add_argument('-albert_xxlarge', type=bool, default=False, help='use albert_xxlarge', required=False)
parser.add_argument('-NEZHA_Large_WWM', type=bool, default=False, help='use NEZHA-Large-WWM', required=False)
parser.add_argument('-chinese_macbert_base', type=bool, default=False, help='use chinese_maxbert_base', required=False)

parser.add_argument('-ema', type=bool, default=False, help='use ema (or not)', required=False)  # 默认启用 ema 和 R-drop
parser.add_argument('-rdrop', type=bool, default=False, help='use rdrop (or not)', required=False)
parser.add_argument('-mode', type=str, default="1", help="use model1 or model2", required=False)
parser.add_argument('-data_k', type=str, default="0", help="use k_th data", required=False)

args = parser.parse_args()

# gpu
if args.gpu is not None:
    config.GPU_INDEX = args.gpu
else:
    config.GPU_INDEX = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_INDEX

import tensorflow as tf
cp = tf.ConfigProto()
cp.gpu_options.allow_growth = True  # 按需动态分配显存
session = tf.Session(config=cp)

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

# roberta
if args.roberta is True:
    config.BERT_VOCAB_PATH = './roberta/vocab.txt'
    config.BERT_CKPT_PATH = './roberta/roberta_zh_large_model.ckpt'
    config.BERT_CONFIG_PATH = './roberta/bert_config_large.json'
elif args.roberta_HIT is True:
    config.BERT_VOCAB_PATH = './chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
    config.BERT_CKPT_PATH = './chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
    config.BERT_CONFIG_PATH = './chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
elif args.electra_180g_large is True:  # 哈工大版的electra-large
    config.BERT_VOCAB_PATH = './electra_180g_large/vocab.txt'
    config.BERT_CKPT_PATH = './electra_180g_large/electra_180g_large.ckpt'
    config.BERT_CONFIG_PATH = './electra_180g_large/large_discriminator_config.json'
elif args.albert_tiny_zh_google:
    config.BERT_VOCAB_PATH = './albert_tiny_zh_google/vocab.txt'
    config.BERT_CKPT_PATH = './albert_tiny_zh_google/albert_model.ckpt'
    config.BERT_CONFIG_PATH = './albert_tiny_zh_google/albert_config_tiny_g.json'
elif args.albert_xxlarge:
    config.BERT_VOCAB_PATH = './albert_xxlarge/vocab_chinese.txt'
    config.BERT_CKPT_PATH = './albert_xxlarge/model.ckpt-best'
    config.BERT_CONFIG_PATH = './albert_xxlarge/albert_config.json'
elif args.NEZHA_Large_WWM:
    config.BERT_VOCAB_PATH = './NEZHA-Large-WWM/vocab.txt'
    config.BERT_CKPT_PATH = './NEZHA-Large-WWM/model.ckpt-346400'
    config.BERT_CONFIG_PATH = './NEZHA-Large-WWM/bert_config.json'
elif args.chinese_macbert_base:
    config.BERT_VOCAB_PATH = './chinese_macbert_base/vocab.txt'
    config.BERT_CKPT_PATH = './chinese_macbert_base/chinese_macbert_base.ckpt'
    config.BERT_CONFIG_PATH = './chinese_macbert_base/macbert_base_config.json'

# batch size
config.BATCH_SIZE = int(args.bsz)

# epoch
config.EPOCH = int(args.epoch)

# mask rate
config.MASK_RATE = float(args.mask_rate)

# learn rate
config.LEARN_RATE = float(args.lr)

config.data_k = args.data_k

if args.ema:
    AdamEma = extend_with_exponential_moving_average(Adam, name='AdamEma')
    config.OPTIMIZER = AdamEma(config.LEARN_RATE)  # 默认1e-5
else:
    config.OPTIMIZER = Adam(config.LEARN_RATE)

if args.rdrop:
    config.LOSS = ce_with_rdrop
    config.RDROP = True

config.RANDOM_MASK_RATE = float(args.rmask)

config.SAVE_NAME = args.save_name

config.mode = args.mode

x = import_module(args.model_name)
x.train()
# x.get_valid()

# session.run(x.train())
