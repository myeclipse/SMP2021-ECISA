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
parser.add_argument('-save_name', type=str, help='model save name(not path)', required=True)
parser.add_argument('-gpu', type=str, help='index of gpu', required=False)

parser.add_argument('-roberta', type=bool, default=False, help='use  roberta', required=False)
parser.add_argument('-macbert', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-nezha_base', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-nezha_large', type=bool, default=False, help='use  bert base', required=False)
parser.add_argument('-roformer_char', type=bool, default=False, help='use  roformer char level', required=False)

parser.add_argument('-no_softmax', type=bool, default=False, help='do not use softmax', required=False)

args = parser.parse_args()

# gpu
if args.gpu is not None:
    config.GPU_INDEX = args.gpu
else:
    config.GPU_INDEX = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_INDEX

# save path
config.MODEL_SAVE_PATH = './outputs/' + args.save_name

if args.no_softmax:
    config.NO_SOFTMAX = True

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

x = import_module(args.model_name)
x.predict()
