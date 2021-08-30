# @Time     : 2021/8/5 9:56 上午
# @Author   : yxq
# @FileName : predict_final.py


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

args = parser.parse_args()

config.MODEL_NAME = args.save_name

if args.gpu is not None:
    config.GPU_INDEX = args.gpu
else:
    config.GPU_INDEX = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_INDEX

config.SAVE_NAME = args.save_name

config.MODEL_SAVE_PATH = './outputs/' + args.save_name
x = import_module(args.model_name)
x.predict_final()