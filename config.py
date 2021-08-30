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

MODEL_SAVE_PATH = './outputs/bert_base.weights'

REMOVE_EMOJI = False

BERT_VOCAB_PATH = './bert/vocab.txt'
# BERT_VOCAB_PATH = './macbert/vocab.txt'
# BERT_VOCAB_PATH = './roberta/vocab.txt'
#
BERT_CKPT_PATH = './bert/bert_model.ckpt'
# BERT_CKPT_PATH = './macbert/chinese_macbert_base.ckpt'
# BERT_CKPT_PATH = './roberta/roberta_zh_large_model.ckpt'
#
BERT_CONFIG_PATH = './bert/bert_config.json'
# BERT_CONFIG_PATH = './macbert/macbert_base_config.json'
# BERT_CONFIG_PATH = './roberta/bert_config_large.json'

DROPOUT_RATE = 0.2

SAVE_MODEL = True

MODEL_NAME = ''

GPU_INDEX = "1"

MASK_RATE = 0.3

# BATCH_SIZE = 2
BATCH_SIZE = 32

EPOCH = 10

LEARN_RATE = 1e-5
# 生成训练数据阈值
TRAIN_DATA_THRESHOLD = 0.95
# 生成数据比例（留用比）
RANDOM_TRAIN_DATA = 0
# 生成训练数据中0的比例（留用比）
ZERO_RATE = 0
# 使用更多的数据
# WITH_MORE_DATA = False
WITH_MORE_DATA = False
WITH_TRANSLATE = False
TRANSLATE_THRESHOLD = 0.5
NEZHA = False
ROFOMER = False
ELECTRA = False
GRAD_STEP = 4
NO_SOFTMAX = False

K_fold = 0

RDROP = False

