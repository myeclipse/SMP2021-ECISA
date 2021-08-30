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
import re

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config import *


def read_xml_v1(filename):
    """
    第一版读取xml文件方法，采用正则表达式，
    只选取标注过的句子
    :param filename:
    :return:
    """
    # 最大长度225 最小7
    # 0  1  2    6545    3749   3882
    # val        2377    1212   1344
    #            2554    1233   1358
    import re
    with open(filename, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    labeled_data = re.findall('<Sentence ID="\d+" label="(\d)">(.*?)</Sentence>', xml_data)
    # labeled_data = labeled_data[:20]
    text = [each[1] for each in labeled_data]
    labels = [int(each[0]) for each in labeled_data]
    return text, labels


def read_csv_data(filename='./data/nCov_10k_test.csv'):
    pass


def data_clean_v1(text, remove_emoji=False):
    """
    第一版数据清洗方法，清洗掉 回复符（//）用户名    emoji（optional）
    :param text:
    :return:
    """
    if text.startswith('//@') or text.startswith('回复') or text.startswith('\\\\@'):
        try:
            del_idx = text.index(':') + 1
        except:
            try:
                del_idx = text.index(' ')
            except:
                del_idx = 0
        text = text[del_idx:]
    final_text = ''
    from_idx = -1
    if remove_emoji:
        for i, each_char in enumerate(text):
            if each_char not in ['[', ']'] and from_idx == -1:
                final_text += each_char
                continue
            if each_char == '[':
                from_idx = i
                continue
            if each_char == ']':
                from_idx = -1
        return final_text
    else:
        return text


def data_clean_v3(text, remove_emoji=False):
    """
    第三版数据清洗策略，采用比赛成熟方法
    :param text:
    :param remove_emoji:
    :return:
    """
    text = data_clean_v1(text, remove_emoji)
    res_text = re.sub(r"(\\\\)?(回复)?(//)?\s*@\S*?\s*:", "@ ", text)
    res_text1 = re.sub(r"@\S*", "@", res_text)
    return res_text1


def tokenize_v1(text, max_length=256):
    """
    第一版tokenize，采用transformers
    :param text:
    :param max_length:
    :return:
    """
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    token_ids = []
    segment_ids = []
    attention_masks = []
    for each in tqdm(text):
        temp = tokenizer.encode_plus(text=each, padding='max_length',
                                     truncation='longest_first', max_length=max_length)
        token_ids.append(temp['input_ids'])
        segment_ids.append(temp['token_type_ids'])
        attention_masks.append(temp['attention_mask'])
    token_ids = np.asarray(token_ids, dtype=np.int32)
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    attention_masks = np.asarray(attention_masks, dtype=np.int32)
    return [token_ids, segment_ids, attention_masks]


def tokenize_v2(text, max_length=256):
    """
    第二版tokenize,采用bert4keras
    :param text:
    :param max_length:
    :return:
    """
    from bert4keras.tokenizers import Tokenizer
    from bert4keras.snippets import sequence_padding
    tokenizer = Tokenizer(token_dict=BERT_VOCAB_PATH)
    token_ids, segment_ids = [], []
    for each in tqdm(text, desc='get token'):
        token_id, segment_id = tokenizer.encode(each, maxlen=max_length)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    token_ids = sequence_padding(token_ids, length=max_length)
    segment_ids = sequence_padding(segment_ids, length=max_length)
    return [token_ids, segment_ids]


def get_data_v1(filename):
    """
    获取数据
    :param filename:
    :return:
    """
    text, labels = read_xml(filename)
    text = [data_clean(each, remove_emoji=REMOVE_EMOJI) for each in text]
    text = tokenize(text)
    labels = tf.keras.utils.to_categorical(labels, num_classes=3)
    return text, labels


# method map
read_xml = read_xml_v1
data_clean = data_clean_v3
tokenize = tokenize_v2
get_data = get_data_v1

if __name__ == '__main__':
    get_data('./data/SMP2019_ECISA_Train.xml')
    # data_clean('//@nokki卡布诺琪: 领悟透了~~[泪] [泪]')
