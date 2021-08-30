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
import ntpath
import os.path

import tensorflow as tf

from data_utils import data_clean, tokenize
from config import *
import config

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def read_xml(filename='./data/SMP2019_ECISA_Test.xml'):
    """
    读取xml，测试集定制，读取文章ID并生成label tag
    只选取标注过的句子
    :param filename:
    :return:
    """
    # 最大长度225 最小7
    import re
    with open(filename, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    labeled_data = re.findall('<Doc ID="(\d+)">|<Sentence ID="(\d+)".*?>(.*?)</Sentence>', xml_data)
    return_data_label = []
    return_data = []
    current_id = -1
    for each in labeled_data:
        if each[0] != '':
            current_id = each[0]
            continue
        return_data.append(each[2])
        return_data_label.append(str(current_id) + '-' + str(each[1]) + '\t')
    # return_data = return_data[:20]
    # return_data_label = return_data_label[:20]
    return return_data, return_data_label


def start_predict(model):
    text, label = read_xml()
    text = [data_clean(each, remove_emoji=REMOVE_EMOJI) for each in text]
    text = tokenize(text)
    res = model.predict(text, verbose=1, batch_size=256)

    pred = []
    with open(os.path.join('./cache_results', config.SAVE_NAME + '.txt'), 'w', encoding='utf-8') as f:  # 将数据存入'/cache_results'
        line = []
        for each in res:
            pred.append(each.argmax())
            line.append(str(each[0]) + "\t" + str(each[1]) + '\t' + str(each[2]) + '\n')
        f.writelines(line)
        f.close()

    # res = [each.argmax() for each in res]
    res = pred

    # assert len(res) == len(label)
    print_res = [label[i] + str(res[i]) + '\n' for i in range(len(label))]
    with open(os.path.join('./results', config.SAVE_NAME+'_result.txt'), 'w', encoding='utf-8') as f:
        f.writelines(print_res)


def predict_final_test(model):
    text, label = read_xml('./data/ECISA2021-Test.xml')
    text = [data_clean(each, remove_emoji=REMOVE_EMOJI) for each in text]
    text = tokenize(text)
    res = model.predict(text, verbose=1, batch_size=256)

    pred = []
    with open(os.path.join('./cache_final_results', config.SAVE_NAME + '.txt'), 'w',
              encoding='utf-8') as f:  # 将数据存入'/cache_final_results'
        line = []
        for each in res:
            pred.append(each.argmax())
            line.append(str(each[0]) + "\t" + str(each[1]) + '\t' + str(each[2]) + '\n')
        f.writelines(line)
        f.close()

    # res = [each.argmax() for each in res]
    res = pred

    # assert len(res) == len(label)
    print_res = [label[i] + str(res[i]) + '\n' for i in range(len(label))]
    with open(os.path.join('./final_results', config.SAVE_NAME + '_result.txt'), 'w', encoding='utf-8') as f:
        f.writelines(print_res)
