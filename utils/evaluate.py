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
import os.path

import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
from bert4keras.optimizers import Adam
from config import *
from data_utils import get_data
import config

v_text1, v_label1 = get_data('./data/SMP2019_ECISA_Dev.xml')


def evaluate(model, data=None, label=None):
    if data is not None and label is not None:
        v_text = data
        v_label = label
    else:
        v_text = v_text1
        v_label = v_label1
    evaluate_res = model.predict(v_text)
    evaluate_pred = [each.argmax() for each in evaluate_res]
    evaluate_true = [each.argmax() for each in v_label]
    macro_f1 = f1_score(evaluate_pred, evaluate_true, average="macro")
    report = classification_report(evaluate_true, evaluate_pred, target_names=["0", "1", "2"])
    print(report)
    return macro_f1, evaluate_res


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, model, optimizer, data=None, label=None):
        super(Evaluate, self).__init__()
        self.best_macro_f1 = 0
        self.model = model
        self.optimizer = optimizer
        if type(optimizer) == Adam:
            self.ema = False
        else:
            self.ema = True
        # self.ema = False

        self.data = data
        self.label = label

    def on_epoch_end(self, epoch, logs=None):
        if self.ema:
            # self.optimizer.optimizer.apply_ema_weights()   # 修改优化器，以适应 ema+梯度累加
            self.optimizer.apply_ema_weights()

        macro_f1, evaluate_res = evaluate(self.model, self.data, self.label)

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            if SAVE_MODEL:
                self.model.save_weights(config.MODEL_SAVE_PATH)
                print('save model at', config.MODEL_SAVE_PATH)

                dev_line = []
                for each in evaluate_res:
                    dev_line.append(str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n')

                with open(os.path.join('./cache_evaluate_results', config.SAVE_NAME + '.txt'), 'w',
                          encoding='utf-8') as f:
                    f.writelines(dev_line)
                    f.close()
        res = {
            'best_macro_f1': self.best_macro_f1,
            'current_macro_f1': macro_f1
        }
        print('\n', res)

        if self.ema:
            # self.optimizer.optimizer.reset_old_weights()
            self.optimizer.reset_old_weights()

        with open(os.path.join('./process_records', config.MODEL_NAME + '.txt'), 'a') as f:
            text = 'epoch:' + str(epoch) + ' model name:' + config.MODEL_NAME + ' dropout rate:' + str(config.DROPOUT_RATE) + \
                   ' res:' + str(res) + '\n'
            f.write(text)


