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
import jieba
import tensorflow as tf  # 1.15

from bert4keras.models import build_transformer_model, Model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.layers import GlobalAveragePooling1D, Dense, Dropout
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import is_string, truncate_sequences, sequence_padding
from sklearn.metrics import f1_score

from config import *
from data_utils import *

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Model

bert = build_transformer_model(
    checkpoint_path=BERT_CKPT_PATH,
    config_path=BERT_CONFIG_PATH,
)

output = GlobalAveragePooling1D()(bert.output)
output = Dropout(DROPOUT_RATE)(output)
if NO_SOFTMAX:
    output = Dense(3)(output)
else:
    output = Dense(3, activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()


def read_xml(filename='./data/SMP2019_ECISA_Train.xml'):
    import re
    with open(filename, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    htmls = re.findall('<Doc ID="\d+">.*?</Doc>', xml_data.replace('\n', ''))
    labeled_data = [
        [[i[0], i[1]] for i in re.findall(r'<Sentence ID="\d+"[ label="]*([\d]*)["]*>(.*?)</Sentence>', html)] for
        html in htmls]

    return labeled_data


class CustomTokenizer(Tokenizer):
    def __init__(self, token_dict):
        super(CustomTokenizer, self).__init__(token_dict)

    def encode(
            self,
            texts,
            label_idx,
            maxlen=512,
            pattern='S*E*E',
            truncate_from='right'
    ):
        tokens = []
        for each in texts:
            tokens.append(self.tokenize(each))

        label_token = tokens[label_idx][1:]
        if label_token != 0:
            front_token = ['[CLS]']
            for i in range(label_idx):
                front_token.extend(tokens[i][1:])
        else:
            front_token = []
        if label_token != len(tokens) - 1:
            back_token = ['[CLS]']
            for i in range(label_idx + 1, len(tokens)):
                back_token.extend(tokens[i][1:])
        else:
            back_token = []

        other_len = maxlen - len(label_token)
        # 前后剪枝
        truncate_sequences(other_len, int(self._token_start is not None), front_token, back_token)

        front_token_ids = self.tokens_to_ids(front_token)
        front_segment_ids = [0] * len(front_token_ids)

        label_token_ids = self.tokens_to_ids(label_token)
        label_segment_ids = [1] * len(label_token_ids)

        back_token_ids = self.tokens_to_ids(back_token)
        back_segment_ids = [0] * len(back_token_ids)

        front_token_ids.extend(label_token_ids)
        front_token_ids.extend(back_token_ids)
        front_segment_ids.extend(label_segment_ids)
        front_segment_ids.extend(back_segment_ids)

        return front_token_ids, front_segment_ids


def generate_train_data(label_data, for_test=False):
    tokenizer = CustomTokenizer(BERT_VOCAB_PATH)
    token_ids = []
    segment_ids = []
    train_label = []
    for doc in tqdm(label_data):
        texts = [data_clean(each[1]) for each in doc]
        for i, each in enumerate(doc):
            # is label
            if each[0] != '':
                token_id, segment_id = tokenizer.encode(texts, i)
                token_ids.append(token_id)
                segment_ids.append(segment_id)
                train_label.append([int(each[0])])
                continue
            if for_test:
                token_id, segment_id = tokenizer.encode(texts, i)
                token_ids.append(token_id)
                segment_ids.append(segment_id)
                train_label.append([0])
                continue
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    # train_label = sequence_padding(train_label)
    train_label = tf.keras.utils.to_categorical(train_label, num_classes=3)
    return [token_ids, segment_ids], train_label


def evaluate(model, epoch, data, label):
    evaluate_res = model.predict(data)

    cache_path = MODEL_SAVE_PATH.replace('.weights', '_' + str(epoch) + '_evaluate.txt').replace('outputs',
                                                                                                 'cache_file')
    cache_data_list = [str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n' for each in evaluate_res]
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.writelines(cache_data_list)
        f.write('=' * 20 + '\n\n\n')

    evaluate_pred = [each.argmax() for each in evaluate_res]
    evaluate_true = [each.argmax() for each in label]
    macro_f1 = f1_score(evaluate_pred, evaluate_true, average="macro")
    return macro_f1


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, model, optimizer, val_data, val_label):
        super(Evaluate, self).__init__()
        self.best_macro_f1 = 0
        self.model = model
        self.optimizer = optimizer
        if type(optimizer) == Adam:
            self.ema = False
        else:
            self.ema = True
        self.data = val_data
        self.label = val_label

    def on_epoch_end(self, epoch, logs=None):
        if self.ema:
            self.optimizer.apply_ema_weights()

        macro_f1 = evaluate(self.model, epoch, self.data, self.label)


        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            if SAVE_MODEL:
                self.model.save_weights(MODEL_SAVE_PATH)
                print('save model at', MODEL_SAVE_PATH)
        if self.ema:
            self.optimizer.reset_old_weights()
        res = {
            'best_macro_f1': self.best_macro_f1,
            'current_macro_f1': macro_f1
        }
        print('\n', res)
        with open(MODEL_NAME + '.txt', 'a') as f:
            text = 'epoch:' + str(epoch) + ' model name:' + MODEL_NAME + ' dropout rate:' + str(DROPOUT_RATE) + \
                   ' res:' + str(res) + '\n'
            f.write(text)


def train():
    xml_data = read_xml('./data/SMP2019_ECISA_Train.xml')
    train_data, train_label = generate_train_data(xml_data)

    xml_data = read_xml('./data/SMP2019_ECISA_Dev.xml')
    val_data, val_label = generate_train_data(xml_data)

    AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    optimizer = AdamEMA(LEARN_RATE)

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['acc'],

    )

    evaluator = Evaluate(model, optimizer, val_data, val_label)
    callbacks = [
        evaluator,
    ]
    print('train model bert_dense_segment_v1, dropout rate:', DROPOUT_RATE, 'save at:', MODEL_SAVE_PATH)
    model.fit(
        x=train_data,
        y=train_label,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        callbacks=callbacks,
        # shuffle=False
    )


def read_xml_for_label(filename='./data/SMP2019_ECISA_Test.xml'):
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
    current_id = -1
    for each in labeled_data:
        if each[0] != '':
            current_id = each[0]
            continue
        return_data_label.append(str(current_id) + '-' + str(each[1]) + '\t')
    return return_data_label


def start_predict(model):
    xml_data = read_xml('./data/SMP2019_ECISA_Test.xml')
    train_data, _ = generate_train_data(xml_data, for_test=True)
    label = read_xml_for_label()

    res = model.predict(train_data, verbose=1, batch_size=BATCH_SIZE)

    if NO_SOFTMAX:
        cache_path = MODEL_SAVE_PATH.replace('.weights', '_predict_no.txt').replace('outputs', 'cache_file')
    else:
        cache_path = MODEL_SAVE_PATH.replace('.weights', '_predict.txt').replace('outputs', 'cache_file')

    cache_data_list = [str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n' for each in res]
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.writelines(cache_data_list)
        f.write('=' * 20 + '\n\n\n')

    res = [each.argmax() for each in res]
    assert len(res) == len(label)
    print_res = [label[i] + str(res[i]) + '\n' for i in range(len(label))]
    with open('./result.txt', 'w', encoding='utf-8') as f:
        f.writelines(print_res)


def predict():
    # from utils.model_predict import start_predict
    AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    optimizer = AdamEMA(LEARN_RATE)
    print('predict model bert_dense_segment_v1', 'save at:', MODEL_SAVE_PATH)
    model.load_weights(MODEL_SAVE_PATH)
    start_predict(model)


if __name__ == '__main__':
    # train()
    # read_xml_for_label('./data/SMP2019_ECISA_Train.xml')
    generate_train_data(read_xml('./data/SMP2019_ECISA_Train.xml'),for_test=True)