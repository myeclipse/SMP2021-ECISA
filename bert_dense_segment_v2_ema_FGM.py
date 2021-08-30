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
import random

import jieba
import tensorflow as tf  # 1.15

from bert4keras.backend import K, search_layer
from bert4keras.models import build_transformer_model, Model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.layers import GlobalAveragePooling1D, Dense, Dropout
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import is_string, truncate_sequences, sequence_padding
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import *
from data_utils import *

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Model
if NEZHA:
    model_name = 'nezha'
elif ROFOMER:
    model_name = 'roformer'
elif ELECTRA:
    model_name = 'electra'
else:
    model_name = 'bert'

bert = build_transformer_model(
    checkpoint_path=BERT_CKPT_PATH,
    config_path=BERT_CONFIG_PATH,
    model=model_name
)

output = GlobalAveragePooling1D()(bert.output)
output = Dropout(DROPOUT_RATE)(output)
if NO_SOFTMAX:
    output = Dense(3)(output)
else:
    output = Dense(3, activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def read_xml(filename='./data/SMP2019_ECISA_Train.xml'):
    import re
    with open(filename, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    htmls = re.findall('<Doc ID="\d+">.*?</Doc>', xml_data.replace('\n', ''))
    labeled_data = [
        [[i[0], i[1]] for i in re.findall(r'<Sentence ID="\d+"[ label="]*([\d]*)["]*>(.*?)</Sentence>', html)] for
        html in htmls]

    return labeled_data


def read_txt(filename='./data/temp_train.txt'):
    import random
    with open(filename, 'r', encoding='utf-8') as f:
        txt_data = f.read()
    txt_data = txt_data.split('\n\n\n')[:-1]
    with open('./data/translated_ja_train.txt', 'r', encoding='utf-8') as f:
        ja_data = f.read()
    ja_data = ja_data.split('\n\n\n')[:-1]

    labeled_data = []
    for i, doc in enumerate(txt_data):
        doc_res = []
        doc_res1 = []
        for each in doc.split('\n'):
            label, text = each.split('\t')
            label = label.replace('-', '')
            doc_res.append([label, text])
            doc_res1.append([label, text])
        if WITH_TRANSLATE:
            for j, each in enumerate(ja_data[i].split('\n')):
                if random.random() < TRANSLATE_THRESHOLD:
                    label, text = each.split('\t')
                    label = label.replace('-', '')
                    doc_res[j] = [label, text]
                else:
                    label, text = each.split('\t')
                    label = label.replace('-', '')
                    doc_res1[j] = [label, text]
        labeled_data.append(doc_res)
        labeled_data.insert(0, doc_res1)
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
        truncate_sequences(maxlen-2, 1, label_token)

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

        front_len = min(len(front_token), int(other_len / 2))
        back_len = other_len - front_len

        # 向前剪枝
        try:
            truncate_sequences(front_len, 1, front_token)
        except:
            pass
        # 向后剪枝
        try:
            truncate_sequences(back_len, -2, back_token)
        except:
            pass

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
    print(classification_report(evaluate_pred, evaluate_true))
    print(confusion_matrix(evaluate_pred, evaluate_true))
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
    if WITH_MORE_DATA:
        from utils.generate_new_train_file import generate
        generate()
        xml_data = read_txt()
    else:
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

    # FGM
    adversarial_training(model, 'Embedding-Token', 0.5)

    evaluator = Evaluate(model, optimizer, val_data, val_label)
    callbacks = [
        evaluator,
    ]
    print('train model bert_dense_segment_v2_ema_FGM, dropout rate:', DROPOUT_RATE, 'save at:', MODEL_SAVE_PATH)
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
    # Todo
    xml_data = read_xml('./data/SMP2019_ECISA_Test.xml')
    train_data, _ = generate_train_data(xml_data, for_test=True)
    # Todo
    label = read_xml_for_label()

    res = model.predict(train_data, verbose=1, batch_size=BATCH_SIZE)

    # Todo
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

    print('predict model bert_dense_segment_v2_ema_FGM', 'save at:', MODEL_SAVE_PATH)
    model.load_weights(MODEL_SAVE_PATH)
    start_predict(model)


if __name__ == '__main__':
    train()
