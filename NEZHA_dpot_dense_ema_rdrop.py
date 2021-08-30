"""
python train.py -model_name NEZHA_dpot_dense_ema_rdrop -save_name NEZHA_dpot_dense_ema_rdrop.weights -NEZHA_Large_WWM True -gpu 3
"""
import os.path

from bert4keras.layers import GlobalAveragePooling1D, Dense, Dropout, GlobalMaxPooling1D, concatenate, Lambda
from bert4keras.models import Model, build_transformer_model
from bert4keras.tokenizers import Tokenizer

from utils.common_function import data_generator, ce_with_rdrop, random_mask
from utils.evaluate import Evaluate
from utils.model_predict import start_predict, predict_final_test

import config
from data_utils import *

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.allow_growth = True


config.BERT_VOCAB_PATH = './NEZHA-Large-WWM/vocab.txt'
config.BERT_CKPT_PATH = './NEZHA-Large-WWM/model.ckpt-346400'
config.BERT_CONFIG_PATH = './NEZHA-Large-WWM/bert_config.json'
# config.BERT_CONFIG_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/nezha/bert_config.json'
# config.BERT_CKPT_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/nezha/model.ckpt-691689'
# config.BERT_VOCAB_PATH = '/home/hadoop-aipnlp/cephfs/data/zhaoyu76/Matchv120210629/nezha/vocab.txt'


nezha = build_transformer_model(
    config_path=config.BERT_CONFIG_PATH,
    checkpoint_path=config.BERT_CKPT_PATH,
    model='nezha',
    dropout_rate=0.3
)

# if config.CLS is "0":
#     output = GlobalAveragePooling1D()(nezha.output)
# else:
#     # output = nezha.output[:, 0, :]
#     output = Lambda(lambda x: x[:, 0])(nezha.output)

output = nezha.output
x_mean = GlobalAveragePooling1D()(output)
# x_max = GlobalMaxPooling1D()(output)
# output = concatenate([x_max, x_mean], axis=-1)

output = Dropout(DROPOUT_RATE)(x_mean)
output = Dense(3, activation='softmax')(output)

model = Model(nezha.inputs, output)
model.summary()

tokenizer = Tokenizer(token_dict=config.BERT_VOCAB_PATH)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """

    from bert4keras.backend import search_layer, K
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


def train():
    xml_data = read_xml('./data/SMP2019_ECISA_Train.xml')
    xml_data = [(xml_data[0][i], xml_data[1][i]) for i in range(len(xml_data[0]))]
    train_generator = data_generator(xml_data, config.BATCH_SIZE)

    model.compile(
        loss=config.LOSS,
        optimizer=config.OPTIMIZER,
        metrics=['sparse_categorical_accuracy']
    )

    evaluator = Evaluate(model, config.OPTIMIZER)

    callbacks = [
        evaluator
    ]

    print('train model NEZHA_dpot_dense_ema_rdrop, dropout_data is : ', config.DROPOUT_RATE, 'save at: ', config.MODEL_SAVE_PATH)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=config.EPOCH,
        callbacks=callbacks
    )


def predict_ML(save_name, dev, test):
    model.load_weights('./outputs/' + save_name)
    dev_text = dev
    test_text = test
    # dev_text = [dev[0][:64], dev[1][:64]]  # 测试
    # test_text = [test[0][:64], test[1][:64]]
    dev_feature = model.predict(dev_text, verbose=1, batch_size=256)
    test_feature = model.predict(test_text, verbose=1, batch_size=256)

    dev_line = []
    for each in dev_feature:
        dev_line.append(str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n')

    test_line = []
    for each in test_feature:
        test_line.append(str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n')

    with open(os.path.join('./cache_evaluate_results', save_name.replace('weights', '.txt')), 'w', encoding='utf-8') as f:
        f.writelines(dev_line)
        f.close()

    with open(os.path.join('./cache_test_results', save_name.replace('weights', '.txt')), 'w',encoding='utf-8') as f:
        f.writelines(test_line)
        f.close()


def predict():
    print('predict model NEZHA_dpot_dense_ema_rdrop, save at: ', config.MODEL_SAVE_PATH)
    model.load_weights(config.MODEL_SAVE_PATH)
    start_predict(model)


def predict_final():
    print('predict model NEZHA_dpot_dense_ema_rdrop, save at: ', config.MODEL_SAVE_PATH)
    model.load_weights(config.MODEL_SAVE_PATH)
    predict_final_test(model)


if __name__ == '__main__':
    train()
