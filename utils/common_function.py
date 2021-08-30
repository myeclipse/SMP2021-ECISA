# -*- coding:utf-8 -*-
import functools

import jieba
from bert4keras.backend import K
from bert4keras.snippets import DataGenerator, sequence_padding, truncate_sequences
from bert4keras.tokenizers import Tokenizer
from data_utils import *
import config
jieba.initialize()


def adversarial_training(model, embedding_name="Embedding-Token", epsilon=1):
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


def ce_with_rdrop(y_true, y_pred, alpha=4.):
    """ r_drop策略的损失函数 """
    from tensorflow.keras.losses import kullback_leibler_divergence as kld
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss_ce = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss_kl = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss_ce + K.mean(loss_kl) / 4 * alpha


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        tokenizer = Tokenizer(token_dict=config.BERT_VOCAB_PATH)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        if config.RANDOM_MASK_RATE != 0:
            import random, collections
            rng = random.Random(12345)
            vocab_list = []
            with open(config.BERT_VOCAB_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    vocab_list.append(line.strip(" ").strip('\n'))
                f.close()

        loop_times = 2  # 是否启用R-drop
        if not config.RDROP:
            loop_times = 1

        for is_end, (text, label) in self.sample(random):

            if config.RANDOM_MASK_RATE != 0:  # 加入random_mask
                text_list = random_mask(text, config.RANDOM_MASK_RATE, vocab_list, rng)
                text = ''.join(word for word in text_list)

            token_ids, segment_ids = tokenizer.encode(text, maxlen=256)  # 语料单句的最大长度为225

            for i in range(loop_times):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def random_mask(text, mask_prob, vocab_words, rng=None, MaskedLmInstance=None):
    sequence = text
    tokens = []
    cand_indexes = []
    for id, word in enumerate(sequence):
        cand_indexes.append(id)
        tokens.append(word)
    rng.shuffle(cand_indexes)
    # output_tokens = list(tokens)
    output_tokens = tokens
    num_to_predict = int(len(tokens) * mask_prob)  # 需要mask的数量

    # if num_to_predict == 0:  # 提前结束
    #     return output_tokens, tokens

    masked_lms = []  # 保存进行了mask的数据： 在sequence中的 index 和 真实token
    covered_indexes = set()  # 存放被mask的id
    for index in cand_indexes:
        if len(covered_indexes) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        # 对于mask_prob要替换的数据，其中0.8替换为[MASK], 0.1替换为其它的Token，0.1不作替换
        # masked_token = None  # 用masked_token标记被替换为了什么？ [MASK] / 其它Token / 不替换
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            if rng.random() < 0.5:
                id = rng.randint(0, len(vocab_words) - 1)
                while id == index:
                    id = rng.randint(0, len(vocab_words) - 1)
                masked_token = vocab_words[id]
            else:
                masked_token = tokens[index]

        output_tokens[index] = masked_token
        # masked_lms.append(MaskedLmInstance(index=index, token=tokens[index]))

    # masked_lms = sorted(masked_lms, key=lambda x: x.index)  # 根据下标索引恢复sequence的原来token顺序

    # +++ 标记mask了的index和token ++++
    # masked_lm_positions = []
    # masked_lm_labels = []
    # for p in masked_lms:
    #     masked_lm_positions.append(p.index)
    #     masked_lm_labels.append(p.token)
    # ++++++++++++++++++++++++++++++++
    return output_tokens


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):  # 设置epoch从0到4000时lr的变化函数, warmup_steps越大，最大的lr越小
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# learning_rate = CustomSchedule(300, 100)  # 学习率的自变量为step，且不同epoch间的step会累加


class CustomTokenizer(Tokenizer):
    def __init__(self, token_dict, **kwargs):
        super(CustomTokenizer, self).__init__(token_dict, **kwargs)

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
        if label_idx != 0:
            front_token = ['[CLS]']
            for i in range(label_idx):
                front_token.extend(tokens[i][1:])
        else:
            front_token = []
        if label_idx != len(tokens) - 1:
            back_token = ['[CLS]']
            for i in range(label_idx + 1, len(tokens)):
                back_token.extend(tokens[i][1:])
        else:
            back_token = []

        other_len = maxlen - len(label_token)

        front_len = min(len(front_token), int(other_len / 2))
        back_len = other_len - front_len

        # 向前剪枝
        truncate_sequences(front_len, -2, front_token)
        # 向后剪枝
        truncate_sequences(back_len, 1, back_token)

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


class c2Tokenizer(Tokenizer):
    """
    根据bert/Nezha的结构构造的segment数据 --》 在我的理解下，认为segment用于训练的过程类似于Bert/Nezha的NSP任务，因此只能有上下文的信息，而不能有其它的语句
      --》 为此可以做一个数据增强，即【a,b]预测b和【b,c】预测b两种情况（两条数据）==》 此处为了简单，只做了第一种情况
    """

    def __init__(self, token_dict):
        super(c2Tokenizer, self).__init__(token_dict)

    def encode(self, texts, label_idx, maxlen=512):
        tokens = []
        for text in texts:
            tokens.append(self.tokenize(text))
        if label_idx == 0:
            first_tokens = tokens[label_idx]
            first_token_ids = self.tokens_to_ids(first_tokens)
            # first_segment_ids = [1] * len(first_token_ids)
            first_segment_ids = [0] * len(first_token_ids)
            return first_token_ids, first_segment_ids
        else:
            first_tokens = tokens[label_idx - 1]
            second_tokens = tokens[label_idx]
            first_token_ids = self.tokens_to_ids(first_tokens)
            second_token_ids = self.tokens_to_ids(second_tokens)
            first_segment_ids = [0] * len(first_token_ids)
            # second_segment_ids = [1] * len(second_token_ids)
            second_segment_ids = [0] * len(second_token_ids)  # 将segment_ids修改为全[0]
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
            return first_token_ids, first_segment_ids


def read_xml_common(filename='./data/SMP2019_ECISA_Train.xml'):
    import re
    with open(filename, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    htmls = re.findall('<Doc ID="\d+">.*?</Doc>', xml_data.replace('\n', ''))
    labeled_data = [
        [[i[0], i[1]] for i in re.findall(r'<Sentence ID="\d+"[ label="]*([\d]*)["]*>(.*?)</Sentence>', html)] for
        html in htmls]

    return labeled_data


def generate_train_data(label_data, for_test=False):
    """
    这里和Data generator并存的原因是想赋予wwm跟随batch变化的能力，但是又不想丢失验证和测试的方便性
    :param label_data:
    :param for_test:
    :return:
    """
    tokenizer = CustomTokenizer(config.BERT_VOCAB_PATH, pre_tokenizer=lambda s: jieba.cut(s, HMM=False))
    # tokenizer = c2Tokenizer(config.BERT_VOCAB_PATH)
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
            if for_test:
                token_id, segment_id = tokenizer.encode(texts, i)
                token_ids.append(token_id)
                segment_ids.append(segment_id)
                train_label.append([0])
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    train_label = sequence_padding(train_label)
    # train_label = tf.keras.utils.to_categorical(train_label, num_classes=3)
    return [token_ids, segment_ids], train_label


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
