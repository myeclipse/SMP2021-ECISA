# ECISA2021-myeclipse

## 文件说明

    |bert # bert的预训练模型存储位置
        |bert_model.ckpt
        |bert_model.config
        |vocab.txt
    |cache_file # 用来保存模型输出概率值的文件，模型融合使用
        |...
        |...
        |...
    |data  # 用来保存训练数据的文件
        |SMP2019_ECISA_Train.xml
        |SMP2019_ECISA_Dev.xml
        |SMP2019_ECISA_Test.xml
        |ECISA2021-Test.xml
    |nezha # nezha base的预训练模型存储位置
        |...
        |...
        |...
    |NEZHA-Large-WWM # nezha的预训练模型存储位置
        |...
        |...
        |...
    |outputs # 模型输出保存到的文件夹
        |...
        |...
        |...
    |results # 存放模型（可提交）输出的文件夹
        |...
        |...
        |...
    |roberta # roberta large的预训练模型存储位置
        |...
        |...
        |...
    |utils  # 工具包
        |common_function.py  # 一些模型需要的公共方法（仅模型3使用）
        |evaluate.py # 用于模型验证的一些公共方法（仅模型3使用）
        |model_predict.py  # 用于模型预测的一些公共方法（仅模型3使用）
    |add_two_files.py  # 将多模型结果进行平均加权，输出最终结果
    |bert_dense_segment_v1_ema.py  # 模型1
    |bert_dense_segment_v2_ema_FGM.py  # 模型2
    |config.py  # 配置文件
    |data_utils.py  # 数据处理工具类
    |NEZHA_dpot_dense_ema_rdrop.py  # 模型3
    |predict.py  # 预测文件（模型1，模型2使用）
    |predict_final.py  # 预测文件（模型3使用）
    |requirements.txt  # 环境文件
    |train.py   # 训练文件（模型1，模型2使用）
    |train_.py   # 训练文件（模型3使用）

## 预训练权重下载

bert base：[google](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

roberta large: [brightmart](https://drive.google.com/open?id=1W3WgPJWGVKlU9wpUYsdZuurAIFKvrl_Y)

nezha base: [huawei](https://drive.google.com/drive/folders/1tFs-wMoXIY8zganI2hQgDBoDPqA8pSmh?usp=sharing)

nezha large: [huawei](https://drive.google.com/drive/folders/1LOAUc9LXyogC2gmP_q1ojqj41Ez01aga?usp=sharing)

## 使用方法

1.初始化环境 `pip install -r requirements.txt`

2.下载预训练权重并存放到对应的文件夹下

3.开始训练，

- 模型1 模型2采用`train.py`进行训练 。


    python train.py 
        -model_name bert_dense_segment_v2_ema_FGM
        -save_name test_model.weights
        -lr 1e-5
        -bsz 32
        -epoch 10
        -dpot 0.2
        -gpu 0,1
        -nezha True
        -roberta True

- 模型3 采用`train_.py`进行训练。


    python train_.py 
        -model_name NEZHA_dpot_dense_ema_rdrop
        -save_name test_model.weights
        -lr 1e-5
        -bsz 16
        -epoch 10
        -gpu 0,1
        -ema True
        -rdrop True
        -rmask 0


## 其他说明

最终的结果为6个模型的融合，分别为

- nezha base（模型1）

  `python train.py -model_name bert_dense_segment_v2_ema_FGM -save_name nezha_ema_FGM_dp_0.2_time0.weights -gpu 0 -lr 1e-5 -epoch 10 -bsz 32 -dpot 0.2 -nezha True`

  `python predict.py -model_name bert_dense_segment_v2_ema_FGM -save_name nezha_ema_FGM_dp_0.2_time0.weights -gpu 0 -nezha True`
- bert base （模型1）

  `python train.py -model_name bert_dense_segment_v2_ema_FGM -save_name bert_ema_FGM_dp_0.2_time4.weights -gpu 0 -lr 1e-5 -epoch 10 -bsz 32 -dpot 0.2`

  `python predict.py -model_name bert_dense_segment_v2_ema_FGM -save_name bert_ema_FGM_dp_0.2_time4.weights -gpu 0`
- bert base （模型1）

  `python train.py -model_name bert_dense_segment_v2_ema_FGM -save_name bert_ema_FGM_dp_0.2_time2.weights -gpu 0 -lr 1e-5 -epoch 10 -bsz 32 -dpot 0.2`

  `python train.py -model_name bert_dense_segment_v2_ema_FGM -save_name bert_ema_FGM_dp_0.2_time2.weights -gpu 0`
- bert base （模型2）

  `python train.py -model_name bert_dense_segment_v1_ema -save_name segment_v1_ema_time0.weights -gpu 0 -lr 1e-5 -epoch 10 -bsz 32 -dpot 0.2`

  `python predict.py -model_name bert_dense_segment_v1_ema -save_name segment_v1_ema_time0.weights -gpu 0`
- nezha large（模型3）

  `python train_.py -model_name NEZHA_dpot_dense_ema_rdrop -save_name nezha_dpot_dense_ema.weghts -bsz 16 -epoch 10 -lr 1e-5 -ema True -gpu 0 -rdrop True -rmask 0`

  `python predict_final.py -model_name NEZHA_dpot_dense_ema_rdrop -save_name nezha_dpot_dense_ema.weghts -gpu 0`
- roberta large（模型2）

  `python train.py -model_name bert_dense_segment_v1_ema -save_name segment_large_v1_ema_time1.weights -gpu 0 -lr 1e-5 -epoch 10 -bsz 8 -dpot 0.2 -roberta True`

  `python predict.py -model_name bert_dense_segment_v1_ema -save_name segment_large_v1_ema_time1.weights -gpu 0 `

获取到6个模型的predict输出后，调用`add_two_files.py`（需要更改对应文件名）来计算六个模型的加权平均作为最终结果。