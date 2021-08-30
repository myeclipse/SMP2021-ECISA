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

import numpy as np

input_file_list = [
    r'./cache_file/nezha_ema_FGM_dp_0.2_time0_predict_8060.txt',
    r'./cache_file/bert_ema_FGM_dp_0.2_time4_predict_8019.txt',
    r'./cache_file/bert_ema_FGM_dp_0.2_time2_predict_7997.txt',
    r'./cache_file/segment_v1_ema_time0_predict_7992.txt',
    r'./cache_file/NEZHA_dpot_dense_ema_rdrop0.7983.txt',
    r'./cache_file/segment_large_v1_ema_time1_predict_7951.txt',
]

file_data = []
print(len(input_file_list))
for each in input_file_list:
    with open(each, 'r', encoding='utf-8') as f:
        s = f.read()
        s = s.split('\n')
        file_data.append(s)
file_data = [each[:-1] for each in file_data]
res = []
float_res = []
for i, each in enumerate(file_data):
    if float_res == []:
        for line in each:
            temp = []
            s0, s1, s2 = line.split('\t')
            s0, s1, s2 = float(s0), float(s1), float(s2)
            temp.append([s0, s1, s2])
            float_res.extend(temp)
    else:
        for j, line in enumerate(each):
            temp = []
            s0, s1, s2 = line.split('\t')
            s0, s1, s2 = float(s0), float(s1), float(s2)
            float_res[j][0] += s0
            float_res[j][1] += s1
            float_res[j][2] += s2
            if i == len(file_data) - 1:
                res.append(
                    str(float_res[j][0] / len(file_data)) + '\t' + str(float_res[j][1] / len(file_data)) + '\t' + str(
                        float_res[j][2] / len(file_data)) + '\n')

# with open('cache_file/dev_8187.txt', 'w', encoding='utf-8') as f:
#     f.writelines(res)

with open('./results/result_segment_ema_20210708_0.7951.txt', 'r', encoding='utf-8') as f:
    # with open('./results/test_result_8522.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.split('\n')
tag = []
for each in text:
    tag.append(each.split('\t')[0])
tag = tag[:-1]
res = np.asarray(float_res, dtype=np.float64)
r = [each.argmax() for each in res]

with open('results/result.txt', 'w', encoding='utf-8') as f:
    f.writelines([tag[i] + '\t' + str(r[i]) + '\n' for i in range(len(tag))])
