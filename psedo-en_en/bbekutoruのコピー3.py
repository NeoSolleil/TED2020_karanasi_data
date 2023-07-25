#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:42:13 2023

@author: fujidai
"""


import torch
from sentence_transformers import SentenceTransformer, InputExample, losses,models
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import numpy as np
from sklearn.preprocessing import MinMaxScaler




#with open('/Users/fujidai/TED2020_data/wmt-qe-2022-data/test_data-gold_labels/task1_da/en-ja/test.2022.mt', 'r') as f:#
#with open('/Users/fujidai/TED2020_data/wmt21/test.2022.mt', 'r') as f:#
with open('/Users/fujisakiharuto/aq/karanasi/en-10000ごと/TED-en-140001-150000.txt', 'r') as f:#
    left = f.read()
left_lines = left.splitlines()
#print(left_lines[-1])
#print(len(left_lines))
#left_lines.pop()
#print(len(left_lines))   #2383690

#with open('/Users/fujidai/TED2020_data/wmt-qe-2022-data/test_data-gold_labels/task1_da/en-ja/test.2022.src', 'r') as f:#
#with open('/Users/fujidai/TED2020_data/wmt21/test.2022.src', 'r') as f:#
with open('/Users/fujisakiharuto/aq/karanasi/擬似-en/TED-ja-140001-150000 の翻訳コピー.txt', 'r') as f:#

    right = f.read()
right_lines = right.splitlines()#改行コードごとにリストに入れている

#print(right_lines[-1])  #（拍手）
#print(len(right_lines))  #2383690


#first='/Users/fujidai/sinTED/paraphrase-mpnet-base-v2__完成2-MarginMSELoss-finetuning-6-30_optimizer_params-nasi_epoch-3'
#second='/Users/fujidai/TED2020_data/parafalese/paraphrase-multilingual-mpnet-base-v2'
#second='/Users/fujidai/teacher_finetuning/paraphrase-mpnet-base-v2'
#second='/Users/fujidai/teacher_finetuning/6-sikiiti>=0.1-paraphrase-mpnet-base-v2_finetuning-25480_完成2-MarginMSELoss-finetuning-6-30-2/7032'
second='/Users/fujisakiharuto/aq/paraphrase-mpnet-base-v2'
#second='/Users/fujidai/teacher_finetuning/4_3-paraphrase-mpnet-base-v2_finetuning-2-61950_完成2-MarginMSELoss-finetuning-6-30-2/7032'

#model1 = SentenceTransformer(first)
model2 = SentenceTransformer(second)


import numpy as np


import numpy as np
'''
def calculate_cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)

    return similarity
'''
import torch
import torch.nn.functional as F

def calculate_cosine_similarity(vector1, vector2):
    tensor1 = torch.tensor(vector1)
    tensor2 = torch.tensor(vector2)

    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    similarity = dot_product / (norm1 * norm2)

    return similarity

# 使用例
v1 = [1, 2, 3]
v2 = [4, 5, 6]

#similarity_score = calculate_cosine_similarity(v1, v2)

def cos_sim(v1, v2):
    #print(v1)
    #print(v1.dtype)
    #print(np.linalg.norm(v1))
    #print(np.linalg.norm(v2))
    #print(np.dot(v1, v2))
    #print(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

from sentence_transformers import SentenceTransformer, util

def aqz(sw,sx):
    cosine_scores = util.cos_sim(sw, sx)
    return cosine_scores

import torch.nn.functional as F
train_examples = []
for i in range(len(left_lines)):


    pair=[]
    pair.append(left_lines[i])#left_lines側のi行目をtextsに追加している
    pair.append(right_lines[i])#right_lines側のi行目をtextsに追加している
    #print(pair)#
    #embeddings1 = model1.encode(pair)#
    #print(embeddings1[0])
    embeddings2 = model2.encode(pair)
    #X = np.array(embeddings1[0])
    #Y = np.array(embeddings1[1])
    #print(embeddings1[0].dtype)
    #print(X.dtype)

    #print('l',X)
    #print(Y)
    #print(aq[i])
    embedding1_tensor = torch.tensor(embeddings2[0]).unsqueeze(0)
    embedding2_tensor = torch.tensor(embeddings2[1]).unsqueeze(0)
    #print(embedding2_tensor)


    #print(cos_sim(embeddings1[0], embeddings1[1]))#0.77356255
    print(cos_sim(embeddings2[0], embeddings2[1]))
    #print(i)
    #print(F.cosine_similarity(embedding1_tensor, embedding2_tensor))
    #print("")
    #print(embeddings1[0].size())
    #print(aqz(embeddings1[0], embeddings1[1]))
    #print(aqz(embeddings1[0], embeddings1[1]).size())

    #print(calculate_cosine_similarity(embeddings1[0], embeddings1[1]))#tensor(0.7736)

    XX = np.array(embeddings2[0])
    YY = np.array(embeddings2[1])
    '''
    if i ==20:
        break
    ''' 
    #print(Y)
    #print(second)
    #print(cos_sim(XX, YY))
    #print('')
    



    #print(embeddings)
    #print(aq[i].dtype)

    #print(pair)
    #このように出力される→['Thank you so much, Chris. ', 'Спасибо, Крис. ']　←１行目
    #example = InputExample(texts=pair, label=float(aq[i]))#textsをラベル付きで追加している
    #print(example.dtype)
    #print(example)
    #example_s = tokenizer(texts, truncation=True, padding=True)
    #train_examples.append(example)#学習として入れるものに入れている
    #print(train_examples)
    #if i ==5:
        #break
   #





print(second)





'''

print(model1)




print(len(train_examples))#出力の数字はleftlinesと同じ数になっている
print('')
print(model.dtype)
print(aq.dtype)
print(example.dtype)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
#print(train_dataloader)
train_objectives=[(train_dataloader, train_loss)]

#Tune the model
model.fit(train_objectives, epochs=10, warmup_steps=100)
model.save("para_seikih-10")
'''
