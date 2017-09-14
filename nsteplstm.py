#!/bin/env python
# coding:utf-8
# http://qiita.com/aonotas/items/8e38693fb517e4e90535

import numpy as np
from chainer import Variable
import chainer.functions as F
import chainer.links as L

# 入力データの準備
x_list = [[0, 1, 2, 3], [4, 5, 6], [7, 8]] # 可変長データ (4, 3, 2)の長さのデータとする
x_list = [np.array(x, dtype=np.int32) for x in x_list] # numpyに変換する


n_vocab = 500
emb_dim = 100
word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1)

use_dropout = 0.25
in_size = 100
hidden_size = 200
n_layers = 1

bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=in_size,
                      out_size=hidden_size, dropout=use_dropout)

# Noneを渡すとゼロベクトルを用意してくれます. Encoder-DecoderのDecoderの時は初期ベクトルhxを渡すことが多いです.
hx = None
cx = None

xs_f = []
for i, x in enumerate(x_list):
    x = word_embed(Variable(x)) # Word IndexからWord Embeddingに変換
    x = F.dropout(x, ratio=use_dropout)
    xs_f.append(x)

# xs_fのサイズは
# [(4, 100), (3, 100), (2, 100)]というVariableのリストになっている

hy, cy, ys = bi_lstm(hx=hx, cx=cx, xs=xs_f)



for h in ys:
    print (h.data.shape)

print(hy.shape)
