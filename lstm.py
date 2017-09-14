#!/bin/env python
# coding:utf-8
# http://qiita.com/aonotas/items/8e38693fb517e4e90535
# http://www.monthly-hack.com/entry/2016/10/24/200000

import numpy as np
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import argparse
import chainer

class LSTM(chainer.Chain):
    def __init__(self, n_vocab=500, emb_dim = 100, hidden_dim=200, n_layers=1, use_dropout=0.25):
        super(LSTM, self).__init__(
            embed = L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            l1 = L.NStepBiLSTM(n_layers=n_layers, in_size=emb_dim,
                                    out_size=hidden_dim, dropout=use_dropout),
            l2 = L.Linear(hidden_dim, n_vocab),
        )

    def __call__(self, xs):
        # Noneを渡すとゼロベクトルを用意してくれます. Encoder-DecoderのDecoderの時は初期ベクトルhxを渡すことが多いです.
        hx = None
        cx = None

        exs=[self.embed(i) for i in xs]
        hy,cy,ys=self.l1(hx=hx,cx=cx,sx=exs)
        y = [self.l2(i) for i in ys]
        return y












def main():
    parser = argparse.ArgumentParser(description='Chainer example: NStepLSTM')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')

    args = parser.parse_args()


if __name__ == '__main__':
    main()
