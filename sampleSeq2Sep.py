# coding: utf-8
"""
Sequence to Sequenceのchainer実装

Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le.,
"Sequence to sequence learning with neural networks.",
Advances in neural information processing systems. 2014.
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import datetime
import glob
import random
import argparse


class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """

        :param x: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        e = functions.tanh(self.xe(x))
        return functions.lstm(c, self.eh(e) + self.hh(h))

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Decoder, self).__init__(
            ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size),
            he = links.Linear(hidden_size, embed_size),
            ey = links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """

        :param y: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れそう
        :return: 予測単語、次の内部メモリ、次の隠れ層
        """
        e = functions.tanh(self.ye(y))
        c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h

class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, gpu):
        """
        Seq2Seqの初期化
        :param vocab_size: 語彙サイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 中間ベクトルのサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Seq2Seq, self).__init__(
            # Encoderのインスタンス化
            encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Decoderのインスタンス化
            decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
        if gpu>=0:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def encode(self, words):
        """
        Encoderを計算する部分
        :param words: 単語が記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        # エンコーダーに単語を順番に読み込ませる
        for w in words:
            c, h = self.encoder(w, c, h)

        # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
        self.h = h
        # 内部メモリは引き継がないので、初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        デコーダーを計算する部分
        :param w: 単語
        :return: 単語数サイズのベクトルを出力する
        """
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()


def forward(enc_words, dec_words, model, ARR):
    """
    順伝播の計算を行う関数
    :param enc_words: 発話文の単語を記録したリスト
    :param dec_words: 応答文の単語を記録したリスト
    :param model: Seq2Seqのインスタンス
    :param ARR: cuda.cupyかnumpyか
    :return: 計算した損失の合計
    """
    # バッチサイズを記録
    batch_size = len(enc_words[0])
    # model内に保存されている勾配をリセット
    model.reset()
    # 発話リスト内の単語を、chainerの型であるVariable型に変更
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    # エンコードの計算 ①
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos>をデコーダーに読み込ませる ②
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする ③
        y = model.decode(t)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))
        # 正解単語と予測単語を照らし合わせて損失を計算 ④
        loss += functions.softmax_cross_entropy(y, t)
    return loss

def forward_test(enc_words, model, ARR):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    counter = 0
    while counter < 50:
        y = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label], dtype='int32'))
        counter += 1
        if label == 1:
            counter = 50
    return ret

# trainの関数
def make_minibatch(minibatch):
    # enc_wordsの作成
    enc_words = [row[0] for row in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    # dec_wordsの作成
    dec_words = [row[1] for row in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    return enc_words, dec_words

def train(infile,indict,outfile,gpu,embed,hidden,batch,epoch):
    dict={}
    # 辞書の読み込み
    with open(indict,"r") as f:
        for line in f.readlines():
            items=line.replace("\n","").split("\t")
            dict[items[0]]=items[1]

    data=[]
    # データの読み込み
    with open(infile,"r") as f:
        for line in f.readlines():
            wakati=[]
            for str in line.replace("\n","").split("\t"):
                w=str.split(" ") #[]
                wakati.append(w) # [[],[]]
            data.append(wakati)  #[[[],[]],[[],[]],...

    # flatten
    # http://d.hatena.ne.jp/xef/20121027/p2
    #from itertools import chain
    #words=list(chain.from_iterable(dict))

    # 語彙数
    vocab_size = len(dict.keys())
    # モデルのインスタンス化
    model = Seq2Seq(vocab_size=vocab_size,
                    embed_size=embed,
                    hidden_size=hidden,
                    batch_size=batch,
                    gpu=gpu)
    model.reset()
    # GPUのセット
    if gpu>=0:
        ARR = cuda.cupy
        cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)
    else:
        ARR = np

    # 学習開始
    for epoch in range(epoch):
        # ファイルのパスの取得
        # エポックごとにoptimizerの初期化
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))

        random.shuffle(data)
        for num in range(len(data)//batch):
                minibatch = data[num*batch: (num+1)*batch]
                # 読み込み用のデータ作成
                enc_words, dec_words = make_minibatch(minibatch)
                # modelのリセット
                model.reset()
                # 順伝播
                total_loss = forward(enc_words=enc_words,
                                     dec_words=dec_words,
                                     model=model,
                                     ARR=ARR)
                # 学習
                total_loss.backward()
                opt.update()
                #opt.zero_grads()
                # print (datetime.datetime.now())
        print ('Epoch %s 終了' % (epoch+1))


    serializers.save_hdf5(outfile, model)

def test(infile,dict,model,gpu):
    pass

def main():
    p = argparse.ArgumentParser(description='seq2seq')


    p.add_argument('-m','--mode', default="train",help='train or test')
    p.add_argument('-i','--infile', default="/Users/admin/Downloads/chat/txt/init100.txt",help='input file')
    p.add_argument('-d','--dict', default="/Users/admin/Downloads/chat/txt/init100.dict",help='dict file')
    p.add_argument('--model',default="/Users/admin/Downloads/chat/txt/init100.model", help="output file when train mode ,input file when test mode")
    p.add_argument('-g','--gpu',default=-1)
    p.add_argument('--embed',default=300, help="only train mode")
    p.add_argument('--hidden',default=150, help="only train mode")
    p.add_argument('--batch',default=40, help="only train mode")
    p.add_argument('--epoch',default=1000, help="only train mode")
    args = p.parse_args()

    print ('開始: ', datetime.datetime.now())
    try:
        if args.mode == "train":
            train(args.infile,args.dict,args.model,args.gpu,args.embed,args.hidden,args.batch,args.epoch)
        else:
            test(args.infile,args.dict,args.model,args.gpu)
    except:
        import traceback
        print( traceback.format_exc())

    print ('終了: ', datetime.datetime.now())


if __name__ == '__main__':
    main()