# coding: utf-8
"""
Attention Model + Seq2Seqのコード
Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
"Neural machine translation by jointly learning to align and translate."
arXiv preprint arXiv:1409.0473 (2014).
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links,optimizer, optimizers, serializers

from sampleSeq2Seq import LSTM_Encoder
import sampleSeq2Seq
import random
import datetime
import argparse
import sampleSeq2Seq_data

EMBED=300
HIDDEN=150
BATCH=40
EPOCH=20




class Att_LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        Attention ModelのためのDecoderのインスタンス化
        :param vocab_size: 語彙数
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(Att_LSTM_Decoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            ye=links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh=links.Linear(embed_size, 4 * hidden_size),
            # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            hh=links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            fh=links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            bh=links.Linear(hidden_size, 4 * hidden_size),
            # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
            he=links.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙数サイズのベクトルに変換する層
            ey=links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, f, b):
        """
        Decoderの計算
        :param y: Decoderに入力する単語
        :param c: 内部メモリ
        :param h: Decoderの中間ベクトル
        :param f: Attention Modelで計算された順向きEncoderの加重平均
        :param b: Attention Modelで計算された逆向きEncoderの加重平均
        :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        """
        # 単語を単語ベクトルに変換
        e = functions.tanh(self.ye(y))
        # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
        c, h = functions.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b))
        # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h


class Attention(Chain):
    def __init__(self, hidden_size, gpu):
        """
        Attentionのインスタンス化
        :param hidden_size: 隠れ層のサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Attention, self).__init__(
            # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            fh=links.Linear(hidden_size, hidden_size),
            # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            bh=links.Linear(hidden_size, hidden_size),
            # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hh=links.Linear(hidden_size, hidden_size),
            # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
            hw=links.Linear(hidden_size, 1),
        )
        # 隠れ層のサイズを記憶
        self.hidden_size = hidden_size
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        if gpu>=0:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def __call__(self, fs, bs, h):
        """
        Attentionの計算
        :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        :param h: Decoderで出力された中間ベクトル
        :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        """
        # ミニバッチのサイズを記憶
        batch_size = h.data.shape[0]
        # ウェイトを記録するためのリストの初期化
        ws = []
        # ウェイトの合計値を計算するための値を初期化
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = functions.tanh(self.fh(f)+self.bh(b)+self.hh(h))
            # softmax関数を使って正規化する
            w = functions.exp(self.hw(w))
            # 計算したウェイトを記録
            ws.append(w)
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        for f, b, w in zip(fs, bs, ws):
            # ウェイトの和が1になるように正規化
            w /= sum_w
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
            att_b += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
        return att_f, att_b


class Att_Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, gpu=-1):
        """
        Seq2Seq + Attentionのインスタンス化
        :param vocab_size: 語彙数のサイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Att_Seq2Seq, self).__init__(
            # 順向きのEncoder
            f_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Attention Model
            attention = Attention(hidden_size, gpu),
            # Decoder
            decoder = Att_LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # GPUを使うときはcupy、使わないときはnumpy
        if gpu>=0:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

    def encode(self, words):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 先ずは順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.fs.append(h)

        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.bs.insert(0, h)

        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        Decoderの計算
        :param w: Decoderで入力する単語
        :return: 予測単語
        """
        att_f, att_b = self.attention(self.fs, self.bs, self.h)
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        #lambda_ = functions.sigmoid(self.predictor(self.h))
        return t

    def reset(self):
        """
        インスタンス変数を初期化する
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        # 勾配の初期化
        self.zerograds()


def forward(enc_words, dec_words, model, ARR):
    batch_size = len(enc_words[0])
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
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



def train(datafile,dictfile,modelfile,gpu,embed,hidden,batch,epoch):

    dict=sampleSeq2Seq_data.load_dict(dictfile)
    data=sampleSeq2Seq.load_data(datafile)

    # 語彙数
    vocab_size = len(dict.keys())
    # モデルのインスタンス化
    model = Att_Seq2Seq(vocab_size=vocab_size,
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
                enc_words, dec_words = sampleSeq2Seq.make_minibatch(minibatch)
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

        serializers.save_hdf5(modelfile+"."+str(epoch), model)
    serializers.save_hdf5(modelfile, model)

def test(datafile,dictfile,modelfile,gpu):

    dict=sampleSeq2Seq_data.load_dict(dictfile)


    # モデルのインスタンス化
    model = Att_Seq2Seq(vocab_size=len(dict.keys()),
                    embed_size=EMBED,
                    hidden_size=HIDDEN,
                    batch_size=1,
                    gpu=gpu)
    serializers.load_hdf5(modelfile,model)

    data=[]  # [["a","b",..],["c","d",..],..]
    with open(datafile,"r") as f:
        for line in f.readlines():
            items=sampleSeq2Seq_data.wakati_list(line,dict)
            data.append(items)


    if gpu>=0:
        ARR = cuda.cupy
        cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)
    else:
        ARR = np

    # change key and val,
    # http://www.lifewithpython.com/2014/03/python-invert-keys-values-in-dict.html
    dict_inv={v:k for k,v in dict.items()}
    for dt in data:
        enc_word=np.array([dt],dtype="int32").T
        predict=forward_test(enc_words=enc_word,model=model,ARR=ARR)
        inword=sampleSeq2Seq.to_word(dt,dict_inv)
        outword=sampleSeq2Seq.to_word(predict,dict_inv)
        print("input:"+str(inword)+",output:"+str(outword))


def main():
    p = argparse.ArgumentParser(description='seq2seq')


    #p.add_argument('--mode', default="test",help='train or test')
    #p.add_argument('--data', default="/Users/admin/Downloads/chat/txt/test.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--mode', default="train",choices=["train","test"],help='train or test')
    p.add_argument('--data', default="/Volumes/DATA/data/chat/txt/init100.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--dict', default="/Volumes/DATA/data/chat/txt/init100.dict",help='word dictionay file, word and word id ')
    p.add_argument('--model',default="/Volumes/DATA/data/chat/txt/att.model", help="in the case of train mode this file is output,in the case of test mode this file is input")
    p.add_argument('-g','--gpu',type=int, default=-1)
    p.add_argument('--embed',default=EMBED, type=int, help="only train mode")
    p.add_argument('--hidden',default=HIDDEN, type=int, help="only train mode")
    p.add_argument('--batch',default=BATCH, type=int, help="only train mode")
    p.add_argument('--epoch',default=EPOCH, type=int, help="only train mode")
    args = p.parse_args()

    print ('開始: ', datetime.datetime.now())
    try:
        if args.mode == "train":
            train(args.data,args.dict,args.model,args.gpu,args.embed,args.hidden,args.batch,args.epoch)
        else:
            test(args.data,args.dict,args.model,args.gpu)
    except:
        import traceback
        print( traceback.format_exc())

    print ('終了: ', datetime.datetime.now())


if __name__ == '__main__':
    main()