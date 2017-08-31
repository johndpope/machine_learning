# coding: utf-8
"""
CopyNet + Seq2Seqのコード
Cao, Ziqiang, et al.
"Joint Copying and Restricted Generation for Paraphrase."
arXiv preprint arXiv:1611.09235 (2016).
"""
import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers

from sampleSeq2Seq import LSTM_Encoder
from sampleAttSeq2Seq import Attention, Att_LSTM_Decoder
import argparse
import datetime
import sampleSeq2Seq_data
import random
import sampleSeq2Seq

EMBED=300
HIDDEN=150
BATCH=40
EPOCH=20




class Copy_Attention(Attention):

    def __call__(self, fs, bs, h):
        """
        Attentionの計算
        :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        :param h: Decoderで出力された中間ベクトル
        :return att_f: 順向きのEncoderの中間ベクトルの加重平均
        :return att_b: 逆向きのEncoderの中間ベクトルの加重平均
        :return att: 各中間ベクトルの重み
        """
        # ミニバッチのサイズを記憶
        batch_size = h.data.shape[0]
        # ウェイトを記録するためのリストの初期化
        ws = []
        att = []
        # ウェイトの合計値を計算するための値を初期化
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = self.hw(functions.tanh(self.fh(f)+self.bh(b)+self.hh(h)))
            att.append(w)
            # softmax関数を使って正規化する
            w = functions.exp(w)
            # 計算したウェイトを記録
            ws.append(w)
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        for i, (f, b, w) in enumerate(zip(fs, bs, ws)):
            # ウェイトの和が1になるように正規化
            w /= sum_w
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
            att_b += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
        att = functions.concat(att, axis=1)
        return att_f, att_b, att


class Copy_Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, gpu=-1):
        super(Copy_Seq2Seq, self).__init__(
            # 順向きのEncoder
            f_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Attention Model
            attention=Copy_Attention(hidden_size, gpu),
            # Decoder
            decoder=Att_LSTM_Decoder(vocab_size, embed_size, hidden_size),
            # λの重みを計算するためのネットワーク
            predictor=links.Linear(hidden_size, 1)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
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
        :return t: 予測単語
        :return att: 各単語のAttentionの重み
        :return lambda_: Copy重視かGenerate重視かを判定するための重み
        """
        # Attention Modelで入力ベクトルを計算
        att_f, att_b, att = self.attention(self.fs, self.bs, self.h)
        # Decoderにベクトルを入力
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        # 計算された中間ベクトルを用いてλの計算
        lambda_ = self.predictor(self.h)
        return t, att, lambda_

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
    """
    forwardの計算をする関数
    :param enc_words: 入力文
    :param dec_words: 出力文
    :param model: モデル
    :param ARR: numpyかcuda.cupyのどちらか
    :return loss: 損失
    """
    # バッチサイズを記録
    batch_size = len(enc_words[0])
    # モデルの中に記録されている勾配のリセット
    model.reset()
    # 入力文の中で使用されている単語をチェックするためのリストを用意
    enc_key = enc_words.T
    # Encoderに入力する文をVariable型に変更する
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    # Encoderの計算
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos>をデコーダーに読み込ませる
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする
        y, att, lambda_ = model.decode(t)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))

        # Generative Modeにより計算された単語のlog_softmaxをとる
        s = functions.log_softmax(y)
        # Attentionの重みのlog_softmaxをとる
        att_s = functions.log_softmax(att)
        # lambdaをsigmoid関数にかけることで、0~1の値に変更する
        lambda_s = functions.reshape(functions.sigmoid(lambda_), (batch_size,))
        # Generative Modeの損失の初期化
        Pg = Variable(ARR.zeros((), dtype='float32'))
        # Copy Modeの損失の初期化
        Pc = Variable(ARR.zeros((), dtype='float32'))
        # lambdaのバランスを学習するための損失の初期化
        epsilon = Variable(ARR.zeros((), dtype='float32'))
        # ここからバッチ内の一単語ずつの損失を計算する、for文を回してしまっているところがダサい・・・
        counter = 0
        for i, words in enumerate(w):
            # -1は学習しない単語につけているラベル
            if words != -1:
                # Generative Modeの損失の計算
                Pg += functions.get_item(functions.get_item(s, i), words) * functions.reshape((1.0 - functions.get_item(lambda_s, i)), ())
                counter += 1
                # もし入力文の中に出力したい単語が存在すれば
                if words in enc_key[i]:
                    # Copy Modeの計算をする
                    Pc += functions.get_item(functions.get_item(att_s, i), list(enc_key[i]).index(words)) * functions.reshape(functions.get_item(lambda_s, i), ())
                    # ラムダがCopy Modeよりになるように学習
                    epsilon += functions.log(functions.get_item(lambda_s, i))
                # 入力文の中に出力したい単語がなければ
                else:
                    # ラムダがGenerative Modeよりになるように学習
                    epsilon += functions.log(1.0 - functions.get_item(lambda_s, i))
        # それぞれの損失をバッチサイズで割って、合計する
        Pg *= (-1.0 / np.max([1, counter]))
        Pc *= (-1.0 / np.max([1, counter]))
        epsilon *= (-1.0 / np.max([1, counter]))
        loss += Pg + Pc + epsilon
    return loss

def forward_test(enc_words, model, ARR,dict):
    ret = []
    ret_mode = []
    model.reset()
    enc_key = enc_words
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    while True:
        y, att, lambda_ = model.decode(t)
        lambda_ = functions.sigmoid(lambda_)
        s = functions.softmax(y)
        att_s = functions.softmax(att)
        prob = lambda_.data[0][0]
        flag = np.random.choice(2, 1, p=[1.0 - prob, prob])[0]
        #if prob > 0.5:
         #   flag = 1
        #else:
        #    flag = 0
        if  flag == 0:
            label = s.data.argmax()
            ret.append(label)
            ret_mode.append('gen')
            t = Variable(ARR.array([label], dtype='int32'))
        else:
            col = att_s.data.argmax()
            label = enc_key[col][0]
            ret.append(label)
            ret_mode.append('copy')
            t = Variable(ARR.array([label], dtype='int32'))
        if label == dict["</s>"]:
            break
    return ret, ret_mode






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

def load_data(infile):
    data=[]
    # データの読み込み
    with open(infile,"r") as f:
        for line in f.readlines():
            wakati=[]
            for str in line.replace("\n","").split("\t"):
                w=str.split(" ") #[]
                wakati.append(w) # [[],[]]
            data.append(wakati)  #[[[],[]],[[],[]],...
    return(data)

def train(datafile,dictfile,modelfile,gpu,embed,hidden,batch,epoch):

    dict=sampleSeq2Seq_data.load_dict(dictfile)
    data=load_data(datafile)

    # 語彙数
    vocab_size = len(dict.keys())
    # モデルのインスタンス化
    model = Copy_Seq2Seq(vocab_size=vocab_size,
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

        serializers.save_hdf5(modelfile+"."+str(epoch), model)
    serializers.save_hdf5(modelfile, model)

def test(datafile,dictfile,modelfile,gpu):

    dict=sampleSeq2Seq_data.load_dict(dictfile)


    # モデルのインスタンス化
    model = Copy_Seq2Seq(vocab_size=len(dict.keys()),
                    embed_size=EMBED,
                    hidden_size=HIDDEN,
                    batch_size=1,
                    gpu=gpu)
    serializers.load_hdf5(modelfile,model)

    data=[]  # [["a","b",..],["c","d",..],..]
    with open(datafile,"r") as f:
        for line in f.readlines():
            items=sampleSeq2Seq_data.wakati_list(line,dict,True,False)
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
        predict=forward_test(enc_words=enc_word,model=model,ARR=ARR,dict=dict)
        inword=sampleSeq2Seq.to_word(dt,dict_inv)
        outword=sampleSeq2Seq.to_word(predict,dict_inv)
        print("input:"+str(inword)+",output:"+str(outword))


def main():
    p = argparse.ArgumentParser(description='copy_seq2seq')


    #p.add_argument('--mode', default="test",help='train or test')
    #p.add_argument('--data', default="/Users/admin/Downloads/chat/txt/test.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--mode', default="train",choices=["train","test"], help='train or test')
    p.add_argument('--data', default="/Volumes/DATA/data/chat/txt/init100.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--dict', default="/Volumes/DATA/data/chat/txt/init100.dict",help='word dictionay file, word and word id ')
    p.add_argument('--model',default="/Volumes/DATA/data/chat/txt/copynet.model",help="in the case of train mode this file is output,in the case of test mode this file is input")
    p.add_argument('-g','--gpu',default=-1, type=int)
    p.add_argument('--embed',default=EMBED, type=int,help="only train mode")
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
