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
import random
import argparse
import sampleSeq2Seq_data
import chainermn
import mpi4py
import chainer

EMBED=300
HIDDEN=150
BATCH=40
EPOCH=20


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


def forward(enc_words, dec_words, model, ARR, dict):
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
    t = Variable(ARR.array([dict["<eos>"] for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする ③
        y = model.decode(t)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))
        # 正解単語と予測単語を照らし合わせて損失を計算 ④
        loss += functions.softmax_cross_entropy(y, t)
    return loss

def forward_test(enc_words, model, ARR, dict):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    y = model.decode(t)
    label = y.data.argmax()
    ret.append(label)

    counter = 0
    while True:
        y = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label], dtype='int32'))
        if label == dict["</s>"]:
            break
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



def train(datafile,dictfile,modelfile,gpu,embed,hidden,batchsize,epoch,communicator):

    dict=sampleSeq2Seq_data.load_dict(dictfile)
    data=load_data(datafile)

    # 語彙数
    vocab_size = len(dict.keys())
    # モデルのインスタンス化
    model = Seq2Seq(vocab_size=vocab_size,
                    embed_size=embed,
                    hidden_size=hidden,
                    batch_size=batchsize,
                    gpu=gpu)
    model.reset()
    # GPUのセット
    if gpu>=0:
        ARR = cuda.cupy
        cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)

        if communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(communicator)
        #device = comm.intra_rank
    else:
        ARR = np
        if communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        #device = -1


    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(mpi4py.MPI.COMM_WORLD.Get_size()))
        if gpu>=0:
            print('Using GPUs')
        print('Using {} communicator'.format(communicator))
        print('Num Minibatch-size: {}'.format(batchsize))
        print('Num epoch: {}'.format(epoch))
        print('==========================================')


    random.seed(123)

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=args.out)

    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=gpu)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.dump_graph('main/loss'))
        trainer.extend(chainer.training.extensions.LogReport())
        trainer.extend(chainer.training.extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()
    serializers.save_hdf5(modelfile, model)







    """
    # 学習開始
    for epoch in range(epoch):
        # ファイルのパスの取得
        # エポックごとにoptimizerの初期化
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))

        random.shuffle(data)
        for num in range(len(data)//batchsize):
                #print(str(num))
                minibatch = data[num*batchsize: (num+1)*batchsize]
                # 読み込み用のデータ作成
                enc_words, dec_words = make_minibatch(minibatch)
                # modelのリセット
                model.reset()
                # 順伝播
                total_loss = forward(enc_words=enc_words,
                                     dec_words=dec_words,
                                     model=model,
                                     ARR=ARR,
                                     dict=dict)
                # 学習
                total_loss.backward()
                opt.update()
                #opt.zero_grads()
                # print (datetime.datetime.now())
        print ('Epoch %s 終了' % (epoch+1))

        serializers.save_hdf5(modelfile+"."+str(epoch), model)
    serializers.save_hdf5(modelfile, model)
    """

def test(datafile,dictfile,modelfile,gpu):

    dict=sampleSeq2Seq_data.load_dict(dictfile)


    # モデルのインスタンス化
    model = Seq2Seq(vocab_size=len(dict.keys()),
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
        dt.reverse()
        inword=to_word(dt,dict_inv)
        outword=to_word(predict,dict_inv)
        print("input:"+str(inword)+",output:"+str(outword))

def to_word(id_list,dict):
    res=[]
    for id in id_list:
        res.append(dict[id])
    return (res)

def main():
    p = argparse.ArgumentParser(description='seq2seq')


    #p.add_argument('--mode', default="test",help='train or test')
    #p.add_argument('--data', default="/Volumes/DATA/data/chat/txt/test.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--mode', default="train",choices=["train","test"], help='train or test')
    p.add_argument('--data', default="/Volumes/DATA/data/chat/txt/init.txt",help='in the case of input this file has two sentences a column, in the case of output this file has one sentence a column  ')
    p.add_argument('--dict', default="/Volumes/DATA/data/chat/txt/init.dict",help='word dictionay file, word and word id ')
    p.add_argument('--model',default="/Volumes/DATA/data/chat/txt/init.model",help="in the case of train mode this file is output,in the case of test mode this file is input")
    p.add_argument('-g','--gpu',default=-1, type=int)
    p.add_argument('--embed',default=EMBED, type=int,help="only train mode")
    p.add_argument('--hidden',default=HIDDEN, type=int, help="only train mode")
    p.add_argument('--batch',default=BATCH, type=int, help="only train mode")
    p.add_argument('--epoch',default=EPOCH, type=int, help="only train mode")
    p.add_argument('--communicator', type=str, default='hierarchical', help='Type of communicator')
    p.add_argument('--outdir', type=str, default='/Volumes/DATA/data/chat/outfiles/', help='Type of communicator')
    args = p.parse_args()

    print ('開始: ', datetime.datetime.now())
    try:
        if args.mode == "train":
            train(args.data,args.dict,args.model,args.gpu,args.embed,args.hidden,args.batch,args.epoch,args.communicator)
        else:
            test(args.data,args.dict,args.model,args.gpu)
    except:
        import traceback
        print( traceback.format_exc())

    print ('終了: ', datetime.datetime.now())


if __name__ == '__main__':
    main()
