import argparse,collections
import os
import sys

import numpy as np
import tensorflow.contrib.rnn as rnn
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

class Data:
    def __init__(self, FILE):
        self.BATCH_SIZE = 64
        self.MAX_LENGTH = 100
        self.MIN_LENGTH = 10
        self.MAX_WORDS_SIZE = 3500
        self.FILE = FILE
        self.BEGIN = '^'
        self.END = '$'
        self.UNK = '*'
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            if line and len(line) > self.MAX_LENGTH:
                index = line.rfind('。', 0, self.MAX_LENGTH)
                index = index if index>0 else self.MAX_LENGTH
                line = line[:index+1]
            return self.BEGIN + line + self.END
        self.poetrys = [line.strip().replace(' ','').split(':')[1] for line in open(self.FILE, encoding='utf-8')]
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > self.MIN_LENGTH]

        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_paris = sorted(counter.items(), key=lambda x:-x[1])
        words, _ = zip(*count_paris)

        words_size = min(self.MAX_WORDS_SIZE, len(words))
        self.words = words[:words_size] + (self.UNK,)
        self.words_size = len(self.words)

        self.c2i_dict = {w:i for i,w in enumerate(self.words)}
        self.i2c_dict = {i:w for i,w in enumerate(self.words)}
        self.UNK_ID = self.c2i_dict.get(self.UNK)
        self.c2i = lambda c : self.c2i_dict.get(c, self.UNK_ID)
        self.i2c = lambda i : self.i2c_dict.get(i)
        self.portrys = sorted(self.poetrys, key=lambda line:len(line))
        self.poetrys_vector = [list(map(self.c2i, poetry)) for poetry in self.poetrys]

    def create_batches(self):
        self.N_BATCH_SIZE = len(self.poetrys_vector)//self.BATCH_SIZE
        self.poetrys_vector = self.poetrys_vector[:self.N_BATCH_SIZE*self.BATCH_SIZE]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.N_BATCH_SIZE):
            batches = self.poetrys_vector[i*self.BATCH_SIZE : (i+1)*self.BATCH_SIZE]
            length = max(map(len, batches))
            for j in range(self.BATCH_SIZE):
                if len(batches[j]) < length:
                    batches[j][len(batches[j]) : length] = [self.UNK_ID] * (length - len(batches[j]))
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:,:-1] = xdata[:,1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

class Model:
    def __init__(self, data, infer=False):
        self.hiddin_dim = 128
        self.embedding_dim = 128
        self.n_layers = 2
        self.lr = 0.002
        if infer:
            self.BATCH_SIZE = 1
        else:
            self.BATCH_SIZE = data.BATCH_SIZE

        embedding = tf.get_variable(name='embedding', shape=[data.words_size, self.embedding_dim])
        self.x = tf.placeholder(tf.int32, [self.BATCH_SIZE, None])
        inputs = tf.nn.embedding_lookup(embedding, self.x)
        self.y = tf.placeholder(tf.int32, [self.BATCH_SIZE, None])
        cell = rnn.BasicLSTMCell(self.hiddin_dim)
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers)
        self.initial_state = self.cell.zero_state(self.BATCH_SIZE, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(self.cell, inputs=inputs, initial_state=self.initial_state)
        self.final_state = final_state

        self.output = tf.reshape(outputs, [-1, self.hiddin_dim])
        W = tf.get_variable(name='W', shape=[self.hiddin_dim, data.words_size])
        b = tf.get_variable(name='b', shape=[data.words_size])
        self.logits = tf.matmul(self.output, W) + b
        self.probs = tf.nn.softmax(self.logits)

        pred = tf.reshape(self.y, [-1])
        loss = seq2seq.sequence_loss_by_example([self.logits], [pred], [tf.ones_like(pred, dtype=tf.float32)])
        self.cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.cost)


def train(data, model):
    epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint('log')
        if model_file:
            print('load model...')
            saver.restore(sess, model_file)
        for epoch in range(epochs):
            for i in range(data.N_BATCH_SIZE):
                feed_dict = {model.x:data.x_batches[i], model.y:data.y_batches[i]}
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{}(eopch {}) | train_loss {:.3f}".format(epoch * data.N_BATCH_SIZE + i, epochs * data.N_BATCH_SIZE, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                if (epoch * data.N_BATCH_SIZE + i + 1)%2000 == 0 or (epoch==epochs-1 and i==data.N_BATCH_SIZE-1):
                    checkpoint_path = os.path.join('log', 'model.ckpt')
                    saver.save(sess, checkpoint_path)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')

def choice_word(data, weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sa = int(np.searchsorted(t, np.random.rand(1)*s))
    return data.i2c(sa)

def sample(data, model, head=''):
    for word in head:
        if word not in data.words:
            return '\'{}\'不在字典中'%word

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint('log'))
        # 当输入长度不为 5 或者 7 的时候，为藏头诗
        if head and  len(head)!=5 and len(head)!=7:
            print('开头 --->', head)
            poem = data.BEGIN
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.c2i, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = choice_word(data, probs[-1])
                while word!='，' and word!='。':
                    poem+=word
                    x = np.zeros((1,1))
                    x[0, 0] = data.c2i(word)
                    [probs, state] = sess.run([model.probs, model.final_state], {model.x:x, model.initial_state: state})
                    word = choice_word(data, probs[-1])
                poem+=word
            return poem[1:]
        # 啥也不输入，随机生成诗
        elif len(head)==0:
            poem = ''
            head = data.BEGIN
            x = np.array([list(map(data.c2i, head))])
            feed_dict = {model.x:x,  model.initial_state: sess.run(model.cell.zero_state(1, tf.float32))}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict=feed_dict)
            word = choice_word(data, probs[-1])
            while word!=data.END:
                poem += word
                x = np.zeros((1,1))
                x[0,0] = data.c2i(word)
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict={model.x:x, model.initial_state:state})
                word = choice_word(data, probs[-1])
            return poem
        # 长度为 5 或者 7 ，写五言诗或者七言绝句
        elif head and  (len(head)==5 or len(head)==7):
            poem = data.BEGIN
            L = len(head)
            poem += head + '，'
            x = np.array([list(map(data.c2i, poem))])
            feed_dict = {model.x: x, model.initial_state: sess.run(model.cell.zero_state(1, tf.float32))}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict=feed_dict)
            word = choice_word(data, probs[-1])
            while len(poem)<2*L+2 and word!=data.END:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.c2i(word)
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict={model.x: x, model.initial_state: state})
                word = choice_word(data, probs[-1])
            return poem[1:]




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ample', help='train\sample(default)')
    parser.add_argument('--head', type=str, default='', help='first sentence')
    args = parser.parse_args()
    data = Data('poetry.txt')
    if args.mode == 'sample':
        infer = True
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        model = Model(data=data, infer=infer)
        print(train(data, model))

if __name__ == '__main__':
    main()