import tensorflow as tf
import numpy as np
import pandas as pd


class DemoPredict():
    def __init__(self, filtered_prod, prod_info2, model_path, w2v):
        self.filtered_prod = filtered_prod
        self.prod_info2 = prod_info2
        self.model_path = model_path
        self.w2v = w2v

    def creat_SentenceVec(self, sentence, model=None):
        if model is None:
            model = self.w2v
        vocab_ls = list(model.wv.vocab.keys())
        word_vecs = np.array([model[word] for word in sentence if word in vocab_ls])
        if len(word_vecs) == 0:
            return 'err'
        else:
            sentence_vec = word_vecs.mean(axis=0)
            return sentence_vec

    def creat_Xdemo(self, rec, n_steps):
        rec_num = [str(self.filtered_prod.query(f"filtered_name=='{prod}'").clean_product_code.iloc[0]) for prod in rec
                   if len(self.filtered_prod.query(f"filtered_name=='{prod}'")) > 0]
        X_demo = np.array([self.creat_SentenceVec(rec_num)])
        X_demo = np.array([np.pad(X_demo, [(0, n_steps - len(X_demo)), (0, 0)], 'constant')])
        return X_demo

    def prediction(self, X_demo, seq_len_demo):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            demo_pred = final_output.eval(feed_dict={X: X_demo, seq_length: seq_len_demo})
        return demo_pred

    def display_result(self, demo_pred):
        pred = pd.Series(demo_pred[0]).sort_values()[::-1].reset_index()
        pred.columns = ['clean_product_code2', 'score']
        result = pd.merge(self.prod_info2, pred).sort_values('score')[::-1].drop_duplicates(['clean_product_code'])
        result = result[~ result.product_name.isin(['チャージ', 'レジ袋'])]
        return result[['clean_product_name', 'score']]


class DefineGraph:
    def _reset_graph(self, seed=None):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def create_graph(self, n_inputs, n_outputs, n_steps, n_layers, n_neurons, learning_rate):
        global X, y, seq_length, basic_cells, multi_basic_cell, outputs, states
        global top_layer_h_state, logits, final_output, xentropy, loss, optimizer, training_op
        self._reset_graph(seed=None)

        # 入力と出力の型を作る
        X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
        y = tf.placeholder(tf.int32, shape=[None, n_outputs])
        # 層としては以下の二つ
        ## 再帰ニューロン層の定義
        seq_length = tf.placeholder(tf.int32, [None])
        basic_cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
        multi_basic_cell = tf.contrib.rnn.MultiRNNCell(basic_cells)
        outputs, states = tf.nn.dynamic_rnn(multi_basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
        top_layer_h_state = states[-1]
        ## 全結合層の定義
        logits = tf.layers.dense(top_layer_h_state, n_outputs)
        final_output = tf.contrib.layers.softmax(logits)
        # 損失関数の定義
        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='xent')
        # 損失関数の最適化手法の定義
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 訓練
        training_op = optimizer.minimize(loss)
