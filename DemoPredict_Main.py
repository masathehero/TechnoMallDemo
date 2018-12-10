import numpy as np
import pandas as pd
from DemoPredict_Module import DemoPredict
from DemoPredict_Module import DefineGraph
import gensim


class Params:
    n_inputs = 300  # X_train.shape[2]
    n_outputs = 21625  # y_train.shape[1]
    n_steps = 50  # X_train.shape[1]
    n_layers = 1
    n_neurons = 250
    learning_rate = 0.001


def data_load():
    global prod_info2, filtered_prod, model_path, w2v
    # 推定したものが具体的に何かを特定
    # path_prod_info2 = '/home/jovyan/work/data/technomall_demo_data/prod_info2.csv'
    path_prod_info2 = './technomall_demo_data/prod_info2.csv'
    prod_info2 = pd.read_csv(path_prod_info2, index_col=0)
    # 商品名フィルター結果のロード
    # path_filtered_prod = '/home/jovyan/work/data/prod_name_filter/prod_filter_ver4.csv'
    path_filtered_prod = './technomall_demo_data/prod_filter_ver4.csv'
    filtered_prod = pd.read_csv(path_filtered_prod, index_col=0)
    # モデルのパス
    # model_path = '/home/jovyan/work/code/trained_models/RNNmodel_ep=20_lr=0.001_loss=xent_all'
    model_path = './technomall_demo_data/trained_models/RNNmodel_ep=20_lr=0.001_loss=xent_all'
    # w２vのインポート
    # path_w2v = '/home/jovyan/work/data/word_vec/embedding_size=300_window=100_minc=5_iter=100_filter=ver4.vec'
    path_w2v = './technomall_demo_data/embedding_size=300_window=100_minc=5_iter=100_filter=ver4.vec'
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_w2v)


def main(rec, n_inputs, n_outputs, n_steps, n_layers, n_neurons, learning_rate):
    DefineGraph().create_graph(n_inputs, n_outputs, n_steps, n_layers, n_neurons, learning_rate)
    DP = DemoPredict(filtered_prod, prod_info2, model_path, w2v)
    X_demo = DP.creat_Xdemo(rec, n_steps)
    seq_len_demo = np.array([1])
    demo_pred = DP.prediction(X_demo, seq_len_demo)
    result = DP.display_result(demo_pred)
    return result


if __name__ == '__main__':
    data_load()
    rec = ['チョコ', 'ケーキ', 'アイス']
    result = main(rec,
                  Params.n_inputs,
                  Params.n_outputs,
                  Params.n_steps,
                  Params.n_layers,
                  Params.n_neurons,
                  Params.learning_rate)
    print(result.head())
    # result.to_csv('/home/jovyan/work/data/demo_data/demo_result.csv')
