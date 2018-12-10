# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import MeCab
import gensim
import sys; sys.path.append('/Users/Yoshida/github/NLP')
import NLP_tool
import ProdNameFilter_Module
import ProdNameFilter_Main
import DemoPredict_Main as DP


def load_module():
    global pr_filter, bumon_candidate
    # dict_path = ''  # 通常辞書
    dict_path = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd/'  # neologdの辞書
    mecabrc_tagger = MeCab.Tagger(f"-d {dict_path} mecabrc")
    # w2v_path = '/home/jovyan/work/data/word_vec/FastText_Wiki_Neologd_model.vec'
    w2v_path = '~/Documents/NLP_model/word2vec_model/FastText_Wiki_Neologd_model.vec'
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    # product_filter用
    onehot = NLP_tool.One_HotVector(mecabrc_tagger=mecabrc_tagger)
    wakati = onehot.wakati_noun

    bumon_candidate = ProdNameFilter_Main.Params.bumon_candidate
    stop_words = ProdNameFilter_Main.Params.stop_words
    replace_prod_dict = ProdNameFilter_Main.Params.replace_dict
    pr_filter = ProdNameFilter_Module.ProdNameFilter()
    pr_filter.fit(w2v, wakati, stop_words, replace_prod_dict)
    print('Finished Loading')


def filter_prod(product, rank_mod=False, nbest=1, weight_mod=False):
    filtering = pd.concat([pd.concat([pr_filter.predict(product, bn, rank_mod=True, nbest=nbest),
                                      pd.Series(None)]).fillna(bn) for bn in bumon_candidate]).rename(columns={0: 'category'})
    if weight_mod:
        filtering_main = filtering[filtering.category.isin(['菓子', 'パン'])]
        filtering_main['similarity'] = filtering_main.similarity + 0.1
        filtering_sub = filtering[~ filtering.category.isin(['菓子', 'パン'])]
        filtering = pd.concat([filtering_main, filtering_sub])
    filtering = filtering.sort_values('similarity')[::-1].reset_index(drop=True)
    if rank_mod:
        return filtering
    else:
        return tuple([product] + list(filtering.loc[0, ['words', 'category']]))


def filter_receipt(receipt):
    filtered_receipt = pd.DataFrame([filter_prod(prod, weight_mod=False) for prod in receipt], columns=['product', 'key_word', 'category'])
    return filtered_receipt


def rnn_prediction(rec):
    predict_receipt = DP.main(rec,
                              DP.Params.n_inputs,
                              DP.Params.n_outputs,
                              DP.Params.n_steps,
                              DP.Params.n_layers,
                              DP.Params.n_neurons,
                              DP.Params.learning_rate)
    return predict_receipt[:10].reset_index(drop=True)


# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
    title = "Pusrchase Data Storage"
    message = 'あなたのレシート情報を入力してください'
    # index.html をレンダリングする
    return render_template('index_TechnoMall.html',
                           message=message,
                           title=title)


# /page2 にアクセスしたときの処理
@app.route('/page2', methods=['GET', 'POST'])
def post():
    title = "Purchase Data Storage"
    message = 'あなたのレシート情報を入力してください'
    if request.method == 'POST':
        # リクエストフォームから、データを取得
        item_ls = [request.form['item%s' % item_id] for item_id in range(1, item_len+1)
                   if request.form['item%s' % item_id] != '']
        print(item_ls)

        # 商品名のフィルター
        filtered_receipt = filter_receipt(item_ls)
        filter_result = filtered_receipt.to_html(classes='filter_table')
        item_ls = item_ls + [None] * (item_len - len(item_ls))
        # filter_result = pd.DataFrame(item_ls).to_html(classes='filter_table')  # 動作確認用

        # 商品のpredict
        rec = list(filtered_receipt['key_word'])
        predict_receipt = rnn_prediction(rec)
        predict_result = predict_receipt.to_html(classes='pred_table')
        # index.html をレンダリングする
        return render_template('index_TechnoMall.html',
                               message=message,
                               title=title,
                               filter_result=filter_result,
                               predict_result=predict_result,
                               it1=item_ls[0],
                               it2=item_ls[1],
                               it3=item_ls[2],
                               it4=item_ls[3],
                               it5=item_ls[4]
                               )
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))


if __name__ == '__main__':
    load_module()
    DP.data_load()
    item_len = 5
    app.debug = True  # デバッグモード有効化
    app.run(host='localhost', port=5555)  # どこからでもアクセス可能に
