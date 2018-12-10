# coding: utf-8
import pandas as pd
import MeCab
import gensim
import collections
import sys; sys.path.append('../')
import NLP_tool
import ProdNameFilter_Module
import pickle

path_pos_data = '~/work/data/complete_pos_data.csv'
path_w2v = '~/work/data/word_vec/FastText_Wiki_Neologd_model.vec'
path_dict = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd/'  # neologd辞書
# path_dict = ''  # 通常辞書
path_result = '~/work/data/prod_name_filter/prod_filter_ver2.csv'


class Params():
    # 部門フィルターのパラメータ
    bumon_candidate = [
        '食', '野菜', '飲料', '調味料', 'パン',
        'お菓子', '揚げ物', '肉', '魚', '衣類', '雑貨', '洗剤'
        ]
    # 商品名フィルタのパラメータ
    stop_words = [
        'オーガニック', '有機', '農場', '薬味', '水煮', '栽培',
        '製菓', '炒め', 'ビタミン', '食品', '燻製', '煮', '葉', '袋',
        'ゆで', '茸', '茶', 'フルーツ', '甘露煮', '味', '包装', '干し',
        '県', '産', '飲料', 'サラダ', '野菜', '煮込み', '肉', '食'
        ]
    replace_dict = {
        'キウィ': 'キウイ',
        'たねなし': '',
        'こまつな': '小松菜',
        '絹とうふ': '絹豆腐',
        '木綿とうふ': '木綿豆腐',
        'おおば': '大葉',
        'シークワーサー': 'シークヮーサー',
        'ブロッコリ': 'ブロッコリー',
        'アンパンマン': '',
        'ティシュー': 'ティッシュ',
        '虫よけ': '虫除け',
        'シューマイ': '焼売',
        'シウマイ': '焼売',
        'シュウマイ': '焼売'
        }
    nbest = 50
    save_mod = True


# データの準備
def data_load():
    global w2v, unique_prod, bc_ls
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_w2v, binary=False)
    pos_data = pd.read_csv(path_pos_data, index_col=0)
    # ユニークな商品名
    count_prod = collections.Counter(pos_data.product_name)
    count_prod_df = pd.DataFrame([(key, val) for key, val in count_prod.items()], columns=['product_name', 'num'])
    unique_prod = pos_data.loc[:, ['bumon_code', 'product_name']].drop_duplicates()
    unique_prod = pd.merge(unique_prod, count_prod_df).sort_values('num')[::-1].reset_index(drop=True)
    # ユニークな部門コード
    bc_ls = list(unique_prod.pivot_table(index='bumon_code', values='num', aggfunc='sum').sort_values('num')[::-1].index)


def idetify_bumon(bumon_candidate, save_mod=False):
    '''
    部門名の決定
    '''
    bm_identifier = ProdNameFilter_Module.BumonNameIdentifier()
    wakati_tagger = MeCab.Tagger(f"-d {path_dict} -Owakati")
    dsim = NLP_tool.Sentence_similarity(wakati_tagger=wakati_tagger, model=w2v)
    bm_identifier.fit(unique_prod, bumon_candidate, wakati_tagger, dsim)
    if save_mod:
        fpath = '/home/jovyan/work/data/bm_identifier.pickle'
        with open(fpath, mode='wb') as f:
            pickle.dump(bm_identifier, f)
    pred_bumon_name = [(bc, bm_identifier.predict(bc)) for bc in bc_ls]
    return pred_bumon_name


def filter_prod(pred_bumon_name, stop_words, replace_dict, nbest=50, save_mod=False):
    '''
    各部門ごとに、フィルタリングを実施
    '''
    pr_filter = ProdNameFilter_Module.ProdNameFilter()
    mecabrc_tagger = MeCab.Tagger(f"-d {path_dict} mecabrc")
    onehot = NLP_tool.One_HotVector(mecabrc_tagger=mecabrc_tagger)
    wakati = onehot.wakati_noun
    pr_filter.fit(w2v, wakati, stop_words, replace_dict)
    if save_mod:
        fpath = '/home/jovyan/work/data/pr_filter.pickle'
        with open(fpath, mode='wb') as f:
            pickle.dump(pr_filter, f)

    def iter_prod_filtering(hp, bc):
        target_prod = list(unique_prod[unique_prod.bumon_code == bc]['product_name'])
        filtered_prodname = [([bc] + [hp] + [prod] +
            pr_filter.predict(prod, hp, nbest=nbest, rank_mod=None)) for prod in target_prod]
        return filtered_prodname

    filtered_prodname = [iter_prod_filtering(hp, bc) for bc, hp in pred_bumon_name]
    filtered_prodname_flatt = [prod for prod_ls in filtered_prodname for prod in prod_ls]
    col = ['bumon_code', 'hypernym', 'product_name', 'filtered_name', 'similarity']
    filtered_prod = pd.DataFrame(filtered_prodname_flatt, columns=col)
    filtered_prod['clean_product_name'] = filtered_prod.bumon_code.astype(str) + '_' + filtered_prod.filtered_name
    return filtered_prod


if __name__ == '__main__':
    data_load()
    # bc_ls = bc_ls[:5]  # テスト用 (全部でやる時は削除すること)
    pred_bumon_name = idetify_bumon(Params.bumon_candidate, Params.save_mod)
    filtered_prod = filter_prod(pred_bumon_name, Params.stop_words, Params.replace_dict, Params.nbest, Params.save_mod)
    filtered_prod.to_csv(path_result)
