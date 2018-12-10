import pandas as pd


class ProdNameFilter:
    def fit(self, w2v_model, wakati_model, stop_words=[], replace_dict={}):
        self.w2v_model = w2v_model
        self.wakati_model = wakati_model
        self.stop_words = stop_words
        self.replace_dict = replace_dict

    def predict(self, item, hypernym, rank_mod=False, nbest=20):
        if len(self.replace_dict) != 0:
            for key, val in self.replace_dict.items():
                item = item.replace(key, val)
        item_wakati = self.wakati_model(item, double_dict=True, letter_conv=True, nbest=nbest)
        similarity_list = list()
        # 分かち書きした単語とhypernymの類似度を計算
        for noun in item_wakati:
            if noun in self.stop_words:
                continue
            try:
                similarity = self.w2v_model.similarity(w1=hypernym, w2=noun)
                similarity_list.append((noun, similarity))
            except KeyError:
                continue
        # エラー処理とデータの整形
        if len(similarity_list) == 0:
            similarity_df = pd.DataFrame([('error'), (1)], index=['words', 'similarity']).T
        else:
            similarity_df = pd.DataFrame(similarity_list, columns=['words', 'similarity'])
        similarity_df = similarity_df.drop_duplicates().sort_values('similarity')[::-1].reset_index(drop=True)
        # 結果を返す
        if rank_mod is None:
            return [similarity_df.iloc[0, 0], similarity_df.iloc[0, 1]]
        if rank_mod:
            return similarity_df
        elif not rank_mod:
            return similarity_df.iloc[0, 0]


class BumonNameIdentifier:
    def fit(self, prod_ls, bumon_name_cand, wakati_tagger, dsim):  # wakati_modelとwakati_taggerは違う
        self.prod_ls = prod_ls
        self.bumon_name_cand = bumon_name_cand
        self.wakati_tagger = wakati_tagger
        self.dsim = dsim

    def predict(self, bc, rank_mod=False):
        bumon_prod_ls = self.prod_ls[self.prod_ls.bumon_code == bc]
        text = self.wakati_tagger.parse(' '.join(list(bumon_prod_ls.loc[:, 'product_name'])))
        category_sim = [(bumon, self.dsim.sentence_similarity(text, bumon, nbest=1)) for bumon in self.bumon_name_cand]
        category_rank = pd.DataFrame(
                category_sim, columns=['category', 'similarity']).sort_values('similarity')[::-1].set_index('category')
        if rank_mod:
            return category_rank
        else:
            return category_rank.index[0]
