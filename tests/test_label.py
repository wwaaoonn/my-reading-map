"""reading_map/label.py のテスト。"""

import pandas as pd

from reading_map.label import tokenize_japanese, top_words_per_cluster


class TestTokenizeJapanese:
    def test_extracts_nouns_and_verbs_in_base_form(self):
        """名詞と動詞を原形で取り出す。"""
        tokens = tokenize_japanese("猫が屋根を歩いた。")
        assert "猫" in tokens
        assert "屋根" in tokens
        assert "歩く" in tokens

    def test_drops_particles(self):
        """助詞は残らない。"""
        tokens = tokenize_japanese("猫が屋根を歩いた。")
        assert not {"が", "を", "た"} & set(tokens)

    def test_keeps_adjectives_in_base_form(self):
        """形容詞は原形で残る。"""
        assert "美しい" in tokenize_japanese("美しかった風景。")

    def test_drops_dependent_words(self):
        """非自立の語は落ちる。"""
        assert "こと" not in tokenize_japanese("本を読むことが好きだ。")

    def test_drops_suffixes(self):
        """接尾の語は落ちる。"""
        tokens = tokenize_japanese("子どもたちの楽しさ。")
        assert "たち" not in tokens
        assert "さ" not in tokens

    def test_drops_pronouns(self):
        """代名詞は落ちる。"""
        assert "それ" not in tokenize_japanese("それを読んだ。")

    def test_drops_numerals(self):
        """数は落ちる。"""
        assert "一" not in tokenize_japanese("一冊の本。")

    def test_returns_empty_list_for_empty_text(self):
        """空文字なら空のリスト。"""
        assert tokenize_japanese("") == []


class TestTopWordsPerCluster:
    def _df(self, texts_by_cluster):
        rows = []
        for cluster_id, texts in texts_by_cluster.items():
            rows.extend({"クラスタID": cluster_id, "説明文": t} for t in texts)
        return pd.DataFrame(rows)

    def test_returns_words_per_cluster(self):
        """クラスタごとに語を返す。"""
        df = self._df({0: ["猫が眠る。"], 1: ["宇宙を旅する。"]})
        result = top_words_per_cluster(df)
        assert set(result) == {0, 1}
        assert "猫" in result[0]
        assert "宇宙" in result[1]

    def test_demotes_words_common_to_all_clusters(self):
        """全クラスタに出る語は上位から外れる。"""
        # 「物語」はどちらにも出る。IDFが1.0になり、固有の語より下がる
        df = self._df(
            {
                0: ["物語 物語 物語 猫 猫 猫 猫"],
                1: ["物語 物語 物語 宇宙 宇宙 宇宙 宇宙"],
            }
        )
        result = top_words_per_cluster(df)
        assert result[0][0] == "猫"
        assert result[1][0] == "宇宙"

    def test_limits_to_top_n(self):
        """top_n で件数を絞る。"""
        df = self._df({0: ["猫 犬 鳥 馬 牛 羊 豚 鹿"]})
        assert len(top_words_per_cluster(df, top_n=3)[0]) == 3

    def test_excludes_zero_score_words(self):
        """スコアが0の語は含めない。"""
        df = self._df({0: ["猫が眠る。"], 1: ["宇宙を旅する。"]})
        result = top_words_per_cluster(df, top_n=50)
        # クラスタ0に出ない「宇宙」はTF-IDFが0になる
        assert "宇宙" not in result[0]

    def test_returns_empty_words_when_no_word_is_found(self):
        """説明文がすべて空でも落ちず、空のリストを返す。"""
        df = self._df({0: ["", ""], 1: [""]})
        assert top_words_per_cluster(df) == {0: [], 1: []}

    def test_cluster_ids_are_int(self):
        """クラスタIDは int。"""
        df = self._df({0: ["猫が眠る。"], 1: ["宇宙を旅する。"]})
        assert all(isinstance(k, int) for k in top_words_per_cluster(df))
