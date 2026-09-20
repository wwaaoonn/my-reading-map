"""src/visualize.py のテスト。描画そのものは確かめない。"""

import numpy as np
import pandas as pd
import pytest

from src.visualize import (
    PERPLEXITY_MAX,
    auto_perplexity,
    centroid_similarity,
    cluster_positions,
    representative_books,
)


class TestAutoPerplexity:
    def test_下限は5(self):
        assert auto_perplexity(3) == 5

    def test_上限を超えない(self):
        assert auto_perplexity(10000) == PERPLEXITY_MAX

    def test_冊数から決まる(self):
        # (61 - 1) // 3 = 20
        assert auto_perplexity(61) == 20

    @pytest.mark.parametrize("n_books", [3, 10, 50, 100, 1000])
    def test_常に5以上かつ上限以下(self, n_books):
        assert 5 <= auto_perplexity(n_books) <= PERPLEXITY_MAX


class TestClusterPositions:
    def test_該当する行位置を返す(self):
        df = pd.DataFrame({"クラスタID": [0, 1, 0, 1, 0]})
        assert cluster_positions(df, 0).tolist() == [0, 2, 4]

    def test_該当が無ければ空(self):
        df = pd.DataFrame({"クラスタID": [0, 0]})
        assert cluster_positions(df, 9).size == 0


class TestCentroidSimilarity:
    def test_同一ベクトルなら1(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert np.allclose(centroid_similarity(embeddings, np.array([0, 1])), 1.0)

    def test_重心に近い順を判別できる(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sims = centroid_similarity(embeddings, np.array([0, 1, 2]))
        assert sims[0] > sims[2]

    def test_対象の行数と同じ長さ(self):
        embeddings = np.eye(4)
        assert len(centroid_similarity(embeddings, np.array([0, 2]))) == 2


class TestRepresentativeBooks:
    def test_重心に近い順に返す(self):
        df = pd.DataFrame({"クラスタID": [0, 0, 0]})
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert representative_books(df, embeddings, top_n=3)[0][-1] == 2

    def test_top_nで件数を絞る(self):
        df = pd.DataFrame({"クラスタID": [0, 0, 0]})
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert len(representative_books(df, embeddings, top_n=2)[0]) == 2

    def test_冊数より多いtop_nでも落ちない(self):
        df = pd.DataFrame({"クラスタID": [0]})
        assert len(representative_books(df, np.array([[1.0, 0.0]]), top_n=8)[0]) == 1

    def test_クラスタごとに返す(self):
        df = pd.DataFrame({"クラスタID": [0, 1]})
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert set(representative_books(df, embeddings)) == {0, 1}

    def test_返すのはクラスタ内の順番ではなく行位置(self):
        # クラスタ1は0行目と2行目。類似度が同じなので順序は問わない
        df = pd.DataFrame({"クラスタID": [1, 0, 1]})
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        assert set(representative_books(df, embeddings, top_n=2)[1]) == {0, 2}
