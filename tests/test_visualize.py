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
    def test_floor_is_five(self):
        """下限は5。"""
        assert auto_perplexity(3) == 5

    def test_does_not_exceed_maximum(self):
        """上限を超えない。"""
        assert auto_perplexity(10000) == PERPLEXITY_MAX

    def test_derives_from_book_count(self):
        """冊数から決まる。"""
        # (61 - 1) // 3 = 20
        assert auto_perplexity(61) == 20

    @pytest.mark.parametrize("n_books", [3, 10, 50, 100, 1000])
    def test_stays_within_bounds(self, n_books):
        """常に5以上かつ上限以下。"""
        assert 5 <= auto_perplexity(n_books) <= PERPLEXITY_MAX


class TestClusterPositions:
    def test_returns_matching_row_positions(self):
        """該当する行位置を返す。"""
        df = pd.DataFrame({"クラスタID": [0, 1, 0, 1, 0]})
        assert cluster_positions(df, 0).tolist() == [0, 2, 4]

    def test_returns_empty_when_no_match(self):
        """該当が無ければ空。"""
        df = pd.DataFrame({"クラスタID": [0, 0]})
        assert cluster_positions(df, 9).size == 0


class TestCentroidSimilarity:
    def test_identical_vectors_score_one(self):
        """同一ベクトルなら1。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert np.allclose(centroid_similarity(embeddings, np.array([0, 1])), 1.0)

    def test_ranks_by_closeness_to_centroid(self):
        """重心に近い順を判別できる。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sims = centroid_similarity(embeddings, np.array([0, 1, 2]))
        assert sims[0] > sims[2]

    def test_length_matches_positions(self):
        """対象の行数と同じ長さ。"""
        embeddings = np.eye(4)
        assert len(centroid_similarity(embeddings, np.array([0, 2]))) == 2


class TestRepresentativeBooks:
    def test_orders_by_closeness_to_centroid(self):
        """重心に近い順に返す。"""
        df = pd.DataFrame({"クラスタID": [0, 0, 0]})
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert representative_books(df, embeddings, top_n=3)[0][-1] == 2

    def test_limits_to_top_n(self):
        """top_n で件数を絞る。"""
        df = pd.DataFrame({"クラスタID": [0, 0, 0]})
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert len(representative_books(df, embeddings, top_n=2)[0]) == 2

    def test_handles_top_n_larger_than_cluster(self):
        """冊数より多い top_n でも落ちない。"""
        df = pd.DataFrame({"クラスタID": [0]})
        assert len(representative_books(df, np.array([[1.0, 0.0]]), top_n=8)[0]) == 1

    def test_returns_entry_per_cluster(self):
        """クラスタごとに返す。"""
        df = pd.DataFrame({"クラスタID": [0, 1]})
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert set(representative_books(df, embeddings)) == {0, 1}

    def test_returns_row_positions_not_cluster_indices(self):
        """返すのはクラスタ内の順番ではなく行位置。"""
        # クラスタ1は0行目と2行目。類似度が同じなので順序は問わない
        df = pd.DataFrame({"クラスタID": [1, 0, 1]})
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        assert set(representative_books(df, embeddings, top_n=2)[1]) == {0, 2}
