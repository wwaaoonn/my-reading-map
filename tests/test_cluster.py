"""src/cluster.py のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.cluster import (
    MIN_BOOKS_PER_CLUSTER,
    cluster_cohesion,
    elbow_k,
    evaluate_k,
    fit_kmeans,
    suggest_k,
)


def scores_df(inertias, silhouettes=None, start_k=2):
    ks = list(range(start_k, start_k + len(inertias)))
    if silhouettes is None:
        silhouettes = [0.5] * len(inertias)
    return pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": silhouettes})


class TestElbowK:
    def test_returns_k_at_sharpest_bend(self):
        """曲がりが最も急な k を返す。"""
        # k=3 で下がり方が鈍くなる
        assert elbow_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0])) == 3

    def test_returns_first_k_with_single_candidate(self):
        """候補が1つなら最初の k を返す。"""
        assert elbow_k(scores_df([100.0])) == 2

    def test_handles_two_candidates(self):
        """候補が2つでも落ちない。"""
        assert elbow_k(scores_df([100.0, 50.0])) == 2

    def test_returns_k_within_range_for_linear_decline(self):
        """直線的に下がる場合も範囲内の k を返す。"""
        result = elbow_k(scores_df([100.0, 80.0, 60.0, 40.0]))
        assert 2 <= result <= 5


class TestSuggestK:
    def test_uses_elbow_value(self):
        """エルボー法の値を採用する。"""
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=100)
        assert chosen == 3

    def test_caps_by_book_count(self):
        """冊数から決まる上限で頭を押さえる。"""
        # 10冊なら 10 // 5 = 2 が上限
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=10)
        assert chosen == 2

    def test_floor_is_two(self):
        """下限は2。"""
        chosen, _ = suggest_k(scores_df([100.0, 50.0]), n_books=3)
        assert chosen == 2

    def test_keeps_minimum_books_per_cluster(self):
        """1クラスタあたりの冊数が下限を下回らない。"""
        n_books = 40
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0, 41.0, 40.0]), n_books)
        assert n_books // chosen >= MIN_BOOKS_PER_CLUSTER

    def test_returns_reasons(self):
        """判断に使った値を文章で返す。"""
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0]), n_books=100)
        text = "\n".join(reasons)
        assert "エルボー法" in text
        assert "シルエット係数" in text
        assert "採用" in text

    def test_warns_when_silhouette_is_low(self):
        """シルエット係数が低いと注意を出す。"""
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0], [0.05, 0.04, 0.03]), n_books=100)
        assert any("--k" in line for line in reasons)

    def test_does_not_warn_when_silhouette_is_fine(self):
        """シルエット係数が十分なら注意を出さない。"""
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0], [0.5, 0.4, 0.3]), n_books=100)
        assert not any("--k" in line for line in reasons)


class TestFitKmeans:
    def test_separates_two_distant_groups(self):
        """離れた2群を分ける。"""
        embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [9.1, 9.1]])
        labels = fit_kmeans(embeddings, k=2, random_state=0)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_is_deterministic(self):
        """同じ入力なら同じ結果。"""
        rng = np.random.default_rng(0)
        embeddings = rng.random((20, 4))
        first = fit_kmeans(embeddings, k=3, random_state=0)
        assert np.array_equal(first, fit_kmeans(embeddings, k=3, random_state=0))

    def test_labels_are_below_k(self):
        """クラスタIDは k 未満。"""
        rng = np.random.default_rng(0)
        labels = fit_kmeans(rng.random((20, 4)), k=3, random_state=0)
        assert set(labels) <= {0, 1, 2}


class TestEvaluateK:
    def test_evaluates_k_from_two_to_below_count(self):
        """k は2から冊数未満まで。"""
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((8, 3)), k_max=14, verbose=False)
        assert scores["k"].tolist() == [2, 3, 4, 5, 6, 7]

    def test_stops_at_k_max(self):
        """k_max で打ち切る。"""
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((20, 3)), k_max=4, verbose=False)
        assert scores["k"].max() == 4

    def test_returns_expected_columns(self):
        """指標の列がそろう。"""
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((10, 3)), k_max=3, verbose=False)
        assert list(scores.columns) == ["k", "inertia", "silhouette"]


class TestClusterCohesion:
    def test_identical_vectors_score_one(self):
        """同一ベクトルのクラスタは1に近い。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0]))
        assert result.loc[0, "平均コサイン類似度"] == pytest.approx(1.0)

    def test_single_book_cluster_scores_one(self):
        """1冊だけのクラスタは1。"""
        result = cluster_cohesion(np.array([[1.0, 0.0]]), np.array([0]))
        assert result.loc[0, "平均コサイン類似度"] == 1.0

    def test_counts_books_per_cluster(self):
        """クラスタごとに冊数を数える。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0, 1]))
        assert result["冊数"].tolist() == [2, 1]

    def test_orthogonal_vectors_score_zero(self):
        """直交するベクトルの類似度は0。"""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0]))
        assert result.loc[0, "平均コサイン類似度"] == pytest.approx(0.0)
