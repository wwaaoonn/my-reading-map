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
    def test_曲がりが最も急なkを返す(self):
        # k=3 で下がり方が鈍くなる
        assert elbow_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0])) == 3

    def test_候補が1つなら最初のkを返す(self):
        assert elbow_k(scores_df([100.0])) == 2

    def test_候補が2つでも落ちない(self):
        assert elbow_k(scores_df([100.0, 50.0])) == 2

    def test_直線的に下がる場合も範囲内のkを返す(self):
        result = elbow_k(scores_df([100.0, 80.0, 60.0, 40.0]))
        assert 2 <= result <= 5


class TestSuggestK:
    def test_エルボー法の値を採用する(self):
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=100)
        assert chosen == 3

    def test_冊数から決まる上限で頭を押さえる(self):
        # 10冊なら 10 // 5 = 2 が上限
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=10)
        assert chosen == 2

    def test_下限は2(self):
        chosen, _ = suggest_k(scores_df([100.0, 50.0]), n_books=3)
        assert chosen == 2

    def test_1クラスタあたりの冊数が下限を下回らない(self):
        n_books = 40
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0, 41.0, 40.0]), n_books)
        assert n_books // chosen >= MIN_BOOKS_PER_CLUSTER

    def test_判断に使った値を文章で返す(self):
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0]), n_books=100)
        text = "\n".join(reasons)
        assert "エルボー法" in text
        assert "シルエット係数" in text
        assert "採用" in text

    def test_シルエット係数が低いと注意を出す(self):
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0], [0.05, 0.04, 0.03]), n_books=100)
        assert any("--k" in line for line in reasons)

    def test_シルエット係数が十分なら注意を出さない(self):
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0], [0.5, 0.4, 0.3]), n_books=100)
        assert not any("--k" in line for line in reasons)


class TestFitKmeans:
    def test_離れた2群を分ける(self):
        embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [9.1, 9.1]])
        labels = fit_kmeans(embeddings, k=2, random_state=0)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_同じ入力なら同じ結果(self):
        rng = np.random.default_rng(0)
        embeddings = rng.random((20, 4))
        first = fit_kmeans(embeddings, k=3, random_state=0)
        assert np.array_equal(first, fit_kmeans(embeddings, k=3, random_state=0))

    def test_クラスタIDはk未満(self):
        rng = np.random.default_rng(0)
        labels = fit_kmeans(rng.random((20, 4)), k=3, random_state=0)
        assert set(labels) <= {0, 1, 2}


class TestEvaluateK:
    def test_kは2から冊数未満まで(self):
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((8, 3)), k_max=14, verbose=False)
        assert scores["k"].tolist() == [2, 3, 4, 5, 6, 7]

    def test_k_maxで打ち切る(self):
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((20, 3)), k_max=4, verbose=False)
        assert scores["k"].max() == 4

    def test_指標の列がそろう(self):
        rng = np.random.default_rng(0)
        scores = evaluate_k(rng.random((10, 3)), k_max=3, verbose=False)
        assert list(scores.columns) == ["k", "inertia", "silhouette"]


class TestClusterCohesion:
    def test_同一ベクトルのクラスタは1に近い(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0]))
        assert result.loc[0, "平均コサイン類似度"] == pytest.approx(1.0)

    def test_1冊だけのクラスタは1(self):
        result = cluster_cohesion(np.array([[1.0, 0.0]]), np.array([0]))
        assert result.loc[0, "平均コサイン類似度"] == 1.0

    def test_クラスタごとに冊数を数える(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0, 1]))
        assert result["冊数"].tolist() == [2, 1]

    def test_直交するベクトルの類似度は0(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = cluster_cohesion(embeddings, np.array([0, 0]))
        assert result.loc[0, "平均コサイン類似度"] == pytest.approx(0.0)
