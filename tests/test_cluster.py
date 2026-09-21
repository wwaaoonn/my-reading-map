"""src/cluster.py のテスト。"""

import numpy as np
import pandas as pd

from src.cluster import (
    MIN_BOOKS_PER_CLUSTER,
    MIN_CLUSTER_BOOKS,
    elbow_k,
    evaluate_k,
    fit_kmeans,
    silhouette_for_labels,
    suggest_k,
)


def scores_df(inertias, silhouettes=None, start_k=2, min_cluster_sizes=None):
    ks = list(range(start_k, start_k + len(inertias)))
    if silhouettes is None:
        silhouettes = [0.5] * len(inertias)
    if min_cluster_sizes is None:
        # 既定では、どの k も最低冊数を満たしている
        min_cluster_sizes = [MIN_CLUSTER_BOOKS] * len(inertias)
    return pd.DataFrame(
        {
            "k": ks,
            "inertia": inertias,
            "silhouette": silhouettes,
            "min_cluster_size": min_cluster_sizes,
        }
    )


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

    def test_finds_bend_late_in_the_curve(self):
        """曲がる位置が後ろでも、そこを返す（小さい k に張り付かない）。"""
        # k=6 まで一定のペースで下がり、そこから鈍くなる
        assert elbow_k(scores_df([100.0, 80.0, 60.0, 40.0, 21.0, 20.0, 19.0, 18.0])) == 6

    def test_ignores_local_wiggle(self):
        """途中の小さな凹凸ではなく、曲線全体の形で決める。"""
        # 本当の曲がりは k=3。k=6 付近の僅かな段差に引っ張られない
        assert elbow_k(scores_df([100.0, 40.0, 38.0, 36.0, 33.0, 32.0, 31.0])) == 3

    def test_is_unchanged_by_inertia_scale(self):
        """inertia を定数倍しても選ぶ k は変わらない。"""
        inertias = [100.0, 50.0, 45.0, 43.0, 42.0]
        assert elbow_k(scores_df(inertias)) == elbow_k(scores_df([v * 1000 for v in inertias]))


class TestSuggestK:
    def test_uses_elbow_value(self):
        """エルボー法の値を採用する。"""
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=100)
        assert chosen == 3

    def test_caps_by_book_count(self):
        """冊数から決まる上限で頭を押さえる。"""
        # 10冊なら 10 // 4 = 2 が上限
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0]), n_books=10)
        assert chosen == 2

    def test_floor_is_two(self):
        """下限は2。"""
        chosen, _ = suggest_k(scores_df([100.0, 50.0]), n_books=3)
        assert chosen == 2

    def test_keeps_average_books_per_cluster(self):
        """1クラスタあたりの平均冊数が下限を下回らないkを選ぶ。"""
        n_books = 40
        chosen, _ = suggest_k(scores_df([100.0, 50.0, 45.0, 43.0, 42.0, 41.0, 40.0]), n_books)
        assert n_books // chosen >= MIN_BOOKS_PER_CLUSTER

    def test_caps_by_searched_k_max(self):
        """冊数から決まる上限より、試した k の範囲が狭ければそちらで止める。"""
        # 100冊なら 100 // 4 = 25 だが、k は 14 までしか試していない
        inertias = [100.0 - 6.0 * i for i in range(13)]  # k=2..14 を直線的に下げる
        scores = scores_df(inertias)
        assert scores["k"].max() == 14
        chosen, reasons = suggest_k(scores, n_books=100)
        assert chosen <= 14
        assert "探索したkの最大=14" in "\n".join(reasons)

    def test_reasons_show_both_upper_bounds(self):
        """2つの上限を両方とも表示する。"""
        _, reasons = suggest_k(scores_df([100.0, 50.0, 45.0]), n_books=100)
        text = "\n".join(reasons)
        assert f"1クラスタ平均{MIN_BOOKS_PER_CLUSTER}冊以上になるk=25" in text
        assert "探索したkの最大=4" in text

    def test_lowers_k_until_minimum_cluster_size_is_met(self):
        """最小クラスタが最低冊数を下回る k は選ばない。"""
        # エルボーは k=5。k=5 と k=4 は1冊のクラスタを含む
        scores = scores_df(
            [100.0, 80.0, 60.0, 40.0, 39.0, 38.0], min_cluster_sizes=[5, 4, 1, 1, 3, 2]
        )
        chosen, reasons = suggest_k(scores, n_books=100)
        assert chosen == 3
        assert "最低2冊を満たす最大のk: 3" in "\n".join(reasons)

    def test_keeps_k_when_minimum_cluster_size_is_met(self):
        """最低冊数を満たしていれば k を下げない。"""
        scores = scores_df([100.0, 50.0, 45.0, 43.0, 42.0], min_cluster_sizes=[9, 8, 7, 6, 5])
        chosen, reasons = suggest_k(scores, n_books=100)
        assert chosen == 3
        assert not any("最低" in line for line in reasons)

    def test_floor_is_two_even_when_minimum_is_unmet(self):
        """下限の2でも満たせないときは2を採用し、その旨を返す。"""
        scores = scores_df([100.0, 50.0, 45.0], min_cluster_sizes=[1, 1, 1])
        chosen, reasons = suggest_k(scores, n_books=100)
        assert chosen == 2
        assert any("満たすkが無い" in line for line in reasons)

    def test_uses_given_minimum_cluster_books(self):
        """最低冊数は引数で変えられる。"""
        # エルボーは k=5。k=4 と k=5 は最小2冊
        scores = scores_df(
            [100.0, 80.0, 60.0, 40.0, 39.0, 38.0], min_cluster_sizes=[9, 3, 2, 2, 1, 1]
        )
        assert suggest_k(scores, n_books=100)[0] == 5
        assert suggest_k(scores, n_books=100, min_cluster_books=3)[0] == 3

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
        assert list(scores.columns) == ["k", "inertia", "silhouette", "min_cluster_size"]


class TestSilhouetteForLabels:
    def test_separated_groups_score_high(self):
        """離れた2群のシルエット係数は高い。"""
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
        assert silhouette_for_labels(embeddings, np.array([0, 0, 1, 1])) > 0.5

    def test_mixed_groups_score_low(self):
        """入り混じった割り当てのシルエット係数は低い。"""
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
        assert silhouette_for_labels(embeddings, np.array([0, 1, 0, 1])) < 0.0

    def test_returns_nan_for_single_cluster(self):
        """クラスタが1つでは計算できないので nan。"""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert np.isnan(silhouette_for_labels(embeddings, np.array([0, 0])))

    def test_returns_nan_when_every_book_is_its_own_cluster(self):
        """クラスタ数が冊数と同じでは計算できないので nan。"""
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        assert np.isnan(silhouette_for_labels(embeddings, np.array([0, 1, 2])))

    def test_is_deterministic(self):
        """同じ入力なら同じ値。"""
        rng = np.random.default_rng(0)
        embeddings = rng.random((20, 4))
        labels = np.array([0] * 10 + [1] * 10)
        assert silhouette_for_labels(embeddings, labels) == silhouette_for_labels(
            embeddings, labels
        )
