"""src/visualize.py のテスト。描画そのものは確かめない。"""

import numpy as np
import pandas as pd
import pytest

from src.visualize import (
    ELLIPSE_MIN_AXIS_RATIO,
    PERPLEXITY_MAX,
    _plot_spread_ellipse,
    auto_perplexity,
    centroid_similarity,
    cluster_positions,
    plain_text,
    plot_cluster_maps,
    representative_books,
)


class TestPlainText:
    def test_removes_dollar_signs(self):
        """mathtextとして解釈される $ を取り除く。"""
        assert plain_text(r"$\frac$") == r"\frac"

    def test_keeps_other_characters(self):
        """それ以外は変えない。"""
        assert plain_text("猫をめぐる物語") == "猫をめぐる物語"

    def test_accepts_non_string(self):
        """文字列以外も受け取れる。"""
        assert plain_text(3) == "3"


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


class TestPlotClusterMaps:
    def _df(self):
        return pd.DataFrame(
            {
                "タイトル": ["猫", "犬", "宇宙", "銀河"],
                "クラスタID": [0, 0, 1, 1],
                "tsne_x": [0.0, 0.1, 5.0, 5.1],
                "tsne_y": [0.0, 0.1, 5.0, 5.1],
            }
        )

    def test_writes_one_file_per_cluster(self, tmp_path):
        """クラスタごとに1枚ずつ出力する。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
        plot_cluster_maps(self._df(), embeddings, {0: "猫の話", 1: "宇宙の話"}, tmp_path, dpi=50)
        assert sorted(f.name for f in tmp_path.iterdir()) == ["cluster_0.png", "cluster_1.png"]

    def test_puts_stamp_in_file_names(self, tmp_path):
        """stamp を渡すと、ファイル名に実行時刻が入る。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
        plot_cluster_maps(
            self._df(), embeddings, {0: "猫の話", 1: "宇宙の話"}, tmp_path,
            stamp="20260921-104300", dpi=50,
        )
        assert sorted(f.name for f in tmp_path.iterdir()) == [
            "cluster_0_20260921-104300.png",
            "cluster_1_20260921-104300.png",
        ]


class TestPlotSpreadEllipse:
    def _ellipses(self, x, y):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _plot_spread_ellipse(np.array(x), np.array(y), ax)
        ellipses = list(ax.patches)
        plt.close(fig)
        return ellipses

    def test_draws_ellipse_for_two_books(self):
        """2冊のクラスタにも楕円を描く。"""
        assert len(self._ellipses([0.0, 1.0], [0.0, 1.0])) == 1

    def test_two_book_ellipse_has_width_and_height(self):
        """2冊でもつぶれない（短軸が0にならない）。"""
        ellipse = self._ellipses([0.0, 1.0], [0.0, 1.0])[0]
        assert ellipse.width > 0
        assert ellipse.height >= ellipse.width * ELLIPSE_MIN_AXIS_RATIO

    def test_draws_nothing_for_single_book(self):
        """1冊では描かない。"""
        assert self._ellipses([0.0], [0.0]) == []

    def test_covers_collinear_books(self):
        """直線状に並んだクラスタでも短軸を持つ。"""
        ellipse = self._ellipses([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])[0]
        assert ellipse.height > 0
