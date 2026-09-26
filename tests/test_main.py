"""main.py のテスト（MapResult から出力する表とファイル）。"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import main as main_mod
from main import (
    books_frame,
    escape_csv_formulas,
    k_scores_frame,
    pairs_frame,
    result_json,
    stamped,
    summary_frame,
    write_outputs,
)
from reading_map.pipeline import Book, Cluster, KScore, MapResult, SimilarPair


@pytest.fixture
def df():
    """4冊の読書記録。"""
    return pd.DataFrame(
        {
            "タイトル": ["猫の本", "犬の本", "星の本", "月の本"],
            "説明文": ["猫を飼う", "犬と歩く", "星を見る", "月を眺める"],
            "著者": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture
def embeddings():
    """4冊ぶんのベクトル。前の2冊と後ろの2冊が近い。"""
    return np.array(
        [[1.0, 0.1], [0.9, 0.2], [0.1, 1.0], [0.2, 0.9]], dtype=np.float32
    )


@pytest.fixture
def result():
    """4冊を2クラスタに分けた結果。"""
    return MapResult(
        books=[
            Book(position=0, title="猫の本", cluster_id=0, x=1.5, y=-2.25),
            Book(position=1, title="犬の本", cluster_id=0, x=1.0, y=-2.0),
            Book(position=2, title="星の本", cluster_id=1, x=-3.0, y=4.0),
            Book(position=3, title="月の本", cluster_id=1, x=-2.5, y=3.5),
        ],
        clusters=[
            Cluster(0, "動物", 2, ["猫", "犬", "飼う"], [1, 0]),
            Cluster(1, "天体", 2, ["星", "月"], [2, 3]),
        ],
        similar_pairs=[SimilarPair(2, 3, 0.95), SimilarPair(0, 1, 0.9)],
        k=2,
        silhouette=0.5,
        naming="top_words",
        k_scores=[KScore(k=2, inertia=1.5, silhouette=0.5, min_cluster_size=2)],
    )


class TestBooksFrame:
    def test_adds_cluster_and_coordinates(self, df, result):
        """入力の列のあとに、クラスタIDと座標の列を足す。"""
        books = books_frame(df, result)
        assert list(books.columns) == [
            "タイトル", "説明文", "著者", "クラスタID", "tsne_x", "tsne_y",
        ]
        assert books["クラスタID"].tolist() == [0, 0, 1, 1]
        assert books["tsne_x"].tolist() == [1.5, 1.0, -3.0, -2.5]

    def test_writes_coordinates_as_float32(self, df, result):
        """座標は float32 の列にする。"""
        books = books_frame(df, result)
        assert books["tsne_x"].dtype == np.float32
        assert books["tsne_y"].dtype == np.float32

    def test_does_not_modify_input(self, df, result):
        """渡されたDataFrameは変えない。"""
        books_frame(df, result)
        assert list(df.columns) == ["タイトル", "説明文", "著者"]


class TestSummaryFrame:
    def test_lists_each_cluster(self, result):
        """クラスタごとに名前・冊数・代表本・頻出語を並べる。"""
        summary = summary_frame(result)
        assert list(summary.columns) == ["クラスタID", "冊数", "クラスタ名", "代表本", "頻出語"]
        assert summary["クラスタ名"].tolist() == ["動物", "天体"]
        assert summary["冊数"].tolist() == [2, 2]

    def test_uses_first_representative_title(self, result):
        """代表本は、重心に最も近い本のタイトル。"""
        assert summary_frame(result)["代表本"].tolist() == ["犬の本", "星の本"]

    def test_joins_top_words_with_space(self, result):
        """頻出語は空白でつなぐ。"""
        assert summary_frame(result)["頻出語"].tolist() == ["猫 犬 飼う", "星 月"]


class TestPairsFrame:
    def test_uses_titles(self, result):
        """本は位置ではなくタイトルで示し、順番は変えない。"""
        pairs = pairs_frame(result)
        assert list(pairs.columns) == ["本1", "本2", "類似度"]
        assert pairs.values.tolist() == [["星の本", "月の本", 0.95], ["猫の本", "犬の本", 0.9]]

    def test_keeps_columns_when_empty(self, result):
        """ペアが無くても列はある。"""
        result.similar_pairs = []
        pairs = pairs_frame(result)
        assert list(pairs.columns) == ["本1", "本2", "類似度"]
        assert pairs.empty


class TestKScoresFrame:
    def test_has_one_row_per_k(self, result):
        """kごとに1行で、指標を列に持つ。"""
        scores = k_scores_frame(result)
        assert list(scores.columns) == ["k", "inertia", "silhouette", "min_cluster_size"]
        assert scores["k"].tolist() == [2]


class TestWriteOutputs:
    def test_writes_k_selection_files_with_stamp(self, df, embeddings, result, tmp_path):
        """k を自動で決めたときは、k の検討に使った指標も出力する。"""
        write_outputs(df, embeddings, result, tmp_path, "20260921-104300")
        assert (tmp_path / "k_selection_20260921-104300.csv").exists()
        assert (tmp_path / "k_selection_20260921-104300.png").exists()

    def test_skips_k_selection_files_when_k_given(self, df, embeddings, result, tmp_path):
        """k を指定したとき（k_scores が None）は、k の検討に使うファイルを作らない。"""
        result.k_scores = None
        write_outputs(df, embeddings, result, tmp_path, "20260921-104300")
        assert not list(tmp_path.glob("k_selection*"))

    def test_writes_maps_and_tables(self, df, embeddings, result, tmp_path):
        """マップのPNGと、4種類のCSVを出力する。"""
        write_outputs(df, embeddings, result, tmp_path, "20260921-104300")
        names = {path.name for path in tmp_path.iterdir()}
        for name in [
            "reading_map", "cluster_0", "cluster_1",
        ]:
            assert f"{name}_20260921-104300.png" in names
        for name in [
            "clustered_books", "clustered_books_public", "cluster_summary", "similar_pairs",
        ]:
            assert f"{name}_20260921-104300.csv" in names


class TestResultJson:
    def test_can_be_loaded(self, result):
        """json.loads で読め、MapResult の値が入っている。"""
        data = json.loads(result_json(result))
        assert data["k"] == 2
        assert [c["name"] for c in data["clusters"]] == ["動物", "天体"]

    def test_keeps_japanese_unescaped(self, result):
        """日本語はエスケープしない。"""
        text = result_json(result)
        assert "動物" in text
        assert "\\u" not in text

    def test_writes_nan_as_null(self, result):
        """nan の silhouette は null にする。"""
        result.silhouette = float("nan")
        result.k_scores[0].silhouette = float("nan")
        text = result_json(result)
        assert "NaN" not in text
        data = json.loads(text)
        assert data["silhouette"] is None
        assert data["k_scores"][0]["silhouette"] is None


class TestMainWithJson:
    @pytest.fixture
    def run_main(self, df, embeddings, result, tmp_path, monkeypatch):
        """埋め込みの作成と計算を差し替えて、--json 付きで main() を呼ぶ関数を返す。"""
        csv_path = tmp_path / "log.csv"
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(main_mod, "load_or_build_embeddings", lambda *a, **kw: embeddings)
        monkeypatch.setattr(main_mod, "build_reading_map", lambda *a, **kw: result)
        monkeypatch.setattr(
            sys, "argv",
            ["main.py", str(csv_path), "--no-ai-names", "--json", "--out", str(out_dir)],
        )

        def run():
            main_mod.main()
            return out_dir

        return run

    def test_writes_only_json_to_stdout(self, run_main, capsys):
        """標準出力にはJSONだけを出し、末尾は改行。"""
        run_main()
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert json.loads(out)["k"] == 2

    def test_does_not_write_png_or_csv(self, run_main):
        """PNG・CSVは書き出さない。--out のディレクトリは作る。"""
        out_dir = run_main()
        assert out_dir.is_dir()
        assert not list(out_dir.glob("*.png"))
        assert not list(out_dir.glob("*.csv"))


class TestStamped:
    def test_inserts_stamp_before_extension(self):
        """拡張子の前に実行時刻を挟む。"""
        assert stamped("reading_map.png", "20260921-104300") == "reading_map_20260921-104300.png"

    @pytest.mark.parametrize("stamp", [None, ""])
    def test_keeps_name_without_stamp(self, stamp):
        """stamp が無ければ名前を変えない。"""
        assert stamped("reading_map.png", stamp) == "reading_map.png"


class TestEscapeCsvFormulas:
    def test_escapes_formula_prefixes(self):
        """数式として解釈される文字で始まる文字列に ' を前置する。"""
        df = pd.DataFrame({"名": ['=HYPERLINK("http://x")', "+1+1", "-1", "@SUM(A1)"]})
        assert escape_csv_formulas(df)["名"].tolist() == [
            '\'=HYPERLINK("http://x")',
            "'+1+1",
            "'-1",
            "'@SUM(A1)",
        ]

    def test_keeps_normal_strings(self):
        """普通の文字列は変えない。"""
        df = pd.DataFrame({"名": ["猫をめぐる物語", ""]})
        assert escape_csv_formulas(df)["名"].tolist() == ["猫をめぐる物語", ""]

    def test_keeps_numeric_columns(self):
        """数値の列は変えない（負の数を壊さない）。"""
        df = pd.DataFrame({"tsne_x": [-1.5, 2.0]})
        assert escape_csv_formulas(df)["tsne_x"].tolist() == [-1.5, 2.0]

    def test_does_not_modify_input(self):
        """渡されたDataFrameは変えない。"""
        df = pd.DataFrame({"名": ["=1"]})
        escape_csv_formulas(df)
        assert df["名"].tolist() == ["=1"]
