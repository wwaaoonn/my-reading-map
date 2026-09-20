"""main.py のテスト（クラスタ数の決定まわり）。"""

import numpy as np
import pandas as pd
import pytest

from main import decide_k, escape_csv_formulas


@pytest.fixture
def embeddings():
    """10冊ぶんのベクトル。"""
    rng = np.random.default_rng(0)
    return rng.random((10, 4))


class TestDecideKWithRequestedK:
    def test_uses_requested_k(self, embeddings, tmp_path):
        """指定された k をそのまま使う。"""
        assert decide_k(embeddings, tmp_path, 3) == 3

    def test_accepts_k_equal_to_book_count(self, embeddings, tmp_path):
        """冊数と同じ k は受け付ける。"""
        assert decide_k(embeddings, tmp_path, 10) == 10

    @pytest.mark.parametrize("requested_k", [0, 1, 11, 50])
    def test_rejects_k_outside_range(self, embeddings, tmp_path, requested_k):
        """2未満、または冊数を超える k は止める。"""
        with pytest.raises(SystemExit) as error:
            decide_k(embeddings, tmp_path, requested_k)
        assert "--k" in str(error.value)

    def test_does_not_write_k_selection_files(self, embeddings, tmp_path):
        """k を指定したときは、k の検討に使うファイルを作らない。"""
        decide_k(embeddings, tmp_path, 3)
        assert list(tmp_path.iterdir()) == []


class TestDecideKAutomatically:
    def test_returns_k_within_range(self, embeddings, tmp_path):
        """自動で決めた k は2以上、冊数以下。"""
        k = decide_k(embeddings, tmp_path, None)
        assert 2 <= k <= len(embeddings)

    def test_writes_k_selection_files(self, embeddings, tmp_path):
        """k の検討に使った指標を出力する。"""
        decide_k(embeddings, tmp_path, None)
        assert (tmp_path / "k_selection.csv").exists()
        assert (tmp_path / "k_selection.png").exists()


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
