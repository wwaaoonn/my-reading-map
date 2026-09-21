"""src/embed.py のテスト。埋め込みモデルは読み込まない。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.embed import (
    cache_mismatch,
    count_distinct_descriptions,
    descriptions_digest,
    l2_normalize,
    load_or_build_embeddings,
    load_reading_log,
    meta_path,
    similar_pairs,
)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoadReadingLog:
    def test_reads_csv_with_required_columns(self, tmp_path):
        """必須列がそろっていれば読める。"""
        path = write_csv(tmp_path / "a.csv", [{"タイトル": "本1", "説明文": "説明1"}])
        df = load_reading_log(path)
        assert list(df["タイトル"]) == ["本1"]

    @pytest.mark.parametrize("missing", ["タイトル", "説明文"])
    def test_raises_when_required_column_missing(self, tmp_path, missing):
        """必須列が欠けると ValueError。"""
        row = {"タイトル": "本1", "説明文": "説明1"}
        del row[missing]
        path = write_csv(tmp_path / "a.csv", [row])
        with pytest.raises(ValueError, match=missing):
            load_reading_log(path)

    def test_fills_missing_description_with_empty_string(self, tmp_path):
        """説明文の欠損は空文字になる。"""
        path = write_csv(
            tmp_path / "a.csv",
            [{"タイトル": "本1", "説明文": None}, {"タイトル": "本2", "説明文": "説明2"}],
        )
        df = load_reading_log(path)
        assert df["説明文"].tolist() == ["", "説明2"]

    def test_keeps_extra_columns(self, tmp_path):
        """余分な列はそのまま残る。"""
        path = write_csv(
            tmp_path / "a.csv", [{"タイトル": "本1", "説明文": "説明1", "著者": "著者1"}]
        )
        assert "著者" in load_reading_log(path).columns


class TestL2Normalize:
    def test_makes_each_row_unit_length(self):
        """各行のノルムが1になる。"""
        result = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_handles_zero_vector(self):
        """ゼロベクトルでゼロ除算にならない。"""
        result = l2_normalize(np.array([[0.0, 0.0]]))
        assert np.isfinite(result).all()

    def test_is_idempotent(self):
        """正規化済みの値は変わらない。"""
        once = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.allclose(l2_normalize(once), once)

    def test_preserves_direction(self):
        """向きは変わらない。"""
        result = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.allclose(result, [[0.6, 0.8]])


class TestSimilarPairs:
    def _df(self, n):
        return pd.DataFrame({"タイトル": [f"本{i}" for i in range(n)]})

    def test_excludes_pairs_below_threshold(self):
        """しきい値以下のペアは出ない。"""
        # 直交する3本のベクトル。類似度はすべて0
        pairs = similar_pairs(self._df(3), np.eye(3), threshold=0.5)
        assert pairs.empty
        assert list(pairs.columns) == ["本1", "本2", "類似度"]

    def test_sorts_by_similarity_descending(self):
        """類似度の降順に並ぶ。"""
        embeddings = np.array([[1.0, 0.0], [0.99, 0.14], [0.7, 0.71]])
        pairs = similar_pairs(self._df(3), embeddings, threshold=0.5)
        assert pairs["類似度"].is_monotonic_decreasing

    def test_returns_each_pair_once(self):
        """同じペアは一度しか出ない。"""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        pairs = similar_pairs(self._df(2), embeddings, threshold=0.5)
        assert len(pairs) == 1


class TestCacheMismatch:
    MODEL = "model-a"

    def _df(self):
        return pd.DataFrame({"説明文": ["説明1", "説明2"]})

    def _meta(self, **overrides):
        df = self._df()
        meta = {
            "model": self.MODEL,
            "dim": 2,
            "count": 2,
            "digest": descriptions_digest(df["説明文"]),
        }
        meta.update(overrides)
        return meta

    def test_returns_none_when_everything_matches(self):
        """すべて一致すれば None。"""
        assert cache_mismatch(self._meta(), self._df(), self.MODEL, np.zeros((2, 2))) is None

    def test_detects_count_change(self):
        """件数が違う。"""
        reason = cache_mismatch(self._meta(), self._df(), self.MODEL, np.zeros((3, 2)))
        assert "件数" in reason

    def test_detects_missing_meta(self):
        """メタ情報が無い。"""
        reason = cache_mismatch(None, self._df(), self.MODEL, np.zeros((2, 2)))
        assert "メタ情報" in reason

    def test_detects_model_change(self):
        """モデルが違う。"""
        reason = cache_mismatch(
            self._meta(model="model-b"), self._df(), self.MODEL, np.zeros((2, 2))
        )
        assert "モデル" in reason

    def test_detects_description_change(self):
        """説明文が変わっている。"""
        reason = cache_mismatch(self._meta(digest="x"), self._df(), self.MODEL, np.zeros((2, 2)))
        assert "説明文" in reason


class TestDescriptionsDigest:
    def test_is_stable_for_same_content(self):
        """同じ内容なら同じ値。"""
        assert descriptions_digest(["a", "b"]) == descriptions_digest(["a", "b"])

    def test_changes_with_one_character(self):
        """1文字でも変われば別の値。"""
        assert descriptions_digest(["a", "b"]) != descriptions_digest(["a", "c"])

    def test_distinguishes_boundary_shift(self):
        """区切りをまたいだ連結を区別する。"""
        assert descriptions_digest(["ab", "c"]) != descriptions_digest(["a", "bc"])


class TestLoadOrBuildEmbeddings:
    """埋め込みの生成だけ差し替えて、キャッシュの判定を確かめる。"""

    MODEL = "model-a"

    @pytest.fixture
    def df(self):
        return pd.DataFrame({"説明文": ["説明1", "説明2"]})

    @pytest.fixture
    def fake_build(self, monkeypatch):
        calls = []

        def build(descriptions, model_name):
            calls.append(model_name)
            return np.ones((len(list(descriptions)), 2), dtype=np.float32)

        monkeypatch.setattr("src.embed.build_embeddings", build)
        return calls

    def test_builds_and_writes_meta_on_first_run(self, tmp_path, df, fake_build):
        """初回は生成してメタ情報を書く。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)

        assert fake_build == [self.MODEL]
        meta = json.loads(meta_path(cache).read_text(encoding="utf-8"))
        assert meta == {
            "model": self.MODEL,
            "dim": 2,
            "count": 2,
            "digest": descriptions_digest(df["説明文"]),
        }

    def test_reuses_cache_on_second_run(self, tmp_path, df, fake_build):
        """2回目は生成しない。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, self.MODEL, cache)
        assert len(fake_build) == 1

    def test_rebuilds_when_model_changes(self, tmp_path, df, fake_build):
        """モデルを変えると作り直す。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, "model-b", cache)
        assert fake_build == [self.MODEL, "model-b"]

    def test_rebuilds_when_descriptions_change(self, tmp_path, df, fake_build):
        """説明文を変えると作り直す。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        edited = pd.DataFrame({"説明文": ["説明1", "別の説明"]})
        load_or_build_embeddings(edited, self.MODEL, cache)
        assert len(fake_build) == 2

    def test_rebuilds_when_meta_missing(self, tmp_path, df, fake_build):
        """メタ情報が無ければ作り直す。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        meta_path(cache).unlink()
        load_or_build_embeddings(df, self.MODEL, cache)
        assert len(fake_build) == 2

    def test_rebuilds_when_forced(self, tmp_path, df, fake_build):
        """force なら作り直す。"""
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, self.MODEL, cache, force=True)
        assert len(fake_build) == 2

    def test_returns_normalized_embeddings(self, tmp_path, df, fake_build):
        """戻り値はL2正規化されている。"""
        result = load_or_build_embeddings(df, self.MODEL, tmp_path / "embeddings.npy")
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


class TestCountDistinctDescriptions:
    def _df(self, descriptions):
        return pd.DataFrame({"説明文": descriptions})

    def test_counts_distinct_texts(self):
        """内容の異なる説明文の件数を返す。"""
        assert count_distinct_descriptions(self._df(["猫", "犬", "猫"])) == 2

    def test_ignores_empty_texts(self):
        """空の説明文は数えない。"""
        assert count_distinct_descriptions(self._df(["猫", "", "  "])) == 1

    def test_returns_zero_when_all_empty(self):
        """すべて空なら0。"""
        assert count_distinct_descriptions(self._df(["", "", ""])) == 0

    def test_ignores_surrounding_spaces(self):
        """前後の空白の違いは同じ説明文として数える。"""
        assert count_distinct_descriptions(self._df(["猫", " 猫 ", "猫"])) == 1
