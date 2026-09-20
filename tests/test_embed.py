"""src/embed.py のテスト。埋め込みモデルは読み込まない。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.embed import (
    cache_mismatch,
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
    def test_必須列がそろっていれば読める(self, tmp_path):
        path = write_csv(tmp_path / "a.csv", [{"タイトル": "本1", "説明文": "説明1"}])
        df = load_reading_log(path)
        assert list(df["タイトル"]) == ["本1"]

    @pytest.mark.parametrize("missing", ["タイトル", "説明文"])
    def test_必須列が欠けるとValueError(self, tmp_path, missing):
        row = {"タイトル": "本1", "説明文": "説明1"}
        del row[missing]
        path = write_csv(tmp_path / "a.csv", [row])
        with pytest.raises(ValueError, match=missing):
            load_reading_log(path)

    def test_説明文の欠損は空文字になる(self, tmp_path):
        path = write_csv(
            tmp_path / "a.csv",
            [{"タイトル": "本1", "説明文": None}, {"タイトル": "本2", "説明文": "説明2"}],
        )
        df = load_reading_log(path)
        assert df["説明文"].tolist() == ["", "説明2"]

    def test_余分な列はそのまま残る(self, tmp_path):
        path = write_csv(
            tmp_path / "a.csv", [{"タイトル": "本1", "説明文": "説明1", "著者": "著者1"}]
        )
        assert "著者" in load_reading_log(path).columns


class TestL2Normalize:
    def test_各行のノルムが1になる(self):
        result = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_ゼロベクトルでゼロ除算にならない(self):
        result = l2_normalize(np.array([[0.0, 0.0]]))
        assert np.isfinite(result).all()

    def test_正規化済みの値は変わらない(self):
        once = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.allclose(l2_normalize(once), once)

    def test_向きは変わらない(self):
        result = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.allclose(result, [[0.6, 0.8]])


class TestSimilarPairs:
    def _df(self, n):
        return pd.DataFrame({"タイトル": [f"本{i}" for i in range(n)]})

    def test_しきい値以下のペアは出ない(self):
        # 直交する3本のベクトル。類似度はすべて0
        pairs = similar_pairs(self._df(3), np.eye(3), threshold=0.5)
        assert pairs.empty
        assert list(pairs.columns) == ["本1", "本2", "類似度"]

    def test_類似度の降順に並ぶ(self):
        embeddings = np.array([[1.0, 0.0], [0.99, 0.14], [0.7, 0.71]])
        pairs = similar_pairs(self._df(3), embeddings, threshold=0.5)
        assert pairs["類似度"].is_monotonic_decreasing

    def test_同じペアは一度しか出ない(self):
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

    def test_すべて一致すればNone(self):
        assert cache_mismatch(self._meta(), self._df(), self.MODEL, np.zeros((2, 2))) is None

    def test_件数が違う(self):
        reason = cache_mismatch(self._meta(), self._df(), self.MODEL, np.zeros((3, 2)))
        assert "件数" in reason

    def test_メタ情報が無い(self):
        reason = cache_mismatch(None, self._df(), self.MODEL, np.zeros((2, 2)))
        assert "メタ情報" in reason

    def test_モデルが違う(self):
        reason = cache_mismatch(
            self._meta(model="model-b"), self._df(), self.MODEL, np.zeros((2, 2))
        )
        assert "モデル" in reason

    def test_説明文が変わっている(self):
        reason = cache_mismatch(self._meta(digest="x"), self._df(), self.MODEL, np.zeros((2, 2)))
        assert "説明文" in reason


class TestDescriptionsDigest:
    def test_同じ内容なら同じ値(self):
        assert descriptions_digest(["a", "b"]) == descriptions_digest(["a", "b"])

    def test_1文字でも変われば別の値(self):
        assert descriptions_digest(["a", "b"]) != descriptions_digest(["a", "c"])

    def test_区切りをまたいだ連結を区別する(self):
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

    def test_初回は生成してメタ情報を書く(self, tmp_path, df, fake_build):
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

    def test_2回目は生成しない(self, tmp_path, df, fake_build):
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, self.MODEL, cache)
        assert len(fake_build) == 1

    def test_モデルを変えると作り直す(self, tmp_path, df, fake_build):
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, "model-b", cache)
        assert fake_build == [self.MODEL, "model-b"]

    def test_説明文を変えると作り直す(self, tmp_path, df, fake_build):
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        edited = pd.DataFrame({"説明文": ["説明1", "別の説明"]})
        load_or_build_embeddings(edited, self.MODEL, cache)
        assert len(fake_build) == 2

    def test_メタ情報が無ければ作り直す(self, tmp_path, df, fake_build):
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        meta_path(cache).unlink()
        load_or_build_embeddings(df, self.MODEL, cache)
        assert len(fake_build) == 2

    def test_forceなら作り直す(self, tmp_path, df, fake_build):
        cache = tmp_path / "embeddings.npy"
        load_or_build_embeddings(df, self.MODEL, cache)
        load_or_build_embeddings(df, self.MODEL, cache, force=True)
        assert len(fake_build) == 2

    def test_戻り値はL2正規化されている(self, tmp_path, df, fake_build):
        result = load_or_build_embeddings(df, self.MODEL, tmp_path / "embeddings.npy")
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
