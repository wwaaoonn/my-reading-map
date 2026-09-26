"""reading_map/pipeline.py のテスト。埋め込みモデルと生成AIのAPIは呼ばない。"""

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from reading_map.embed import MIN_BOOKS
from reading_map.pipeline import (
    Book,
    Cluster,
    KScore,
    MapOptions,
    MapResult,
    SimilarPair,
    build_reading_map,
    check_reading_log,
    choose_k,
)

GENERATE_CLUSTER_NAMES = "reading_map.name_clusters.generate_cluster_names"

# 3つのグループ×各4冊。グループごとに話題の違う説明文にする
GROUPS = [
    [
        ("猫の一年", "猫が庭で昼寝をする季節の物語。"),
        ("犬と散歩", "犬と飼い主が公園を散歩する日々の記録。"),
        ("動物園の朝", "動物園の飼育員が動物たちの世話をする一日。"),
        ("小鳥の歌", "小鳥が森の中で仲間を探す冒険の話。"),
    ],
    [
        ("星の地図", "宇宙の星座と惑星の位置を解説する天文学の入門書。"),
        ("月への旅", "宇宙飛行士が月面を探査する記録。"),
        ("銀河の果て", "銀河の誕生と宇宙の歴史をたどる科学読み物。"),
        ("火星の暮らし", "火星の基地で暮らす研究者の日常を描いた小説。"),
    ],
    [
        ("家庭の料理", "野菜と肉を使った家庭料理のレシピ集。"),
        ("パンの時間", "小麦粉と酵母でパンを焼く手順の解説。"),
        ("和食の基本", "出汁と味噌を使った和食の調理法。"),
        ("お菓子の本", "砂糖と卵で作る焼き菓子のレシピ。"),
    ],
]
BOOKS_PER_GROUP = len(GROUPS[0])
BOOK_COUNT = len(GROUPS) * BOOKS_PER_GROUP

# 下の埋め込みでは、類似度は同じグループで 0.9 以上、違うグループで 0.31 以下になる
LOW_THRESHOLD = 0.5
# 同じグループのペアの一部だけが超える値
HIGH_THRESHOLD = 0.94


def group_of(position):
    return position // BOOKS_PER_GROUP


def fake_names(top_words):
    return {cluster_id: f"生成AIの名前{cluster_id}" for cluster_id in top_words}


def assert_same_result(first, second):
    """silhouette が nan でも比べられるように、nan どうしは等しいとみなす。"""
    if math.isnan(first.silhouette) and math.isnan(second.silhouette):
        first = dataclasses.replace(first, silhouette=0.0)
        second = dataclasses.replace(second, silhouette=0.0)
    assert first == second


@pytest.fixture
def df():
    """3つのグループ×各4冊の読書記録。"""
    rows = [book for group in GROUPS for book in group]
    return pd.DataFrame(
        {
            "タイトル": [title for title, _ in rows],
            "説明文": [description for _, description in rows],
        }
    )


@pytest.fixture
def embeddings():
    """はっきり分かれた3グループのベクトル。各行の長さは1。"""
    rng = np.random.default_rng(0)
    dim = 8
    centers = np.repeat(np.eye(dim)[: len(GROUPS)], BOOKS_PER_GROUP, axis=0)
    vectors = centers + rng.normal(scale=0.1, size=(BOOK_COUNT, dim))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def ai_calls(monkeypatch):
    """生成AIの命名を、呼ばれたら失敗する関数に差し替える。呼び出しの記録を返す。"""
    calls = []

    def fake(df, representatives, top_words, model=None):
        calls.append({"model": model})
        raise AssertionError("generate_cluster_names が呼ばれた")

    monkeypatch.setattr(GENERATE_CLUSTER_NAMES, fake)
    return calls


@pytest.fixture
def ai_succeeds(monkeypatch):
    """生成AIの命名を、成功する関数に差し替える。返した名前と渡されたモデルを記録する。"""
    record = {"names": None, "models": []}

    def fake(df, representatives, top_words, model=None):
        record["names"] = fake_names(top_words)
        record["models"].append(model)
        return record["names"], None

    monkeypatch.setattr(GENERATE_CLUSTER_NAMES, fake)
    return record


@pytest.fixture
def ai_fails(monkeypatch):
    """生成AIの命名を、失敗して頻出語の名前を返す関数に差し替える。"""
    record = {"names": None, "reason": "APIキーが設定されていません"}

    def fake(df, representatives, top_words, model=None):
        record["names"] = {
            cluster_id: "・".join(words[:3]) or f"クラスタ{cluster_id}"
            for cluster_id, words in top_words.items()
        }
        return record["names"], record["reason"]

    monkeypatch.setattr(GENERATE_CLUSTER_NAMES, fake)
    return record


@pytest.fixture
def result(df, embeddings, ai_calls):
    """k=3、生成AIを使わずに作った結果。"""
    return build_reading_map(df, embeddings, MapOptions(k=3, ai_names=False))


class TestBuildReadingMapBooks:
    def test_returns_map_result(self, result):
        """MapResult を返す。"""
        assert isinstance(result, MapResult)
        assert all(isinstance(book, Book) for book in result.books)

    def test_keeps_input_row_order(self, result):
        """books は入力の行の順で、position は 0 から冊数-1 まで。"""
        assert [book.position for book in result.books] == list(range(BOOK_COUNT))

    def test_titles_match_input(self, df, result):
        """各本の title は df の「タイトル」と一致する。"""
        assert [book.title for book in result.books] == df["タイトル"].tolist()

    def test_cluster_ids_exist_in_clusters(self, result):
        """各本の cluster_id は clusters のどれかにある。"""
        cluster_ids = {cluster.cluster_id for cluster in result.clusters}
        assert {book.cluster_id for book in result.books} <= cluster_ids


class TestBuildReadingMapClusters:
    def test_clusters_are_sorted_by_id(self, result):
        """clusters はクラスタIDの昇順で、IDは重複しない。"""
        ids = [cluster.cluster_id for cluster in result.clusters]
        assert ids == sorted(set(ids))
        assert all(isinstance(cluster, Cluster) for cluster in result.clusters)

    def test_sizes_add_up_to_book_count(self, result):
        """size の合計は冊数。"""
        assert sum(cluster.size for cluster in result.clusters) == BOOK_COUNT

    def test_size_matches_books_in_cluster(self, result):
        """size はそのクラスタに入った本の数。"""
        for cluster in result.clusters:
            members = [b for b in result.books if b.cluster_id == cluster.cluster_id]
            assert cluster.size == len(members)

    def test_representatives_belong_to_cluster(self, result):
        """representatives はそのクラスタの本の位置だけを含む。"""
        for cluster in result.clusters:
            members = {b.position for b in result.books if b.cluster_id == cluster.cluster_id}
            assert set(cluster.representatives) <= members

    @pytest.mark.parametrize("per_cluster", [1, 2, 8])
    def test_limits_representatives(self, df, embeddings, ai_calls, per_cluster):
        """representatives の件数は representatives_per_cluster 以下。"""
        options = MapOptions(k=3, ai_names=False, representatives_per_cluster=per_cluster)
        result = build_reading_map(df, embeddings, options)
        for cluster in result.clusters:
            assert len(cluster.representatives) <= per_cluster


class TestBuildReadingMapWithRequestedK:
    def test_uses_requested_k(self, result):
        """指定した k を使う。"""
        assert result.k == 3
        assert len(result.clusters) == 3

    def test_has_no_k_scores(self, result):
        """k を指定したときは k_scores が None。"""
        assert result.k_scores is None

    def test_keeps_separated_groups_together(self, result):
        """はっきり分かれた3グループは、同じグループの本が同じクラスタになる。"""
        by_group = {}
        for book in result.books:
            by_group.setdefault(group_of(book.position), set()).add(book.cluster_id)
        assert all(len(ids) == 1 for ids in by_group.values())
        assert len({ids.pop() for ids in by_group.values()}) == len(GROUPS)

    @pytest.mark.parametrize("k", [0, 1, BOOK_COUNT, BOOK_COUNT + 1])
    def test_rejects_k_outside_range(self, df, embeddings, ai_calls, k):
        """2未満、または冊数-1を超える k は ValueError。"""
        with pytest.raises(ValueError):
            build_reading_map(df, embeddings, MapOptions(k=k, ai_names=False))


class TestBuildReadingMapWithAutomaticK:
    @pytest.fixture
    def result(self, df, embeddings, ai_calls):
        return build_reading_map(df, embeddings, MapOptions(ai_names=False))

    def test_k_is_within_range(self, result):
        """自動で決めた k は2以上、冊数-1以下。"""
        assert 2 <= result.k <= BOOK_COUNT - 1

    def test_returns_k_scores_sorted_by_k(self, result):
        """k_scores は KScore のリストで、k の昇順。"""
        assert isinstance(result.k_scores, list)
        assert result.k_scores
        assert all(isinstance(score, KScore) for score in result.k_scores)
        ks = [score.k for score in result.k_scores]
        assert ks == sorted(ks)


class TestBuildReadingMapSimilarPairs:
    def test_sorted_by_similarity_descending(self, result):
        """similar_pairs は類似度の高い順。"""
        similarities = [pair.similarity for pair in result.similar_pairs]
        assert similarities == sorted(similarities, reverse=True)
        assert all(isinstance(pair, SimilarPair) for pair in result.similar_pairs)

    def test_first_book_comes_first(self, result):
        """book1 は book2 より前の行。"""
        assert all(pair.book1 < pair.book2 for pair in result.similar_pairs)

    @pytest.mark.parametrize("threshold", [LOW_THRESHOLD, HIGH_THRESHOLD])
    def test_similarity_exceeds_threshold(self, df, embeddings, ai_calls, threshold):
        """similarity は pair_threshold を超える。"""
        options = MapOptions(k=3, ai_names=False, pair_threshold=threshold)
        result = build_reading_map(df, embeddings, options)
        assert result.similar_pairs
        assert all(pair.similarity > threshold for pair in result.similar_pairs)

    def test_higher_threshold_gives_fewer_pairs(self, df, embeddings, ai_calls):
        """pair_threshold を上げるとペアが減る（または同じ）。"""
        low = build_reading_map(
            df, embeddings, MapOptions(k=3, ai_names=False, pair_threshold=LOW_THRESHOLD)
        )
        high = build_reading_map(
            df, embeddings, MapOptions(k=3, ai_names=False, pair_threshold=HIGH_THRESHOLD)
        )
        low_pairs = {(pair.book1, pair.book2) for pair in low.similar_pairs}
        high_pairs = {(pair.book1, pair.book2) for pair in high.similar_pairs}
        assert len(high_pairs) <= len(low_pairs)
        assert high_pairs <= low_pairs


class TestBuildReadingMapNaming:
    def test_does_not_call_ai_when_disabled(self, df, embeddings, ai_calls):
        """ai_names=False なら生成AIを呼ばず、頻出語から命名する。"""
        result = build_reading_map(df, embeddings, MapOptions(k=3, ai_names=False))
        assert ai_calls == []
        assert result.naming == "top_words"
        assert result.naming_fallback_reason is None

    def test_names_every_cluster_when_disabled(self, result):
        """ai_names=False でも、どのクラスタにも空でない名前が付く。"""
        assert all(cluster.name for cluster in result.clusters)

    def test_uses_ai_names_on_success(self, df, embeddings, ai_succeeds):
        """生成AIが成功したら、その名前を使う。"""
        result = build_reading_map(df, embeddings, MapOptions(k=3))
        assert result.naming == "ai"
        assert result.naming_fallback_reason is None
        assert {c.cluster_id: c.name for c in result.clusters} == ai_succeeds["names"]

    def test_passes_model_to_ai(self, df, embeddings, ai_succeeds):
        """options.model を生成AIの命名に渡す。"""
        build_reading_map(df, embeddings, MapOptions(k=3, model="claude-sonnet-5"))
        assert ai_succeeds["models"] == ["claude-sonnet-5"]

    def test_falls_back_to_top_words_on_failure(self, df, embeddings, ai_fails):
        """生成AIが失敗したら、頻出語の名前と失敗の理由を返す。"""
        result = build_reading_map(df, embeddings, MapOptions(k=3))
        assert result.naming == "top_words"
        assert result.naming_fallback_reason == ai_fails["reason"]
        assert {c.cluster_id: c.name for c in result.clusters} == ai_fails["names"]


class TestBuildReadingMapInput:
    def test_does_not_modify_input(self, df, embeddings, ai_calls):
        """渡された df は変えない。"""
        original = df.copy()
        build_reading_map(df, embeddings, MapOptions(k=3, ai_names=False))
        pd.testing.assert_frame_equal(df, original)

    def test_handles_missing_description(self, df, embeddings, ai_calls):
        """説明文に NaN があっても動き、df の NaN はそのまま残る。"""
        df.loc[1, "説明文"] = np.nan
        original = df.copy()
        result = build_reading_map(df, embeddings, MapOptions(k=3, ai_names=False))
        assert len(result.books) == BOOK_COUNT
        pd.testing.assert_frame_equal(df, original)
        assert pd.isna(df.loc[1, "説明文"])

    def test_does_not_write_files(self, df, embeddings, ai_calls, tmp_path, monkeypatch):
        """ファイルを書き出さない。"""
        monkeypatch.chdir(tmp_path)
        build_reading_map(df, embeddings, MapOptions(ai_names=False))
        assert list(tmp_path.iterdir()) == []

    def test_rejects_too_few_books(self, df, embeddings, ai_calls):
        """冊数が MIN_BOOKS 未満なら ValueError。"""
        count = MIN_BOOKS - 1
        with pytest.raises(ValueError):
            build_reading_map(df.head(count), embeddings[:count], MapOptions(ai_names=False))

    @pytest.mark.parametrize("description", ["同じ説明文の本。", ""])
    def test_rejects_too_few_distinct_descriptions(self, df, embeddings, ai_calls, description):
        """内容の異なる説明文が MIN_DISTINCT_DESCRIPTIONS 未満なら ValueError。"""
        df["説明文"] = description
        with pytest.raises(ValueError):
            build_reading_map(df, embeddings, MapOptions(ai_names=False))

    @pytest.mark.parametrize("column", ["タイトル", "説明文"])
    def test_rejects_missing_column(self, df, embeddings, ai_calls, column):
        """必須列が無ければ ValueError。"""
        with pytest.raises(ValueError):
            build_reading_map(df.drop(columns=column), embeddings, MapOptions(ai_names=False))

    def test_rejects_mismatched_embeddings(self, df, embeddings, ai_calls):
        """embeddings の行数が df と違えば ValueError。"""
        with pytest.raises(ValueError):
            build_reading_map(df, embeddings[:-1], MapOptions(ai_names=False))


class TestBuildReadingMapOutput:
    def test_is_deterministic(self, df, embeddings, ai_calls):
        """同じ入力なら同じ結果。"""
        options = MapOptions(ai_names=False)
        first = build_reading_map(df, embeddings, options)
        second = build_reading_map(df, embeddings, options)
        assert_same_result(first, second)

    def test_silhouette_is_float(self, result):
        """silhouette は float。"""
        assert type(result.silhouette) is float

    def test_uses_python_types(self, df, embeddings, ai_calls):
        """フィールドの値は numpy の型ではなく Python の int / float / str。"""
        result = build_reading_map(df, embeddings, MapOptions(ai_names=False))
        assert type(result.k) is int
        assert type(result.silhouette) is float
        assert type(result.naming) is str
        for book in result.books:
            assert type(book.position) is int
            assert type(book.title) is str
            assert type(book.cluster_id) is int
            assert type(book.x) is float
            assert type(book.y) is float
        for cluster in result.clusters:
            assert type(cluster.cluster_id) is int
            assert type(cluster.name) is str
            assert type(cluster.size) is int
            assert all(type(word) is str for word in cluster.top_words)
            assert all(type(position) is int for position in cluster.representatives)
        for pair in result.similar_pairs:
            assert type(pair.book1) is int
            assert type(pair.book2) is int
            assert type(pair.similarity) is float
        for score in result.k_scores:
            assert type(score.k) is int
            assert type(score.inertia) is float
            assert type(score.silhouette) is float
            assert type(score.min_cluster_size) is int


class TestMapResultToDict:
    @pytest.fixture
    def result(self, df, embeddings, ai_calls):
        """k を自動で決め、生成AIを使わずに作った結果。"""
        return build_reading_map(df, embeddings, MapOptions(ai_names=False))

    @pytest.fixture
    def nan_result(self):
        """silhouette が nan の結果。"""
        return MapResult(
            books=[
                Book(position=0, title="猫の本", cluster_id=0, x=0.0, y=0.0),
                Book(position=1, title="星の本", cluster_id=1, x=1.0, y=1.0),
            ],
            clusters=[
                Cluster(0, "動物", 1, ["猫"], [0]),
                Cluster(1, "天体", 1, ["星"], [1]),
            ],
            similar_pairs=[],
            k=2,
            silhouette=float("nan"),
            naming="top_words",
            k_scores=[KScore(k=2, inertia=0.0, silhouette=float("nan"), min_cluster_size=1)],
        )

    def test_passes_json_dumps(self, result):
        """allow_nan=False の json.dumps が通る。"""
        json.dumps(result.to_dict(), allow_nan=False)

    def test_has_every_field(self, result):
        """全フィールドのキーがあり、books などは dict のリスト。"""
        data = result.to_dict()
        assert set(data) == {field.name for field in dataclasses.fields(MapResult)}
        for key in ["books", "clusters", "similar_pairs", "k_scores"]:
            assert isinstance(data[key], list)
            assert all(isinstance(item, dict) for item in data[key])
        assert set(data["books"][0]) == {"position", "title", "cluster_id", "x", "y"}

    def test_converts_nan_to_none(self, nan_result):
        """nan の silhouette は None にする（k_scores の中も）。"""
        data = nan_result.to_dict()
        assert data["silhouette"] is None
        assert data["k_scores"][0]["silhouette"] is None
        json.dumps(data, allow_nan=False)

    def test_does_not_modify_result(self, nan_result):
        """元の MapResult の nan は変えない。"""
        nan_result.to_dict()
        assert math.isnan(nan_result.silhouette)
        assert math.isnan(nan_result.k_scores[0].silhouette)

    def test_keeps_k_scores_none(self, df, embeddings, ai_calls):
        """k_scores が None なら None。"""
        result = build_reading_map(df, embeddings, MapOptions(k=3, ai_names=False))
        assert result.to_dict()["k_scores"] is None

    def test_round_trips_through_json(self, result):
        """JSONを読み戻した値が元の MapResult の値と一致する。"""
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert data["k"] == result.k
        assert data["naming"] == result.naming
        assert data["silhouette"] == result.silhouette
        assert [c["name"] for c in data["clusters"]] == [c.name for c in result.clusters]
        assert [c["top_words"] for c in data["clusters"]] == [
            c.top_words for c in result.clusters
        ]
        assert [b["title"] for b in data["books"]] == [b.title for b in result.books]
        assert [(b["x"], b["y"]) for b in data["books"]] == [(b.x, b.y) for b in result.books]
        assert [(p["book1"], p["book2"], p["similarity"]) for p in data["similar_pairs"]] == [
            (p.book1, p.book2, p.similarity) for p in result.similar_pairs
        ]
        assert [s["k"] for s in data["k_scores"]] == [s.k for s in result.k_scores]


class TestCheckReadingLog:
    def test_accepts_enough_books(self, df):
        """冊数と説明文が足りていれば何もしない。"""
        check_reading_log(df)

    def test_rejects_too_few_books(self, df):
        """冊数が MIN_BOOKS 未満なら止める。"""
        with pytest.raises(ValueError, match="冊では地図を作れません"):
            check_reading_log(df.head(MIN_BOOKS - 1))

    def test_rejects_same_descriptions(self, df):
        """説明文がすべて同じなら止める。"""
        same = df.assign(説明文="同じ説明文")
        with pytest.raises(ValueError, match="内容の異なる説明文"):
            check_reading_log(same)


@pytest.fixture
def random_embeddings():
    """10冊ぶんのベクトル。"""
    rng = np.random.default_rng(0)
    return rng.random((10, 4))


class TestChooseKWithRequestedK:
    def test_uses_requested_k(self, random_embeddings):
        """指定された k をそのまま使い、指標は返さない。"""
        assert choose_k(random_embeddings, 3) == (3, None)

    def test_accepts_k_one_below_book_count(self, random_embeddings):
        """上限（冊数-1）の k は受け付ける。"""
        assert choose_k(random_embeddings, 9) == (9, None)

    @pytest.mark.parametrize("requested_k", [0, 1, 10, 11, 50])
    def test_rejects_k_outside_range(self, random_embeddings, requested_k):
        """2未満、または冊数-1を超える k は ValueError（冊数と同じ k も含む）。"""
        with pytest.raises(ValueError):
            choose_k(random_embeddings, requested_k)

    def test_does_not_write_files(self, random_embeddings, tmp_path, monkeypatch):
        """ファイルを書き出さない。"""
        monkeypatch.chdir(tmp_path)
        choose_k(random_embeddings, 3)
        assert list(tmp_path.iterdir()) == []


class TestChooseKAutomatically:
    def test_returns_k_within_range(self, random_embeddings):
        """自動で決めた k は2以上、冊数-1以下。"""
        k, _ = choose_k(random_embeddings)
        assert 2 <= k <= len(random_embeddings) - 1

    def test_returns_scores(self, random_embeddings):
        """各kの指標を DataFrame で返す。"""
        _, scores = choose_k(random_embeddings)
        assert isinstance(scores, pd.DataFrame)
        assert {"k", "inertia", "silhouette", "min_cluster_size"} <= set(scores.columns)

    def test_does_not_write_files(self, random_embeddings, tmp_path, monkeypatch):
        """ファイルを書き出さない。"""
        monkeypatch.chdir(tmp_path)
        choose_k(random_embeddings)
        assert list(tmp_path.iterdir()) == []
