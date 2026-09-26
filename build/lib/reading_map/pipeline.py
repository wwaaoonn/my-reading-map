"""読書記録と埋め込みから読書マップの計算結果を作る。ファイルは書き出さない。"""

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import cluster, label, name_clusters, visualize
from .embed import (
    MIN_BOOKS,
    MIN_DISTINCT_DESCRIPTIONS,
    count_distinct_descriptions,
    prepare_reading_log,
    similar_pair_positions,
)
from .name_clusters import MODEL

logger = logging.getLogger(__name__)


@dataclass
class MapOptions:
    """`build_reading_map` の設定。すべて省略できる。"""

    # クラスタ数。None ならデータから決める
    k: int | None = None
    # t-SNEのperplexity。None なら冊数から決める
    perplexity: int | None = None
    # False なら生成AIを呼ばず、頻出語からクラスタ名を作る
    ai_names: bool = True
    # 命名に使うモデル
    model: str = MODEL
    # 類似ペアに含める類似度の下限（この値を超えるペアを含める）
    pair_threshold: float = 0.5
    # クラスタごとに返す代表本の数
    representatives_per_cluster: int = 8
    kmeans_random_state: int = 0
    tsne_random_state: int = 42


@dataclass
class Book:
    """1冊の本の配置。position は入力した df の行の位置（0始まり）。"""

    position: int
    title: str
    cluster_id: int
    x: float
    y: float


@dataclass
class Cluster:
    """1つのクラスタ。representatives は重心に近い順の行の位置。"""

    cluster_id: int
    name: str
    size: int
    top_words: list[str]
    representatives: list[int]


@dataclass
class SimilarPair:
    """説明文の類似度がしきい値を超える2冊。book1・book2 は行の位置。"""

    book1: int
    book2: int
    similarity: float


@dataclass
class KScore:
    """クラスタ数の検討に使った、1つのkの指標。"""

    k: int
    inertia: float
    silhouette: float
    min_cluster_size: int


@dataclass
class MapResult:
    """読書マップの計算結果。"""

    # 入力した df の行の順
    books: list[Book]
    # クラスタIDの順
    clusters: list[Cluster]
    # 類似度の高い順
    similar_pairs: list[SimilarPair]
    k: int
    # 計算できないときは nan
    silhouette: float
    # "ai": 生成AIが命名した / "top_words": 頻出語から命名した
    naming: Literal["ai", "top_words"]
    # 生成AIでの命名に失敗して頻出語に切り替えた理由。切り替えていなければ None
    naming_fallback_reason: str | None = None
    # k を自動で決めたときの各kの指標（kの昇順）。k を指定したときは None
    k_scores: list[KScore] | None = field(default=None)

    def to_dict(self) -> dict:
        """JSONに書ける dict にする。

        キーはフィールド名。books などは dict のリスト。
        silhouette と k_scores の各 silhouette の nan は None にする。
        """
        data = asdict(self)
        data["silhouette"] = nan_to_none(data["silhouette"])
        if data["k_scores"] is not None:
            for score in data["k_scores"]:
                score["silhouette"] = nan_to_none(score["silhouette"])
        return data


def nan_to_none(value):
    """nan なら None、それ以外はそのまま返す。"""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def check_reading_log(df):
    """地図を作れる冊数と説明文があるか確かめる。足りなければ ValueError。

    df は `prepare_reading_log` を通したもの。
    """
    n_books = len(df)
    if n_books < MIN_BOOKS:
        raise ValueError(f"{n_books}冊では地図を作れません。{MIN_BOOKS}冊以上必要です。")

    distinct = count_distinct_descriptions(df)
    if distinct < MIN_DISTINCT_DESCRIPTIONS:
        raise ValueError(
            f"内容の異なる説明文が{distinct}件しかありません。"
            f"{MIN_DISTINCT_DESCRIPTIONS}件以上必要です"
            "（説明文がすべて空、またはすべて同じでは地図を作れません）。"
        )


def choose_k(embeddings, requested_k=None, random_state=0):
    """使うクラスタ数を決める。

    requested_k があればそれを使う。2未満、または冊数-1を超えると ValueError。
    requested_k が None ならデータから決める。

    戻り値: (k, 各kの指標の DataFrame。requested_k を使ったときは None)
    """
    # 上限は冊数-1。冊数と同じkにすると全クラスタが1冊になり、
    # シルエット係数（2以上・冊数-1以下でのみ定義される）が計算できない
    n_books = len(embeddings)
    max_k = n_books - 1
    if requested_k is not None:
        if not 2 <= requested_k <= max_k:
            raise ValueError(
                f"k は2以上、冊数-1以下で指定してください"
                f"（指定された k={requested_k} / 冊数={n_books} / 上限={max_k}）"
            )
        logger.info(f"クラスタ数: 指定された k={requested_k} を使います")
        return requested_k, None

    logger.info("クラスタ数を決めています（kを変えてクラスタリングを試行）")
    scores = cluster.evaluate_k(embeddings, random_state=random_state)
    k, reasons = cluster.suggest_k(scores, n_books)
    for line in reasons:
        logger.info(line)
    return k, scores


def build_reading_map(
    df: pd.DataFrame, embeddings: np.ndarray, options: MapOptions | None = None
) -> MapResult:
    """読書記録と埋め込みから、読書マップの計算結果を作る。

    df には「タイトル」「説明文」の列が必要。embeddings は df と行の順番で対応する。
    df は変更しない。

    冊数が MIN_BOOKS 未満、内容の異なる説明文が MIN_DISTINCT_DESCRIPTIONS 件未満、
    必須列が無い、df と embeddings の行数が違う、options.k が範囲外のときは ValueError。
    """
    if options is None:
        options = MapOptions()

    work = prepare_reading_log(df)
    check_reading_log(work)
    n_books = len(work)

    embeddings = np.asarray(embeddings)
    if len(embeddings) != n_books:
        raise ValueError(
            f"埋め込みの行数が冊数と一致しません（埋め込み {len(embeddings)}行 / 冊数 {n_books}）"
        )

    k, scores = choose_k(embeddings, options.k, options.kmeans_random_state)

    logger.info(f"k={k} でクラスタリング")
    work["クラスタID"] = cluster.fit_kmeans(embeddings, k, options.kmeans_random_state)
    labels = work["クラスタID"].to_numpy()
    silhouette = cluster.silhouette_for_labels(embeddings, labels, options.kmeans_random_state)
    if math.isnan(silhouette):
        logger.info(
            "シルエット係数（コサイン距離）: 計算できません"
            "（クラスタが1つ、または全クラスタが1冊）"
        )
    else:
        logger.info(f"シルエット係数（コサイン距離）: {silhouette:.3f}")
        if silhouette < cluster.LOW_SILHOUETTE:
            logger.warning(
                f"※ {cluster.LOW_SILHOUETTE}未満。このkでは、クラスタは明確に分離していない"
            )

    top_words = label.top_words_per_cluster(work)
    representatives = visualize.representative_books(
        work, embeddings, top_n=options.representatives_per_cluster
    )

    logger.info("クラスタ名を生成")
    naming_fallback_reason = None
    if options.ai_names:
        cluster_names, naming_fallback_reason = name_clusters.generate_cluster_names(
            work, representatives, top_words, model=options.model
        )
        naming = "ai" if naming_fallback_reason is None else "top_words"
    else:
        cluster_names = name_clusters.fallback_names(top_words)
        naming = "top_words"
    logger.info(f"命名: {'生成AI（' + options.model + '）' if naming == 'ai' else '頻出語'}")

    logger.info("t-SNEで2次元化")
    coords = visualize.compute_tsne(embeddings, options.perplexity, options.tsne_random_state)

    pairs = similar_pair_positions(embeddings, options.pair_threshold)

    titles = work["タイトル"].tolist()
    books = [
        Book(
            position=i,
            title=str(titles[i]),
            cluster_id=int(labels[i]),
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
        )
        for i in range(n_books)
    ]
    clusters = [
        Cluster(
            cluster_id=int(cluster_id),
            name=str(cluster_names[cluster_id]),
            size=int((labels == cluster_id).sum()),
            top_words=[str(word) for word in top_words[cluster_id]],
            representatives=[int(pos) for pos in representatives[cluster_id]],
        )
        for cluster_id in sorted(cluster_names)
    ]
    similar_pairs = [
        SimilarPair(book1=int(row.book1), book2=int(row.book2), similarity=float(row.similarity))
        for row in pairs.itertuples(index=False)
    ]
    k_scores = None
    if scores is not None:
        k_scores = [
            KScore(
                k=int(row.k),
                inertia=float(row.inertia),
                silhouette=float(row.silhouette),
                min_cluster_size=int(row.min_cluster_size),
            )
            for row in scores.sort_values("k").itertuples(index=False)
        ]

    return MapResult(
        books=books,
        clusters=clusters,
        similar_pairs=similar_pairs,
        k=int(k),
        silhouette=float(silhouette),
        naming=naming,
        naming_fallback_reason=naming_fallback_reason,
        k_scores=k_scores,
    )
