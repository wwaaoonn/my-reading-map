"""KMeansクラスタリングと、クラスタ数kの決定。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .plot_style import setup_japanese_font

# kを探す上限。冊数が少ないときは自動でさらに小さくなる
DEFAULT_K_MAX = 14
# kの上限を決めるための、1クラスタあたりの平均冊数の下限。
# 各クラスタの冊数は拘束しないので、割り当て次第でこれを下回るクラスタはできる
MIN_BOOKS_PER_CLUSTER = 4
# 各クラスタに最低限必要な冊数。これを下回るクラスタができるkは選ばない。
# `--k` を指定したときは適用しない
MIN_CLUSTER_BOOKS = 2
# シルエット係数（コサイン）がどのkでもこの値を下回るとき、分離が弱い旨を表示する。
# 0.25 は「実質的な構造が見つからない」とされる慣用的な境界
LOW_SILHOUETTE = 0.25


def fit_kmeans(embeddings, k, random_state=0):
    """KMeansでクラスタIDを割り当てる。"""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return kmeans.fit_predict(embeddings)


def evaluate_k(embeddings, k_max=DEFAULT_K_MAX, random_state=0, verbose=True):
    """k ごとに inertia（エルボー法）・シルエット係数・最小クラスタの冊数を計算する。

    シルエット係数はコサイン距離で測る。inertia はKMeansが返すユークリッド基準の値。
    ここでのKMeansは `fit_kmeans` と同じ乱数・同じ設定なので、記録した冊数は
    そのkを採用したときの実際の冊数と一致する。
    """
    k_upper = min(k_max, len(embeddings) - 1)
    rows = []

    for k in range(2, k_upper + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_for_labels(embeddings, labels, random_state)
        min_size = int(np.bincount(labels, minlength=k).min())
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette": score,
                "min_cluster_size": min_size,
            }
        )
        if verbose:
            print(
                f"  k={k:2d} : inertia={kmeans.inertia_:9.2f}"
                f"  silhouette={score:.4f}  最小クラスタ={min_size}冊"
            )


    return pd.DataFrame(rows)


def _unit_scale(values):
    """値を0〜1に収める。すべて同じ値なら0を返す。"""
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def elbow_k(scores):
    """inertia曲線が最も大きくへこむk（エルボー＝ひじ）を返す。

    両端を結んだ直線からの距離が最大になる点を選ぶ。kとinertiaは単位が違うので、
    どちらも0〜1に正規化してから距離を測る。

    隣り合う差分ではなく曲線全体の形で決めるため、局所的な凹凸に振られない。
    へこみが無い（inertiaが直線的に下がる）場合は距離がどこも0になり、
    最小のkを返す。
    """
    ks = scores["k"].to_numpy(dtype=float)
    inertias = scores["inertia"].to_numpy(dtype=float)
    if len(ks) < 3:
        return int(ks[0])

    x = _unit_scale(ks)
    y = _unit_scale(inertias)
    # 両端 (x[0], y[0]) と (x[-1], y[-1]) を結ぶ直線からの距離。
    # 全ての点で共通の分母は、大小の比較には要らないので省く
    distance = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(distance))])


def suggest_k(scores, n_books, min_cluster_books=MIN_CLUSTER_BOOKS):
    """指標からkを1つ選び、判断に使った値を文章で返す。

    エルボー法の値を軸に、2つの上限のうち小さいほうで頭を押さえる。
    上限は「1クラスタ平均 MIN_BOOKS_PER_CLUSTER 冊以上になるk」と
    「探索したkの最大（DEFAULT_K_MAX と冊数で決まる）」の2つ。下限は2。

    そのうえで、最小クラスタが min_cluster_books 冊以上になるまでkを1つずつ下げる。
    下限の2でも満たせない場合は2を採用し、その旨を返す。
    """
    elbow = elbow_k(scores)
    silhouettes = scores["silhouette"]
    has_silhouette = bool(silhouettes.notna().any())
    searched_upper = int(scores["k"].max())
    average_upper = n_books // MIN_BOOKS_PER_CLUSTER
    k_upper = max(2, min(searched_upper, average_upper))
    capped = max(2, min(elbow, k_upper))

    min_sizes = {
        int(k): int(v) for k, v in scores.set_index("k")["min_cluster_size"].items()
    }
    chosen = capped
    while chosen > 2 and min_sizes[chosen] < min_cluster_books:
        chosen -= 1

    if has_silhouette:
        best = int(scores.loc[silhouettes.idxmax(), "k"])
        silhouette_line = f"シルエット係数が最大のk: {best}（値={silhouettes.max():.3f}）"
    else:
        silhouette_line = "シルエット係数: どのkでも計算できない"

    reason = [
        f"エルボー法が示すk: {elbow}",
        silhouette_line,
        f"kの上限: {k_upper}"
        f"（1クラスタ平均{MIN_BOOKS_PER_CLUSTER}冊以上になるk={average_upper} と"
        f" 探索したkの最大={searched_upper} の小さいほう。下限は2）",
    ]
    if min_sizes[chosen] < min_cluster_books:
        reason.append(
            f"※ 最低{min_cluster_books}冊を満たすkが無い"
            f"（下限のk={chosen}でも最小{min_sizes[chosen]}冊）"
        )
    elif chosen != capped:
        skipped = "、".join(
            f"k={k}は最小{min_sizes[k]}冊" for k in range(capped, chosen, -1)
        )
        reason.append(f"最低{min_cluster_books}冊を満たす最大のk: {chosen}（{skipped}）")
    reason.append(f"→ 採用 k={chosen}")
    if has_silhouette and silhouettes.max() < LOW_SILHOUETTE:
        reason.append(
            f"※ シルエット係数がどのkでも{LOW_SILHOUETTE}未満で、"
            "クラスタは明確に分離していない。"
            "図を見て納得できなければ --k で指定すること"
        )
    return chosen, reason


def plot_k_selection(scores, out_path, chosen_k=None, dpi=200):
    """エルボー法とシルエット係数を並べて描き、採用したkを示す。"""
    setup_japanese_font()

    best_k = (
        int(scores.loc[scores["silhouette"].idxmax(), "k"])
        if scores["silhouette"].notna().any()
        else None
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set_title("エルボー法（クラスタ内誤差）")
    axes[0].set_xlabel("クラスタ数 k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="orange")
    if best_k is not None:
        axes[1].axvline(best_k, color="red", linestyle="--", label=f"シルエット最大 k={best_k}")
    axes[1].set_title("シルエット係数（分離の良さ）")
    axes[1].set_xlabel("クラスタ数 k")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(True, alpha=0.3)

    if chosen_k is not None:
        for ax in axes:
            ax.axvline(chosen_k, color="gray", linestyle=":", label=f"採用 k={chosen_k}")
    if best_k is not None or chosen_k is not None:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"k選択グラフを保存しました: {out_path}")


def silhouette_for_labels(embeddings, labels, random_state=0):
    """割り当て済みのクラスタに対するシルエット係数（コサイン距離）。

    `evaluate_k` はkを変えながら計算するが、こちらは採用した1つの割り当てだけを測る。
    計算できるのはクラスタ数が2以上、冊数-1以下のときだけで、それ以外は nan を返す。
    """
    labels = np.asarray(labels)
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    return float(
        silhouette_score(
            embeddings, labels, metric="cosine",
            sample_size=min(500, len(embeddings)), random_state=random_state,
        )
    )
