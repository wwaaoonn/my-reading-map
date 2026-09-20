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
MIN_BOOKS_PER_CLUSTER = 5
# シルエット係数（コサイン）がどのkでもこの値を下回るとき、分離が弱い旨を表示する。
# 0.25 は「実質的な構造が見つからない」とされる慣用的な境界
LOW_SILHOUETTE = 0.25


def fit_kmeans(embeddings, k, random_state=0):
    """KMeansでクラスタIDを割り当てる。"""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return kmeans.fit_predict(embeddings)


def evaluate_k(embeddings, k_max=DEFAULT_K_MAX, random_state=0, verbose=True):
    """k ごとに inertia（エルボー法）とシルエット係数をまとめて計算する。

    シルエット係数はコサイン距離で測る。inertia はKMeansが返すユークリッド基準の値。
    """
    k_upper = min(k_max, len(embeddings) - 1)
    rows = []

    for k in range(2, k_upper + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(
            embeddings, labels, metric="cosine",
            sample_size=min(500, len(embeddings)), random_state=random_state,
        )
        rows.append({"k": k, "inertia": kmeans.inertia_, "silhouette": score})
        if verbose:
            print(f"  k={k:2d} : inertia={kmeans.inertia_:9.2f}  silhouette={score:.4f}")

    return pd.DataFrame(rows)


def elbow_k(scores):
    """inertiaの下がり方が最も鈍くなるk（エルボー＝ひじ）を返す。"""
    inertias = scores["inertia"].to_numpy()
    if len(inertias) < 3:
        return int(scores["k"].iloc[0])
    # 二次差分が最大 = 下がり方の鈍化が最も大きい点。
    # 絶対値を取ると、下がり方が急になる点（エルボーの逆）も選んでしまう
    return int(scores["k"].iloc[int(np.argmax(np.diff(inertias, n=2))) + 1])


def suggest_k(scores, n_books):
    """指標からkを1つ選び、判断に使った値を文章で返す。

    エルボー法の値を軸に、冊数から決まる上限で頭を押さえる。
    """
    elbow = elbow_k(scores)
    best_silhouette = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    k_upper = max(2, min(int(scores["k"].max()), n_books // MIN_BOOKS_PER_CLUSTER))
    chosen = max(2, min(elbow, k_upper))

    reason = [
        f"エルボー法が示すk: {elbow}",
        f"シルエット係数が最大のk: {best_silhouette}（値={scores['silhouette'].max():.3f}）",
        f"1クラスタ平均{MIN_BOOKS_PER_CLUSTER}冊以上になるkの上限: {k_upper}",
        f"→ 採用 k={chosen}",
    ]
    if scores["silhouette"].max() < LOW_SILHOUETTE:
        reason.append(
            f"※ シルエット係数がどのkでも{LOW_SILHOUETTE}未満で、"
            "クラスタは明確に分離していない。"
            "図を見て納得できなければ --k で指定すること"
        )
    return chosen, reason


def plot_k_selection(scores, out_path, chosen_k=None, dpi=200):
    """エルボー法とシルエット係数を並べて描き、採用したkを示す。"""
    setup_japanese_font()

    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set_title("エルボー法（クラスタ内誤差）")
    axes[0].set_xlabel("クラスタ数 k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="orange")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"シルエット最大 k={best_k}")
    axes[1].set_title("シルエット係数（分離の良さ）")
    axes[1].set_xlabel("クラスタ数 k")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(True, alpha=0.3)

    if chosen_k is not None:
        for ax in axes:
            ax.axvline(chosen_k, color="gray", linestyle=":", label=f"採用 k={chosen_k}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"k選択グラフを保存しました: {out_path}")


def silhouette_for_labels(embeddings, labels, random_state=0):
    """割り当て済みのクラスタに対するシルエット係数（コサイン距離）。

    `evaluate_k` はkを変えながら計算するが、こちらは採用した1つの割り当てだけを測る。
    クラスタが1つしかないときは計算できないので nan を返す。
    """
    if len(set(np.asarray(labels).tolist())) < 2:
        return float("nan")
    return float(
        silhouette_score(
            embeddings, labels, metric="cosine",
            sample_size=min(500, len(embeddings)), random_state=random_state,
        )
    )
