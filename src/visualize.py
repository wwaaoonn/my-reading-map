"""t-SNEによる2次元化と、読書マップの描画。"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from .plot_style import setup_japanese_font

CMAP = plt.get_cmap("tab10")

# t-SNEの perplexity の上限。実際の値は冊数から決める
PERPLEXITY_MAX = 30
# 楕円の大きさ（標準偏差の何倍か）
ELLIPSE_N_STD = 1.2
# 楕円の計算に使う点の割合。重心から遠い点を落とす
ELLIPSE_CORE_RATIO = 0.85


def auto_perplexity(n_books, maximum=PERPLEXITY_MAX):
    """冊数に合わせた perplexity。"""
    return max(5, min(maximum, (n_books - 1) // 3))


def compute_tsne(embeddings, perplexity=None, random_state=42):
    """埋め込みを2次元に落とす。perplexity を省略すると冊数から決める。"""
    n_books = len(embeddings)
    if perplexity is None:
        perplexity = auto_perplexity(n_books)
    perplexity = max(2, min(perplexity, n_books - 1))
    print(f"  perplexity={perplexity}")
    tsne = TSNE(
        n_components=2, random_state=random_state, perplexity=perplexity, metric="cosine"
    )
    return tsne.fit_transform(np.asarray(embeddings))


def cluster_positions(df, cluster_id):
    """あるクラスタに属する行の位置（0始まりの行番号）を返す。"""
    return np.flatnonzero(df["クラスタID"].to_numpy() == cluster_id)


def centroid_similarity(embeddings, positions):
    """クラスタ重心と各本のコサイン類似度を返す（クラスタの中心らしさ）。"""
    vectors = np.asarray(embeddings)[positions]
    centroid = vectors.mean(axis=0)
    return cosine_similarity([centroid], vectors)[0]


def representative_books(df, embeddings, top_n=1):
    """クラスタごとに、重心に近い順の行位置を top_n 件返す。

    戻り値: {クラスタID: [行位置, ...]}（重心に近い順）
    """
    result = {}
    for cluster_id in sorted(df["クラスタID"].unique()):
        positions = cluster_positions(df, cluster_id)
        if len(positions) == 0:
            continue
        sims = centroid_similarity(embeddings, positions)
        order = np.argsort(sims)[::-1][: min(top_n, len(positions))]
        result[int(cluster_id)] = [int(positions[i]) for i in order]
    return result


def representative_table(df, embeddings):
    """各クラスタの代表本を一覧表（DataFrame）で返す。"""
    import pandas as pd

    rows = []
    for cluster_id, positions in representative_books(df, embeddings, top_n=1).items():
        pos = positions[0]
        sims = centroid_similarity(embeddings, cluster_positions(df, cluster_id))
        rows.append(
            {
                "クラスタID": cluster_id,
                "代表本タイトル": df.iloc[pos]["タイトル"],
                "重心との類似度": round(float(sims.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def _marker_sizes(sims, base=100, scale=1000):
    """マーカーサイズ。重心に近い本ほど大きく、差は非線形に強調する。"""
    span = sims.max() - sims.min()
    if span == 0:
        return np.full_like(sims, float(base))
    normed = (sims - sims.min()) / span
    return base + scale * normed**2


def _hide_ticks(ax):
    ax.tick_params(
        labelbottom=False, labelleft=False, labelright=False, labeltop=False,
        bottom=False, left=False, right=False, top=False,
    )


def _core_points(x, y, keep=ELLIPSE_CORE_RATIO):
    """重心から遠い点を落として残りを返す。"""
    points = np.column_stack([x, y])
    if keep >= 1.0 or len(points) < 5:
        return points
    distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
    n_keep = max(3, int(round(len(points) * keep)))
    return points[np.argsort(distances)[:n_keep]]


def _plot_confidence_ellipse(x, y, ax, n_std=ELLIPSE_N_STD, **kwargs):
    """点の散らばりを楕円で示す。3点未満はスキップする。

    楕円は重心に近い点だけで計算するので、離れた点は楕円の外に出る。
    """
    if len(x) < 3:
        return
    core = _core_points(x, y)
    x, y = core[:, 0], core[:, 1]
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ax.add_patch(
        patches.Ellipse(
            xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=angle, **kwargs
        )
    )


def plot_reading_map(df, cluster_names, out_path, dpi=200):
    """全体マップ：クラスタを色分けし、散らばりを楕円、名前を重心付近に置く。

    クラスタ名が重なった場合は adjustText が自動でずらす。
    """
    setup_japanese_font()
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(9, 4.8))
    texts = []

    for i, cluster_id in enumerate(sorted(df["クラスタID"].unique())):
        cluster_df = df[df["クラスタID"] == cluster_id]
        color = CMAP(i % 10)
        name = cluster_names.get(int(cluster_id), f"クラスタ{cluster_id}")

        ax.scatter(
            cluster_df["tsne_x"], cluster_df["tsne_y"],
            label=name, color=color, alpha=0.7, s=60, zorder=3,
        )
        _plot_confidence_ellipse(
            cluster_df["tsne_x"].values, cluster_df["tsne_y"].values, ax,
            edgecolor=color, facecolor=color, alpha=0.12, linewidth=1.5, zorder=2,
        )

        texts.append(
            ax.text(
                cluster_df["tsne_x"].mean(), cluster_df["tsne_y"].mean(), name,
                fontsize=15, color=color, fontweight="bold", ha="center", va="center", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
            )
        )

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    _hide_ticks(ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"全体マップを保存しました: {out_path}")


def plot_cluster_maps(df, embeddings, cluster_names, out_dir, top_n_titles=5, dpi=200):
    """クラスタ別マップ：1クラスタだけ色を付け、中心的な本のタイトルを表示する。"""
    setup_japanese_font()
    from adjustText import adjust_text

    saved = []
    for i, cluster_id in enumerate(sorted(df["クラスタID"].unique())):
        positions = cluster_positions(df, cluster_id)
        if len(positions) == 0:
            continue
        cluster_df = df.iloc[positions]
        sims = centroid_similarity(embeddings, positions)

        fig = plt.figure(figsize=(10, 6))
        # 背景として全体を淡く描く
        plt.scatter(df["tsne_x"], df["tsne_y"], color="lightgray", alpha=0.3, s=10)
        plt.scatter(
            cluster_df["tsne_x"], cluster_df["tsne_y"],
            color=CMAP(i % 10), alpha=0.8, s=_marker_sizes(sims, scale=800),
        )

        top = np.argsort(sims)[::-1][: min(top_n_titles, len(positions))]
        texts = [
            plt.text(
                df.iloc[positions[t]]["tsne_x"], df.iloc[positions[t]]["tsne_y"],
                df.iloc[positions[t]]["タイトル"], fontsize=16, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            for t in top
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray"))

        plt.title(cluster_names.get(int(cluster_id), f"クラスタ{cluster_id}"), fontsize=16)
        plt.grid(True)
        _hide_ticks(plt.gca())
        plt.tight_layout()

        out_path = f"{out_dir}/cluster_{cluster_id}.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    print(f"クラスタ別マップを保存しました: {len(saved)}枚")
    return saved
