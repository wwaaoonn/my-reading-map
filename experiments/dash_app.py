"""読書マップをブラウザ上で操作するDashアプリ（試作）。

クラスタの点をクリックすると、そのクラスタだけのマップを下に表示する。

先に `python main.py <CSV>` を実行して、出力先に
clustered_books.csv / embeddings.npy / cluster_summary.csv を作っておくこと。

    python experiments/dash_app.py --out outputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS = px.colors.qualitative.Plotly


def load_data(out_dir):
    """パイプラインの出力（クラスタ結果・埋め込み・クラスタ名）を読み込む。"""
    out_dir = Path(out_dir)
    csv_path = out_dir / "clustered_books.csv"
    npy_path = out_dir / "embeddings.npy"

    for path in (csv_path, npy_path):
        if not path.exists():
            raise SystemExit(
                f"{path} が見つかりません。先に `python main.py <CSV>` を実行してください。"
            )

    df = pd.read_csv(csv_path)
    embeddings = np.load(npy_path)
    if len(df) != len(embeddings):
        raise SystemExit(
            f"件数が合いません（CSV {len(df)}件 / 埋め込み {len(embeddings)}件）。"
            "パイプラインを実行し直してください。"
        )

    # クラスタ名はパイプラインが出す cluster_summary.csv から取る
    cluster_names = {}
    summary_path = out_dir / "cluster_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        cluster_names = dict(zip(summary["クラスタID"], summary["クラスタ名"]))

    return df, embeddings, cluster_names


def centroid_sims(df, embeddings, cluster_id):
    """クラスタ重心との類似度と、対象行の位置を返す。"""
    positions = np.flatnonzero(df["クラスタID"].to_numpy() == cluster_id)
    vectors = embeddings[positions]
    centroid = vectors.mean(axis=0)
    return positions, cosine_similarity([centroid], vectors)[0]


def scaled_sizes(sims, base, span):
    """重心に近いほど大きいマーカーサイズ。"""
    return base + span * (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)


def main_figure(df, embeddings, cluster_names):
    """全体マップ。クラスタごとに色を分け、重心に近い本を大きく描く。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tsne_x"], y=df["tsne_y"], mode="markers",
            marker=dict(size=6, color="lightgray", opacity=0.3), hoverinfo="skip",
            showlegend=False,
        )
    )

    for i, cluster_id in enumerate(sorted(df["クラスタID"].unique())):
        positions, sims = centroid_sims(df, embeddings, cluster_id)
        sub = df.iloc[positions]
        fig.add_trace(
            go.Scatter(
                x=sub["tsne_x"], y=sub["tsne_y"], mode="markers",
                marker=dict(size=scaled_sizes(sims, 20, 50), color=COLORS[i % len(COLORS)], opacity=0.8),
                hovertext=sub["タイトル"], hoverinfo="text",
                customdata=[int(cluster_id)] * len(sub),
                name=cluster_names.get(int(cluster_id), f"クラスタ{cluster_id}"),
            )
        )

    fig.update_layout(
        title="My読書マップ（点をクリックすると下に詳細）",
        width=1000, height=600, plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def detail_figure(df, embeddings, cluster_names, cluster_id, top_n=3):
    """1クラスタだけを取り出したマップ。中心的な本にタイトルを付ける。"""
    positions, sims = centroid_sims(df, embeddings, cluster_id)
    sub = df.iloc[positions]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tsne_x"], y=df["tsne_y"], mode="markers",
            marker=dict(size=6, color="lightgray", opacity=0.2), hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["tsne_x"], y=sub["tsne_y"], mode="markers",
            marker=dict(size=scaled_sizes(sims, 10, 35),
                        color=COLORS[int(cluster_id) % len(COLORS)], opacity=0.8),
            hovertext=sub["タイトル"], hoverinfo="text", showlegend=False,
        )
    )

    for rank in np.argsort(sims)[::-1][: min(top_n, len(positions))]:
        row = df.iloc[positions[rank]]
        fig.add_annotation(
            x=row["tsne_x"], y=row["tsne_y"], text=row["タイトル"],
            showarrow=True, arrowhead=2, ax=0, ay=-20, bgcolor="white", opacity=0.8,
        )

    fig.update_layout(
        title=cluster_names.get(int(cluster_id), f"クラスタ{cluster_id}"),
        width=1000, height=600, plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_app(out_dir):
    df, embeddings, cluster_names = load_data(out_dir)

    app = Dash(__name__, title="My読書マップ")
    app.layout = html.Div(
        [
            html.H2("My読書マップ"),
            # 全体マップは差し替えない。詳細は下の別グラフに出す
            dcc.Graph(id="main-map", figure=main_figure(df, embeddings, cluster_names)),
            dcc.Graph(id="detail-map"),
        ]
    )

    @app.callback(Output("detail-map", "figure"), Input("main-map", "clickData"))
    def show_detail(click_data):
        if not click_data:
            fig = go.Figure()
            fig.update_layout(
                title="上のマップで点をクリックすると、そのクラスタだけを表示します。",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                plot_bgcolor="white", height=200,
            )
            return fig

        point = click_data["points"][0]
        # クリックされた点のクラスタIDを customdata から取る
        cluster_id = point.get("customdata")
        if cluster_id is None:
            title = point["hovertext"]
            cluster_id = int(df.loc[df["タイトル"] == title, "クラスタID"].iloc[0])
        return detail_figure(df, embeddings, cluster_names, int(cluster_id))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="読書マップのDashアプリ")
    parser.add_argument("--out", default="outputs", help="パイプラインの出力先（既定: outputs）")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    build_app(args.out).run(debug=args.debug, port=args.port)
