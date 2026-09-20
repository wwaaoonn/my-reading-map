"""読書記録のCSVから「読書マップ」を作る。

    python main.py 読書記録.csv

CSVに必要な列は「タイトル」と「説明文」の2つだけ。それ以外の指定は
すべて省略できる（クラスタ数はデータから決まり、クラスタ名は生成AIが付ける）。
"""

import argparse
from pathlib import Path

import pandas as pd

from src import cluster as cluster_mod
from src import visualize as viz
from src.embed import MIN_BOOKS, load_or_build_embeddings, load_reading_log, similar_pairs
from src.label import top_words_per_cluster
from src.name_clusters import MODEL, generate_cluster_names

# 公開用CSVから外す列
PUBLIC_EXCLUDE_COLUMNS = ["説明文"]
# 文埋め込みモデル（日本語対応）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TSNE_RANDOM_STATE = 42
KMEANS_RANDOM_STATE = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="読書記録のCSVから読書マップを作る",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python main.py data/sample_reading_log.csv",
    )
    parser.add_argument("csv", help="読書ログCSV（「タイトル」「説明文」の列が必要）")
    parser.add_argument(
        "--k", type=int, default=None,
        help="クラスタ数。省略するとエルボー法とシルエット係数から自動で決める",
    )
    parser.add_argument("--out", default="outputs", help="出力先ディレクトリ（既定: outputs）")
    parser.add_argument(
        "--no-ai-names", action="store_true",
        help="生成AIによる命名をせず、頻出語からクラスタ名を作る",
    )
    parser.add_argument("--model", default=MODEL, help=f"命名に使うモデル（既定: {MODEL}）")
    parser.add_argument(
        "--perplexity", type=int, default=None,
        help="t-SNEのperplexity。省略すると冊数から決める",
    )
    parser.add_argument(
        "--embed-model", default=EMBEDDING_MODEL,
        help=f"文埋め込みモデル（既定: {EMBEDDING_MODEL}）",
    )
    parser.add_argument("--force-embed", action="store_true", help="埋め込みを作り直す")
    return parser.parse_args()


def decide_k(embeddings, out_dir, requested_k):
    """使うクラスタ数を決める。指定があればそれに従う。"""
    if requested_k is not None:
        print(f"\n[3/6] クラスタ数: 指定された k={requested_k} を使います")
        return requested_k

    print("\n[3/6] クラスタ数を決めています（kを変えてクラスタリングを試行）")
    scores = cluster_mod.evaluate_k(embeddings, random_state=KMEANS_RANDOM_STATE)
    scores.to_csv(out_dir / "k_selection.csv", index=False, encoding="utf-8-sig")

    k, reasons = cluster_mod.suggest_k(scores, len(embeddings))
    print()
    for line in reasons:
        print(f"  {line}")
    cluster_mod.plot_k_selection(scores, out_dir / "k_selection.png", chosen_k=k)
    return k


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 読み込み
    print(f"[1/6] 読み込み: {args.csv}")
    df = load_reading_log(args.csv)
    print(f"  {len(df)}冊")
    if len(df) < MIN_BOOKS:
        raise SystemExit(
            f"{len(df)}冊では地図を作れません。{MIN_BOOKS}冊以上のCSVを渡してください。"
        )

    # 2. 説明文をベクトル化（キャッシュがあれば再利用）
    print("\n[2/6] 説明文をベクトル化")
    embeddings = load_or_build_embeddings(
        df, model_name=args.embed_model,
        cache_path=out_dir / "embeddings.npy", force=args.force_embed,
    )

    # 3. クラスタ数を決める
    k = decide_k(embeddings, out_dir, args.k)

    # 4. クラスタリング
    print(f"\n[4/6] k={k} でクラスタリング")
    df["クラスタID"] = cluster_mod.fit_kmeans(embeddings, k, KMEANS_RANDOM_STATE)
    silhouette = cluster_mod.silhouette_for_labels(
        embeddings, df["クラスタID"].to_numpy(), KMEANS_RANDOM_STATE
    )
    print(f"  シルエット係数（コサイン距離）: {silhouette:.3f}")
    if silhouette < cluster_mod.LOW_SILHOUETTE:
        print(
            f"  ※ {cluster_mod.LOW_SILHOUETTE}未満。このkでは、クラスタは明確に分離していない"
        )

    top_words = top_words_per_cluster(df)
    representatives = viz.representative_books(df, embeddings, top_n=8)

    # 5. クラスタ名を付ける
    print("\n[5/6] クラスタ名を生成")
    if args.no_ai_names:
        from src.name_clusters import fallback_names

        cluster_names, by_ai = fallback_names(top_words), False
    else:
        cluster_names, by_ai = generate_cluster_names(
            df, representatives, top_words, model=args.model
        )
    print(f"  命名: {'生成AI（' + args.model + '）' if by_ai else '頻出語'}")
    for cluster_id in sorted(cluster_names):
        count = int((df["クラスタID"] == cluster_id).sum())
        center = df.iloc[representatives[cluster_id][0]]["タイトル"]
        print(f"  {cluster_id}: {cluster_names[cluster_id]}（{count}冊 / 中心: {center}）")
        print(f"     頻出語: {', '.join(top_words[cluster_id][:8])}")

    # 6. 2次元化して描画
    print("\n[6/6] t-SNEで2次元化して描画")
    coords = viz.compute_tsne(embeddings, args.perplexity, TSNE_RANDOM_STATE)
    df["tsne_x"], df["tsne_y"] = coords[:, 0], coords[:, 1]

    viz.plot_reading_map(df, cluster_names, out_dir / "reading_map.png")
    viz.plot_cluster_maps(df, embeddings, cluster_names, out_dir)

    # 結果の保存
    df.to_csv(out_dir / "clustered_books.csv", index=False, encoding="utf-8-sig")
    public_columns = [c for c in df.columns if c not in PUBLIC_EXCLUDE_COLUMNS]
    df[public_columns].to_csv(
        out_dir / "clustered_books_public.csv", index=False, encoding="utf-8-sig"
    )

    summary = pd.DataFrame({"クラスタID": sorted(cluster_names)})
    summary["冊数"] = summary["クラスタID"].map(lambda cid: int((df["クラスタID"] == cid).sum()))
    summary["クラスタ名"] = summary["クラスタID"].map(cluster_names)
    summary["代表本"] = summary["クラスタID"].map(
        lambda cid: df.iloc[representatives[cid][0]]["タイトル"]
    )
    summary["頻出語"] = summary["クラスタID"].map(lambda cid: " ".join(top_words[cid][:10]))
    summary.to_csv(out_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    pairs = similar_pairs(df, embeddings, threshold=0.5)
    pairs.to_csv(out_dir / "similar_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"\n完了しました。{out_dir}/ に出力しました（類似ペア {len(pairs)}組）")


if __name__ == "__main__":
    main()
