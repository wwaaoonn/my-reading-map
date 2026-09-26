"""読書記録のCSVから「読書マップ」を作る。

    python main.py 読書記録.csv

CSVに必要な列は「タイトル」と「説明文」の2つだけ。それ以外の指定は
すべて省略できる（クラスタ数はデータから決まり、クラスタ名は生成AIが付ける）。
"""

import argparse
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from reading_map import cluster as cluster_mod
from reading_map import visualize as viz
from reading_map.embed import load_or_build_embeddings, load_reading_log
from reading_map.name_clusters import MODEL, load_env
from reading_map.pipeline import MapOptions, build_reading_map, check_reading_log

logger = logging.getLogger("reading_map.cli")

# APIキーを書いておくファイル（.gitignore 済み）
ENV_FILE = Path(__file__).resolve().parent / ".env"
# 公開用CSVから外す列
PUBLIC_EXCLUDE_COLUMNS = ["説明文"]
# 文埋め込みモデル（日本語対応）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Excel系の表計算ソフトが数式として解釈する、セル先頭の文字
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")
# 出力ファイル名に挟む実行時刻。例: reading_map_20260921-104300.png
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
# cluster_summary.csv に載せる頻出語の数
SUMMARY_TOP_WORDS = 10


def escape_csv_cell(value):
    """数式として解釈される文字で始まる文字列なら、`'` を前置する。"""
    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def escape_csv_formulas(df):
    """文字列の列にエスケープをかけた複製を返す。数値の列は変えない。"""
    escaped = df.copy()
    for column in escaped.columns:
        if escaped[column].dtype == object:
            escaped[column] = escaped[column].map(escape_csv_cell)
    return escaped


def write_csv(df, path):
    """CSVに書く。数式として解釈される文字列はエスケープする。"""
    escape_csv_formulas(df).to_csv(path, index=False, encoding="utf-8-sig")


def stamped(name, stamp):
    """ファイル名に実行時刻を挟む。stamp が空なら名前を変えない。"""
    if not stamp:
        return name
    path = Path(name)
    return f"{path.stem}_{stamp}{path.suffix}"


def setup_logging():
    """reading_map のログを標準エラーに出す。他のライブラリは警告以上だけ出す。"""
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    logging.getLogger("reading_map").setLevel(logging.INFO)


def books_frame(df, result):
    """入力の df に、クラスタIDと t-SNE の座標の列を足した表。"""
    books = df.copy()
    books["クラスタID"] = [book.cluster_id for book in result.books]
    # t-SNE は float32 で座標を返す。CSVに書く桁をそろえるため float32 に戻す
    books["tsne_x"] = np.array([book.x for book in result.books], dtype=np.float32)
    books["tsne_y"] = np.array([book.y for book in result.books], dtype=np.float32)
    return books


def summary_frame(result):
    """クラスタごとの名前・冊数・代表本・頻出語の表。"""
    return pd.DataFrame(
        {
            "クラスタID": [c.cluster_id for c in result.clusters],
            "冊数": [c.size for c in result.clusters],
            "クラスタ名": [c.name for c in result.clusters],
            "代表本": [result.books[c.representatives[0]].title for c in result.clusters],
            "頻出語": [" ".join(c.top_words[:SUMMARY_TOP_WORDS]) for c in result.clusters],
        }
    )


def pairs_frame(result):
    """類似ペアの表。本は位置ではなくタイトルで示す。"""
    return pd.DataFrame(
        {
            "本1": [result.books[p.book1].title for p in result.similar_pairs],
            "本2": [result.books[p.book2].title for p in result.similar_pairs],
            "類似度": [p.similarity for p in result.similar_pairs],
        }
    )


def k_scores_frame(result):
    """クラスタ数の検討に使った指標の表。"""
    return pd.DataFrame([asdict(score) for score in result.k_scores])


def parse_args():
    parser = argparse.ArgumentParser(
        description="読書記録のCSVから読書マップを作る",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python main.py data/sample_reading_log_1.csv",
    )
    parser.add_argument("csv", help="読書ログCSV（「タイトル」「説明文」の列が必要）")
    parser.add_argument(
        "--k", type=int, default=None,
        help="クラスタ数（2以上・冊数-1以下）。省略するとエルボー法と冊数から自動で決める",
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


def log_clusters(result):
    """クラスタごとの名前・冊数・中心の本・頻出語を表示する。"""
    for c in result.clusters:
        center = result.books[c.representatives[0]].title
        logger.info(f"  {c.cluster_id}: {c.name}（{c.size}冊 / 中心: {center}）")
        logger.info(f"     頻出語: {', '.join(c.top_words[:8])}")


def write_outputs(df, embeddings, result, out_dir, stamp):
    """PNGとCSVを書き出す。"""
    books = books_frame(df, result)
    cluster_names = {c.cluster_id: c.name for c in result.clusters}

    if result.k_scores is not None:
        scores = k_scores_frame(result)
        write_csv(scores, out_dir / stamped("k_selection.csv", stamp))
        cluster_mod.plot_k_selection(
            scores, out_dir / stamped("k_selection.png", stamp), chosen_k=result.k
        )

    viz.plot_reading_map(books, cluster_names, out_dir / stamped("reading_map.png", stamp))
    viz.plot_cluster_maps(books, embeddings, cluster_names, out_dir, stamp=stamp)

    write_csv(books, out_dir / stamped("clustered_books.csv", stamp))
    public_columns = [c for c in books.columns if c not in PUBLIC_EXCLUDE_COLUMNS]
    write_csv(books[public_columns], out_dir / stamped("clustered_books_public.csv", stamp))
    write_csv(summary_frame(result), out_dir / stamped("cluster_summary.csv", stamp))
    write_csv(pairs_frame(result), out_dir / stamped("similar_pairs.csv", stamp))


def main():
    args = parse_args()
    setup_logging()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(RUN_STAMP_FORMAT)
    logger.info(f"出力ファイル名に付ける実行時刻: {stamp}")

    logger.info(f"読み込み: {args.csv}")
    try:
        df = load_reading_log(args.csv)
        check_reading_log(df)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    logger.info(f"  {len(df)}冊")

    logger.info("\n説明文をベクトル化")
    embeddings = load_or_build_embeddings(
        df, model_name=args.embed_model,
        cache_path=out_dir / "embeddings.npy", force=args.force_embed,
    )

    if not args.no_ai_names:
        load_env(ENV_FILE)
    options = MapOptions(
        k=args.k, perplexity=args.perplexity, ai_names=not args.no_ai_names, model=args.model
    )
    try:
        result = build_reading_map(df, embeddings, options)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    log_clusters(result)

    logger.info("\n描画と書き出し")
    write_outputs(df, embeddings, result, out_dir, stamp)

    logger.info(
        f"\n完了しました。{out_dir}/ に出力しました"
        f"（{stamp} / 類似ペア {len(result.similar_pairs)}組）"
    )


if __name__ == "__main__":
    main()
