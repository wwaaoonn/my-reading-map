"""読書ログの読み込みと、説明文の文ベクトル化（埋め込み）。"""

from pathlib import Path

import numpy as np
import pandas as pd

# 読書ログCSVに最低限必要な列
REQUIRED_COLUMNS = ["タイトル", "説明文"]

# クラスタリングと2次元化に必要な最小の冊数
MIN_BOOKS = 3

# e5系のモデルは入力にプレフィックスが必要。クラスタリングは対称タスクなので query: を使う
E5_PREFIX = "query: "



def load_reading_log(path):
    """読書ログCSVを読み込み、必須列があるか確認する。"""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} に必須列がありません: {missing}（必要な列: {REQUIRED_COLUMNS}）"
        )

    # 説明文の欠損は空文字にして、以降の処理で落ちないようにする
    df["説明文"] = df["説明文"].fillna("").astype(str)

    empty = int((df["説明文"].str.strip() == "").sum())
    if empty:
        print(f"注意：説明文が空の行が {empty} 件あります（ベクトルが意味を持ちません）")

    return df


def l2_normalize(embeddings):
    """各ベクトルの長さを1に揃える。既に揃っていれば値は変わらない。"""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def build_embeddings(descriptions, model_name):
    """説明文のリストを文ベクトルに変換する。"""
    # 実際に計算するときだけ読み込む
    from sentence_transformers import SentenceTransformer

    print(f"モデルを読み込み中: {model_name}")
    texts = list(descriptions)
    if "e5" in model_name.lower():
        texts = [E5_PREFIX + t for t in texts]

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def load_or_build_embeddings(df, model_name, cache_path, force=False):
    """埋め込みをキャッシュから読む。無い・冊数が合わない・force なら作り直す。

    df と embeddings は行の順番で対応するため、行数は必ず一致させる。
    """
    cache_path = Path(cache_path)

    if not force and cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(df):
            print(f"埋め込みをキャッシュから読み込みました: {cache_path} {embeddings.shape}")
            return l2_normalize(embeddings)
        print(
            f"キャッシュの件数が合わないので作り直します"
            f"（キャッシュ {embeddings.shape[0]}件 / データ {len(df)}件）"
        )

    embeddings = build_embeddings(df["説明文"], model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"埋め込みを保存しました: {cache_path} {embeddings.shape}")

    assert embeddings.shape[0] == len(df), "件数が一致しません"
    return l2_normalize(embeddings)


def similar_pairs(df, embeddings, threshold=0.5):
    """類似度がしきい値を超える本のペアを、類似度の高い順に返す。"""
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(embeddings)
    titles = df["タイトル"].tolist()

    rows = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if similarity[i, j] > threshold:
                rows.append(
                    {"本1": titles[i], "本2": titles[j], "類似度": round(float(similarity[i, j]), 3)}
                )

    if not rows:
        return pd.DataFrame(columns=["本1", "本2", "類似度"])
    return pd.DataFrame(rows).sort_values("類似度", ascending=False, ignore_index=True)
