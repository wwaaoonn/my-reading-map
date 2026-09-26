"""読書ログの読み込みと、説明文の文ベクトル化（埋め込み）。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["タイトル", "説明文"]

# クラスタリングと2次元化に必要な最小の冊数
MIN_BOOKS = 3

# 2次元化に必要な、内容の異なる説明文の最小件数。
# すべて同じ説明文（空も含む）だとベクトルが1種類になり、t-SNEがネイティブ側で落ちる
MIN_DISTINCT_DESCRIPTIONS = 2

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


def count_distinct_descriptions(df):
    """空でない説明文の種類数を返す。前後の空白は無視する。"""
    filled = df["説明文"].str.strip()
    return int(filled[filled != ""].nunique())


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


def descriptions_digest(descriptions):
    """説明文の中身から決まるハッシュ。1文字でも変われば別の値になる。"""
    digest = hashlib.sha256()
    for text in descriptions:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def meta_path(cache_path):
    """埋め込みキャッシュに対応するメタ情報ファイルのパス。"""
    return Path(cache_path).with_suffix(".json")


def cache_mismatch(meta, df, model_name, embeddings):
    """キャッシュを使えない理由を返す。使えるなら None を返す。"""
    if embeddings.shape[0] != len(df):
        return f"件数が違う（キャッシュ {embeddings.shape[0]}件 / データ {len(df)}件）"
    if meta is None:
        return "メタ情報が無い"
    if meta.get("model") != model_name:
        return f"モデルが違う（キャッシュ {meta.get('model')} / 指定 {model_name}）"
    if meta.get("digest") != descriptions_digest(df["説明文"]):
        return "説明文が変わっている"
    return None


def _read_meta(path):
    """メタ情報を読む。無い・壊れている場合は None を返す。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_or_build_embeddings(df, model_name, cache_path, force=False):
    """埋め込みをキャッシュから読む。使えない・force なら作り直す。

    キャッシュを使うのは、冊数・埋め込みモデル・説明文の内容がすべて
    前回と一致する場合だけ。判定にはメタ情報（`embeddings.json`）を使う。

    df と embeddings は行の順番で対応するため、行数は必ず一致させる。
    """
    cache_path = Path(cache_path)

    if not force and cache_path.exists():
        embeddings = np.load(cache_path)
        reason = cache_mismatch(_read_meta(meta_path(cache_path)), df, model_name, embeddings)
        if reason is None:
            print(f"埋め込みをキャッシュから読み込みました: {cache_path} {embeddings.shape}")
            return l2_normalize(embeddings)
        print(f"キャッシュを使えないので作り直します（{reason}）")

    embeddings = build_embeddings(df["説明文"], model_name)
    assert embeddings.shape[0] == len(df), "件数が一致しません"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    meta = {
        "model": model_name,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "digest": descriptions_digest(df["説明文"]),
    }
    with open(meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"埋め込みを保存しました: {cache_path} {embeddings.shape}")

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
                    {
                        "本1": titles[i],
                        "本2": titles[j],
                        "類似度": round(float(similarity[i, j]), 3),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["本1", "本2", "類似度"])
    return pd.DataFrame(rows).sort_values("類似度", ascending=False, ignore_index=True)
