"""クラスタごとの頻出語抽出（Janomeで形態素解析 → TF-IDF）。"""

from sklearn.feature_extraction.text import TfidfVectorizer

# 除外する語。増やすほど頻出語が読みやすくなる
JAPANESE_STOPWORDS = [
    "こと", "これ", "それ", "あれ", "ため", "よう", "もの", "ところ",
    "そして", "しかし", "また", "ので", "のである", "られる", "れる", "ある", "いる",
    "ます", "です", "する", "した", "して", "できる", "なる",
    "の", "に", "を", "が", "と", "て", "で", "は", "も", "へ", "から", "まで", "より",
]

# 頻出語として残す品詞
TARGET_PARTS = ("名詞", "動詞", "形容詞")

_tokenizer = None


def _get_tokenizer():
    """Janomeの解析器を使い回す（毎回作ると遅い）。"""
    global _tokenizer
    if _tokenizer is None:
        from janome.tokenizer import Tokenizer

        _tokenizer = Tokenizer()
    return _tokenizer


def tokenize_japanese(text):
    """日本語テキストから名詞・動詞・形容詞の原形を取り出す。"""
    tokens = []
    for token in _get_tokenizer().tokenize(text):
        base = token.base_form
        part = token.part_of_speech.split(",")[0]
        if part in TARGET_PARTS and base != "*":
            tokens.append(base)
    return tokens


def top_words_per_cluster(df, top_n=15, max_features=100):
    """クラスタごとに説明文をまとめ、TF-IDF上位語を返す。

    戻り値: {クラスタID: [頻出語, ...]}
    """
    cluster_texts = df.groupby("クラスタID")["説明文"].apply(" ".join)
    result = {}

    for cluster_id, text in cluster_texts.items():
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_japanese,
            stop_words=JAPANESE_STOPWORDS,
            max_features=max_features,
            token_pattern=None,  # tokenizer を渡すときは無効化する（警告回避）
        )
        matrix = vectorizer.fit_transform([text])
        names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        top = scores.argsort()[::-1][:top_n]
        result[int(cluster_id)] = [names[i] for i in top]

    return result
