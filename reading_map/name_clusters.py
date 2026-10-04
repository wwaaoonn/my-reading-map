"""クラスタの名前を生成AI（Claude API）で付ける。

クラスタに入った本の説明文をClaudeに渡して短い見出しを作らせる。
APIキーが無い場合や接続できない場合は、頻出語をつないだ名前にフォールバックする。
"""

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# APIキーを書いておくファイル。リポジトリ直下に置く（.gitignore 済み）
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

MODEL = "claude-opus-5"
# 既定の接続先。違っていれば ANTHROPIC_BASE_URL が効いている
DEFAULT_BASE_URL = "https://api.anthropic.com"
# 1クラスタあたりClaudeに見せる本の数（クラスタ中心に近い順）
BOOKS_PER_CLUSTER = 8
# 説明文は先頭だけ渡す
DESCRIPTION_CHARS = 300
# 受け取った見出しを切る長さ
TITLE_MAX_CHARS = 40
# 見出しの長さの上限を指定しないときの、指示文の長さの条件
DEFAULT_LENGTH_RULE = "8〜20文字程度"

SYSTEM_PROMPT_TEMPLATE = """\
あなたは読書傾向を分析して、本のグループに見出しを付ける編集者です。

渡された各グループについて、そのグループに共通する内容やテーマを表す短い日本語の
見出しを作ってください。条件は次のとおりです。

- {length_rule}。体言止め
- 説明文から読み取れる中身に即した見出しにする。ジャンル名の言い換え
  （「小説」「エッセイ」など）だけで済ませない
- グループ同士で似た見出しにならないよう、違いが分かる言葉を選ぶ
- 番号や記号は付けない。見出しの文字だけを返す"""


class ClusterTitle(BaseModel):
    cluster_id: int
    title: str


class ClusterTitles(BaseModel):
    titles: list[ClusterTitle]


def load_env(path=ENV_FILE):
    """.env ファイルを読み込む。既にある環境変数は上書きしない。"""
    path = Path(path)
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(f"{path.name} がありますが python-dotenv が入っていないため読み込めません")
        return
    load_dotenv(path, override=False)


def one_line(text):
    """改行と連続する空白を1つの空白にまとめる。

    CSV由来の文字列がプロンプトのブロック構造（行頭の `## グループ N` など）を
    作らないようにする。
    """
    return " ".join(str(text).split())


def fallback_names(top_words):
    """APIを使わないときの名前。頻出語を3つつなぐ。"""
    return {
        cluster_id: "・".join(words[:3]) if words else f"クラスタ{cluster_id}"
        for cluster_id, words in top_words.items()
    }


def check_max_title_chars(max_title_chars):
    """見出しの長さの上限が指定できる範囲にあるか確かめる。範囲外なら ValueError。

    None は指定なし。
    """
    if max_title_chars is None:
        return
    if not 1 <= max_title_chars <= TITLE_MAX_CHARS:
        raise ValueError(
            f"max_title_chars は1以上、{TITLE_MAX_CHARS}以下で指定してください"
            f"（指定された max_title_chars={max_title_chars}）"
        )


def check_client_options(timeout, max_retries):
    """タイムアウトと再試行の回数が指定できる範囲にあるか確かめる。範囲外なら ValueError。

    None は指定なし。timeout は0より大きい値、max_retries は0以上。
    """
    if timeout is not None and not timeout > 0:
        raise ValueError(
            f"timeout は0より大きい値で指定してください（指定された timeout={timeout}）"
        )
    if max_retries is not None and not max_retries >= 0:
        raise ValueError(
            f"max_retries は0以上で指定してください（指定された max_retries={max_retries}）"
        )


def build_system_prompt(max_title_chars=None):
    """命名の指示文を作る。

    長さの条件は、max_title_chars が None なら「8〜20文字程度」、
    指定があれば「{max_title_chars}文字以内」。
    """
    if max_title_chars is None:
        length_rule = DEFAULT_LENGTH_RULE
    else:
        length_rule = f"{max_title_chars}文字以内"
    return SYSTEM_PROMPT_TEMPLATE.format(length_rule=length_rule)


def build_prompt(df, representatives, top_words):
    """クラスタごとに、中心に近い本の情報を並べたプロンプトを作る。"""
    blocks = []
    for cluster_id in sorted(df["クラスタID"].unique()):
        cluster_id = int(cluster_id)
        positions = representatives.get(cluster_id, [])[:BOOKS_PER_CLUSTER]
        total = int((df["クラスタID"] == cluster_id).sum())

        lines = [f"## グループ {cluster_id}（全{total}冊）"]
        words = top_words.get(cluster_id)
        if words:
            lines.append(f"頻出語: {', '.join(words[:10])}")
        lines.append(f"グループの中心に近い本（最大{BOOKS_PER_CLUSTER}冊）:")
        for pos in positions:
            row = df.iloc[pos]
            title = one_line(row["タイトル"])
            description = one_line(row["説明文"])[:DESCRIPTION_CHARS]
            lines.append(f"- 『{title}』 {description}")
        blocks.append("\n".join(lines))

    return (
        "次の各グループに見出しを付けてください。"
        "cluster_id は下のグループ番号をそのまま使ってください。\n\n" + "\n\n".join(blocks)
    )


def _fall_back(fallback, reason, *messages):
    """頻出語の名前と切り替えた理由を返す。messages は warning でログに出す。"""
    for message in messages:
        logger.warning(message)
    return fallback, reason


def generate_cluster_names(
    df, representatives, top_words, model=MODEL, max_title_chars=None,
    timeout=None, max_retries=None,
):
    """Claudeにクラスタ名を作らせる。失敗したら頻出語による名前を返す。

    APIキーは環境変数 ANTHROPIC_API_KEY から読む。

    max_title_chars は指示文に入れる見出しの長さの上限（文字数）。1未満、または
    TITLE_MAX_CHARS を超えると ValueError。上限を超えた見出しは切らずに返す。
    頻出語による名前には上限をかけない。

    timeout はAPIの呼び出し1回あたりのタイムアウト（秒）、max_retries は再試行の回数。
    None ならSDKの既定。timeout が0以下、max_retries が0未満だと ValueError。

    戻り値: ({クラスタID: 名前}, 頻出語に切り替えた理由。生成AIで命名できたら None)
    """
    check_max_title_chars(max_title_chars)
    check_client_options(timeout, max_retries)
    fallback = fallback_names(top_words)

    try:
        import anthropic
    except ImportError:
        return _fall_back(
            fallback, "anthropic パッケージが無い",
            "anthropic パッケージが無いので、頻出語からクラスタ名を作ります",
        )

    # 指定のある項目だけ渡す。SDKは timeout=None をタイムアウトなしとして扱う
    client_options = {}
    if timeout is not None:
        client_options["timeout"] = timeout
    if max_retries is not None:
        client_options["max_retries"] = max_retries
    client = anthropic.Anthropic(**client_options)

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=build_system_prompt(max_title_chars),
            messages=[{"role": "user", "content": build_prompt(df, representatives, top_words)}],
            output_format=ClusterTitles,
        )
    except anthropic.AuthenticationError:
        return _fall_back(
            fallback, "APIキーが拒否された",
            "APIキーが拒否されたので、頻出語からクラスタ名を作ります",
        )
    except TypeError as error:
        # 認証情報が1つも見つからないとき、SDKは AuthenticationError ではなく
        # TypeError を投げる。それ以外の TypeError は投げ直す
        if "authentication" not in str(error).lower():
            raise
        return _fall_back(
            fallback, "APIキーが見つからない",
            "APIキーが見つからないので、頻出語からクラスタ名を作ります"
            "（.env か環境変数 ANTHROPIC_API_KEY を設定すると生成AIが命名します）",
        )
    except anthropic.RateLimitError:
        return _fall_back(
            fallback, "APIのレート制限に達した",
            "APIのレート制限に達したので、頻出語からクラスタ名を作ります",
        )
    except anthropic.APIStatusError as error:
        return _fall_back(
            fallback, f"APIがエラーを返した（{error.status_code}）",
            f"APIがエラーを返したので、頻出語からクラスタ名を作ります（{error.status_code}）",
            f"  {error.message}",
        )
    except anthropic.APITimeoutError:
        return _fall_back(
            fallback, "APIへの接続がタイムアウトした",
            "APIへの接続がタイムアウトしたので、頻出語からクラスタ名を作ります",
            f"  接続先: {client.base_url}",
        )
    except anthropic.APIConnectionError as error:
        # 原因（DNS・TLS・接続拒否など）と接続先を出す
        messages = [
            "APIに接続できないので、頻出語からクラスタ名を作ります",
            f"  接続先: {client.base_url}",
            f"  原因: {error.__cause__ or error}",
        ]
        if str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
            messages.append(
                f"  接続先が {DEFAULT_BASE_URL} ではありません。"
                "環境変数 ANTHROPIC_BASE_URL を確認してください"
            )
        return _fall_back(fallback, "APIに接続できない", *messages)

    if response.stop_reason == "refusal":
        return _fall_back(
            fallback, "APIが応答を拒否した",
            "APIが応答を拒否したので、頻出語からクラスタ名を作ります",
        )

    names = dict(fallback)
    for item in response.parsed_output.titles:
        title = one_line(item.title)[:TITLE_MAX_CHARS]
        # 存在しないクラスタIDが返ってきても無視する
        if item.cluster_id in names and title:
            names[item.cluster_id] = title

    missing = [cid for cid in fallback if names[cid] == fallback[cid]]
    if missing:
        logger.warning(f"一部のクラスタ名が生成されなかったので頻出語で補いました: {missing}")

    if max_title_chars is not None:
        too_long = [
            cid for cid in fallback
            if cid not in missing and len(names[cid]) > max_title_chars
        ]
        if too_long:
            logger.warning(
                f"{max_title_chars}文字を超える見出しが返りました（切らずに使います）: {too_long}"
            )

    return names, None


def check_connection(model=MODEL):
    """APIに繋がるかどうかだけを、小さなリクエストで確かめる。

        python -m reading_map.name_clusters
    """
    import anthropic

    client = anthropic.Anthropic()
    print(f"接続先: {client.base_url}")
    if str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
        print("  ※ 接続先が通常と違います。環境変数 ANTHROPIC_BASE_URL を確認してください")

    try:
        response = client.messages.create(
            model=model, max_tokens=16,
            messages=[{"role": "user", "content": "接続確認です。OKとだけ答えてください。"}],
        )
    except TypeError as error:
        if "authentication" not in str(error).lower():
            raise
        print("結果: 認証情報が見つかりません"
              f"（{ENV_FILE.name} か環境変数 ANTHROPIC_API_KEY を設定してください）")
        return False
    except anthropic.AuthenticationError:
        print("結果: APIキーが拒否されました（キーの値を確認してください）")
        return False
    except anthropic.APITimeoutError:
        print("結果: タイムアウトしました")
        return False
    except anthropic.APIConnectionError as error:
        print(f"結果: 接続できません / 原因: {error.__cause__ or error}")
        return False
    except anthropic.APIStatusError as error:
        print(f"結果: APIがエラーを返しました（{error.status_code}）{error.message}")
        return False

    print(f"結果: 接続OK（model={response.model}）")
    return True


if __name__ == "__main__":
    load_env()
    raise SystemExit(0 if check_connection() else 1)
