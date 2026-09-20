"""クラスタの名前を生成AI（Claude API）で付ける。

クラスタに入った本の説明文をClaudeに渡して短い見出しを作らせる。
APIキーが無い場合や接続できない場合は、頻出語をつないだ名前にフォールバックする。
"""

from pathlib import Path

from pydantic import BaseModel

# APIキーを書いておくファイル。リポジトリ直下に置く（.gitignore 済み）
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

MODEL = "claude-opus-5"
# 既定の接続先。違っていれば ANTHROPIC_BASE_URL が効いている
DEFAULT_BASE_URL = "https://api.anthropic.com"
# 1クラスタあたりClaudeに見せる本の数（クラスタ中心に近い順）
BOOKS_PER_CLUSTER = 8
# 説明文は先頭だけ渡す
DESCRIPTION_CHARS = 300

SYSTEM_PROMPT = """\
あなたは読書傾向を分析して、本のグループに見出しを付ける編集者です。

渡された各グループについて、そのグループに共通する内容やテーマを表す短い日本語の
見出しを作ってください。条件は次のとおりです。

- 8〜20文字程度。体言止め
- 説明文から読み取れる中身に即した見出しにする。ジャンル名の言い換え
  （「小説」「エッセイ」など）だけで済ませない
- グループ同士で似た見出しにならないよう、違いが分かる言葉を選ぶ
- 番号や記号は付けない。見出しの文字だけを返す"""


class ClusterTitle(BaseModel):
    cluster_id: int
    title: str


class ClusterTitles(BaseModel):
    titles: list[ClusterTitle]


def load_env():
    """リポジトリ直下の .env を読み込む。既にある環境変数は上書きしない。"""
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(f"{ENV_FILE.name} がありますが python-dotenv が入っていないため読み込めません")
        return
    load_dotenv(ENV_FILE, override=False)


def fallback_names(top_words):
    """APIを使わないときの名前。頻出語を3つつなぐ。"""
    return {
        cluster_id: "・".join(words[:3]) if words else f"クラスタ{cluster_id}"
        for cluster_id, words in top_words.items()
    }


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
            description = str(row["説明文"])[:DESCRIPTION_CHARS]
            lines.append(f"- 『{row['タイトル']}』 {description}")
        blocks.append("\n".join(lines))

    return (
        "次の各グループに見出しを付けてください。"
        "cluster_id は下のグループ番号をそのまま使ってください。\n\n" + "\n\n".join(blocks)
    )


def generate_cluster_names(df, representatives, top_words, model=MODEL):
    """Claudeにクラスタ名を作らせる。失敗したら頻出語による名前を返す。

    戻り値: ({クラスタID: 名前}, AIで生成できたかどうか)
    """
    fallback = fallback_names(top_words)
    load_env()

    try:
        import anthropic
    except ImportError:
        print("anthropic パッケージが無いので、頻出語からクラスタ名を作ります")
        return fallback, False

    client = anthropic.Anthropic()

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(df, representatives, top_words)}],
            output_format=ClusterTitles,
        )
    except anthropic.AuthenticationError:
        print("APIキーが拒否されたので、頻出語からクラスタ名を作ります")
        return fallback, False
    except TypeError as error:
        # 認証情報が1つも見つからないとき、SDKは AuthenticationError ではなく
        # TypeError を投げる。それ以外の TypeError は投げ直す
        if "authentication" not in str(error).lower():
            raise
        print(
            "APIキーが見つからないので、頻出語からクラスタ名を作ります"
            "（.env か環境変数 ANTHROPIC_API_KEY を設定すると生成AIが命名します）"
        )
        return fallback, False
    except anthropic.RateLimitError:
        print("APIのレート制限に達したので、頻出語からクラスタ名を作ります")
        return fallback, False
    except anthropic.APIStatusError as error:
        print(f"APIがエラーを返したので、頻出語からクラスタ名を作ります（{error.status_code}）")
        print(f"  {error.message}")
        return fallback, False
    except anthropic.APITimeoutError:
        print(f"APIへの接続がタイムアウトしたので、頻出語からクラスタ名を作ります")
        print(f"  接続先: {client.base_url}")
        return fallback, False
    except anthropic.APIConnectionError as error:
        # 原因（DNS・TLS・接続拒否など）と接続先を表示する
        print("APIに接続できないので、頻出語からクラスタ名を作ります")
        print(f"  接続先: {client.base_url}")
        print(f"  原因: {error.__cause__ or error}")
        if str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
            print(f"  接続先が {DEFAULT_BASE_URL} ではありません。"
                  "環境変数 ANTHROPIC_BASE_URL を確認してください")
        return fallback, False

    if response.stop_reason == "refusal":
        print("APIが応答を拒否したので、頻出語からクラスタ名を作ります")
        return fallback, False

    names = dict(fallback)
    for item in response.parsed_output.titles:
        # 存在しないクラスタIDが返ってきても無視する
        if item.cluster_id in names and item.title.strip():
            names[item.cluster_id] = item.title.strip()

    missing = [cid for cid in fallback if names[cid] == fallback[cid]]
    if missing:
        print(f"一部のクラスタ名が生成されなかったので頻出語で補いました: {missing}")

    return names, True


def check_connection(model=MODEL):
    """APIに繋がるかどうかだけを、小さなリクエストで確かめる。

        python -m src.name_clusters
    """
    import anthropic

    load_env()
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
    raise SystemExit(0 if check_connection() else 1)
