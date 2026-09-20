"""src/name_clusters.py のテスト。APIは呼ばず、クライアントを差し替える。"""

import builtins

import anthropic
import httpx2
import pandas as pd
import pytest

from src.name_clusters import (
    BOOKS_PER_CLUSTER,
    DESCRIPTION_CHARS,
    ClusterTitle,
    ClusterTitles,
    build_prompt,
    fallback_names,
    generate_cluster_names,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(status, cls):
    return cls("エラー", response=httpx2.Response(status, request=REQUEST), body=None)


class FakeMessages:
    """messages.parse が、渡されたものを返すか例外を投げるだけのスタブ。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeResponse:
    def __init__(self, titles, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.parsed_output = ClusterTitles(
            titles=[ClusterTitle(cluster_id=cid, title=t) for cid, t in titles]
        )


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "クラスタID": [0, 0, 1],
            "タイトル": ["本A", "本B", "本C"],
            "説明文": ["猫の話", "犬の話", "宇宙の話"],
        }
    )


@pytest.fixture
def representatives():
    return {0: [0, 1], 1: [2]}


@pytest.fixture
def top_words():
    return {0: ["猫", "犬", "動物"], 1: ["宇宙", "星", "旅"]}


@pytest.fixture
def fake_client(monkeypatch):
    """anthropic.Anthropic を差し替える。戻り値の messages を各テストが設定する。"""
    messages = FakeMessages()

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = messages
            self.base_url = "https://api.anthropic.com"

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return messages


class TestFallbackNames:
    def test_頻出語を3つつなぐ(self):
        assert fallback_names({0: ["猫", "犬", "鳥", "馬"]}) == {0: "猫・犬・鳥"}

    def test_語が3つ未満でも作れる(self):
        assert fallback_names({0: ["猫"]}) == {0: "猫"}

    def test_語が無ければクラスタIDを使う(self):
        assert fallback_names({3: []}) == {3: "クラスタ3"}


class TestBuildPrompt:
    def test_クラスタごとの見出しが入る(self, df, representatives, top_words):
        prompt = build_prompt(df, representatives, top_words)
        assert "## グループ 0（全2冊）" in prompt
        assert "## グループ 1（全1冊）" in prompt

    def test_頻出語とタイトルが入る(self, df, representatives, top_words):
        prompt = build_prompt(df, representatives, top_words)
        assert "頻出語: 猫, 犬, 動物" in prompt
        assert "『本A』" in prompt

    def test_渡す本はBOOKS_PER_CLUSTERまで(self, top_words):
        df = pd.DataFrame(
            {
                "クラスタID": [0] * 20,
                "タイトル": [f"本{i}" for i in range(20)],
                "説明文": ["説明"] * 20,
            }
        )
        prompt = build_prompt(df, {0: list(range(20))}, {0: ["語"]})
        assert prompt.count("- 『") == BOOKS_PER_CLUSTER

    def test_説明文は先頭だけ渡す(self, top_words):
        long_text = "あ" * (DESCRIPTION_CHARS + 100)
        df = pd.DataFrame({"クラスタID": [0], "タイトル": ["本A"], "説明文": [long_text]})
        prompt = build_prompt(df, {0: [0]}, {0: ["語"]})
        assert "あ" * DESCRIPTION_CHARS in prompt
        assert "あ" * (DESCRIPTION_CHARS + 1) not in prompt


class TestGenerateClusterNamesFallback:
    """APIが使えない場合、頻出語の名前と by_ai=False を返す。"""

    def test_anthropicが入っていない(self, monkeypatch, df, representatives, top_words):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("no anthropic")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is False
        assert names == fallback_names(top_words)

    @pytest.mark.parametrize(
        "error",
        [
            api_error(401, anthropic.AuthenticationError),
            api_error(429, anthropic.RateLimitError),
            api_error(500, anthropic.InternalServerError),
            anthropic.APITimeoutError(request=REQUEST),
            anthropic.APIConnectionError(request=REQUEST),
            TypeError("missing authentication credentials"),
        ],
    )
    def test_APIが失敗したら頻出語に戻す(
        self, fake_client, df, representatives, top_words, error
    ):
        fake_client.error = error
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is False
        assert names == fallback_names(top_words)

    def test_認証以外のTypeErrorは投げ直す(self, fake_client, df, representatives, top_words):
        fake_client.error = TypeError("引数の型が違う")
        with pytest.raises(TypeError, match="引数の型が違う"):
            generate_cluster_names(df, representatives, top_words)

    def test_APIが拒否したら頻出語に戻す(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "猫の本")], stop_reason="refusal")
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is False
        assert names == fallback_names(top_words)


class TestGenerateClusterNamesSuccess:
    def test_返ってきた見出しを使う(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "動物をめぐる物語"), (1, "宇宙への旅")])
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is True
        assert names == {0: "動物をめぐる物語", 1: "宇宙への旅"}

    def test_足りない分は頻出語で補う(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "動物をめぐる物語")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == "動物をめぐる物語"
        assert names[1] == fallback_names(top_words)[1]

    def test_存在しないクラスタIDは無視する(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "動物をめぐる物語"), (99, "知らないクラスタ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert set(names) == {0, 1}

    def test_空の見出しは採用しない(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "   ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == fallback_names(top_words)[0]

    def test_前後の空白を落とす(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "  動物をめぐる物語  ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == "動物をめぐる物語"

    def test_指定したモデルを渡す(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "見出し")])
        generate_cluster_names(df, representatives, top_words, model="claude-sonnet-5")
        assert fake_client.calls[0]["model"] == "claude-sonnet-5"

    def test_構造化出力を指定する(self, fake_client, df, representatives, top_words):
        fake_client.result = FakeResponse([(0, "見出し")])
        generate_cluster_names(df, representatives, top_words)
        assert fake_client.calls[0]["output_format"] is ClusterTitles
