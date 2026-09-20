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
    def test_joins_three_top_words(self):
        """頻出語を3つつなぐ。"""
        assert fallback_names({0: ["猫", "犬", "鳥", "馬"]}) == {0: "猫・犬・鳥"}

    def test_handles_fewer_than_three_words(self):
        """語が3つ未満でも作れる。"""
        assert fallback_names({0: ["猫"]}) == {0: "猫"}

    def test_falls_back_to_cluster_id(self):
        """語が無ければクラスタIDを使う。"""
        assert fallback_names({3: []}) == {3: "クラスタ3"}


class TestBuildPrompt:
    def test_includes_heading_per_cluster(self, df, representatives, top_words):
        """クラスタごとの見出しが入る。"""
        prompt = build_prompt(df, representatives, top_words)
        assert "## グループ 0（全2冊）" in prompt
        assert "## グループ 1（全1冊）" in prompt

    def test_includes_top_words_and_titles(self, df, representatives, top_words):
        """頻出語とタイトルが入る。"""
        prompt = build_prompt(df, representatives, top_words)
        assert "頻出語: 猫, 犬, 動物" in prompt
        assert "『本A』" in prompt

    def test_limits_books_per_cluster(self):
        """渡す本は BOOKS_PER_CLUSTER まで。"""
        df = pd.DataFrame(
            {
                "クラスタID": [0] * 20,
                "タイトル": [f"本{i}" for i in range(20)],
                "説明文": ["説明"] * 20,
            }
        )
        prompt = build_prompt(df, {0: list(range(20))}, {0: ["語"]})
        assert prompt.count("- 『") == BOOKS_PER_CLUSTER

    def test_truncates_descriptions(self):
        """説明文は先頭だけ渡す。"""
        long_text = "あ" * (DESCRIPTION_CHARS + 100)
        df = pd.DataFrame({"クラスタID": [0], "タイトル": ["本A"], "説明文": [long_text]})
        prompt = build_prompt(df, {0: [0]}, {0: ["語"]})
        assert "あ" * DESCRIPTION_CHARS in prompt
        assert "あ" * (DESCRIPTION_CHARS + 1) not in prompt


class TestGenerateClusterNamesFallback:
    """APIが使えない場合、頻出語の名前と by_ai=False を返す。"""

    def test_falls_back_when_anthropic_missing(self, monkeypatch, df, representatives, top_words):
        """anthropic が入っていない。"""
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
    def test_falls_back_on_api_error(self, fake_client, df, representatives, top_words, error):
        """APIが失敗したら頻出語に戻す。"""
        fake_client.error = error
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is False
        assert names == fallback_names(top_words)

    def test_reraises_unrelated_type_error(self, fake_client, df, representatives, top_words):
        """認証以外の TypeError は投げ直す。"""
        fake_client.error = TypeError("引数の型が違う")
        with pytest.raises(TypeError, match="引数の型が違う"):
            generate_cluster_names(df, representatives, top_words)

    def test_falls_back_on_refusal(self, fake_client, df, representatives, top_words):
        """APIが拒否したら頻出語に戻す。"""
        fake_client.result = FakeResponse([(0, "猫の本")], stop_reason="refusal")
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is False
        assert names == fallback_names(top_words)


class TestGenerateClusterNamesSuccess:
    def test_uses_returned_titles(self, fake_client, df, representatives, top_words):
        """返ってきた見出しを使う。"""
        fake_client.result = FakeResponse([(0, "動物をめぐる物語"), (1, "宇宙への旅")])
        names, by_ai = generate_cluster_names(df, representatives, top_words)
        assert by_ai is True
        assert names == {0: "動物をめぐる物語", 1: "宇宙への旅"}

    def test_fills_missing_titles_with_fallback(self, fake_client, df, representatives, top_words):
        """足りない分は頻出語で補う。"""
        fake_client.result = FakeResponse([(0, "動物をめぐる物語")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == "動物をめぐる物語"
        assert names[1] == fallback_names(top_words)[1]

    def test_ignores_unknown_cluster_id(self, fake_client, df, representatives, top_words):
        """存在しないクラスタIDは無視する。"""
        fake_client.result = FakeResponse([(0, "動物をめぐる物語"), (99, "知らないクラスタ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert set(names) == {0, 1}

    def test_rejects_blank_title(self, fake_client, df, representatives, top_words):
        """空の見出しは採用しない。"""
        fake_client.result = FakeResponse([(0, "   ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == fallback_names(top_words)[0]

    def test_strips_surrounding_whitespace(self, fake_client, df, representatives, top_words):
        """前後の空白を落とす。"""
        fake_client.result = FakeResponse([(0, "  動物をめぐる物語  ")])
        names, _ = generate_cluster_names(df, representatives, top_words)
        assert names[0] == "動物をめぐる物語"

    def test_passes_requested_model(self, fake_client, df, representatives, top_words):
        """指定したモデルを渡す。"""
        fake_client.result = FakeResponse([(0, "見出し")])
        generate_cluster_names(df, representatives, top_words, model="claude-sonnet-5")
        assert fake_client.calls[0]["model"] == "claude-sonnet-5"

    def test_requests_structured_output(self, fake_client, df, representatives, top_words):
        """構造化出力を指定する。"""
        fake_client.result = FakeResponse([(0, "見出し")])
        generate_cluster_names(df, representatives, top_words)
        assert fake_client.calls[0]["output_format"] is ClusterTitles
