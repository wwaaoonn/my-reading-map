"""matplotlibの日本語フォント設定。

matplotlib_fontja を優先し、無ければ japanize_matplotlib を使う。
"""

import warnings


def setup_japanese_font():
    """日本語フォントを有効にする。どちらも入っていなければ警告のみ。"""
    try:
        import matplotlib_fontja  # noqa: F401
        return "matplotlib_fontja"
    except ImportError:
        pass

    try:
        import japanize_matplotlib  # noqa: F401
        return "japanize_matplotlib"
    except ImportError:
        warnings.warn(
            "日本語フォントが設定できません。"
            "`pip install matplotlib-fontja` を実行してください（図の日本語が□になります）。",
            stacklevel=2,
        )
        return None
