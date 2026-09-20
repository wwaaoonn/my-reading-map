# -*- coding: utf-8 -*-
"""README に貼る「作成フロー図」を生成するスクリプト。

使い方:
    python docs/make_flow_diagram.py
引数なしで実行すると、このファイルと同じディレクトリに flow.png を書き出す。
依存は matplotlib と matplotlib_fontja（日本語フォント）のみ。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib_fontja  # noqa: F401  日本語フォントを有効化（import するだけで適用）
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- 配色
INK = "#1F2933"          # 見出しの文字色
SUB = "#6B7682"          # 補足行の文字色
ACCENT = "#2F5D8C"       # アクセント（枠・番号）: 落ち着いた青
ACCENT_ALT = "#7A5E8F"   # 外部モデル/APIを使う段のアクセント: 落ち着いた紫
FACE = "#FFFFFF"         # 通常の箱の地色
FACE_ALT = "#F6F3F9"     # 外部モデル/APIを使う段の地色
GRID = "#FAFBFC"         # 背景（ほぼ白）

# ---------------------------------------------------------------- 図の内容
STEPS = [
    ("読書記録CSVの読み込み", "タイトルと説明文", False),
    ("説明文のベクトル化", "Sentence-BERT・384次元", True),
    ("クラスタ数kの決定", "エルボー法・シルエット係数", False),
    ("KMeansでクラスタリング", "内容が近い本をグループ化", False),
    ("クラスタ名の生成", "Claude APIが説明文から命名", True),
    ("t-SNEで2次元化して描画", "読書マップを出力", False),
]

# ---------------------------------------------------------------- 寸法（単位: インチ）
FIG_W = 8.2                  # dpi=200 で横 1640 px
BOX_W = 7.1                  # 箱の横幅
BOX_H = 0.88                 # 箱の高さ
GAP = 0.30                   # 箱と箱のすき間（ここに矢印を描く）
TOP_PAD = 0.50               # 上余白
BOTTOM_PAD = 0.62            # 下余白（注記のぶんを含む）
DPI = 200

# 箱の積み上がりから図の高さを決める（はみ出し防止）
FIG_H = TOP_PAD + len(STEPS) * BOX_H + (len(STEPS) - 1) * GAP + BOTTOM_PAD


def build() -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=GRID)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    ax.set_facecolor(GRID)

    x0 = (FIG_W - BOX_W) / 2.0
    top = FIG_H - TOP_PAD

    for i, (title, note, is_ext) in enumerate(STEPS):
        edge = ACCENT_ALT if is_ext else ACCENT
        face = FACE_ALT if is_ext else FACE

        y_top = top - i * (BOX_H + GAP)
        y_bot = y_top - BOX_H
        y_mid = (y_top + y_bot) / 2.0

        ax.add_patch(
            FancyBboxPatch(
                (x0, y_bot),
                BOX_W,
                BOX_H,
                boxstyle="round,pad=0,rounding_size=0.14",
                linewidth=1.5,
                edgecolor=edge,
                facecolor=face,
                mutation_aspect=1,
                zorder=2,
            )
        )

        # 番号（左端）
        ax.text(
            x0 + 0.46,
            y_mid,
            str(i + 1),
            color=edge,
            fontsize=23,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=3,
        )
        # 番号と本文を分ける細い縦線
        ax.plot(
            [x0 + 0.90, x0 + 0.90],
            [y_bot + 0.17, y_top - 0.17],
            color=edge,
            linewidth=0.8,
            alpha=0.45,
            zorder=3,
        )

        # 見出し（大きめ・太字）
        ax.text(
            x0 + 1.18,
            y_mid + 0.19,
            title,
            color=INK,
            fontsize=17.5,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=3,
        )
        # 補足の一行（小さめ・グレー）
        ax.text(
            x0 + 1.18,
            y_mid - 0.21,
            note,
            color=SUB,
            fontsize=12.5,
            ha="left",
            va="center",
            zorder=3,
        )

        # 次の箱へつなぐ矢印（すき間の中だけに収める）
        if i < len(STEPS) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (FIG_W / 2.0, y_bot - 0.045),
                    (FIG_W / 2.0, y_bot - GAP + 0.045),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.2,
                    color="#9AA5B1",
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                )
            )

    # 淡い色の箱についての注記
    last_bottom = top - (len(STEPS) - 1) * (BOX_H + GAP) - BOX_H
    ax.text(
        FIG_W / 2.0,
        last_bottom / 2.0,
        "淡い紫の箱は外部モデル・APIを利用する段",
        color=SUB,
        fontsize=11.5,
        ha="center",
        va="center",
    )
    return fig


def main() -> None:
    out = Path(__file__).resolve().parent / "flow.png"
    fig = build()
    fig.savefig(out, dpi=DPI, facecolor=GRID, transparent=False)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
