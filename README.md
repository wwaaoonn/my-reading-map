# My読書マップ

![読書マップ](docs/reading_map.png)

読んだ本の「説明文」を文ベクトルに変換し、内容が近い本どうしを集めて、自分の読書傾向を
1枚の地図として描くツールです。点が1冊、色が内容のまとまり（クラスタ）、楕円がその
まとまりの領域を表します。生成AIが本の説明文を読んで、まとまりの名前を付けます。

## 使い方と環境

```bash
pip install -r requirements.txt
python main.py data/sample_reading_log.csv
```

`タイトル` と `説明文` の列があるCSVを渡せば、そのまま動きます。設定ファイルはありません。

```bash
python main.py ~/Documents/読書記録.csv
```

クラスタ名の生成にはClaude APIを使います。APIキーを設定しておくと生成AIが命名し、
無ければ頻出語をつないだ名前で代替します（処理は止まりません）。命名のときは、
クラスタごとに8冊分の説明文をAPIに送ります。`--no-ai-names` を付けるとAPIを呼ばず、
処理はすべて手元で完結します。

キーはリポジトリ直下の `.env` に書きます。`.env` は `.gitignore` 済みです。

```bash
cp .env.example .env     # ANTHROPIC_API_KEY=sk-ant-... を記入する
```

環境変数 `ANTHROPIC_API_KEY` でも動きます。両方ある場合は環境変数が優先されます。

APIに繋がるかどうかだけを先に確かめられます。接続先と失敗の原因を表示します。

```bash
python -m src.name_clusters
```

オプションはすべて省略できます。

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--k N` | 自動 | クラスタ数を指定する（2以上・冊数以下）。省略時はデータから決める |
| `--out DIR` | `outputs` | 出力先ディレクトリ |
| `--no-ai-names` | | 生成AIを使わず、頻出語からクラスタ名を作る（説明文を外部に送らない） |
| `--model` | `claude-opus-5` | 命名に使うモデル |
| `--perplexity N` | 自動 | t-SNEのperplexity。省略すると冊数から決める |
| `--embed-model` | MiniLM | 文埋め込みモデルを差し替える |
| `--force-embed` | | キャッシュを無視してベクトル化をやり直す |

出力されるもの（`outputs/`）:

| ファイル | 中身 |
| --- | --- |
| `reading_map.png` | 全体マップ。クラスタを色分けし、散らばりを楕円で示す |
| `cluster_0.png` … | クラスタ別マップ。1つだけ色を付け、中心的な本のタイトルを表示 |
| `cluster_summary.csv` | クラスタ名・冊数・代表本・頻出語の一覧 |
| `clustered_books.csv` | クラスタIDとt-SNE座標を付けた全データ |
| `clustered_books_public.csv` | 上記から説明文を除いたもの（結果を共有するとき用） |
| `k_selection.png` / `.csv` | クラスタ数の検討に使った指標。`--k` を指定したときは作りません |
| `similar_pairs.csv` | 説明文の類似度が0.5を超える本のペア |
| `embeddings.npy` | ベクトルのキャッシュ |
| `embeddings.json` | キャッシュのメタ情報。モデル名・次元数・件数・説明文のハッシュ |

動作環境は Python 3.12.2（macOS）で確認しています。初回実行時に文埋め込みモデル
（約500MB）のダウンロードが走ります。

## ディレクトリ構成

```text
├── main.py                  実行の入口（python main.py <CSV>）
├── src/
│   ├── embed.py             CSVの読み込みと説明文のベクトル化
│   ├── cluster.py           クラスタ数の決定とKMeans
│   ├── label.py             頻出語の抽出（Janome + TF-IDF）
│   ├── name_clusters.py     クラスタ名の生成（Claude API）
│   ├── visualize.py         t-SNEと各種マップの描画
│   └── plot_style.py        日本語フォントの設定
├── data/
│   ├── sample_reading_log.csv   パブリックドメイン作品20冊のサンプル
│   └── README.md                CSVの仕様
├── tests/                   pytestのテスト（APIと埋め込みモデルは呼ばない）
├── requirements.txt         main.py の実行に必要なすべて
├── requirements-core.txt    埋め込みモデル以外の依存
├── requirements-dev.txt     テストとlintに必要な依存
├── pyproject.toml           ruffとpytestの設定
└── docs/                    READMEのトップに貼る図
```

## 開発

テストとlintは、埋め込みモデルのダウンロードもAPI呼び出しもせずに動きます。
`requirements-dev.txt` には sentence-transformers と torch を含めていません。

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

push と pull request のたびに、GitHub Actions で同じ2つを実行します
（[.github/workflows/test.yml](.github/workflows/test.yml)）。

## 作成フロー

ステップごとに使うプログラム・手法・目的は次のとおりです。

| ステップ | プログラム | 手法 | 目的 |
| --- | --- | --- | --- |
| 1. 読み込み | `src/embed.py` | pandasでCSVを読み、必須列を検査する | 説明文を処理できる形にそろえる |
| 2. ベクトル化 | `src/embed.py` | Sentence-BERTで384次元に変換し、L2正規化する | 説明文の内容どうしを数値で比べられるようにする |
| 3. クラスタ数の決定 | `src/cluster.py` | エルボー法とシルエット係数 | 冊数に合ったまとまりの数を決める |
| 4. クラスタリング | `src/cluster.py` | KMeans（384次元空間、乱数固定） | 内容が近い本を同じグループにする |
| 5. クラスタ名の生成 | `src/label.py`／`src/name_clusters.py` | Janome＋TF-IDFで頻出語、Claude APIで命名 | 図に載せる見出しを作る |
| 6. 2次元化と描画 | `src/visualize.py`／`src/plot_style.py` | t-SNE（コサイン）、matplotlib、adjustText | 地図として1枚の画像にする |

グループ分けはステップ4までに384次元空間で完了します。ステップ6は、その結果を
人が見て分かる形にするための描画です。

### 1. 読書記録CSVの読み込み

`src/embed.py` の `load_reading_log` が担当します。必要な列は `タイトル` と `説明文` の
2つで、3冊以上必要です。条件を満たさなければその場で止まります。

説明文の欠損は空文字として扱い、件数を表示して処理を続けます。ベクトル化の対象は
説明文だけで、タイトルや著者は図のラベルに使います。

### 2. 説明文のベクトル化

`src/embed.py` の `build_embeddings` が担当します。Sentence-BERT
（`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`）で説明文1件を
384次元のベクトルに変換し、長さ1に正規化します。

ベクトルは `outputs/embeddings.npy` にキャッシュします。再利用するのは、冊数・埋め込み
モデル・説明文の内容が前回と一致する場合だけです。判定には `embeddings.json`
（モデル名・次元数・件数・説明文のSHA-256）を使い、一致しない項目があればその理由を
表示して作り直します。

### 3. クラスタ数kの決定

`src/cluster.py` の `evaluate_k` と `suggest_k` が担当します。kを2から14まで
（冊数が少ないときは `冊数 - 1` まで）変えてクラスタリングし、エルボー法と
シルエット係数を計算します。エルボーは、inertiaの曲線と両端を結ぶ直線との距離が
最大になるkです。

エルボー法の値を軸に選び、`冊数 ÷ 5`（小数点以下切り捨て）と試したkの最大値（14）の
小さいほうを上限、2を下限とします。上限は20冊で4、50冊で10、100冊で14です。これはk
の上限で、各クラスタの冊数は拘束しません。対立する候補は両方表示し、`--k`（2以上・
冊数以下）で上書きできます。シルエット係数（コサイン距離）がどのkでも0.25を下回る
場合は、その旨を実行時に表示します。

### 4. KMeansでクラスタリング

`src/cluster.py` の `fit_kmeans` と `silhouette_for_labels` が担当します。決まったkで
KMeansを実行し、各本にクラスタIDを割り当てます。計算は384次元の空間、距離はユークリッド
距離、初期値の選び方は `n_init="auto"`、乱数は固定です。

割り当て後、採用したkでのシルエット係数（コサイン距離）を計算して表示します。0.25を
下回る場合は、分離が弱い旨をあわせて表示します。`--k` を指定した場合はステップ3を
通らず、この値だけが分離の指標になります。

### 5. クラスタ名の生成

`src/label.py`（頻出語）と `src/name_clusters.py`（命名）が担当します。Janomeで説明文を
形態素解析し、名詞・動詞・形容詞の原形だけを取り出します（非自立・接尾・代名詞・数は
落とします）。1クラスタの説明文をつないだものを1文書とし、全クラスタを1つのコーパスと
してTF-IDFを学習して、クラスタごとの上位語を取ります。

次に、クラスタの重心に近い本8冊の説明文と頻出語をClaude APIに渡し、
共通するテーマを表す短い見出しを作らせます。返り値はPydanticのモデルで構造化出力として
受け取ります。APIキーが無い場合やAPIに繋がらない場合は、頻出語をつないだ名前に
フォールバックします。

### 6. t-SNEで2次元化して描画

`src/visualize.py` が担当します。t-SNEで384次元のベクトルを2次元に落とします。距離はコサインを使い、perplexityは冊数から決めます（`min(30, (冊数 - 1) // 3)`、下限5）。`--perplexity` でも指定できます。描くのは全体マップ（クラスタごとに色と楕円、名前は重なりを自動回避して配置）と、クラスタ別マップ（1つだけ色を付け、中心的な本のタイトルを表示）の2種類です。

全体マップ上の楕円はクラスタの領域の目安で、統計的な信頼区間ではありません。

t-SNEの座標は再計算すると配置が変わり、クラスタ名ラベルの位置も描き直すたびに変わります。同じ図を使い続ける場合は、生成された画像と `clustered_books.csv` の座標を保存してください。どの本が同じクラスタに入るかは、再実行しても変わりません。

## ライセンス

コードは MIT License（[LICENSE](LICENSE)）です。

`data/sample_reading_log.csv` の説明文は、パブリックドメイン作品について本リポジトリの
著者が書き下ろしたものです。著者自身の読書記録はこのリポジトリに含めていません。
