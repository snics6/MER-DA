# MER-DA — 少数ターゲットラベル下のマルチモーダル感情認識（MER）転移

新しいコーパス（ターゲット）にラベルが**クラスあたり K 件**しかないとき，
ほかのコーパス（ソース）のラベル付きデータがどれだけ役立つかを，
ドメイン適応（DA）理論の各項と対応づけて**測る**研究である．
DA とメタ学習はどちらも解法候補であり，どちらを採るかはデータで決める．

**このディレクトリは自己完結させる．`~/MER` など他のディレクトリは一切参照しない**
（コード・環境・文書・設定値のどれも）．要るものはここに置く．

---

## 正本と優先順位

| 対象 | 正本 |
|---|---|
| 研究の目的・理論・実験計画・仮説・未決事項 | **`researchvision.md`**（ここを磨いていく） |
| データの中身・扱い方・作り直し方 | **`data/README.md`**（発話一覧の詳細は `data/metadata/README.md`） |
| 特徴の作り方 | `docs/feature-spec.md` |
| 実験の方針・決定事項・未決事項（常に更新．更新記録つき） | **`docs/experiment-policy.md`** |
| 実験の手順（条件・分割・ずれを消す操作） | `docs/experiment-protocol.md` |
| ずれの指標の定義と注意 | `docs/shift-metrics.md` |
| 先行研究の要点 | `docs/related-work.md` |
| 結果の場所・作り方・使ってよい状態（執筆・発表の入口） | `docs/results-index.md`（進み具合は `results/status.md`） |
| 探索的な解析の方法・数値・限界（事前登録の判定とは別） | `docs/exploratory-analyses.md` |

**食い違ったら，データの扱いは `data/README.md`，それ以外は `researchvision.md` が勝つ．**
食い違いを見つけたら，勝った側に合わせて負けた側を直す（放置しない）．
本ファイルは要点と取り決めだけを持つ．**未決事項は期限つきで，研究全体のものは `researchvision.md`，実験のものは `docs/experiment-policy.md` に書き，ここには置かない．**方針を変えたら `docs/experiment-policy.md` の更新記録に 1 行足す．

中核の理論：`A theory of learning from different domains.pdf`（Ben-David et al., 2010．Theorem 3）

---

## データの取り決め（破ると測定が無効になる）

根拠は `data/README.md`．読み込みは `measure/data.py` の `load_group(群, エンコーダ)` が 1〜6 を引き受ける．
**特徴は必ずこれを通して読む．`np.load` した `X` をそのまま使わない．**

1. **ラベルは `data/metadata/*.csv` から貼り直す．**npz の `label_4` は使わない
2. **`sample_id` で並べ直す．**行番号で標本を選ばない
3. `ok` が偽の行と，NaN・Inf を含む行は落とす
4. **映像の `X_std` は使わない．**発話の長さの代理変数になっている
5. **話者は必ず分ける．**同じ話者が学習側と評価側の両方に現れてはならない（`measure/split.py`）
6. **IEMOCAP は即興と台本を別の群として扱う**
7. **K はクラスあたりの本数．**K=25 なら 4 クラス × 25 = 100 本
8. **ラベルは 4 つ**（0=Neutral 1=Happy 2=Sad 3=Angry），**主指標は UAR**（でたらめなら 25%）
9. **エンコーダは微調整していないものだけ**

群は `IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D` の 7 つ．
映像は RAVDESS に無く，言葉は AFEW・RAVDESS・CREMA-D に無い．

---

## ディレクトリ

| 場所 | 中身 | git |
|---|---|---|
| `researchvision.md` | 研究計画（正本） | ○ |
| `data/metadata/` | 発話一覧 7 本（ラベルと話者の正本） | ○ |
| `data/cache/*.npz` | 特徴 28 本（1.1 GB）．消えたら `run/build_data.sh` で作り直す | × |
| `data/cache/faces_*.csv.gz` | 顔の座標．**消さない** | ○ |
| `features/` | 発話一覧と特徴を作るコード | ○ |
| `measure/` | 読み込み（`data.py`）・話者分割（`split.py`）・ずれの指標（`shift.py`）・ずれを消す操作（`ablate.py`）・その検算（`test_shift.py`）・監査 | ○ |
| `exp/` | 実験の台本．地図の実験（`map.py`），GPU で解くロジスティック回帰（`logreg.py`），途中結果のまとめ（`summarize_map.py`），進み具合（`results_status.py`），検算（`test_map.py`・`test_summarize.py`）．結果は `results/map_<エンコーダ>/<ターゲット>.csv` に反復ごとに追記 | ○ |
| `run/build_data.sh` | データの作り直し台本 | ○ |
| `weights/` | 顔検出と表情モデルの重み | × |
| `.venv/` | Python 環境（`requirements.txt` の冒頭に作り方） | × |

**Python 環境は `.venv/` の 1 つだけ．**版は `requirements.txt` で固定する（scikit-learn の版違いで結果がずれうる）．

---

## 進め方

- **実験コードを書く前に `docs/experiment-protocol.md` を読む**
- **ずれの指標は `measure/shift.py`，ずれを消す操作は `measure/ablate.py` を必ず通す．**実験コードで書き直さない．変えたら `measure/test_shift.py` を流し，根拠の md（`docs/shift-metrics.md`・`docs/experiment-protocol.md`）も直す
- **長時間の処理（特徴抽出・学習・評価）は勝手に起動しない．**
  まずコマンド・所要見込み・上書きするファイルを示して確認を取る
- 長時間処理は **tmux セッション `mer-da`** の中で回す
- **数値の主張は，ログか出力ファイルを示してから述べる**
- **仮説（`researchvision.md` の事前登録）は結果を見てから書き換えない**
- ハイパーパラメータはテストラベルで選ばない．実用値とオラクル値を分けて報告する
- このファイルは短く保つ（150 行以内）．増えたら `docs/` に分ける
