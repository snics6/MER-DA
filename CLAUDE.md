# MER-DA — 少数ターゲットラベル下のマルチモーダル感情認識（MER）転移

新しいコーパス（ターゲット）にラベルが**クラスあたり K 件**しかないとき，
ほかのコーパス（ソース）のラベル付きデータを混ぜると成績がどう変わるかを測り，
得と損が生じる仕組みを，ドメイン適応（DA）理論と対応づけた数式モデルで説明する研究である．
**分析の軸は得の仕組みの仮説 MH1〜MH5**（`docs/researchvision.md` 3 節）．その上に，仕組みの式（E15〜E17）と条件の探索（E18〜）を積んでいる．**解析は，合併の C をソースのみの C に固定した地図（`results/fixc_map/`）を土台にし，論文と発表資料の本文もこの地図だけで書く**（交差検証の地図は付録．同 6.2 節）．
論文は分析と手法を 1 本に統合する（2026-10-22 決定．数理を前面に）．2026-10-28 に，今の版では IEEE TAC に出さず，教師なしの DA としても半教師ありの DA としても数学的に裏打ちされた手法の新しさを目指して実験を増やすと決めた（`docs/researchvision.md` 6.1 節）．2026-10-31 に投稿先を Discover Artificial Intelligence に決め，論文は `DAI_paper/` で書くことにした（規則は `DAI_paper/CLAUDE.md`）．TAC 向けの旧版の `paper/` の内容には従わない．2026-11-04 に，Albanie ほか 2018 とその後続研究を土台に研究を組み直すと決め，設計と実験は `DAI/` で行うことにした（規則は `DAI/CLAUDE.md`，計画は `DAI/docs/計画_2026-11-04.pdf`）．
各主張の最新の状態と実験の一覧は `docs/results.md`（3.6 節），分析の層と次の実験の計画は `docs/researchvision.md` 3.6・3.7 節．

**このディレクトリは自己完結させる．`~/MER` など他のディレクトリは一切参照しない**
（コード・環境・文書・設定値のどれも）．要るものはここに置く．

---

## 正本と優先順位

文書は役割ごとに 8 本ある（2026-10-08 に統合，2026-10-28 に台帳を足した．`CLAUDE.md` と `data/README.md` 以外は `docs/` の下）．

| 対象 | 正本 |
|---|---|
| 研究の目的・分析の軸・論文の計画・決定事項・未決事項・更新記録・事前登録した仮説 | **`docs/researchvision.md`** |
| 実験の手順（地図，学び直し，ずれの指標，操作，統計，理論からの予言の導出，実験の検査の関門と不備の目録，新しい定理を探す手順） | **`docs/methods.md`** |
| データの扱い・発話一覧・特徴の作り方・重み・作り直し方 | **`data/README.md`** |
| 各主張の最新の状態，解析の本筋と端，結果の場所と作り方（執筆・発表の入口） | **`docs/results.md`**（進み具合は `results/status.md`） |
| 探索的な解析の方法・数値・経緯（日付順の記録） | `docs/exploratory-analyses.md` |
| 設計の判断のために見たデータの記録と，確認に使えるデータ | `docs/seen-data.md` |
| 先行研究と論文の位置づけ（**論文と発表資料の新しさ・関連研究・限界の書き方はここに合わせる**） | `docs/related-work.md` |

**食い違ったら，データの扱いは `data/README.md`，それ以外は `docs/researchvision.md` が勝つ．**
食い違いを見つけたら，勝った側に合わせて負けた側を直す（放置しない）．古い文は残さずその場で書き直す（前の版は git にある）．
本ファイルは要点と取り決めだけを持つ．**未決事項は期限つきで `docs/researchvision.md` 7 節に書き，ここには置かない．**方針を変えたら `docs/researchvision.md` の決定事項を直し，更新記録に 1 行足す．

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
映像は RAVDESS に無く，言葉は RAVDESS・CREMA-D に無い．**AFEW の言葉だけは人ではなく Whisper large-v3 の書き起こしから作る**（2026-11-04〜．`data/README.md` 3 節）．

---

## ディレクトリ

| 場所 | 中身 | git |
|---|---|---|
| `docs/` | 文書 6 本（研究計画 `researchvision.md`，方法，結果，探索の記録，先行研究，見たデータの台帳 `seen-data.md`）と論文の主張契約の下書き `paper-claim.md`．データの文書は `data/README.md` | ○ |
| `.claude/agents/` | 実験の検査役の定義（関門 1 の `gate-stats`・`gate-leak`・`gate-compare`・`gate-define`・`gate-probe`・`gate-code`，関門 2 の `gate-claim`．手順は `docs/methods.md` 9 節）と，新しい定理を探す担当の定義（`thm-unexplained`・`thm-open-problems`・`thm-claim`・`thm-screen`・`thm-deep`・`thm-check`．手順は同 10 節）．2026-10-28〜 | ○ |
| `checks/` | 検査役の報告（`checks/<E 番号>/<役>.md`） | ○ |
| `A theory of learning from different domains.pdf` | 中核の理論の論文（Ben-David 2010） | ○ |
| `data/metadata/` | 発話一覧 6 本と MOSEI のチャンネル対応表（ラベルと話者の正本），AFEW の Whisper の書き起こし `afew_transcript_whisper.csv` | ○ |
| `data/cache/*.npz` | 特徴 30 本（1.1 GB）．消えたら `run/build_data.sh` で作り直す | × |
| `data/cache/faces_*.csv.gz` | 顔の座標．**消さない** | ○ |
| `features/` | 発話一覧と特徴を作るコード | ○ |
| `measure/` | 読み込み（`data.py`）・話者分割（`split.py`）・ずれの指標（`shift.py`）・ずれを消す操作（`ablate.py`）・その検算（`test_shift.py`）・監査 | ○ |
| `exp/` | 実験の台本．地図の実験（`map.py`），GPU で解くロジスティック回帰（`logreg.py`），途中結果のまとめ（`summarize_map.py`），進み具合（`results_status.py`），ずれの指標を全組で測る（`shift_map.py`），ずれを消す・足す操作（`ops.py`），仕組みの仮説の学び直し（`mech.py`），本筋の探索的な解析（`explore_*.py`．共通の道具は `pair_table.py`，まとめて流すのは `rerun_main.py`．対応は `docs/results.md` 4.5），HuBERT での MH の判定（`check_mh.py`），人工データでの理論の検算（`sim_position.py`・`sim_direction.py`．E15・E16），合併を地図の C で解き直してテストの点数を残す（`merged_scores.py`．E15），手の比較と複数ソース（`interventions.py`・`multisource.py`．E25・E26），仮説転移の学び直しと罰則・切片の格子（`transfer_scores.py`・`transfer_grid.py`），全ソースの合併の点数（`multisource_scores.py`），手法の数理の確かめ（`explore_interventions.py`・`explore_gate.py`・`explore_transfer_theory.py`・`explore_variance.py`・`explore_newton_steps.py`・`explore_uar_link.py`・`explore_label_shift.py`・`explore_room.py`・`explore_multisource_lambda.py`・`explore_selection.py`．E25〜E28），端の解析（`side/`．E1・E3〜E7・E14・H7），検算（`test_map.py`・`test_summarize.py`）．地図の結果は `results/map_<エンコーダ>/<ターゲット>.csv` に反復ごとに追記 | ○ |
| `run/build_data.sh` | データの作り直し台本 | ○ |
| `weights/` | 顔検出と表情モデルの重み | × |
| `results/` | 結果（再計算できる前提．各ファイルに設定とコミット番号を書く） | × |
| `logs/` | 長時間処理のログ | × |
| `DAI_paper/` | Discover Artificial Intelligence に出す論文の執筆フォルダ（2026-10-31〜）．規則は `DAI_paper/CLAUDE.md` | ○ |
| `DAI/` | Albanie 型の研究を土台に組み直す作業場（2026-11-04〜）．規則 `DAI/CLAUDE.md`，計画 `DAI/docs/`，検査の報告 `DAI/checks/`，サブエージェントの写し `DAI/.claude/agents/` | ○（LaTeX の中間生成物は ×） |
| `paper/` | TAC 向けの旧版の執筆フォルダ（2026-10-27〜10-28．2026-10-31 からは従わない）．主張契約 `CLAIM.md`，規則 `CLAUDE.md`，数値台帳 `results/numbers.json`，実験要求 `requests.md`．IEEEtran を同梱．`make check` で検査 | ○（PDF と中間生成物は ×） |
| `slides/` | 発表資料（Beamer，LuaLaTeX．2026-10-27〜）．資料ごとに 1 フォルダ．数値は論文の台帳の `\NUM{key}`，図の値は資料の `scripts/export_figdata.py` が台帳と照合して書き出す．見た目の設定は `~/slides` から 1 回だけ写した（以後は読まない）．規則 `slides/CLAUDE.md` | ○（PDF と中間生成物は ×） |
| `research_notes/`，`reports/` | 調べ直しのノートと報告（2026-10-27：「TAC 論文の先行研究と書き方」と「TAC 論文 先行研究の精読と穴」．要点は `docs/related-work.md` 1.6・1.13・1.14 節） | ○ |
| `.venv/` | Python 環境（`requirements.txt` の冒頭に作り方） | × |

**Python 環境は `.venv/` の 1 つだけ．**版は `requirements.txt` で固定する（scikit-learn の版違いで結果がずれうる）．

---

## 進め方

- **研究の流れは 6 つの問い（2026-10-19 の面談の資料の 5 つの問いに問い 6 を足したもの．`docs/researchvision.md` 3.0 節）で書き，語はスライドの語に合わせる**（対応表は `docs/results.md` 3.6 節末）．研究者への報告もこの流れと語で行う
- **実験コードを書く前に `docs/methods.md` を読む**
- **実験は関門を通してから計算し，結果を書く**（`docs/methods.md` 9 節．2026-10-28〜）．設計を書いたら `docs/seen-data.md` に 1 行足し，本番は関門 1 の 4 体，下調べは `gate-probe` の 1 体に検査させる．**本番でも下調べでも，台本を書いたら計算の前に必ず `gate-code` に台本を検査させる**（2026-11-01〜．目的は計算の後の指摘で計算をやり直さないこと．台本が設計どおりに計算するかに加え，後から求められそうな解析を保存した出力から出せるかを確かめさせる．台本を直したら差分を再検査させ，計算の後には結果ファイルの欠けと重複を点検させる）．結果を文書に書く前に `gate-claim` に照合させる．検査役には設計と台本と台帳だけを渡し，期待する結論は渡さない．**`DAI/` の実験は，さらに ML と感情認識の査読想定の役（`dai-review-ml`・`dai-review-affect`）の両方の OK を計算の前に得る**（2026-11-04〜．`DAI/CLAUDE.md`）
- **ずれの指標は `measure/shift.py`，ずれを消す操作は `measure/ablate.py` を必ず通す．**実験コードで書き直さない．例外は学び直し（E12）の `exp/mech.py`（`docs/methods.md` 3.4）．変えたら `measure/test_shift.py` を流し，`docs/methods.md` も直す
- **長時間の処理（特徴抽出・学習・評価）は勝手に起動しない．**
  まずコマンド・所要見込み・上書きするファイルを示して確認を取る
- 長時間処理は **tmux セッション `mer-da`** の中で回す
- git の remote は `origin`（github.com/snics6/MER-DA，非公開．SSH 鍵 `~/.ssh/github_ed25519`）．**コミットしたら `git push` する**（2026-10-27 から）
- **数値の主張は，ログか出力ファイルを示してから述べる**
- **事前登録した仮説（`docs/researchvision.md` 8 節）は結果を見てから書き換えない**
- ハイパーパラメータはテストラベルで選ばない．実用値とオラクル値を分けて報告する
- このファイルは短く保つ（150 行以内）．増えたら `docs/` に分ける
