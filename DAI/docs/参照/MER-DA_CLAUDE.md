# MER-DA — 少数ターゲットラベル下のマルチモーダル感情認識（MER）転移

新しいコーパス（ターゲット）にラベルが**クラスあたり K 件**しかないとき，
ほかのコーパス（ソース）のラベル付きデータを混ぜると成績がどう変わるかを測り，
得と損が生じる仕組みを，ドメイン適応（DA）理論と対応づけた数式モデルで説明する研究である．
**分析の軸は得の仕組みの仮説 MH1〜MH5**（`docs/researchvision.md` 3 節）．その上に，仕組みの式（E15〜E17）と条件の探索（E18〜）を積んでいる．**解析は，合併の C をソースのみの C に固定した地図（`DAI/results/fixc_map/`）を土台にし，論文と発表資料の本文もこの地図だけで書く**（交差検証の地図は付録．同 6.2 節）．
論文は分析と手法を 1 本に統合する（2026-10-22 決定．数理を前面に）．2026-10-28 に，今の版では IEEE TAC に出さず，教師なしの DA としても半教師ありの DA としても数学的に裏打ちされた手法の新しさを目指して実験を増やすと決めた（`docs/researchvision.md` 6.1 節）．2026-10-31 に投稿先を Discover Artificial Intelligence に決め，論文は `DAI_paper/` で書くことにした．2026-11-14 に，原稿は `DAI/paper/` で書くことにした（規則は `DAI/paper/CLAUDE.md`．旧版の執筆フォルダは 2026-11-15 に `DAI/docs/参照/DAI_paper/` に移し，要項・参照論文・棚卸しを読むだけにする）．TAC 向けの旧版の `archive/paper/` の内容には従わない．2026-11-04 に，Albanie ほか 2018 とその後続研究を土台に研究を組み直すと決め，設計と実験は `DAI/` で行うことにした（規則は `DAI/CLAUDE.md`，計画は `DAI/docs/計画/計画_2026-11-04.pdf`）．2026-11-08 に，時間が無いので，まず解析論文として結論をまとめ，エンコーダの凍結を解析のための設定にすると決めた（`docs/researchvision.md` 6.1 節．作業の記録は `research_notes/解析論文のストーリー_2026-11-08/`）．
各主張の最新の状態と実験の一覧は `docs/results.md`（3.6 節），分析の層と次の実験の計画は `docs/researchvision.md` 3.6・3.7 節．

**このディレクトリは自己完結させる．`~/MER` など他のディレクトリは一切参照しない**
（コード・環境・文書・設定値のどれも）．要るものはここに置く．

**2026-11-15 から，研究の実体は `DAI/` にある．**論文の実験の台本・部品・データの一覧・結果は DAI の中にあり，DAI だけで再現できる（`DAI/README.md`）．共有の文書 `docs/`・ノート `research_notes/`・旧版の執筆フォルダ `DAI_paper/` も同じ日に DAI の中に移し，直下の同じ名前はそこへのリンクにした（古いパスで書かれた文書と定義のため）．直下に実体があるのは Python の環境 `.venv/` と旧版の結果 `results/` だけで，それ以外の旧版の台本・報告・原稿は `archive/` に移した（`archive/README.md`）．

---

## 正本と優先順位

文書は役割ごとに 8 本ある（2026-10-08 に統合，2026-10-28 に台帳を足した．`CLAUDE.md` と `DAI/data/README.md` 以外は `docs/` の下．`docs/` は 2026-11-15 から `DAI/docs/` へのリンク）．

| 対象 | 正本 |
|---|---|
| 研究の目的・分析の軸・論文の計画・決定事項・未決事項・更新記録・事前登録した仮説 | **`docs/researchvision.md`** |
| 実験の手順（地図，学び直し，ずれの指標，操作，統計，理論からの予言の導出，実験の検査の関門と不備の目録，新しい定理を探す手順） | **`docs/methods.md`** |
| データの扱い・発話一覧・特徴の作り方・重み・作り直し方 | **`DAI/data/README.md`** |
| 各主張の最新の状態，解析の本筋と端，結果の場所と作り方（執筆・発表の入口） | **`docs/results.md`**（旧版の TAC 向けの記録．DAI の論文の入口は `DAI/docs/実験の地図.md`） |
| 探索的な解析の方法・数値・経緯（日付順の記録） | `docs/exploratory-analyses.md` |
| 設計の判断のために見たデータの記録と，確認に使えるデータ | `docs/seen-data.md` |
| 先行研究と論文の位置づけ（**論文と発表資料の新しさ・関連研究・限界の書き方はここに合わせる**） | `docs/related-work.md` |

**食い違ったら，データの扱いは `DAI/data/README.md`，それ以外は `docs/researchvision.md` が勝つ．**
食い違いを見つけたら，勝った側に合わせて負けた側を直す（放置しない）．古い文は残さずその場で書き直す（前の版は git にある）．
本ファイルは要点と取り決めだけを持つ．**未決事項は期限つきで `docs/researchvision.md` 7 節に書き，ここには置かない．**方針を変えたら `docs/researchvision.md` の決定事項を直し，更新記録に 1 行足す．

中核の理論：`archive/A theory of learning from different domains.pdf`（Ben-David et al., 2010．Theorem 3）

---

## データの取り決め（破ると測定が無効になる）

根拠は `DAI/data/README.md`．読み込みは `DAI/measure/data.py` の `load_group(群, エンコーダ)` が 1〜6 を引き受ける．
**特徴は必ずこれを通して読む．`np.load` した `X` をそのまま使わない．**

1. **ラベルは `DAI/data/metadata/*.csv` から貼り直す．**npz の `label_4` は使わない
2. **`sample_id` で並べ直す．**行番号で標本を選ばない
3. `ok` が偽の行と，NaN・Inf を含む行は落とす
4. **映像の `X_std` は使わない．**発話の長さの代理変数になっている
5. **話者は必ず分ける．**同じ話者が学習側と評価側の両方に現れてはならない（`DAI/measure/split.py`）
6. **IEMOCAP は即興と台本を別の群として扱う**
7. **K はクラスあたりの本数．**K=25 なら 4 クラス × 25 = 100 本
8. **ラベルは 4 つ**（0=Neutral 1=Happy 2=Sad 3=Angry），**主指標は UAR**（でたらめなら 25%）
9. **エンコーダは微調整していないものだけ**

群は `IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D` の 7 つ．
映像は RAVDESS に無く，言葉は RAVDESS・CREMA-D に無い．**AFEW の言葉だけは人ではなく Whisper large-v3 の書き起こしから作る**（2026-11-04〜．`DAI/data/README.md` 3 節）．

---

## ディレクトリ

| 場所 | 中身 | git |
|---|---|---|
| `DAI/` | **研究の実体**（2026-11-04〜．2026-11-15 に整理し，データを除けば単独で論文の実験を流し直せる形にした）．入口 `DAI/README.md`，手順 `DAI/docs/再現の手順.md`，規則 `DAI/CLAUDE.md`，台本 `DAI/exp/`（論文の前半の写しを含む），部品 `DAI/measure/`・`DAI/features/`，発話一覧 `DAI/data/metadata/`，特徴 `DAI/data/cache/*.npz`（git の外），重み `DAI/weights/`（git の外），設計 `DAI/docs/設計/`，論文の原稿 `DAI/paper/`，検査の報告 `DAI/checks/`，結果 `DAI/results/`，ログ `DAI/logs/` | ○（結果は集計と記録だけ．特徴・重み・ログ・LaTeX の中間生成物は ×） |
| `docs` | `DAI/docs/` へのリンク（2026-11-15 に実体を DAI に移した）．共有の文書 6 本（研究計画 `researchvision.md`，方法 `methods.md`，結果 `results.md`，探索の記録 `exploratory-analyses.md`，先行研究 `related-work.md`，見たデータの台帳 `seen-data.md`）と旧版の主張契約 `paper-claim.md`．は `DAI/docs/` の直下にある．検査役の定義と論文のセッションが引く．中のパスは MER-DA の直下から書いてある | ○（実体は `DAI/docs/`） |
| `.claude/agents/` | 実験の検査役の定義（関門 1 の `gate-stats`・`gate-leak`・`gate-compare`・`gate-define`・`gate-probe`・`gate-code`，関門 2 の `gate-claim`．手順は `docs/methods.md` 9 節），新しい定理を探す担当（`thm-*`．同 10 節），`story-judge`，`prior-work-diff` | ○ |
| `DAI_paper` | `DAI/docs/参照/DAI_paper/` へのリンク（2026-11-15 に中身を DAI に移した．旧版の執筆フォルダで，要項 `docs/venue.md`・参照論文・棚卸しを論文のセッションと査読役が読む）．古いパスで書かれた文書と論文のセッションの参照のために残す．論文の原稿の参照を書き換えたら消してよい | ○（リンク） |
| `research_notes` | `DAI/research_notes/` へのリンク（2026-11-15 に実体を DAI に移した）．調べ直しのノート，論文の相談のノート（論文のセッションが書く） | ○（実体は `DAI/research_notes/`） |
| `results/` | 旧版の実験の結果．論文に使う置き場（93 個）は `DAI/results/` へのリンク | × |
| `logs` | `DAI/logs/` へのリンク | × |
| `archive/` | 使わなくなった旧版のもの（2026-11-15 に移した）：台本 `exp/`・`run/`，部品 `measure/`・`features/`，データの旧い写し `data/`（波形のファイルなど DAI に要らないものを含む），検査の報告 `checks/`（E1〜E87），TAC 向けの原稿 `paper/`，発表資料 `slides/`，`paper_code/`，`reports/`，`requirements.txt`，中核の理論の PDF．`archive/README.md` | ○（元から git に入っていたものだけ） |
| `.venv/` | Python 環境（作り方は `DAI/requirements.txt` の冒頭）．DAI の `.venv` はここへのリンク | × |

**Python 環境は `.venv/` の 1 つだけ．**版は `DAI/requirements.txt` で固定する（scikit-learn の版違いで結果がずれうる）．

---

## 進め方

- **研究の流れは 6 つの問い（2026-10-19 の面談の資料の 5 つの問いに問い 6 を足したもの．`docs/researchvision.md` 3.0 節）で書き，語はスライドの語に合わせる**（対応表は `docs/results.md` 3.6 節末）．研究者への報告もこの流れと語で行う
- **論文の文章（要旨・序論・本文・図表の説明・査読への返事．日本語でも英語でも）を書く・直す前に，スキル `mer-paper-style`（`.claude/skills/mer-paper-style/SKILL.md`）を必ず読む**（2026-11-15 研究者の承認．有名な MER の論文と岡田・石井の論文の書き方の傾向）
- **実験コードを書く前に `docs/methods.md` を読む**
- **実験は関門を通してから計算し，結果を書く**（`docs/methods.md` 9 節．2026-10-28〜）．設計を書いたら `docs/seen-data.md` に 1 行足し，本番は関門 1 の 4 体，下調べは `gate-probe` の 1 体に検査させる．**本番でも下調べでも，台本を書いたら計算の前に必ず `gate-code` に台本を検査させる**（2026-11-01〜．目的は計算の後の指摘で計算をやり直さないこと．台本が設計どおりに計算するかに加え，後から求められそうな解析を保存した出力から出せるかを確かめさせる．台本を直したら差分を再検査させ，計算の後には結果ファイルの欠けと重複を点検させる）．結果を文書に書く前に `gate-claim` に照合させる．検査役には設計と台本と台帳だけを渡し，期待する結論は渡さない．**`DAI/` の実験は，さらに ML と感情認識の査読想定の役（`dai-review-ml`・`dai-review-affect`）の両方の OK を計算の前に得る**（2026-11-04〜．`DAI/CLAUDE.md`）
- **ずれの指標は `DAI/measure/shift.py`，ずれを消す操作は `DAI/measure/ablate.py` を必ず通す．**実験コードで書き直さない．変えたら `DAI/measure/test_shift.py` を流し，`docs/methods.md` も直す（`docs/methods.md` などの文書に出る `exp/`・`measure/`・`features/`・`run/`・`checks/`・`paper/`・`slides/` のパスは，2026-11-15 からは `DAI/` の中の同じ名前か `archive/` の下を指す）
- **長時間の処理（特徴抽出・学習・評価）は勝手に起動しない．**
  まずコマンド・所要見込み・上書きするファイルを示して確認を取る
- 長時間処理は **tmux セッション `mer-da`** の中で回す
- git の remote は `origin`（github.com/snics6/MER-DA，非公開．SSH 鍵 `~/.ssh/github_ed25519`）．**コミットしたら `git push` する**（2026-10-27 から）
- **数値の主張は，ログか出力ファイルを示してから述べる**
- **事前登録した仮説（`docs/researchvision.md` 8 節）は結果を見てから書き換えない**
- ハイパーパラメータはテストラベルで選ばない．実用値とオラクル値を分けて報告する
- このファイルは短く保つ（150 行以内）．増えたら `docs/` に分ける
