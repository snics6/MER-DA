# 結果の索引（執筆・発表の入口）（2026-09-30）

論文（`~/papers/<論文名>/`）やスライド（`~/slides/decks/MER-DA/<用途>/`）を，**別のチャットで `claude --add-dir ~/MER-DA` として作るとき**の入口．
結果そのもの（`results/`）は git の外にあるので，入口はここ（git の中）に置く．

**執筆側は MER-DA を読むだけで，書き換えない．**足りない結果や解析は，実験要求として MER-DA 側のチャットに返す（`~/papers/CCMER` と同じ分担）．

**進み具合**（何回終わったか，いま動いているか）は **`results/status.md`**（機械が読むなら `results/status.json`）．
`./.venv/bin/python exp/results_status.py` で作り直す．この表は UAR などの値を含まない．

---

## 1. 最初に読むもの（この順）

| # | ファイル | 何が分かるか |
|---|---|---|
| 1 | `CLAUDE.md` | データの取り決め（話者分割，ラベル 4 つ，主指標 UAR），正本の優先順位 |
| 2 | `researchvision.md` | 研究の目的・理論（Ben-David 2010 Theorem 3 ほか）・仮説．**2026-09-30 の方針転換は 0.1 節**（主軸は地図と E シリーズ）．事前登録は 6.5 節（H1〜H6），6.5.1 節（予言 P1〜P4），6.5.2 節（H7） |
| 3 | `docs/experiment-policy.md` | 実験の決定事項と未決事項（日付と根拠つき．更新記録あり） |
| 4 | `docs/experiment-protocol.md` | 手順（条件・分割・ずれを消す操作）と，理論からの予言の導出（7 節） |
| 5 | `docs/shift-metrics.md` | ずれの指標の定義と注意 |
| 6 | `docs/related-work.md`，`docs/related-work-notes.md` | 先行研究の要点と，論文ごとの詳しいメモ |
| 7 | `data/README.md` | データの中身と作り方 |
| 8 | `docs/exploratory-analyses.md` | **研究の主軸の分析（E1〜）**：方法，主な数値，発表用のわかりやすい説明，限界 |
| 9 | `docs/paper-positioning.md` | **論文の位置づけ**：先行研究との関係（書き方を直す点，足す文献），研究の価値，IEEE Access で想定される指摘．**本文の新しさ・関連研究・限界はこれに合わせて書く** |

## 2. 結果を使うときの決まり

- **数値は結果ファイルから取り，出典（ファイル・列・条件）を添える．**集計は下の台本の出力から引く．執筆側で新しい値を計算しない
- **状態が `complete` でない結果（`partial`，`quick (R=1)`）は，論文の主張の数値に使わない．**速報（反復 1 回）は傾向を見るためだけのもの
- **K* は反復が 30 回そろうまで暫定**（`kstar.csv` の `status` 列が `provisional`）．**IEMOCAP がターゲットの組は，K* の区間の計算法が未決**（`experiment-policy.md` 3 節．`iemocap_interval_undecided`）
- **オラクルの列（`uar_oracle`，`C_oracle`，`test_uar_C*`）は分析専用．**手法の性能として報告しない（`CLAUDE.md`）
- **予言の判定は，`researchvision.md` 6.5.1 に登録した主判定と Holm の手順でだけ行う．**登録より前に地図の結果を見たのは，動作確認の 2 組（AFEW←RAVDESS，AFEW←MOSEI）の反復 1 回ずつだけ
- UAR はでたらめに当てると 0.25

---

## 3. 結果の一覧

### 3.1 地図（作り方 ①②③ × 全組 × K）

| 項目 | 内容 |
|---|---|
| 場所 | `results/map_<kind>/<ターゲット>.csv`．1 行 1 条件．**反復 1 回が終わるたびに追記** |
| kind | `audio_wavlm`（本番 30 回を実行中），`video`，`text_roberta`，`fuse-audio-video`，`fuse-audio-text`，`fuse-video-text`，`fuse-audio-video-text`（5 回．2026-10-01 に完了）．第 2 エンコーダ（`audio_hubert` など）は未実施 |
| 作り方 | `./.venv/bin/python exp/map.py --kind <kind> [--reps …] [--resume]`（コードは `exp/map.py`・`exp/logreg.py`） |
| 手順の根拠 | `experiment-protocol.md` 2〜3 節，`experiment-policy.md` 2 節 |
| 追跡 | 各ファイルの `#` で始まる行（実行ごとの設定とコミット番号），`results/map_<kind>/code/<開始時刻>/`（実行したコードの写し）．**`map_audio_wavlm` は未コミットの状態で起動した**（記録は `6845855+dirty`）が，写しの `map.py`・`logreg.py` はコミット `ea8a79a` と同じ（2026-09-30 確認） |
| 読み方 | `pd.read_csv(path, comment="#", dtype={"K": str, "source": str})` |

**列の意味**

| 列 | 意味 |
|---|---|
| `target`，`source` | ターゲット／ソースの群．`tgt_only` の行は `source` が空（ソースに依らない） |
| `method` | `tgt_only`（① ターゲットの K 本だけ），`src_only`（② ソースだけ），`merged`（③ ソース全件＋K 本） |
| `K` | クラスあたりの本数（`1`〜`50`，`all`＝学習プール全件）．`src_only` は空（K に依らない） |
| `rep` | 反復番号 0〜29．テストは話者 5 袋のうち rep mod 5 番．K 本の乱数はターゲットと rep だけで決まる（入れ子） |
| `kind` | 特徴の種類 |
| `n_src`，`n_tgt`，`n_test` | 学習に使ったソース・ターゲットの本数，テストの本数 |
| `C`，`C_rule` | 選ばれた正則化の強さと，その決まり方（`cv`：交差検証，`tie`：同点の規則，`fixed`：交差検証できず 1.0） |
| `uar` | **選ばれた C でのテストの UAR（主指標）** |
| `C_oracle`，`uar_oracle` | テストで最良の C とその UAR（オラクル．分析専用） |
| `cv_uar_C0.01` 〜 `cv_uar_C10` | 交差検証の UAR（C の候補ごと） |
| `test_uar_C0.01` 〜 `test_uar_C10` | テストの UAR（C の候補ごと．オラクル用） |
| `sec`，`n_iter`，`n_loop` | 計算時間と反復回数（計算量の記録） |
| `init`，`cached` | 温め直しの出発点（`none`／`folds`／`final`），② を使い回した行か |

合併とターゲットのみの比較は，`(target, K, rep)` で対にして差をとる（`experiment-policy.md` 2 節の t 区間）．

**まとめ**：`./.venv/bin/python exp/summarize_map.py --dir results/map_<kind>` で `results/map_<kind>/summary/` に次を作る（保存済みの反復の分だけで作る）．

| ファイル | 中身 |
|---|---|
| `pairs.csv` | 組 × K ごとの，終わった反復の数，UAR の平均と 95% 区間（`tgt_*`・`src_*`・`merged_*`），合併 − ターゲットのみ の差（`diff_*`），オラクルの平均 |
| `kstar.csv` | 組ごとの K*（差の 95% 区間の下端が 0 を上回る最大の K．無ければ `none`）と状態（`final`／`provisional`／`iemocap_interval_undecided`） |
| `c_rule.csv` | 作り方 × K ごとの，C の決まり方の割合 |
| `fig_<ターゲット>.png` | ソースごとの K–UAR の曲線（誤差棒は 95% 区間） |
| `fig_gain.png` | 全組の「合併 − ターゲットのみ」（UAR の点．`*` は区間が 0 をまたがない） |

### 3.2 データの監査（完了）

| ファイル | 作ったコード | 中身 |
|---|---|---|
| `results/manifest_audit.csv` | `measure/audit_manifest.py` | 発話一覧の検査 |
| `results/speaker_audit.csv` | `measure/check_speakers.py` | 話者の診断 |
| `results/split_summary.csv` | `measure/split.py` | 話者 5 分割の大きさと，4 感情がそろうか |

### 3.3 使わないもの

| ファイル | 理由 |
|---|---|
| `results/map_timing_audio_wavlm.csv` | 計算時間の試走（2 ターゲット・反復 1 回で中断）．分析に使わない |

### 3.4 ずれの指標（軸 1〜4）

| 項目 | 内容 |
|---|---|
| 場所 | `results/shift_<kind>/` |
| 作り方 | `./.venv/bin/python exp/shift_map.py {gpu,input,space} --kind <kind>`（段ごとの中身は `exp/shift_map.py` の先頭） |
| 根拠 | `docs/shift-metrics.md`（0.1 節に測る空間），`researchvision.md` 6.5.1 共通の決まり 6（空間の選び方） |
| ファイル | `labeling.csv`（軸 4：ソースのみの分類器の UAR・混同行列・Δ_CE・BER・P4-1 の検算．`cross_fit` が真の組は，同じ人物を除いて学び直した分類器で採点），`label.csv`（軸 3：割合の差．記述用と MLLS の推定），`input.csv`（軸 1・2．`space` は raw／pca64／logit，`balanced` は割合をそろえた版），`zero.csv`（ゼロ点），`space_choice.json`（規則の当てはめ），`logits/`（空間 logit の点数） |
| 注意 | 空間 logit は 4 つの点数を和 0 の 3 次元の座標で表したもの．MLLS による割合の推定は，コーパス間では誤差が大きい（音声で ‖q − p_t‖₁ が 0.34〜1.71） |

### 3.5 ずれを消す・足す操作（速報）

| 項目 | 内容 |
|---|---|
| 場所 | `results/ops_<kind>/<ターゲット>.csv`（地図と同じ列に `op` を足したもの） |
| 作り方 | `./.venv/bin/python exp/ops.py --kind <kind> --reps 0`（操作の一覧は `exp/ops.py` の先頭） |
| 比べる相手 | 地図（3.1）の同じ `(target, source, K, rep)` の行．テストと K 本の抜き方は地図と同じ |
| 注意 | 速報は反復 1 回．主判定に使う設定だけ（`experiment-policy.md` 2 節） |

### 3.6 探索的な解析（途中の見立て）

| 項目 | 内容 |
|---|---|
| 説明と主な数値 | **`docs/exploratory-analyses.md`**（E1：組をそのまま並べた順位相関，E2：ターゲットとソースの効果を分けた解析，E3：新しいターゲットに対するソースの順位の予想，E4：どの指標が予想に役立つか，E5：ソースの選び方の比較，E6：音声と映像の違いの切り分け，E7：RAVDESS を除いた音声でのソースの選び方，E8：足し算の分け方の妥当性，E9：特徴空間の形（感情の向きのそろい方）による説明，E10：ソースの境目の向きがターゲットの感情を分けるか，E11：得の仕組みの仮説 MH1〜MH5 の予言，E12：MH3・MH4 の学び直し，E13：ソースのみの分類器の d と o を感情の対ごとに測る，E14：話者ごとの位置のずれ（MH6）） |
| 保存した結果 | `results/explore_relation/snapshots/<日時>/`（その時点の地図の回数とコミット番号は `state.json`）．`results/explore_relation/` の直下は，台本を流すたびに上書きされる |
| 作り方 | `./.venv/bin/python exp/explore_relation.py` → `exp/explore_decomp.py` → `exp/explore_select.py` → `exp/explore_metrics.py` → `exp/explore_strategies.py` |
| 注意 | **確かめた結果として扱わない．**E2 は HuBERT で確かめる仮説 H7 として `researchvision.md` 6.5.2 に事前登録した |

### 3.6.1 仕組みの仮説の学び直し（E12．実行中）

| 項目 | 内容 |
|---|---|
| 場所 | `results/mech_<kind>/<ターゲット>.csv`（地図と同じ列に `op`，`alpha` と α・C ごとの交差検証とテストの UAR を足したもの） |
| 作り方 | `./.venv/bin/python exp/mech.py --kind <kind> --reps 0 1 2`（操作の一覧は `exp/mech.py` の先頭） |
| 比べる相手 | 地図（3.1）の同じ `(target, source, K, rep)` の行．テストと K 本の抜き方は地図と同じ．`wt` の α=nat の列は地図の合併と一致する（確認済み） |
| 注意 | 反復 0〜2 だけ．音声と映像 |

### 3.6.2 仮説 H7 の判定（HuBERT）

| 項目 | 内容 |
|---|---|
| 場所 | `results/check_h7/`（`primary.csv`：主判定 4 つと判定，`secondary.csv`：副判定，`share.csv`：ばらつきの内訳，`raw_corr.csv`：組をそのまま並べた順位相関，`fig_h7.png`，`provenance.json`） |
| 作り方 | `./.venv/bin/python exp/summarize_map.py --dir results/map_audio_hubert` → `./.venv/bin/python exp/check_h7.py` |
| 根拠 | `researchvision.md` 6.5.2（登録した手順．末尾に判定の結果を追記した） |
| 説明と主な数値 | `docs/exploratory-analyses.md` の「HuBERT での確かめ」（2026-10-06）．4 つとも「判定できない」 |
| 注意 | 比べるために WavLM の値も同じ表に出す．WavLM の値は結果を見た後に方法を決めた探索的な値で，判定には使わない |

### 3.7 これから作るもの（予定）

| 予定の場所 | 中身 | 根拠 |
|---|---|---|
| 未定 | 事前登録した予言の判定 | `researchvision.md` 6.5.1 |

作ったら，この節から上と同じ形の節に移す．
