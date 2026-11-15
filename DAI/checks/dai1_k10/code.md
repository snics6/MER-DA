# 関門 1：実装と出力（実験 1 の K = 10 の族の計算の後の解析，2026-11-16，コミット e13266f）

台本：`tools/dai1_k10_families.py`（最後に変えたコミット e13266f．検査の時点の HEAD は f874dfd で，e13266f から HEAD までに台本・`exp/dai1_stats.py`・`exp/dai1.py`・`exp/dai_data.py` の差分は無い）．
入力：`results/dai1/eval/units/`（評価の回．単位の npz の commit は ba98974，dirty は偽），`results/dai1/select/frozen.json`（sha256 6926fb3e…，`frozen/dai1_frozen.json` と同じ），`results/dai1/stats/families_flagged.csv`．

## 判定
判定：直してから通す．台本は K = 10 の族 1・3・4 を固定の設定のとおりに正しく計算する（差は 3 通りの独立の照合で 2e-15 以内）が，同じ量は設計（`docs/設計/実験1_設計.md` 9.3 節の「族の外の数（S5）」）で評価の回の前に決められ，評価のコミット ba98974 の `exp/dai1_desc.py` が `results/dai1/stats/desc/outside_families.csv` に既に書いており，台本の出力を `results/dai1/stats/` に置くと MOSEI の区間が別の乱数の列で 2 通りになり，族の外の数に付けない決まりの p 値も出るためである．

## 要点（大学 1 年生向けの言い換え）

- 台本の計算は正しい．K = 5 で作り直すと元の表と一致し（差の最大 4.4e-16），K = 10 の差は rows.csv の UAR から別の道で作り直した値とも一致した（差の最大 1.8e-15）．
- ただし，K = 10 の族 1〜4 の差と区間は，評価の回の 2026-11-05 に「族の外の記述」として既に計算されていた（`results/dai1/stats/desc/outside_families.csv`）．台本の冒頭の「K = 10 の差と区間は集計していなかった」は誤りである．台本の出力はこの表と，差と jackknife の区間で完全に一致し，MOSEI の区間の端だけが最大 0.021 点（標準誤差の 2.3%）ずれる．ずれの原因は区間を作る乱数の種の違いである．
- 族 4（選んだ設定 − 既定の設定）の K = 10 は 16 条件すべてで「選んだ設定 = 既定の設定」なので，差は定義から 0 である．情報を持たない．

## A．設計と台本の対応表

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 族 1：提案（同時に解く版，主の版，選んだ設定）− K 本だけの基準（引き戻しの C は固定の tether） | `tools/dai1_k10_families.py:34` の FAMS，`exp/dai1_stats.py:106-118` の `unit_preds`（`prop`，`none`） | 合う．K = 10 の `sel["s|10|joint|one"]`，`fz["main"]["10"]`，`fz["tether"]["s|10"]` がすべて固定のファイルにあり，npz の `p3_<s>_K10_items`・`p1_<s>_items` に該当の行が 1 つずつある（`p3item` の assert が通った） |
| 族 3：提案 − 相手（固定の partner の手法と版，選んだ設定） | 同 `:34`，`exp/dai1_stats.py:112-124` | 合う．K = 10 の相手は 4 コーパス × 4 生徒の 16 条件すべてで温度つき蒸留（kd）．版は IEMOCAP の A・AVT と MELD の AVT が sel，ほかは one |
| 族 4：提案 − 選択なし（版 one，ρ = 1，a = 1） | 同 `:34`，`exp/dai1_stats.py:118` | 合う．ただし K = 10 の選んだ設定は 16 条件すべてで ρ = 1，a = 1，版 one で既定と同じ（指摘 [L] 2） |
| 族 2（役割を分ける版 − 同時に解く版） | 無い | 依頼の範囲（族 1・3・4）では外してよい．設計 S5 では K = 10 の族 2 も族の外の数に入り，`outside_families.csv` にある |
| 効果 θ = 分割の平均（種の平均），区間は G_eff ≥ 20 で塊の引き直し B = 2,000 回 + 種の項，G_eff < 20 で塊の jackknife + 種の項 | `exp/dai1_stats.py:160-220` の `effect_stats` をそのまま呼ぶ（台本 `:60`） | 合う．G_eff は AFEW 15.0，IEMOCAP 4.9，MOSEI 783.2，MELD 8.6 で，MOSEI だけが引き直し |
| 単位：5 分割 × 種 10 = 50 | 台本 `:48-52` | 合う．4 コーパスとも 50 単位．`n_units` は 50，`n_folds` は 5 |
| 保存した判定と rows.csv の照合（S3） | `exp/dai1_stats.py:264-273`（`collect` の中．KS_MAIN を回す） | 合う．KS_MAIN を (10,) に替えると K = 10 の格子の行を照合する．4 コーパスとも食い違い 0/200 |
| p 値を付けない（設計 9.3 節 S5「族の外の数 … p 値は付けない」） | 台本 `:61` が `effect_stats` の戻りを全部書く | 合わない．出力に補正していない `p` と `t` の列が残る（指摘 [I] 3） |
| 出力の置き場と記録（DAI/CLAUDE.md：`provenance.json`・`provenance_log.jsonl` にコミット，dirty，台本と入力の sha256） | 台本 `:65-66`（commit と dirty の列だけ） | 足りない．`dai1.git_commit` の dirty は `exp splits measure features data/metadata` だけを見るので，`tools/` の未コミットの変更を拾わない．台本と入力の sha256 が無い（指摘 [I] 4） |

数える組：設計（S5）の K = 10 は族 1〜4 × 生徒 4 × コーパス 4 = 64 行．台本は族 1・3・4 の 48 行を作る．族 2 の 16 行を作らない理由は依頼の文にある（族 1・3・4 だけ）が，台本の冒頭には書かれていない．

### (1) KS_MAIN を一時に替える作りについて

- `collect`（`exp/dai1_stats.py:260`）は `for K in KS_MAIN` を関数の実行のときにモジュールの大域変数から読むので，`ST.KS_MAIN = Ks`（台本 `:41`）は効く．`finally` で元に戻す（`:63-64`）．
- モジュールの `ST.FAMS`（`exp/dai1_stats.py:52`）は読み込みの時点の (0, 5, 25) を持ったままだが，台本は自分の FAMS を使い，`ST.FAMS` を使う `evaluate`・`detect`・`expected_effects` を呼ばないので影響しない．
- `unit_preds` は族 2 を使わなくても `rolesep` の行を毎回取り出す（`exp/dai1_stats.py:117`）．K = 10 の `sel["s|10|rolesep|one"]` と npz の `p3r_<s>_K10_pred_b` はどちらもあり，止まらなかった．`p3c_K10_*`，`p3g_K10_*` も保存されている（相手が kd なので使われない）．
- 小さく流した結果で，K = 10 の 4 コーパスすべてが最後まで動き，保存と rows.csv の照合の食い違いは 0/200 だった．

### (2) K = 5 の照合が意味のある照合か

- 意味はあるが範囲は限られる．K = 5 の照合が確かめるのは「KS_MAIN を替えても `collect` と `effect_stats` が元の `evaluate` と同じ数を出す」ことである．K = 10 だけで使う鍵（`main["10"]`，`tether["s|10"]`，`select["s|10|…"]`，`partner["s|10"]`，npz の K10 の行）が正しく引かれるかは確かめない．しかも照合の相手 `families_flagged.csv` は同じ `effect_stats` と同じ `unit_preds` で作った表なので，選び方の誤りが両方にあれば一致してしまう．
- これを補うために，検査役が 3 つの照合を足した（どれも `checks/dai1_k10/code_run/` に置いた）．
  1. **rows.csv からの独立の作り直し**（`checks/dai1_k10/code_run/indep_check.py`）：npz の判定を使わず，各単位の `p1.csv`・`p3.csv` の UAR の列から，固定のファイルの設定に当たる行を列の値で選び，θ を作り直した．K = 10 の 48 行で台本の差との差の最大は 1.78e-15，K = 5 の 48 行で `families_flagged.csv` との差の最大は 8.88e-16 だった．選び方の誤りは無い．
  2. **`outside_families.csv` との照合**：評価のコミット ba98974 の `exp/dai1_desc.py` が作った K = 10 の族 1・3・4 の 48 行と，台本の 48 行を突き合わせた．A と B の名前，`n_units`，`folds_dropped`，`identical`（相手の表では `se_zero`）はすべて一致した．差の最大 0.0，AFEW・IEMOCAP・MELD の区間の端と標準誤差の差の最大 0.0，MOSEI の区間の端の差の最大 0.0206 点（標準誤差に対して最大 2.3%）だった．
  3. **MOSEI の区間の再現**：`exp/dai1_stats.py eval` を一時の置き場（`checks/dai1_k10/code_run/reeval/`．`eval` と `select` は `results/dai1/` への記号のリンク）に流すと，`families_flagged.csv` の 176 行のすべての列（MOSEI の lo・hi・se を含む）が 4.4e-16 以内で一致した．MOSEI の区間の端の差は乱数の種 20261116 と 20261104 の違いだけで生じる．
- 台本の照合の結果：K = 5 の 48 行で差の最大 4.44e-16，jackknife の区間の端の差の最大 8.88e-16，MOSEI の区間の端の差の最大 0.042 点（標準誤差に対して最大 6.1%）．台本の止める条件は通った．

### (3) 族 3・4 の K = 10 の行の解釈で気をつける点

- **族 4 は 16 条件すべてで同じ識別器の比較である．**K = 10 の選んだ設定は 4 コーパス × 4 生徒のすべてで版 one，ρ = 1，a = 1 で，既定の設定と同じである．台本の出力でも族 4 の 16 行すべてで `identical` が真（差 0，標準誤差 0）だった．設計 9.2 節は K = 0，5，25 で同じ状況のセルを「同じ水準」と数えず「同じ識別器」として別に数えると決めている．K = 10 も同じ扱いにし，「選ぶことの効果は 0」「同じ水準」と書かない．値を見なくても，この結果は固定のファイルだけから分かる．
- **族 3 の相手は K = 10 の 16 条件すべてで温度つき蒸留（kd，温度は選んだ値）である．**「5 手法のうち仮のターゲットで最も良かったもの」の名前を並べた形で書く設計 9.3 節の規則をそのまま当てる．IEMOCAP の A・AVT と MELD の AVT では相手の版が sel，提案の版が one で版が違う．版をそろえた値は `results/dai1/stats/desc/fam3_notes.csv` の K = 10 の 16 行（`aligned_*` の列）に既にある．
- **K = 5，10，25 は独立の証拠ではない．**K 本は K について入れ子に抜く（`exp/dai_data.py:8`，`:132`）．K = 10 の K 本は K = 5 の K 本を含み，テストの話者，分割，種も同じである．「3 つの K で同じ向き」を 3 回の独立の確認として数えない．K の間の差の区間が要るなら，単位ごとの差から対にして作る（B の表の 9 行目）．
- **K = 10 は計算の前の族に入っていない．**設計は K = 10 を族の外の数として区間だけを添え，p 値を付けないと決めた（S5）．研究者の 2026-11-16 の指示で K = 10 を K = 5 と同じ重さで論文に載せても，K = 10 の行は Holm の補正の族に入らず，検定の結果の文（「上回った」）の根拠にならない．台本の `p` の列を使わない．
- **記述に回す印が K = 10 に無い．**`families_flagged.csv` の `descriptive` と `same_level_possible` は固定のファイルの `detect` から来るが，`detect` は K = 0，5，25 のセルしか持たない（4 コーパスとも `detect.cells` に K = 10 の鍵が 0 個）．K = 10 では検出力の見込みに基づく「記述に回す」の区別ができない．
- **MOSEI の区間の端は乱数に依って 0.02〜0.04 点ほど動く．**B = 2,000 回の引き直しの誤差である．論文に 2 桁目まで書く場合，どの表の値を使うかを 1 つに決める（指摘 [I] 1）．

### (4) 後で求められそうな量が出力にそろうか

出力 `families_k10.csv` の列：family，T，student，K，A，B，g_eff，mismatch，checked，effect，n_folds，n_units，seed_varies，lo，hi，p，se，method，df，v_jk，v_seed，v_seed_all，folds_dropped，identical，t，boot_lo，boot_hi，commit，dirty，units_commit，posthoc．区間，単位の数，分割の数，区間の作り方，自由度，分散の内訳，同じ識別器の印はそろう．無いもの：使った設定の文字列（`outside_families.csv` の `spec_A`・`spec_B` にある），版，固定のファイルの sha256，台本の sha256，`descriptive` の印，単位ごとの差 d_fr．詳細は B の表．

## A．小さく流した結果

- GPU は別の計算が使っていた（13.2 GB，使用率 90%）ので，`CUDA_VISIBLE_DEVICES=""` で CPU で流した．台本は GPU を使わない．
- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W default tools/dai1_k10_families.py --out-dir checks/dai1_k10/code_run`
- 所要：1 分 24 秒（実時間．K = 5 の照合と K = 10 の本計算を合わせて）．ログは `checks/dai1_k10/code_run/run.log`．
- 警告：`ResourceWarning: unclosed file`（台本 `:39` の `json.load(open(...))`）だけ．収束，NaN，型の変換の警告は無い．
- 出力：`checks/dai1_k10/code_run/families_k10.csv`（48 行 = 族 3 × コーパス 4 × 生徒 4，K は 10 だけ，同じ識別子の重複 0），`checks/dai1_k10/code_run/families_k10_check.csv`（48 行）．
- NaN：`df` 12 個（MOSEI は正規の区間で自由度を持たない），`t` 24 個（MOSEI の 12 と，族 4 で標準誤差が 0 の jackknife の 12），`boot_lo`・`boot_hi` 各 36 個（jackknife の 3 コーパス）．どれも作りの上で空になる列である．Inf は 0 個．
- 退化した場合の検算：族 4 は選んだ設定 = 既定なので，2 つの行が同じ識別器になる退化の場合そのものである．16 行すべてで差が 0，標準誤差が 0，`identical` が真になった．
- 保存済みの値の再現：K = 5 の 48 行で `families_flagged.csv` との差の最大 4.44e-16（区間の端は jackknife で 8.88e-16）．`exp/dai1_stats.py eval` の流し直しで `families_flagged.csv` の 176 行のすべての列が 4.4e-16 以内（所要 2 分 2 秒）．K = 10 の 48 行で `outside_families.csv` との差の最大 0.0．
- 手法の間で同じ単位を使っているか：`effect_stats` は 2 つの行の同じ (分割，種) の組を突き合わせ，rows.csv からの作り直しでも (fold，seed) の鍵で 50 単位がそろった．
- K = 10 の差と区間の値は画面に出していない（見たのは行の数，照合の差，NaN の数，`identical` の数，`folds_dropped` の文字列だけ）．

## A．本番の規模の見積もり

本番は小さく流した規模と同じ（全部の単位）である．CPU で約 1.5 分，GPU とメモリの心配は無い．出力は 2 ファイルで合わせて約 30 KB．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| K = 10 の族 1・3 の表 | 区間を話者（塊）を単位に作っているか | tac-round1-status（話者を単位の区間），9.3 節 S1 | (a)．`families_k10.csv`（または `outside_families.csv`）の lo，hi，method の列．AFEW・IEMOCAP・MELD は話者の塊の jackknife，MOSEI は塊の引き直し | — | — |
| 同 | K = 10 にも Holm の補正をした p を出せ／K = 10 を含めた族で補正せよ | 9.3 節 S5，L1 | (a)．補正していない `p` の列から族ごとに `exp/dai1_stats.py:231` の `holm` で計算できる．ただし設計は族の外の数に p 値を付けないと決めたので，出すなら計算の後の解析と明記する | — | — |
| 同 | K = 10 のセルも検出力の見込みで「記述に回す」か判定したか | 9.2 節（記述に回すセル），`families_flagged.csv` の `descriptive` | (b) | 固定のファイルを書き換えない形で，`detect` と `expected_effects` の計算を K = 10 のセルにも当てた表を別のファイルに書く（仮のターゲットの単位 `results/dai1/pseudo/units/` は保存済み．評価のテストは使わない）．`detect` は frozen.json を上書きするので，そのままは呼べない | CPU で数分，ディスク数 KB |
| 同 | MOSEI の区間は乱数の種で変わらないか | S1，R（感度） | (a)．種を替えて台本を流し直すだけで出る（1.5 分）．検査役の実測で，種 20261104 と 20261116 の違いで端が最大 0.042 点（K = 5），0.021 点（K = 10 の `outside_families.csv` との比較）動く | — | — |
| 同 | ターゲットの感情ごとの内訳（再現率）を K = 10 で出せ | review-affect の例，9.3 節 S5「感情ごとの再現率」 | (a)．`results/dai1/stats/desc/recall.csv` と `confusion.csv` に K = 10 の行がある．無い組み合わせも npz の発話ごとの判定（`p3_<s>_K10_pred`，`p1_<s>_pred`）と `data/metadata` のラベルから作れる | — | — |
| 同 | 感度の解析（MOSEI のチャンネルが分かる発話，IEMOCAP のマーカー，書き起こしの注記，AFEW の書き起こしの疑い，MELD の顔が競る発話，MELD の対話の塊）を K = 10 でも出せ | 設計 9.4 節，tac-round1-status（話者を絞った測り直し） | (b) | `results/dai1/stats/desc/sensitivity.csv` と `sensitivity_folds.csv` は K = 0，5，25 だけを持つ．`exp/dai1_desc.py` の感度の表の K の並びに 10 を足した版を，置き場を替えて流す．学び直しは要らない（npz の発話ごとの判定と `frozen/subsets.csv` から作れる） | CPU で数分〜十数分，ディスク数十 KB |
| 同 | 性別ごとに出せ | tac-round1-status（性別ごとの UAR） | (a)．npz に発話ごとの判定と `sample_id` があるので，`data/metadata` に性別の列があるコーパスでは単位ごとの UAR を性別ごとに作れる（性別の列の有無は読んでいない） | — | — |
| K = 10 の族 3 | 相手は提案と同じ教師（予測）を受け取っているか．版が違うセルはそろえよ | 9.3 節 族 3 の注（m57，m58），比較の検査 | (a)．`fam3_notes.csv` の K = 10 の 16 行に，相手の手法・版・設定と，版をそろえた差（`aligned_effect`，`aligned_lo`，`aligned_hi`）がある | — | — |
| K = 10 の族 4 | 選ぶことに意味があったか | 9.2 節（同じ識別器） | (a)．固定のファイルだけで答えが出る：16 条件すべてで選んだ設定 = 既定，差は定義から 0．`identical` の列で数えられる | — | — |
| K = 5・10・25 を並べた図 | K の間の差（例：K = 10 の得 − K = 5 の得）に区間を付けよ／K = 10 は K = 5 と独立か | S（独立でない比較），`exp/dai_data.py:132` の入れ子 | (b)（検査役が流した計算では (a) にできる） | 単位ごとの差 d_fr（family，T，student，K，fold，seed，d）を長い形の csv に保存する．保存しなくても npz から 1.5 分で作り直せるので，学び直しにはならない | 4 コーパス × 3 族 × 4 生徒 × 3 K × 50 単位 = 7,200 行，約 300 KB |
| K = 10 の族 2 | 役割を分ける版も K = 10 で同等に扱え | 研究者の指示（K = 10 も同等），設計 S5 | (a)．`outside_families.csv` の family = 2，K = 10 の 16 行 | — | — |
| K = 10 の表全体 | 第 2 のエンコーダ（HuBERT，CLIP，DeBERTa）でも K = 10 を出せ | 設計 9.5 節 | 未確認（`results/dai1_enc2` の有無は読んでいない） | — | — |
| K = 10 の表全体 | 新しいコーパス，エンコーダの微調整 | — | (c)．範囲の外は DAI の設定（凍結したエンコーダ，4 コーパス）で決まっている | — | — |

「止める」に当たる行は無い．(b) の 3 行（K = 10 の記述に回す印，K = 10 の感度の解析，単位ごとの差）はどれも保存済みの npz と仮のターゲットの単位から CPU で作れ，学び直しを要しない．

## 指摘

### [I] 1 同じ量の表が 2 つになり，MOSEI の区間の端が表ごとに違う
- 重さ：直してから進む
- 該当箇所：`tools/dai1_k10_families.py:3-6`（「K = 10 の差と区間は集計していなかった」），`:14`（出力を `results/dai1/stats/families_k10.csv` に書く），`:33`（`RNG_SEED = 20261116`）
- 問題：K = 10 の族 1〜4 の差と区間は，設計 9.3 節 S5（「族の外の数（S5）：表と図のすべての数（K = 1・10，…，役割を分ける版の K = 0・1・10，…）に同じ規則の区間と単位の数を添え，p 値は付けない」）で評価の回の前に決められ，`exp/dai1_desc.py:617-625` の `t_outside` が評価のコミット ba98974（dirty 偽）で `results/dai1/stats/desc/outside_families.csv` に書いている．流して確かめた結果，台本の K = 10 の 48 行は，この表と差が完全に一致し，jackknife の区間の端も完全に一致し，MOSEI の区間の端だけが最大 0.0206 点違う．台本の出力を `results/dai1/stats/` に置くと，同じセルの MOSEI の区間が 2 通りになり，論文の数値の検査（`paper-numbers`）でどちらが正本かが決まらない．また台本は出力に `posthoc=True` を付けるが，量そのものは計算の前に決めた族の外の記述である（計算の後に決めたのは「論文で主の表と同じ重さで載せる」ことだけ）．
- 計算の後で直せるか：直せる（どちらの表も保存済みの単位から作り直せる）
- 求める対応：K = 10 の正本を `outside_families.csv` にする．台本は `results/dai1/stats/` に新しい表を書かず，`outside_families.csv` との照合（差 1e-9，jackknife の端 1e-9，MOSEI の端は記録）をする道具に変えるか，台本の出力を使うなら `outside_families.csv` を使わないと結果の文に書く．台本の冒頭の「集計していなかった」の文を直し，`posthoc` の意味（量は計算の前に決めた族の外の記述で，主の表に格上げしたのが計算の後の判断）を書く．

### [L] 2 族 4 の K = 10 は 16 条件すべてで同じ識別器の比較である
- 重さ：直してから進む
- 該当箇所：`results/dai1/select/frozen.json` の `select["<s>|10|joint|one"]`（4 コーパス × 4 生徒のすべてで ρ = 1.0，a = 1.0），`main["10"] = "one"`．台本 `:34` の族 4
- 問題：族 4 の A（選んだ設定）と B（既定：版 one，ρ = 1，a = 1）が 16 条件すべてで同じ設定の同じ行になる．流した出力で族 4 の 16 行すべての `identical` が真だった．これを表に「差 0.0 [0.0, 0.0]」と載せると，「選んでも選ばなくても同じ水準」という結果に読める．設計 9.2 節は K = 0，5，25 のこの種のセルを「同じ水準」と数えず「同じ識別器」として別に数えると決めている．
- 計算の後で直せるか：直せる（書き方の問題）
- 求める対応：K = 10 の族 4 は「選んだ設定は 16 条件すべてで既定と同じ（同じ識別器）」と 1 文で書き，区間の表に載せないか，載せるなら同じ識別器の印を付ける．台本の出力でこの 16 行を区別できるよう，`identical` と別に「設定が同じ」の列（`spec_A == spec_B`）を足すとよい．

### [I] 3 族の外の数に付けない p 値が出力に残る
- 重さ：直してから進む
- 該当箇所：`tools/dai1_k10_families.py:12`（「Holm の補正の後の p は書かない … 補正していない 95% 区間だけを書く」），`:61`（`**e` で `effect_stats` の戻りを全部書く）
- 問題：冒頭の説明は区間だけを書くと言うが，出力には補正していない `p` と `t` の列が入る．設計 S5 は K = 10 の行に p 値を付けないと決めている．`outside_families.csv` は p の列を持たない（`exp/dai1_desc.py` の `_fmt` が落とす）．
- 計算の後で直せるか：直せる
- 求める対応：`p` と `t` の列を落とすか，残すなら列の名前を `p_unadjusted_not_for_claims` のようにして結果の文で使わないと台本の冒頭に書く．

### [I] 4 記録が足りない
- 重さ：記録する
- 該当箇所：`tools/dai1_k10_families.py:65-66`，`exp/dai1.py:70-75`
- 問題：(1) `dai1.git_commit` の dirty は `exp splits measure features data/metadata` だけを見るので，`tools/dai1_k10_families.py` を書き換えたまま流しても `dirty` が偽になる．(2) DAI/CLAUDE.md の取り決め（`results/<実験>/provenance.json` か `provenance_log.jsonl` に台本・共通の部品・入力の sha256 を残す）を満たさない．(3) `units_commit="ba98974"` は手で書いた値である（npz の `commit` の値 ba98974583df… と一致することは確かめた）．(4) 使った設定の文字列（`spec_A`・`spec_B`）と固定のファイルの sha256 の列が無い．(5) 照合で止まると `families_k10_check.csv` だけが残り，次に流すと「既にある」で止まる．
- 計算の後で直せるか：直せる
- 求める対応：`results/dai1/provenance_log.jsonl` に 1 行足す（台本，`exp/dai1_stats.py`，`frozen.json`，`families_flagged.csv`，`outside_families.csv` の sha256，HEAD，tools を含めた dirty，乱数の種）．`units_commit` は npz の `commit` から読む．`spec_A`・`spec_B` の列を足す．

### [S] 5 K = 10 の行に検出力の見込みに基づく印が無い
- 重さ：記録する
- 該当箇所：`results/dai1/select/frozen.json` の `<T>.detect.cells`（K = 10 の鍵が 4 コーパスとも 0 個），`results/dai1/stats/families_flagged.csv` の `descriptive` の列
- 問題：K = 5，25 のセルには評価の前に「記述に回す」の印が付いたが，K = 10 には付いていない．研究者の指示で K = 10 を K = 5 と同じ重さで扱うと，K = 5 のセルは「記述」，K = 10 のセルは印なし，という違いが表に出る．
- 計算の後で直せるか：直せる（仮のターゲットの単位から CPU で作れる．評価のテストを使わない）
- 求める対応：K = 10 は族の外の記述として扱い，検定の文を書かないと結果の文に書く．印を揃えたいなら，固定のファイルを書き換えない別のファイルに `detect` の計算を K = 10 で当てた表を作る（B の表の 3 行目）．

## 目録に無い種類の不備

- **計算の後の解析が，計算の前に決めて既に作った表と同じ量を別の乱数で作り直す**．同じセルに 2 つの区間ができる．`docs/methods.md` 9.3 節に次の項目を足す案：「計算の後の解析の台本は，書く量が既にある結果の表（とくに設計で族の外と決めて作った表）に無いかを，置き場（`results/<実験>/stats/desc/`）の列と K の値で確かめる．あるなら新しい表を作らず，既にある表との照合にとどめる．作るなら乱数の種と流れを元の表と同じにし，端の値まで一致させる」．
