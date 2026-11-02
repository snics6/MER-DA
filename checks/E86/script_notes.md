# E86 台本の記録（`exp/analyze_pair_routing_e86.py`．台本を書いた担当のメモ．2026-11-04）

設計の正本：`research_notes/研究提案_2026-11-04/段0a_設計案.md`（E86 1〜5 節）．台本は設計の 2・3 節のとおりに計算する．設計に無い決め事は台本の冒頭の docstring（D1〜D14）に列挙し，本記録の 2 節に写した．
値は何も見ていない（試走の出力は行数・列名・NaN の有無・恒等式だけを確かめ，削除した）．`results/` の下には何も書いていない．git のコミットはしていない．

## 1. 設計の項と台本の対応

| 設計の項（2・3 節） | 台本の関数・場所 | 出力 |
|---|---|---|
| 入力：E80 の単位ファイル（`cand`・`te`・`y`・`yte`・`lab__K`・`P__<kind>`・`ref__K`・`none__<f>__K`） | `Unit.__init__`（`np.load`．行番号・ラベル・`lab__K`・`P__` の鍵の数・行の和を照合） | `checks.csv` の行 `rows_labels_vs_manifest`・`labK_subset_of_cand`・`P_keys_present`・`prob_rows_sum_to_one` |
| 入力：`samples_<c>.csv` の `speaker_group`（区間の話者群） | `main`（`sample_id` と `row` を発話一覧と照合），`Unit.__init__`（`self.groups = speaker_group[cand]`） | `checks.csv` の行 `samples_sample_id_vs_manifest` |
| 入力：特徴（`load_any(c, <fused>)` の行を `cand` で引き，f のブロックだけ） | `feats_for`（E85 と同じ．kind の順に 1024・512・768 のブロック），`Unit.feat` | — |
| ラベルは `load_any` が貼り直したもの（npz の `y` と一致を確かめる） | `Unit.__init__`（一致しなければ止まる．作業用のラベルは発話一覧のもの） | — |
| 世界・f・K・単位（世界，T，c，K）・反復 0〜4 | `WORLDS`，`fused_parts(fused)[1]`（f），`KS`，`main` のループ（T → c → r），`--reps` | — |
| K ≥ 1 では候補から `lab__K` を除く | `process_unit`（`keep[u.labs[K]] = False`） | `pairs.csv`・`base.csv` の `n_cand_kept` |
| 比較の基準 a ＝ `none__<f>__K`（指数を取って正規化） | `to_prob`（最小値が負なら指数．行の和 10⁻⁴），`process_unit` の `A` | `pairs.csv` の `acc_a` |
| 既定 q⁰（主 `ref__K`，副 `P__<fused>`） | `process_unit` の `q0`（`DEFAULTS = ("ref", "pF")`） | `default` 列 |
| 候補 s ＝ 残りの出発点の積（av は残り 1 つ） | `product_model`（要素ごとの積を正規化） | `model = s` |
| 記述：残りの単モダリティ（avt だけ） | `process_unit`（`len(others) >= 2` のとき `other_<m>`） | `model = other_<kind>` |
| 階層の形 h(s, α)，α ∈ {0.5, 1}，既定ごと | `hier`（怒り：α·s＋(1−α)·q⁰．怒り以外：(1−h(怒))·q⁰(c)/(1−q⁰(怒))） | `model = h1`，`h05` |
| 対ごとの補い J_kl（2 感情の判定．E83 (C) と同じ） | `pair_right`（k と l の確率の大きい方．同点は k），`j_stat`（100·[P(当たり∧外す) − P(外し∧当たる)]） | `pairs.csv` の `J`・`acc_model`・`n_pair` |
| 差 Δ_kl ＝ J(モデル) − J(q⁰) | `process_unit`（`J[name][p] - J["q0"][p]`） | `pairs.csv` の `Delta`・`J_q0` |
| δ_J：J(q⁰) の反復の SD の（T，c，対）の中央値 × 2 ÷ √5．感度 ½・2 倍 | `delta_table`（ddof=1．÷√(反復の数)） | `delta.csv`（`delta_J`・`delta_J_half`・`delta_J_double`） |
| 区間：反復ごとに話者群を復元抽出して Δ を引き直し，反復で平均．1,000 回．種は（世界，T，c）．2.5・97.5 % | `main`（`rng = default_rng([86, crc32(world), crc32(T), crc32(c)])`，反復ごとに `W = multinomial(G, 1/G, size=B)`），`process_unit`（話者群ごとの当たりの数 `num`・`den` を先に集計し `W @ num / W @ den`），`finalize_boot`（反復の平均 → 百分位） | `units.csv` の `lo`・`hi`・`n_boot_valid` |
| 話者群が 10 未満の単位の印 | `units_table`（`few_groups = n_groups_min < MIN_GROUPS`） | `units.csv` の `few_groups` |
| 判定（上回った／下回った／同じ水準／判定できない） | `judge`（E79・E83 と同じ 4 区分 pos/neg/same/undet．欠けは missing），`units_table` | `units.csv` の `judge_main`・`judge_half`・`judge_double` |
| 土台 D_kl（話者群で 2 分，Ledoit–Wolf の LDA，入れ替えて平均） | `base_D`（`speaker_folds(k=2, seed=0)` → `explore_pairs.own_separation` をそのまま呼ぶ．標準化は K 本を除いた候補の平均・標準偏差） | `base.csv` の `D`・`n_half0/1`・`n_groups_half0/1` |
| 土台 d_kl（`none__<f>__K` の対数確率の差．`d_o_pairs` と同じ式） | `pair_d`，`to_logp` | `base.csv` の `d` |
| 方向の誤り ＝ 100·[Φ(D/2) − Φ(d/2)]，反復で平均．条件：≥ δ_J．負の数と割合 | `dir_err`（`explore_pairs.ba`），`units_table`（`dir_err_mean`，`base_ok_<scale>`，`dir_err_neg`），`counts_table`（`n_dir_err_neg`・`share_dir_err_neg`） | `base.csv` の `dir_err`，`units.csv`，`counts.csv` |
| 分母と偶然の見込み（候補ごと・既定ごと．怒りを含む 3 対：avt 72，av 216．土台後の分母）．怒り以外の 3 対も | `counts_table`（K ごとと `K = all`．`pair_type` ∈ {anger, nonanger}．`chance = n_cells × 0.025`，`chance_base = n_base × 0.025`） | `counts.csv` |
| 判定の規則 R86-1・R86-2（偶然の見込みを超えるか） | `counts_table` の列 `exceeds_chance_base`（土台後．主）・`exceeds_chance_all` | `counts.csv` |
| コーパスの塊（c のコーパス，対）：1 回でも／過半で上回った．塊の数 × 0.025 と注 | `corpus_table`（`level = block` の行と `level = total` の行） | `corpus_counts.csv` |
| 怒りと答える割合（q⁰・s・h．本当の割合も） | `process_unit`（argmax ＝ 3 の割合），`anger_table`（反復の平均と反復ごとの列．R86-4 の s との差 `diff_vs_s`・`within_2pt_of_s`） | `anger_share.csv` |
| 階層の形の切り分け R86-3（s が上回ったうち h(s,1)・h(s,0.5) も上回った数） | `hier_table`（全体と土台後．`majority_h1`，`h05_le_h1`） | `hier_counts.csv` |
| 照合：`results/e83/` の (C) の J と 10⁻⁶ で一致（無ければ飛ばす） | `compare_e83`（`c_complementarity_units.csv`．`g = ref` ↔ `model = q0`（既定 ref），`g = other_<m>` ↔ `other_<m>`．E83 が値を出す n_pair ≥ 10 の行だけ） | `checks.csv` の行 `e83_compare`（今は `skipped`：ファイルが無い） |
| 退化の検算 `--selftest` | `selftest`（J(a, a) ＝ 0，J ＝ 正解率の差，dir_err(D, D) ＝ 0，`base_D`・`pair_d` が既知の離れ方を再現，α ＝ 0 は q⁰，α ＝ 1 の h(怒) ＝ s(怒)，行の和 1，怒りを含まない対の判定は q⁰ と同じ，`to_prob` の対数／確率と和の検査，同じモデルのブートストラップ Δ ＝ 0，`judge` の 4 区分） | 標準出力 |
| 試走 `--smoke` | `main`（avt の最初の T，反復 1，ブートストラップ 50．既定の出力先 `checks/E86/code_run/smoke`） | — |
| 記録（provenance，台本の写し，CSV の先頭行，古い出力があれば止まる） | `main`（`provenance.json`：commit，dirty（台本が読む `SCRIPT_DEPS` に絞る），台本の sha256，入力の sha256，引数，判定の定数 `CONSTANTS`，スレッド数，版，所要時間），`shutil.copy` → `script_copy.py`，`write_csv`（`# E86 commit=... dirty=... worlds=... reps=... boot=... targets=... smoke=...`） | — |

## 2. 設計に無い決め事（台本の docstring の D1〜D14 と同じ）

- D1 対数で保存された確率の見分け方：配列の最小値が負なら対数（log_softmax）とみなして指数を取る．行の和の検査（10⁻⁴）は指数を取った後の和に当てる．d_kl に使う対数確率は，対数で保存された配列はそのまま，確率の配列は log(max(p, 1e−12))．
- D2 階層の形で q⁰(怒) が 1 に極めて近い行：q⁰ を 1e−12 で下から切って正規化してから使う．怒り以外の比 q⁰(c)/(1 − q⁰(怒)) の分母は怒り以外の 3 つの和で計算する（同じ値．1 − q⁰(怒) の桁落ちを避ける．最初の版で `--selftest` が端の行の和の検査に落ちたので直した）．
- D3 2 感情の判定の同点は番号の小さい k とする（E83 (C) の `>=` と同じ）．
- D4 対の最小の本数は設けない（E83 (C) は 10）．対のどちらかの感情が候補に無い反復は J・Δ を NaN にし，`n_pair` を列に残す．点の値は NaN を除いた反復の平均（`n_reps_finite`）．E83 との照合は n_pair ≥ 10 の行だけ．
- D5 土台の 2 分割は，K 本を除いた候補の話者群に `speaker_folds(k=2, seed=0)` を当てる（K ごとに分け直す）．どちらかの半分でどれかの感情が 2 本未満なら D は NaN（`checks.csv` に記録）．特徴の標準化の統計も K 本を除いた候補から取る（標準偏差に 1e−8．E85 と同じ）．
- D6 ブートストラップの重みは，反復ごとに候補（K 本を除く前）の話者群の全体に多項分布を 1 回引き，同じ重みを全 f・K・既定・候補・対に使う（K 本を除いて空になった群は分母に入らない）．種は [86, crc32(world), crc32(T), crc32(c)]．反復 r の重みは r 番目の引き．分母が 0 になった引きは NaN とし，反復の平均と百分位から除く（`n_boot_valid`）．
- D7 δ_J の標準偏差は ddof ＝ 1．反復が 2 未満の (T, c, 対) は材料から除く（試走の反復 1 つでは δ_J が無く判定は missing）．÷√5 は ÷√(反復の数)（本番は 5）．
- D8 判定の呼び方は E79・E83 と同じ pos・neg・same・undet．点・区間・δ_J のどれかが無ければ missing．
- D9 コーパスの塊の「過半」は，塊の中の (T, K) のセルのうち pos が厳密に半分を超えること．土台で条件づけた版も並べる．
- D10 怒りと答える割合は反復ごとの値も列に残す（`share_anger_r<r>`）．`other_<m>`（avt）についても出す．R86-4 の ±2 点は s との差 `diff_vs_s` の列で読む．
- D11 候補のモデル s の積が全て 0 の行があれば止まる．
- D12 偶然の見込みは 1 セルあたり 0.025．塊の偶然の見込みは 塊の数 × 0.025（T・K をまたぐ数え上げの偶然の見込みはこれより大きい，と `note` 列に書く）．
- D13 `--targets`（T を絞る）と `--boot`（回数）を足した．部分的な流し方（`--smoke`，`--targets`，`--reps < 5`，`--worlds` の一部）の出力は `results/` の下に書けない．
- D14 BLAS のスレッド数を 4 に固定する（環境変数が既にあればそれに従う）．理由は 3 節．

## 3. 試走の所要時間と本番の見込み

- 環境：WSL2，16 論理コア，`.venv`（numpy 2.5.2，sklearn 1.9.1，scipy 1.18.1，pandas 3.0.6）．GPU は使わない．
- 最初の試走（既定の 16 スレッド）は 2 単位ファイルで 152 秒（52.7 秒と 93.7 秒．CPU 時間 30 分）だった．cProfile で時間の 95% が `own_separation` の中の Ledoit–Wolf の `pinvh`（1024×1024 の固有値分解．1 回 3.2 秒）と `np.linalg.solve`（1.8 秒）にあると分かった．人工の 4600×1024 の行列で測ると，既定のスレッド数では固有値分解 3.36 秒・solve 1.58 秒，4 スレッドでは 0.43 秒・0.02 秒で，共分散の値は同一（`np.array_equal`）だった．スレッドの取り合いが原因なので，台本の冒頭でスレッド数を 4 に固定した（D14）．
- 直した後の試走（`--smoke`：avt の IEMOCAP-impro，c ＝ MELD・MOSEI，反復 1，ブートストラップ 50）：2 単位ファイルで 29 秒（10.8 秒と 13.5 秒．特徴の読み込みを含めた実時間 49 秒）．検査用の別の流し（`--worlds avt --targets IEMOCAP-impro --reps 2 --boot 50`）：4 単位ファイルで 52 秒（10.7〜13.1 秒／ファイル）．この 2 つの c は候補が最も多い部類（MELD・MOSEI）である．
- 本番の見込み：単位ファイル 160 本（avt 40，av 120）．1 ファイル 8〜13 秒（av は f が 2 つなので LDA が 2/3，AFEW・CREMA-D・IEMOCAP は候補が少ない）＋ ブートストラップ 1,000 回（話者群ごとの当たりの数を先に集計して行列積で引くので 1 ファイル 1 秒以内の見込み）＋ 特徴の読み込み（c ごとに数秒）．合計 25〜35 分．tmux の `mer-da` で流す．

## 4. 確かめた行数と構造（値は見ていない）

`--selftest`：7 項目すべて通過（1 節の表の「退化の検算」の行）．

`--smoke`（2 単位ファイル．出力は scratchpad に書いて確かめた後に削除）：

| ファイル | 行数 | 列数 | 確かめたこと |
|---|---|---|---|
| `pairs.csv` | 1,296（＝ 2 ファイル × f 3 × K 3 × 既定 2 × モデル 6 × 対 6） | 21 | NaN なし．鍵の重複なし．`J == 100·(acc_model − acc_a)` が全行で 10⁻⁹ 以内．q⁰ の行の Δ ＝ 0．怒りを含まない対の h1・h05 の Δ ＝ 0．K ＝ 0 で `n_cand_kept == n_cand`，K ＝ 5 で −20，K ＝ 25 で −100 |
| `base.csv` | 108（＝ 2 × 3 × 3 × 6） | 21 | NaN なし（D の飛ばしは無かった） |
| `units.csv` | 1,080（＝ 1,296 × 5/6．q⁰ を除く） | 37 | `delta_J` だけ NaN（反復 1 つなので想定どおり．判定は全行 missing）．`lo ≤ hi` 全行．`n_boot_valid` ＝ 50 全行 |
| `delta.csv` | 0 | 10 | 反復 1 つなので空（想定どおり） |
| `counts.csv` | 720 | 26 | `n_cells = n_pos + n_neg + n_same + n_undet + n_missing` 全行．`K = all` の怒りの対の分母 18（＝ 2 単位 × 3 K × 3 対．本番の avt では 72）．`share_pos_base` の NaN は `n_base = 0` の行だけ |
| `corpus_counts.csv` | 1,260（block 1,080 ＋ total 180） | 27 | NaN は block／total で使わない列だけ |
| `anger_share.csv` | 216（＝ 2 × 3 × 3 × 2 × 6） | 15 | NaN なし．鍵の重複なし |
| `hier_counts.csv` | 288 | 17 | 割合の NaN は `n_s_pos = 0` の行だけ |
| `checks.csv` | 11 | 7 | 状態は ok と skipped（E83 の照合は `results/e83/c_complementarity_units.csv` が無いので skipped） |
| `provenance.json` | — | — | commit・dirty（`?? exp/analyze_pair_routing_e86.py` だけ）・台本の sha256・入力 4 本の sha256・引数・定数・スレッド数・版・所要時間 |

反復 2 つの検査用の流し（4 単位ファイル）：`pairs.csv` 2,592 行，`base.csv` 216 行，`units.csv` 1,080 行で NaN なし（`delta_J` は全行有限，`judge_main` の missing は 0），`delta.csv` 18 行（＝ f 3 × K 3 × 既定 2）で全列有限，`anger_share.csv` に `share_anger_r0`・`share_anger_r1` の列．

## 5. 本番の前に本体のセッションが決めること

- `results/e83/` に (C) の出力が無いので，E83 との照合は本番でも skipped になる（`checks.csv` に残る）．E83 を先に流すなら，その後で E86 を流せば照合が働く．
- 本番の命令：`cd ~/MER-DA && ./.venv/bin/python exp/analyze_pair_routing_e86.py`（既定：両世界，反復 5，ブートストラップ 1,000，出力 `results/e86/`）．出力先に CSV があれば止まる．
- 関門：`gate-probe`（設計）と `gate-code`（台本）の検査の後に流す．
