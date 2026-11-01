# E82 の実装の記録（2026-11-02．台本を書いた担当．関門 1 の前．コミットしていない）

設計は `docs/exploratory-analyses.md` の「E82．P4 の手法の出発点への罰則を選び直す」，事実の調べは `checks/E81/step7_facts.md` の 5 節．
台本は 2 本：GPU で E82 の行を作る `exp/s43_penalty.py` と，E79 の関数で判定を出す CPU の読み手 `exp/analyze_e82.py`．
`exp/compare_dai.py`・`exp/compare_dai2.py`・`exp/e80_stage_c.py`・`exp/analyze_e79.py` は変えていない（読み込んで関数を使うだけ）．
この記録には評価のターゲットの値（UAR・正解率・向きの損）を書かない．書くのは差・許容幅・数・形・所要だけである．
試しの出力はすべて `/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/e82_test/`（以下 `$SP`）にあり，`results/` には何も書いていない．

## 1. `exp/s43_penalty.py`（GPU．E82 の行を E77 と同じ形で書く）

### 1.1 何をするか

| 設計（E82 の 2 節） | 台本の箇所 |
|---|---|
| entmin・IM・自己学習の C′ だけを倍率倍にする．`Prob2.solve`・`im_solve` が解くときに読む `pr.s["cp"]` を書き換え，解き終えたら戻す | `run_methods`（`s43_penalty.py:102-147`．書き換えは `:110`，戻しは `:147` の finally．書き換える前に C′ が E39 の値かを確かめる `:108-109`） |
| self は τ 3 × m_eff 4 × 3 巡，entmin は m_eff 4，IM は α 3 × m_eff 4．式と束ね方は `compare_dai.run_ab` の 397〜427 行と同じ（関数 `hard_rows`・`CD.im_solve`・`CD.shaken` をそのまま呼ぶ） | `:113-145` |
| TIM（α 0.1，総重み 40K）は解かない（E79 の im の族は m_eff ∈ MEFFS だけを選ぶので候補にならない） | `:133-134` |
| 評価の単位では entmin・IM を種 1・2 の揺らした出発点からも解く（E77 の `--nc-seeds 1 2`） | `:136-145`（`--nc-seeds`） |
| none・自己学習の出発点 P0・entmin と IM の初期値は E39 の C′ で解いた none | `solve_student`（`:150-161`）．none は E77 の本番と同じ束（K 0・5・25，`--none-ks`．`:65`）で解き，行は `--ks`（既定 0・5）だけを書く |
| C_b・ρ_M・罰則の中心（E39 の下書き）・生徒・候補・K 本・テストは E77 と同じ | 生徒は `kind_student`（`:166-171`，`compare_dai.run_kind` の student_of と同じ式）と `fused_student`（`:174-185`，`make_students` の融合の生徒だけ）．単位の作り方は `run_kind`（`:228-272`）・`run_world`（`:275-318`）で，`compare_dai.run_kind`・`run_world` と同じ（`CB.Ctx`，`BD.label_sets_of`，`M5.pool_info`，仮のターゲットは合併の C = C_FIT，種類の評価は C_T） |
| 範囲：種類の A・V・T と，AV・AVT の融合の生徒 | `--kind {audio_wavlm, video, text_roberta}`・`--world {fuse-audio-video, fuse-audio-video-text}`（`:428-429`）．世界では融合の生徒だけを作って解く（モダリティの生徒は作らない） |
| 仮のターゲットの単位は E77 と同じ．IEMOCAP-script は即興の写し（短縮 a） | `--copy-pseudo` は `compare_dai.copy_pseudo` をそのまま呼ぶ（`pseudo_header`，`:220-225`）．写しの照合（コミット，`_settings.json`，起動の引数，組，K，npz）も E77 と同じ |
| 行の鍵：param に倍率の印，種の印は最後 | `ctag`（`:70-72`）：entmin `-C3x`・`-C3xs1`，im `a0.1C3x`・`a0.1C3xs1`，self `0.5C3x`．`analyze_e79.strip_seed`・`seed_of` がそのまま働く（種は `s\d+$`） |
| 倍率 1 の行は作らない | 既定は `--mults 3 10`．`--mults 1` は恒等の試し専用で，E77 と同じ param を書く（読み手は E77 と重なる鍵で止まる） |

出力（`--out`，既定 `results/e82_<名前>`）は E77 と同じ名前と形：`<T>.csv`，`<T>_r<r>_logp.npy`・`_pred.npy`・`_rowids.npy`・`_te.npy`・`_W_<生徒>.npy`・`_b_<生徒>.npy`・`_wrows_<生徒>.npy`（`CD.Rec.save`），`<T>_pseudo.csv`，`<T>_ps_<c>_r<r>_none.npz`（`CD.Rec.save_pseudo`），`<T>_samples.csv`，`<T>_settings.json`（E39 の設定と倍率ごとの C′），`<T>_conv.csv`（解きごとの所要と収束．E77 に無い IM の所要も `im_timed`（`:91-99`）で記録），`<T>_convp.csv`（問ごとの反復の回数・勾配・解いた C′．`Prob3.note`，`:79-86`），`<T>_diag.csv`（自己学習の疑似ラベル），`<T>_cp.csv`（単位 × 生徒 × 倍率の E39 の C′ と使った C′），`provenance.json`・`provenance_log.jsonl`．
`convp` の C′ は `fit_batch` の結果の辞書の鍵（`f.n_iter[cp]`）から読むので，解きに使った C′ と違えば KeyError で止まる．

1 単位の行の数：評価 242 行（none 2，倍率ごとに self 24・entmin 24（種 0・1・2）・IM 72（同））．仮のターゲット 114 行（none 2，倍率ごとに self 24・entmin 8・IM 24）．

### 1.2 フラグ

`--kind`/`--world`，`--targets`（`--target` も可），`--reps`（既定 0〜9），`--pseudo-reps`（既定 0〜4），`--pseudo-targets`（試し用．`none` で回さない），`--ks`（既定 0 5），`--none-ks`（既定 0 5 25），`--mults`（既定 3 10），`--copy-pseudo`，`--nc-seeds`（既定 1 2），`--out`（`--out-dir` も可），`--gpu-fraction`（既定 0.3），`--allow-dirty`（試し専用），`--check <E77 の場所>`（GPU を使わない照合）．
本番では `exp/`・`measure/` に未コミットの変更があるか，この台本が未追跡だと止まる（`:463-466`）．provenance にはコミット，dirty の印，台本と `compare_dai.py` の sha256，argv，K・none の K・倍率・種・格子，ライブラリと GPU の版，ターゲットごとの所要を書く（`:481-490`，終わりに `:505-511` で所要と GPU の最大を足す）．

### 1.3 照合（`--check`）

`run_checks`（`:328-420`）：E82 の none の行が E77 の none と同じか（仮のターゲットの UAR の差 ≤ 1e−9，`_ps_*_none.npz` の対数の確率の差 ≤ 1e−5，評価の対数の確率の差 ≤ 1e−5．E79 の許容幅と同じ），相手の無い none が無いか，テストの行（`_te.npy`）と `_samples.csv` が同じか，`<T>.csv` の行と `logp` の行の数がそろうか，倍率 3・10 の行が E77 の鍵と重ならないか（倍率 1 を含む出力では重なった行の差を手法ごとに測る）．`check.csv` に書き，画面には差と数だけを出す．食い違いがあれば終了の値 1．

## 2. `exp/analyze_e82.py`（CPU．E79 の P4 の段を 2 つの版で流して比べる）

### 2.1 何をするか

- 版 `--grid fixed`：E79 の関数をそのまま使う（E77・E78 の行だけ）．E79 の P4 の判定の作り直し．compare の段では `--fixed-from <E79 の --out>` で E79 の本番の出力を読むこともできる（同じファイル名を読む）．
- 版 `--grid reselect`：E79 の関数に次を差し込む（`install`，`analyze_e82.py:200-211`）．E79 は 3 つ目の波を E80 の段 c の場所（大域の `E80C`，波の名前 `e80c`）から読む仕組みを持つので，E82 の場所をこの枠に入れる（`SLOT`，`:63`．本物の段 c の場所は読まない．P4 の判定に使わない行である）．
  - `E80C_METHODS` を entmin・im・self にする（`load_pseudo` と `measure_target` が段 c の行から残す手法．none は E79 のとおり E77 の none と照合してから落とす）
  - `sel_defs` を `sel_defs_e82`（`:107-113`）にする：entmin・self・im の族に，E77 の行と同じ判定の関数で e80c の波の行を足す．ほかの族と規則（excl の変種，none を除いて選ぶ，ns_excl，区間，Holm の族）は E79 のまま
  - `check_e80c_prov` を `check_e82_prov`（`:116-146`）にする：実験 E82，実行の名前，K ＝ 0・5，倍率 3・10，dirty や `--allow-dirty` で流していないか
  - `diagnostics` を `diagnostics_e82`（`:155-197`）にする：E79 の診断のあとに，e80c の波から選んだ設定の「働かなかった」の材料（none とのテストの予測の食い違い．判定と同じ種の平均と種 0）を E79 と同じ式（`analyze_e79.py:1745-1763`）で足す．E79 の診断は e77 以外の波の行を E78 の表で探すので，足さないと E82 から選んだ単位が「働かなかった」に替わらない．同じ材料を E79 が既に書いていたら止める
  - `choose` を `choose_e82`（`:90-104`）にする（2026-11-02 の本体の決定．7 節）：E82 の行（e80c の波）を含む候補だけ，仮のターゲットの点数がちょうど同じ候補が並んだら倍率の小さい方（1 → 3 → 10）を選び，倍率が同じ候補の間は E79 の 'method|pnorm|meff' の文字列の順の先頭．E82 の行を含まない族（K 25 を含む）は E79 の choose をそのまま呼ぶ．E79 の `pick_one`（`analyze_e79.py:904`）が大域の `choose` を呼ぶので，台本を変えずに差し替えられる
- 段：delta・select・measure・judge は E79 の段そのもの（`<out>/<grid>/<段>/`．E79 の段の間の照合とコミットの確認もそのまま）．追加の照合：reselect の delta で E82 の `_ps_*_none.npz` を E77 と照らす（`check_ps_none`，`:240-268`），`--e79-delta` を渡すと E79 の delta.csv と sha256 が同じかを見る，select・measure の前に E82 の出力がターゲットごとにあるかを見る（`precheck_e82`，`:227-237`．無いと E82 の候補が黙って抜け，固定の版と同じ選択になるため）．
- compare の段（`stage_compare`，`:371-488`．見張りは select の段と同じ：入力の場所では仮のターゲットの行と provenance しか開かず，data/ も開かない）：
  - `judgments.csv`：3 手法 × 族（P4・P4′）× K × 感度 × 版ごとの判定の数（pos＝減らした，same＝同じ水準，neg＝増やした，undet＝判定できない，inactive＝働かなかった，missing）
  - `judgment_units.csv`：単位ごとの 2 つの版の判定と，変わったか
  - `statements.csv`：筋 4 の文の鍵．族の全手法（E79 の `statements.csv`）と，手法ごと（E79 の `statements_p4` と種の感度の書き直しを 1 手法の組み合わせに当てる．`statements_of`・`adjust_nc_seed0`，`:273-307`）．2 つの版の rule・文の鍵・組み合わせが同じか
  - `decision.csv`：tag main・subset all・P4 と P4′ で，3 手法のすべてで文の鍵（rule）が同じか → 設計の 3 節の書き方の規則の文．10 倍が過半か（excl）
  - `multipliers.csv`・`multiplier_units.csv`：倍率（1・3・10）ごとに選ばれた単位の数（族 × 手法 × K × 版，手法ごと，3 手法の合計），10 倍の数と過半か（新しい同点の規則で選んだ数．`decision.csv` の `majority_mult10_excl`・`grid_note` もこの数），同点に複数の倍率が並んだ単位の数（`n_tie_multi_mult`），同点で倍率の小さい方を選んだ単位の数（`n_tie_chose_lower_mult`．E79 の文字列の順なら選ばれた倍率は `multiplier_units.csv` の `mult_string_order`），固定の版と同じ設定を選んだ単位の数
  - 検算：`statements_recheck.csv`（E79 の statements.csv を combos.csv から作り直して rule が一致するか），ほかの 5 手法（t3a・lp・coral・dann・mme）の判定が 2 つの版で同じか，選択を仮のターゲットの値から（reselect の規則で）作り直して chosen.csv と一致するか（`tie_info`，`:328-368`）
- 画面には行の数・場所・食い違いの数だけを出す．`e82_provenance.json` を段ごとに書く（台本と analyze_e79.py の sha256，コミット，差し込んだもの，E82 の場所，照合の結果）．

### 2.2 フラグ

`--stage {delta, select, measure, judge, compare}`，`--grid {fixed, reselect}`（compare 以外で必須），`--runs`，`--e77`・`--e78`・`--e82`（名前=場所），`--targets`，`--reps`，`--pseudo-reps`，`--out`（既定 `results/e82_analysis`），`--e79-delta`，`--fixed-from`，`--boot`，`--seed`，`--chunk`，`--no-bias-fixed`，`--allow-incomplete`・`--skip-commit-check`・`--allow-dirty`（試し用．E79 と同じ意味）．`--families` は p4 に固定．

本番の流れ（例）：`--stage delta --grid reselect --e79-delta results/e79/delta/delta.csv` → `--stage select --grid reselect` → chosen.csv・fixed.csv の sha256 の先頭 16 文字を文書に書いてコミット（E79 と同じ規則．`require_committed`）→ `--stage measure --grid reselect` → `--stage judge --grid reselect` → `--stage compare --fixed-from results/e79`．

## 3. 試しと結果（A5000，`--gpu-fraction 0.15`．同じ時間に別の GPU の仕事 2 つ（`relearn_e76_finish.py`，別の担当の `g3_split.py`）が動いていた）

台本の版：`exp/s43_penalty.py` sha256 先頭 `c1779329ad3c41da`，`exp/analyze_e82.py` `7f8b84696d7ac8c4`（HEAD 2790344 の上．どちらも未追跡なので試しは `--allow-dirty`）．試しの補助の台本は `$SP/scripts/`．

### 3.1 恒等の試し（倍率 1 の行 ＝ compare_dai.py の行）

```
./.venv/bin/python exp/compare_dai.py --kind video --target AFEW --reps 0 --ks 0 5 --pseudo-targets none --out-dir $SP/id_e77 --gpu-fraction 0.15
./.venv/bin/python exp/s43_penalty.py --kind video --targets AFEW --reps 0 --ks 0 5 --none-ks 0 5 --pseudo-targets none --mults 1 --out $SP/id_e82_final --gpu-fraction 0.15 --allow-dirty
./.venv/bin/python exp/s43_penalty.py --kind video --out $SP/id_e82_final --check $SP/id_e77
./.venv/bin/python $SP/scripts/cmp_wb.py $SP/id_e82_final $SP/id_e77 AFEW video
```

結果：E82 の 122 行（none 2，self 24，entmin 24（種 0・1・2），IM 72（同））がすべて E77 に相手を持ち，対数の確率の差の最大 0，予測の食い違い 0，UAR の差の最大 0，W・b の差の最大 0（122 行）．テストの行と `_samples.csv` も同じ．
IM の束から TIM の 1 問を抜いても，ほかの問の解はビット単位で変わらなかった．最初の版と最終の版（IM の所要の記録を足し，使わない定数を消した版）の両方で同じ結果．

### 3.2 倍率 3・10 の試し

```
./.venv/bin/python exp/compare_dai.py --kind video --target AFEW --reps 0 --ks 0 5 25 --pseudo-targets none --out-dir $SP/ref_e77_video --gpu-fraction 0.15
./.venv/bin/python exp/s43_penalty.py --kind video --targets AFEW --reps 0 --pseudo-targets none --out $SP/mult_video --gpu-fraction 0.15 --allow-dirty
./.venv/bin/python exp/compare_dai.py --world fuse-audio-video --target AFEW --reps 0 --ks 0 5 25 --pseudo-targets none --coreg-eval-only --out-dir $SP/ref_e77_av --gpu-fraction 0.15
./.venv/bin/python exp/s43_penalty.py --world fuse-audio-video --targets AFEW --reps 0 --pseudo-targets none --out $SP/mult_av --gpu-fraction 0.15 --allow-dirty
（それぞれ --check と $SP/scripts/check_cp.py）
```

- 使った C′（`convp` の cp．解きの結果の鍵）：A（video）・AV の AFEW は none の 2 問は 0.003，倍率 3 の 168 問すべて 0.009，倍率 10 の 168 問すべて 0.03（`geom_e0.settings` の 0.003 の 3 倍・10 倍．期待と違う問 0/338）．AVT の IEMOCAP 即興（3.4 節）は none 0.01，倍率 3 で 0.03，倍率 10 で 0.1（期待と違う問 0/548）．T・AVT の MOSEI・MELD（E39 の値 0.03 → 0.09・0.3）は台本を流していないが，`settings` の値は設計の 2 節の表と同じことを確かめた（A・V は全ターゲット 0.003，T は IEMOCAP 0.01・MOSEI と MELD 0.03，AV は全部 0.003，AVT は IEMOCAP 0.01・MOSEI と MELD 0.03．IEMOCAP 即興と台本の E39 の設定と C_T はどの実行でも同じなので写しの照合を通る）．
- none の行：E77（`--ks 0 5 25` の本番と同じ束）とビット単位で同じ（種類 2 行，世界の融合の生徒 2 行．対数の確率の差の最大 0，予測の食い違い 0）．`--none-ks` で none の束を E77 の本番と同じにしたので，`--ks 0 5` の行だけを書いても none がそろう．
- 倍率 3・10 の行（self 48，entmin 48，IM 144）は E77 の鍵と 1 つも重ならない．1 単位の行の数 242．
- 世界の道（融合の生徒だけを作る `fused_student`）の none が E77 の `world_unit` の none とビット単位で同じなので，生徒も同じに作られている．

### 3.3 読み手の試し（E77 と E82 の試しの出力の上で 2 つの版を最後まで流す）

入力：映像の種類，評価のターゲット IEMOCAP-impro・IEMOCAP-script・AFEW，反復 0・1，仮のターゲット AFEW・CREMA-D の反復 0・1（台本は即興の写し）．

```
./.venv/bin/python exp/compare_dai.py --kind video --target IEMOCAP-impro IEMOCAP-script AFEW --reps 0 1 --pseudo-reps 0 1 --pseudo-targets AFEW CREMA-D --ks 0 5 25 --copy-pseudo --coreg-eval-only --out-dir $SP/rt_e77_video --gpu-fraction 0.15
./.venv/bin/python exp/s43_penalty.py --kind video --targets IEMOCAP-impro IEMOCAP-script AFEW --reps 0 1 --pseudo-reps 0 1 --pseudo-targets AFEW CREMA-D --copy-pseudo --out $SP/rt_e82_video --gpu-fraction 0.15 --allow-dirty
./.venv/bin/python exp/s43_penalty.py --kind video --out $SP/rt_e82_video --check $SP/rt_e77_video
bash $SP/scripts/reader_test.sh     # analyze_e82 を fixed・reselect × delta・select・measure・judge，compare（--boot 50 --allow-incomplete --skip-commit-check --allow-dirty）
```

- E82 の出力の照合：none は仮のターゲット 20 行（写した 8 行を含む）の UAR の差の最大 0，`_ps_*_none.npz` 10 本の対数の確率の差の最大 0，評価 12 行（6 反復）の差の最大 0．倍率 3・10 の行（仮のターゲット 1,120 行，評価 1,440 行）は E77 と重ならない．台本の仮のターゲットの写し（456 行，組 4，npz 4 本）が働いた．
- 9 段すべてが終了の値 0（各段 20〜31 秒）．
- E79 の照合の記録（reselect）：`none_e77_vs_e80c_pseudo` 差の最大 0（20 行すべてに相手），`none_e77_vs_e80c_logp` 6 反復で差の最大 0，`e80c_rows_not_used` 1（E82 の none の行を照合の後に落とした記録），`e82_inactive_material_added` 3 ターゲットで 10・10・16 行，`e82_ps_none_check.csv` 10 本すべて ok．δ は 2 つの版でバイト単位で同じ（`other_grid_delta_same`）．
- E82 の行が流れたこと（数だけ）：reselect の chosen.csv の 3 手法・excl の 27 行のうち 11 行が E82 の行（K 0・5 の 18 行のうち 11．K 25 は E82 の行が無いので E77），測る鍵のうち 18 個が E82 の行，「働かなかった」の材料 36 行が E82 の表から作られた（fixed の版の材料 120 行 ＝ reselect の E79 の材料 84 行 ＋ E82 の材料 36 行）．
- compare の検算：単位 180 行のうち 2 つの版で相手の無い行 0，ほかの 5 手法の判定が 2 つの版で違う単位 0，E79 の statements.csv を combos.csv から作り直したときの rule の食い違い 0（2 つの版），仮のターゲットの値から作り直した選択と chosen.csv の食い違い 0（36 単位）．
- `--fixed-from` で固定の版を別の場所から読んでも，比べの表 6 つがバイト単位で同じ．最後の修正（pandas の警告の書き方）の後に compare を流し直しても 7 つの表がバイト単位で同じ．
- 試しの範囲で出る欠けの記録（`--allow-incomplete` で止めずに記録）：`ps_missing` 16，`pseudo_units_mismatch`，`selector_without_candidates` 90（E78 が無い CORAL・DANN・MME など），`e78_missing` 1，`e82_dirty_or_allow_dirty` 3（試しは `--allow-dirty`）．どれも試しの範囲を絞ったことによるもの．

### 3.4 止まる条件の試し（`$SP/scripts/neg_tests.sh`）

| 壊し方 | 結果 |
|---|---|
| E82 の仮のターゲットの none の行を 1 つ 1e−6 ずらす | `none_e77_vs_e80c_pseudo_over_tol` を記録（`--allow-incomplete` が無ければ止まる） |
| E82 の entmin の `-C3x` の 16 行を E77 の param `-` に替える | `--allow-incomplete` でも終了の値 1 で止まる（E79 の `check_overlap`：「同じ鍵が 16 個重なる」．文言は「E80 の段 c」と出る） |
| E82 の `AFEW_pseudo.csv` を消す | `e82_file_missing` を記録（`--allow-incomplete` が無ければ止まる） |

## 4. 所要（A5000）と H100 の見積もり

1 単位の所要（`<T>_conv.csv` の unit_total．生徒の作成と解き．ソースの分類器 `pool_info` とデータの読み込みはターゲットごとに 13〜44 秒）：

| 実行 | 単位 | 候補の数 n | 次元 d | 秒 |
|---|---|---|---|---|
| video | 仮 AFEW（2 単位） | 636 | 512 | 27 |
| video | 仮 CREMA-D（4） | 3,927 | 512 | 24 |
| audio_wavlm | 仮 MOSEI（1） | 15,135 | 1,024 | 146 |
| fuse-audio-video-text | 仮 MELD（1） | 9,193 | 2,304 | 231 |
| video | 評価 AFEW（3） | 636 | 512 | 40 |
| video | 評価 IEMOCAP-impro（2） | 2,327 | 512 | 39 |
| video | 評価 IEMOCAP-script（2） | 2,025 | 512 | 63 |
| audio_wavlm | 評価 AFEW（1） | 636 | 1,024 | 52 |
| fuse-audio-video | 評価 AFEW（1） | 636 | 1,536 | 46 |
| fuse-audio-video-text | 評価 IEMOCAP-impro（1） | 2,327 | 2,304 | 104 |

比べ：同じ AV の AFEW の評価の 1 単位は E77（全手法）で 298 秒，E82 で 74 秒（ターゲットの読み込みを含む）．
見積もりの方法（`$SP/scripts/estimate.py`，`$SP/unit_sizes.csv`）：全単位（`map.test_mask` と同じ分割で候補の数を数えた．仮のターゲット 495（A 175，V 120，T 40，AV 120，AVT 40．IEMOCAP-script の写しは数えない），評価 270（70，60，40，60，40））に，測った単位の所要を n·d で直線に当てはめた式（仮：10.5 + 9.8×10⁻⁶·n·d 秒，評価：37 + 1.2×10⁻⁵·n·d 秒）を当てた．
- A5000（共有）：仮のターゲット 9.9 時間，評価 9.3 時間，ターゲットごとの読み込み 27 × 19 秒，合計 19.3 時間
- H100（A5000 の 3.5 倍の速さ．`checks/E80/code.md:234`）：仮 2.8 時間，評価 2.7 時間，合計 **約 5.5 時間**（実行ごと：A 1.4，V 0.7，T 0.5，AV 1.6，AVT 1.3 時間）
- 幅：測った単位は 18 個で，評価の MOSEI・MELD（n 9,000〜15,000）は外挿である．L-BFGS の反復の回数で同じ大きさの単位でも 1.5 倍ほど違い（評価の IEMOCAP 即興と台本），共有の GPU の上の値なので，H100 の合計は 4〜8 時間ほどと読む．読み手（CPU）は E79 の P4 の段と同じ程度（reselect の 1 つの版の measure が E79 の measure の P4 の部分とほぼ同じ鍵の数）．

## 5. 試していないこと（理由）

- 読み手を世界（P4′．融合の生徒）の出力に当てていない．世界の E77 の試しの出力（仮のターゲット込み）は A5000 で 1 評価の単位 5 分（共正則化）かかり，時間の枠を超えるため．E82 の世界の道（AV・AVT の融合の生徒）は台本の側で none の一致と C′ を確かめた．読み手のコードは種類と世界で同じ関数を通る（`role_of` が fused を返すだけ）
- ns_excl の版（MOSEI・MELD の選び直し）の差し込み（`diagnostics_e82` の ns_excl の分岐）は，試しのターゲットに MOSEI・MELD が無く動いていない
- E78 の出力が無い入力でしか読み手を流していない（CORAL・DANN・MME の族は欠け）．本番では E78 の場所を渡す
- 本番のコミットの確認（`require_committed`：δ・chosen.csv・fixed.csv の sha256 を文書に書いてコミット），`--e79-delta` の照合（手元に E79 の出力が無い），`--fixed-from` に E79 の本番の出力を渡すことは試していない
- text_roberta と AVT の MOSEI・MELD の評価の単位は流していない（台本の道は種類と世界で同じ．C′ の値は settings で確かめた）
- 本番の大きさの GPU メモリ：試しの最大は 2.7 GB（AVT の MELD の仮のターゲット，0.15 の枠の中）．MOSEI の AVT（n 約 15,000，d 2,304）は試していない．E77 の本番（0.3）より束が小さい（1 生徒，1 回に束ねる問は最大 24）ので 0.3 で足りる見込み

## 6. 設計からのずれ（1・2 は 2026-11-02 に本体が決めた．7 節）

1. **同点の決め方**：最初の版は E79 の文字列の順のままで，設計の印では同点のとき 10 倍 → 3 倍 → 1 倍の順に選ばれた．2026-11-02 の本体の決定で reselect の版の同点を倍率の小さい方に変えた（7 節）
2. **筋 4 の文の鍵を手法ごとに当てる読み方**（2026-11-02 の本体の決定で，この読み方でよいとなった）：E79 の文の鍵は族（P4・P4′）ごとに 8 手法をまとめて 1 つ決める．設計の 3 節は「3 手法のすべてで文の鍵が固定の版と同じなら」なので，E79 の規則（`statements_p4` と種の感度の書き直し）を 1 手法の組み合わせだけに当てた鍵を手法ごとに作り，`decision.csv` は tag main・subset all・P4 と P4′ の手法ごとの rule が 2 つの版で同じかで決める．族の全手法の鍵（E79 の statements.csv）も並べて出す（`family_level_same_rule`）．
3. E82 の行は E79 の 3 つ目の波の枠（`E80C`，名前 e80c）から入れる．reselect の版では本物の E80 段 c の行を読まない（P4 の判定に使わない参照の行（`refs`）の教師の行が減るだけ）．fixed の版を analyze_e82 で作り直すときも段 c を読まないので，E79 の本番の出力と違うのは参照の記述の行だけで，P4 の行は同じはず（本番では `--fixed-from results/e79` で読むことを勧める）
4. TIM（IM の 1 つの点）は倍率 3・10 で解かない（E79 の im の族の候補でないため．設計の格子の 12 通りと同じ）
5. none は E77 の本番の束（K 0・5・25）で解き，K 0・5 の行だけを書く（none を E77 とビット単位でそろえるため．K 25 の none の 1 問だけ計算が増える）
6. 世界の実行では融合の生徒だけを作る（設計の範囲は融合の生徒．モダリティの生徒は作らないので計算が減る）
7. 「働かなかった」の材料の行に `source`（e82）の列を足した．`_conv.csv` に IM の所要の行（E77 には無い），`_convp.csv` に解いた C′ の列，`_cp.csv` を足した．どれも E79 の読みには影響しない

## 7. 本体の決定を入れた直し（2026-11-02）

決定：(1) reselect の版の選び直しで，仮のターゲットの点数がちょうど同じ候補が並んだら倍率の小さい方（1 → 3 → 10）を選ぶ．倍率が同じ候補の間は E79 の規則（文字列の順）のまま．`analyze_e79.py` は変えない．読み方の規則の数（10 倍の過半）は新しい規則で選んだ数にし，同点で倍率の小さい方を選んだ単位の数を別の列に出す．(2) 文の鍵は手法ごとに比べる読み方のまま（族ごとの鍵も並べる）．

直したところ（`exp/analyze_e82.py`．sha256 先頭 `68e4442d44216ba9`．`exp/s43_penalty.py` は変えていない）：
- `choose_e82`（`:90-104`）を足し，`install`（`:211`）で reselect の版だけ `A.choose` を差し替える（fixed の版は E79 の choose のまま）．provenance の `patched` に `choose` を足した
- `tie_info`（`:328-368`）：新しい規則で選択を作り直して chosen.csv と照らし，あわせて E79 の文字列の順だけで決めた場合の倍率（`mult_string_order`）を出す．列 `tie_multi_mult`（同点に複数の倍率）と `tie_chose_lower_mult`（新しい規則で選んだ倍率が文字列の順の倍率より小さい）を足した
- `multipliers.csv` に `n_tie_multi_mult`・`n_tie_chose_lower_mult` を足した．`n_mult10`・`majority_mult10` は新しい規則で選んだ数（chosen.csv をそのまま数える）．`n_mult10_strict` は残した（新しい規則では選んだ 10 倍の同点に小さい倍率が残らないので `n_mult10` と同じになる）．`decision.csv` の `grid_note` は `majority_mult10_excl` だけで決める

試し（評価の値は出していない）：
- 人工の候補の表（`$SP/scripts/test_choose.py`．点数は作った数）：倍率 1・3・10 が同点 → 倍率 1（文字列の順なら `-C10x`），3 倍と 10 倍が同点で最良 → 3 倍（同 10 倍），同じ倍率の中の同点 → E79 と同じ文字列の順，10 倍がはっきり上 → 10 倍，E82 の行が無い族 → E79 の choose と同じ．5 つの場合すべて決定どおり
- 3.3 節の試しの入力で reselect の版を delta → select → measure → judge → compare（`--fixed-from` に 3.3 節の fixed）と流し直した（`$SP/rt_analysis2`．各段 20〜30 秒，終了の値 0）．この入力には倍率をまたぐ同点が無く，chosen.csv の 198 行は前の版と 1 行も変わらなかった．作り直しと chosen.csv の食い違い 0，`n_mult10` と `n_mult10_strict` が違う行 0/20，ほかの検算も食い違い 0
- 同点を作った入力（`$SP/tie_case`．E82 の entmin の仮のターゲットの 160 行の値を E77 の倍率 1 の同じ設定の値に置き換え，どの設定でも 3 つの倍率が同点になるようにした）：entmin の excl・K 0・5 の 6 単位すべてが倍率 1（E77 の行）を選び（n_tie は 3 以上），E79 の文字列の順なら 6 単位とも 10 倍だった（`mult_string_order`）．`n_tie_chose_lower_mult` は entmin 6，im 0，self 0．entmin 以外の族の選択は 180 行すべて前と同じ．measure・judge・compare も終了の値 0，作り直しの食い違い 0
