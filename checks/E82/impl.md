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

## 8. 関門 1 の後の直し（2026-11-02．設計はコミット 4a52140 の E82 の 2〜4 節．報告は `checks/E82/{stats,leak,compare,define}.md`）

直す前の版は 4a52140 に入っている．直した版は下のコミットに入れてから試した（`docs/methods.md` 9.1 節．評価のターゲットで試す前に台本をコミットする）．

`exp/s43_penalty.py`（sha256 先頭 `3bf22778f93dd0d4`）：
- **none の対照**（define D10，compare）：`run_methods`（`:131-`）の最初で，倍率 3・10 のそれぞれについて none を同じ C′ で解き，method none・param `-C3x`・`-C10x`・meff 0 の行を K 0・5 で書く（仮のターゲットと評価の両方．`:142-153`）．倍率 1（恒等の試し専用）では書かない
- **K=0 の検算**：倍率の none の K=0 は出発点そのもので C′ に依らないので，W・b が E39 の none とビット単位で同じことを台本の中で確かめ，`<T>_e82check.csv`（check `none_K0_mult_vs_e39`，W・b の差の最大，ok）に書く．違えば止まる（`:146-153`）
- **出発点からの離れ**（設計の 2 節の記述の列，define D8）：`Rec2`（`:93-`）が W・b のある行ごとに ‖W − W_出発点‖_F ÷ ‖W_出発点‖_F（`dW_rel`）と ‖b − b_出発点‖（`db`）を `<T>_dist.csv` に書く（生徒の座標 Z の上．出発点は E39 の下書き＝罰則の中心＝K=0 の none．`one_unit` の `:234`）．none（E39・倍率），self，entmin（種 0・1・2），IM の全行
- 倍率の none の行は，仮のターゲットの `_ps_*_none.npz`（E39 の none．δ の材料）を上書きしない（`Rec2.add`）
- `--copy-pseudo` の写しは `_dist.csv`・`_e82check.csv` の仮のターゲットの行も写す（`copy_extra`，`:253-`）．`--check` は倍率の none を `none_mult` として数え（E77 に相手が無いはず），`_e82check.csv` を読む
- 1 単位の行の数：評価 246 行（242 ＋ 倍率の none 4），仮のターゲット 118 行（114 ＋ 4）

`exp/analyze_e82.py`（sha256 先頭 `29cdbe44999e6abe`）：
- **比べ (b)**：新しい段 `judge_b`（reselect の版だけ）．E79 の判定の段を `StoreB`（`:138-`）で流す．E79 の判定の compare は単位の群ごとに手法の鍵を読んだすぐ後に none の鍵を読む（`analyze_e79.py:2084-2085`）ので，直前の手法の鍵の倍率が 1 でなければ none の鍵を同じ倍率の none（`none_cp`）に替える．倍率 1 の行（E77・E78 の行）と K=0 では (a) と同じ値になり，compare の段でこれを確かめる（`n_units_a_b_differ_K0`，ほかの 5 手法で (a) と (b) の点の値が違う単位の数）
- 倍率の none の読み方：E82 の場所の表では倍率の none を method `none_cp` に読み替える（`read_rows_e82`，`:117-`．E79 は e80c の波の none を 1 つの鍵に 1 行と決めて E77 の none と照らす（`analyze_e79.py:606`・`:1470`）ので，そのままでは重複で壊れる）．`key_table_e82`（`:128-`）が none_cp の鍵を測る鍵に足す
- **「働かなかった」**：E82 から選んだ設定は，同じ C′ の none（none_cp）との予測の食い違いで材料を作る（`diagnostics_e82`，`:211-`）．倍率 1 の設定は E79 のまま E39 の none（同じ C′）と比べる．材料は (a)・(b) で共通
- **書き方の規則**（設計の 3 節）：`decision.csv` は手法 × 族ごとに tag main・subset all の手法ごとの鍵で分ける（`category_of`，`:454-`）：same_a（(a) が固定の版と同じ → 限定なし），limit（(a) が変わり (b) も同じ向き → 限定あり），k_label（(a) が変わり (b) は固定と同じ → 限定なしで「罰則を弱めて K 本のラベルで動いた分」の内訳を添える），opposite（(a) と (b) が逆の向き．設計に無い場合で本体が判断）．向きは鍵が筋 4 を狭める強さ（(3) < (4) < (2′) < (2) < (1)．E79 の印の優先の順）の増減で決める．unseen_only の鍵と (2′) の same_combos は記述の列．全体の行に「限定を外せるのは entmin・IM・自己学習だけ（T3A・LP・DANN・MME には残す）」と，固定と (a) がともに (4) なら「約 1.5δ_d より大きな減りは無かった」までと書く注を出す
- **倍率の数**：`multipliers.csv` に（実行，生徒，K，手法）ごとの行を足し（stats S7．単位は仮のターゲットを共有するので独立でないと注），過半の母数を手法が選ばれた単位（__none__ を除く）にした（define D1）．`decision.csv` に（実行，生徒，K，手法）の組のうち 10 倍が過半の組の数を出す
- **候補の数**（compare C2）：`candidates.csv`（固定と選び直した版の n_cand．族 × 手法 × K）
- **oracle_dir**（stats S4）：`oracle_dir_both.csv`（2 つの版を格子の大きさ `grid_size` を添えて並べる）と `oracle_dir_summary.csv`（版 × 手法 × K の δ_d 以上の単位の数）
- `decision.csv` の分け方を作った鍵の組で確かめた：(4)(4)(4) → same_a，(4)(2′)(2) → limit，(4)(2′)(4) → k_label，(3)(4)(1) → limit，(2)(4)(1) → opposite，鍵の欠け → missing

試しの結果は 9 節．

## 9. 直した版の試し（コミット a06c165 の台本．A5000，`--gpu-fraction 0.15`．評価の値は開いていない）

前の試しの出力は消し，E77 の試しの入力（`$SP/id_e77`，`$SP/ref_e77_av`，`$SP/rt_e77_video`．3 節で作ったもの）だけを残して使った．

- 恒等（`--mults 1`，映像の AFEW の評価の反復 0，`$SP/id_e82_v2`）：122 行すべてが `compare_dai.py` の行とビット単位で一致（対数の確率・予測・UAR・W・b の差 0）．倍率 1 では倍率の none を書かない
- 倍率 3・10（AV の世界の AFEW の評価の反復 0，`$SP/mult_av_v2`）：1 単位 246 行．none（E39）は E77 と差 0，倍率の none 4 行（K 0・5 × 倍率 2）は E77 と重ならない．解いた C′ は none 0.003（2 問），noneC3x 0.009，noneC10x 0.03（期待と違う問 0/340）．台本の中の K=0 の検算 2 件が ok（W・b の差の最大 0）．`_dist.csv` は 246 行（csv の行と同じ数）で，K=0 の none（E39 と倍率）の `dW_rel` は 0，NaN 0
- 読み手の試しの範囲（映像の種類，評価のターゲット IEMOCAP-impro・IEMOCAP-script・AFEW の反復 0・1，仮のターゲット AFEW・CREMA-D の反復 0・1，`$SP/rt_e82_v2`）：`--check` で none は仮のターゲット 20 行・npz 10 本・評価 12 行が E77 と差 0（npz は E39 の none のまま上書きされていない），倍率の none（仮のターゲット 40 行，評価 24 行）は E77 と重ならない，台本の中の K=0 の検算 32 件（仮のターゲット 6 単位＋写した 4 単位＋評価 6 単位，× 倍率 2）がすべて ok．写しは 472 行
- 読み手（`$SP/scripts/reader_test_v2.sh`，出力 `$SP/rt_analysis_v2`）：fixed の 4 段，reselect の 4 段と judge_b，compare の 10 段がすべて終了の値 0（各段 20〜32 秒）
  - judge_b で none の鍵を同じ C′ の none に替えた回数 88．測る鍵に none_cp が 12 個入った．「働かなかった」の材料のうち E82 から選んだ設定の 36 行は，同じ倍率の none（`-C10x` 32 行，`-C3x` 4 行）と比べた
  - 検算：K=0 で (a) と (b) の判定か点の値が違う単位 0，ほかの 5 手法で 3 つの判定（固定・(a)・(b)）が違う単位 0，(a) と (b) で点の値が違う単位 0，E79 の statements.csv の作り直しの食い違い 0（3 つの版），選択の作り直しの食い違い 0．3 手法の主の単位 12 のうち K=5 で (a) と (b) が違う単位は 3（倍率の none が E39 の none と違うので，違うのが期待どおり）
  - 候補の数（K 0・5 の中央値）：entmin 4 → 12，IM 12 → 36，自己学習 12 → 36（設計の 3 節と同じ）．`oracle_dir_both.csv` は 2 つの版を同じ行の数で並べ，格子の大きさの中央値も同じ（4・12・12 と 12・36・36）
  - 出力の形：`decision.csv` 7 行（手法 × 族の 3 行，手法ごとのまとめ 3 行，全体 1 行）23 列，`statements.csv` 114 行（scope：族 38，IM 38，entmin 19，self 19．entmin と self は P4 で「見た後の予言」なので unseen_only が無い），`multipliers.csv` の level は（実行，生徒，K，手法）12 行，族 × K 12 行，手法 6 行，3 手法 2 行

所要：同じ読み手の試しの範囲の E82 の計算は，3 節では 8 分 11 秒（別の GPU の仕事 2 つと共有），今回は 2 分 47 秒だった．4 節の見積もり（H100 で約 5.5 時間）は混んだ GPU の測りなので多めに出ている．倍率の none は 1 単位 2 問を足すだけで所要はほとんど変わらない．

試していないこと（5 節に足す）：judge_b の `StoreB` は E79 の判定の compare が「手法の鍵 → none の鍵」の順に読むことに頼る（`analyze_e79.py:2084-2085`）．E79 の台本が変わればこの前提を確かめ直す（今は analyze_e79.py の sha256 を provenance に残す）．世界の出力・ns_excl・E78 がそろった入力で読み手を流していないのは 5 節と同じ．

見たデータの台帳（`docs/seen-data.md`）に足す行の材料（本体が書く）：評価のターゲットで試した単位は，映像の種類の AFEW の反復 0（恒等），AV の世界の AFEW の反復 0，映像の種類の IEMOCAP-impro・IEMOCAP-script・AFEW の反復 0・1（仮のターゲット AFEW・CREMA-D の反復 0・1）．出力は scratchpad の `e82_test/` で，画面と記録に出したのは差・数・形・所要・C′ の値だけ（評価の値は誰も開いていない）．このうち 3 節の試しは台本のコミットの前に流した（関門 1 の leak の指摘）．9 節の試しはコミット a06c165 の後に流した．

## 10. gate-code の指摘の後の直し（2026-11-02．報告 `checks/E82/code.md`．設計はコミット 9ce63fa）

`exp/analyze_e82.py`（sha256 先頭 `dfa4969c3992cc8d`）：
- **10 倍の端の規則**（R4・R7．設計の 3 節）：`decision.csv` の分け方を，倍率の数え上げの後で決めるように移した．手法ごとに excl の版で 10 倍が過半（母数はその手法が選ばれた単位，同点の規則で選んだ数）なら，限定を外す分け方を外さない側に替える（`edge_category`・`EDGE_OF`，`:456`・`:465`）：same_a → `same_but_edge10`，k_label → `k_label_but_edge10`（limit・opposite は元から限定を付ける）．列 `category_keys_only`（鍵だけで決めた分け方），`edge10`，`edge_note`（「C′ を E39 の値の 10 倍まで広げた格子では」と，格子がまだ足りないことを限界に書く注）を足した
- **opposite の文**（I1-opposite）：「E39 の罰則で出発点につないだとき」と限定を付け，(a)・(b) の鍵と判定を並べて書く（設計の 3 節）に直した（`CATEGORY`，`:447-`）．説明の文に 4 つの場合と 10 倍の規則を書いた
- **compare の段の照合**（I12・I14）：固定の版（`--fixed-from`）と選び直した版の measure の段の条件（`--boot`・`--seed`・反復・評価のターゲット・切片だけを直した UAR の有無・δ の sha256．実行は選び直した版 ⊆ 固定の版）が違えば，表を作る前に止める（`check_conditions`，`:499-`）．ほかの 5 手法の判定か点の値（固定と (a)，(a) と (b)），K=0 の (a) と (b)，相手の無い単位，文の鍵の作り直し，選択の作り直しが 1 つでも食い違えば，表と provenance を書いてから終了の値 1 で止まる（`:787` 付近）
- **provenance**（I9）：compare の段でも差し込んだ名前を `patched` に書く
- **収束の表**（I11）：`convergence_e82.csv`（`convergence_e82`，`:514-`）．E82 の `_convp.csv` を実行 × 側 × 手法 × 倍率ごとに，E79 と同じ列（`n_hit_max_iter`，`n_grad_over_tol`，`n_stopped_early`）と反復の最大・中央値，勾配の最大，C′ の範囲で数える．写した仮のターゲットの行は数えない．compare の段の見張りが開いてよいファイルに `_convp.csv` を足した（値の列は無い）

`exp/s43_penalty.py`（sha256 先頭 `c5a99690b753d280`）：
- **GPU の型**（I8．設計の 4 節）：`--check` で，E82 と E77 の `provenance.json`・`provenance_log.jsonl` の `gpu` がすべて同じ 1 つの型かを照らし（check `gpu_type`），違えば終了の値 1（`run_checks`，`:382-389`）

`run/e82h.sh`（新規．sha256 先頭 `8fbcd5c33789883d`）：
- 冒頭の注に準備の手順（本体が流す）：`/workspace/MER-DA` で `git worktree add --detach /workspace/MER-DA-e82 <コミット>`，`.venv` と `results` は `/workspace/MER-DA` のものへのリンク，`data/cache` は特徴の npz だけをリンク（追跡している顔の座標の csv は作業木のもの），`logs` は作業木のもの，`git status -- exp measure` が空であること
- 引数は流れの番号 1・2．配分は流れ 1 ＝ AVT・WavLM，流れ 2 ＝ AV・映像・RoBERTa（H100 の見積もり：AVT 2.2，WavLM 1.8，AV 2.2，映像 0.9，RoBERTa 0.8 時間．4 節の式に gate-code の T の MELD の評価の 1 単位（305 秒）を足して当てはめ直した．A5000 の共有の GPU の測りを 3.5 で割った値で，合計約 7.9 時間，2 本で壁時計約 4 時間）
- 各実行：E77 の `results/e77_<実行>/IEMOCAP-impro.csv` が無ければ流さない → `--targets IEMOCAP-impro` → `--check results/e77_<実行>`（none・鍵の重なり・K=0 の検算・GPU の型．落ちたらその実行の残りを流さない）→ 残りのターゲット（終わった IEMOCAP-impro は check_resume で飛ばす）→ 全ターゲットの `--check`．旗は `--ks 0 5 --copy-pseudo --nc-seeds 1 2 --gpu-fraction 0.2`．ログ `logs/e82h_<名前>.log`，失敗 `logs/e82h_fail.log`，終わりに `logs/e82h_<番号>.done`
- 準備の確かめ（results がリンクか，`.venv/bin/python` があるか，特徴の npz があるか，exp/・measure/ に変更が無いか，E77 の作業木でないか）が落ちたら止まる
- `E82H_DRY=1` で，コマンドの組み立てと準備の確かめを表示するだけにできる（ログにも何も書かない）

試しの結果は 11 節．

## 11. 10 節の直しの試し（コミット 4b27fc1 の台本．CPU と照合だけ．GPU の計算はしていない．評価の値は開いていない）

- compare の段を 9 節の出力（`$SP/rt_analysis_v2`）で流し直した：終了の値 0．検算はすべて食い違い 0（ほかの 5 手法の判定・点の値，K=0 の (a) と (b)，相手の無い単位，文の鍵と選択の作り直し）．`decision.csv` の手法 × 族の 3 行で，鍵だけで決めた分け方（`category_keys_only`）は直す前の版の分け方と 3/3 同じ．10 倍が過半の手法の行は分け方が `same_but_edge10`・`k_label_but_edge10` の側に替わり（替わった行 2，`edge_note` のある行 2），10 倍が過半なのに限定を外す分け方（same_a・k_label）のまま残った行は 0．ほかの表（judgments，judgment_units，statements，multipliers，multiplier_units，candidates，oracle_dir_both）は直す前の版とバイト単位で同じ．`convergence_e82.csv` は 18 行（問 3,312．反復の上限に当たった問 0，勾配が許容を超えて早く止まった問 0）．provenance の `patched` に 8 つの名前が入り，条件（`--boot 50`・`--seed 0`・反復 0・1・3 ターゲット・切片の UAR あり・δ の sha256 が同じ）が残った
- 止まる条件：
  - 固定の版の measure の `boot` を 60 に書き換えた写し（`$SP/neg_cond`）を `--fixed-from` に渡す → 表を作る前に終了の値 1（「測った条件が違う」）
  - 固定の版の `units.csv` の t3a の単位の判定を 1 つ書き換えた写し（`$SP/neg_other`）→ 表 11 本と provenance を書いてから終了の値 1（`n_other_method_units_judgment_differ` 1）
  - E82 の出力の `provenance.json` の `gpu` を H100 に書き換えた写し（`$SP/neg_gpu`）で `--check` → `gpu_type` が食い違い（E82 は 2 つの型，E77 は A5000），終了の値 1．書き換えない出力では `gpu_type` が ok（A5000 ／ A5000）
- `run/e82h.sh`：`bash -n` が通った．`E82H_DRY=1 bash run/e82h.sh 1`・`2` で，各実行 4 つのコマンド（IEMOCAP-impro → `--check results/e77_<実行>` → 残り → 全体の `--check`）と旗（`--ks 0 5 --copy-pseudo --nc-seeds 1 2 --gpu-fraction 0.2`）の組み立てを確かめた．研究室では準備の確かめが「results が共有のリンクでない」「exp/・measure/ に変更がある」と出て，E77 の出力の無い実行は「流さない」と出る（期待どおり）．引数 3 は終了の値 2．DRY ではログを 1 本も作らない．RunPod では流していない
- [I7] のとおり K=0 の倍率の none の検算は定義から成り立つ（重みが全部 0 の問は解かずに下書きを返す）．倍率が効いたことは `_convp.csv` の cp（解きの結果の鍵）と，K=5 で (a) と (b) が違う単位があること（9 節）で確かめた
