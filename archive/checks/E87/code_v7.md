# 関門 1：実装と出力（E87 第 7 版，2026-11-03，コミット 45586c4）

## 判定
直してから進む．前回（第 6 版）の指摘 1（none を選んだ群の直接の差）は直り，人工のデータと none を選んだ 6 群の試走で，手法の側の点の限り・読めるか・印・抽出の限り・抽出の印が none の読みを独立に計算した値と食い違いなく一致した．包絡の新しい式も設計 2.4 節と一致し，主の族では抽出ごとに L ≤ T ≤ U が丸めなしで保たれた．ただし第 7 版の変更で，同じ単位の中の差（dir_minus_pos・dir_minus_posloss）と ΔS の直接の差の行の `flag_*_method` が NaN から真になり，同じ単位の中の差の直接の差の抽出の印がすべて 0 になって 2.5% の基準が当たらなくなった（指摘 1）．判定の表（`judgments.csv`）には入らず，保存した出力から後で直せるので「止める」にはしない．

検査の範囲：依頼のコミットは 96cdf57 だったが，検査の途中で依頼者から対象を 45586c4 に替えると連絡を受けた．96cdf57 の台本は `values_group` の複製の枝（96cdf57 の 663 行）で `read_bounds` の 8 個の戻り値を 7 個でほどくため，試走が ValueError で止まる（人工の W0b で同じほどき方が ValueError になることを確かめた）．45586c4 はその 1 行だけを直したもので（`git diff 96cdf57 45586c4` は 1 行），`read_bounds` を呼ぶ箇所は 616 行と 663 行の 2 つだけで，どちらも 8 個でほどいている．設計は `docs/exploratory-analyses.md` の E87（96cdf57 の版．その後の aa60756・53b41f0 は 2.4 節末の経緯の 1 文だけの変更で，台本は 45586c4 から変わっていない）．台本は `exp/analyze_value_split_e87.py`（45586c4）で，`git diff 9c892ab 45586c4` の全部と，`read_bounds`・`values_group`・`aggregate_unit`・`direct_units`・`family_stats`・`judge` を読んだ．検査の台本とログは `checks/E87/code_run_v7/` に置いた．流したものはすべて CPU（`CUDA_VISIBLE_DEVICES=""`．この環境に `nvidia-smi` は無い．依頼者の試走が 1 本同時に動いていたが，その出力には触れていない）．`results/` の下は読むだけにした．試走の出力（`smoke/`，`smoke_none_av/`，`smoke_none_avt/`）は消し，検査の台本とログだけを残した．ログからは判定の区分の種類の行を消した（`chk_struct_v7.log`）．

**検査の後に台本が変わった．**報告を書き終えた時点（2026-11-03 23:23 以降）で，コミット a79839c（「envelopes centred on the unfavourable point bound; both-none direct differences zero by definition; NaN method flags treated as false」）が入っていた．差分は，包絡を不利な側の点の限りのまわりで広げる（L ＝ pt_up ＋ r·(LB − pt_up)，U ＝ pt_lo ＋ r·(UB − pt_lo)．点の限りが ±10⁶/2 を超えるときは x₀），両方が none を選んだ直接の差を定義から 0 にする，`mflag_*` を `is True` で読む，の 3 つである．この報告の数値と判定は 45586c4 のものであり，a79839c は検査していない．差分を読んだだけでは，指摘 1 の (i)（NaN が真になる）は a79839c で直り，(ii)（同じ単位の中の差の直接の差の抽出の印が 0）は直っていないように見える．a79839c で本番を流すなら，その差分をこの役に再検査させる．

## 前回（第 6 版）の指摘の判定

| 前回の指摘 | 判定 | 根拠 |
|---|---|---|
| 1 [I1] none を選んだ群の直接の差が none の読みの限りと印を落とす | **直った** | `read_bounds` が none の限り `(n_lo_b, n_up_b)` を 8 番目に返し（A:578），`defn_zero` の枝で `m_lo, m_up` ＝ none の限り，`method_readable` ＝ none の 3 つの印のどれも無い，`mflag` ＝ none の印を手法のビットに写したもの（A:619–624）．抽出では none の抽出の読みを `read_K`（最初と最後）で読み直し，あいまいの印を `read_bounds` と同じ規則で足し，`MLB`・`MUB` ＝ `bounds_of`，`mmask` ＝ none の印を手法のビットに写したもの（A:648–652）．`aggregate_unit` は `|mmask` を保存し（A:819），`mflag_*` を `units.csv` に書く（A:852）．`direct_units` は `|mmask` の論理和を抽出の印にし（A:891–896），`flag_*_method` を `mflag_*` から作る（A:881）．人工のデータ（W1〜W2h の 15 項目）がすべて通った．試走で none を選んだ 6 群（AV の CREMA-D 2 群，AVT の MELD・MOSEI 4 群．前回と同じ群）で，点の読み 84 個（印あり 30）と抽出の読み 48 組（none の印が 2.5% を超えるもの 30）について，台本の手法の側の値と独立に計算した none の限りと印の食い違いは点 0，抽出 0（`chk_nonegroup_v7_*.log`）．この群を含む直接の差 56 行（QUANTS_B）のすべてで点の限りが手法の限りから作られ（前回は AVT の 32 行で抽出だけ lo(A) − up(B) に戻っていた），`|mlo`・`|mmask` の欠けは 0，`is_readable` の食い違い 0，flag の食い違い 0，抽出の印と `|mmask` の論理和の食い違い 0，読めて区間が使える行で none を選んだ側の印が 2.5% を超えるもの 0 |
| 2（記録）同じ単位の中の差の直接の差で none の扱いがそろわない | **悪くなった（指摘 1）** | 限りと読めるかは none を含むまま，抽出の印は `|mmask` が無いので 0 になり，flag は NaN から真になる |
| 3（記録）`direct.csv` の `flag_*` に none の印が入る | 直った | `flag_*_none` は偽に固定（A:881）．試走で直接の差 QUANTS_B 1,152 行のうち `flag_*_none` が真の行 0，`flag_*_method` と `mflag_*` の論理和の食い違い 0．`direct_family.csv` の `n_corp_*_none` が正の行 0 |
| 4（記録）直接の差の族の包絡が float32 の丸めで記録を超える | 変わらない（大きさは倍率の分だけ増えた） | 試走で直接の差の族の超過の最大 3.8×10⁻⁴ 本（前回 6.1×10⁻⁵ 本．限りのずれにも倍率 r が掛かるため），10⁻³ 本を超える項目 0．主の族の超過は 0（`chk_envelope_direct_mag_v7.log`） |
| 5（記録）provenance と冒頭の説明の限りの作り方 | 一部直った（記録 3） | provenance の `bounds`・`readable` は第 7 版の文になった．provenance の `family` と台本の冒頭の説明（A:12–13）は古いまま |
| 6（記録）`*_final_by`・`contrast_B_robust` の列が判定できる行が無いと出ない | 変わらない | IEMOCAP だけの試走の `judgments.csv` にこれらの列は無かった．後の集計で列が無い場合を空として扱えばよい |

## A．設計と台本の対応表（第 7 版で変わった所）

台本は `exp/analyze_value_split_e87.py`（45586c4．A:行）を指す．

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 2.4 直接の差の限り・読めるか・抽出の印は手法の読みだけから（none は差で消える） | A:872–875（点の限り），A:885–890（抽出の限り），A:891–896（抽出の印 ＝ `|mmask` の論理和），A:881（flag），A:922（`is_readable`） | QUANTS_B（uar・dir・pos・posloss）では合う．同じ単位の中の差と ΔS では合わない（指摘 1） |
| 2.4 none を選んだ群（手法 ＝ none）では none の読みの限りと印を手法の側として引き継ぐ | A:616–624（点），A:646–652（抽出），A:774（`mflag` の単位への集め方），A:819（`|mmask`） | 合う（W1・W1b〜W1e，試走の 6 群）．saerens・massbal の向きの量（定義から 0 だが手法を選んだ群）も同じ枝を通り，手法の側に none の読みが入る（W1g）．設計の「手法の向き ＝ none の向き」と合う．saerens・massbal は直接の差に入らない |
| 2.4 none を選んだ群の n の側は定義から 0 で読める | A:618–627，A:646–647 | 合う（W1f．n ＝ 0，読める，印なし，`mask`・`B`・`LB`・`UB` は 0） |
| 2.4 包絡：r は記録した複製と点 x₀ から，L ＝ x₀ ＋ r·(LB − x₀)，U ＝ x₀ ＋ r·(UB − x₀) | A:1019–1028 | 合う．人工のコーパス 300 個（G_eff < 20，倍率 > 1 が 300/300）で式どおりの値と一致し，抽出ごとに L ≤ T ≤ U の破れは許容なしで 0（W4・W4b）．−10⁶ に置いた抽出は反対の側へ動かず，さらに外へ動く（W4c）．試走の主の族で L ＞ T・U ＜ T の超過は 0 |
| 2.5 出力 `units.csv` の `mflag_*`，`boot_units.npz` の `|mmask` | A:852，A:819 | 合う．試走で `|mlo`・`|mup`・`|mmask` 各 2,016（定義から 0 の単位 144 を含む）．定義から 0 でない単位で `|mmask` ＝ `|mask` の手法の 3 ビット（食い違い 0）．手法の印の無い抽出で mlo ≠ mup は 0，mlo ＞ mup は 0（`chk_mmask_v7.log`） |

計算する組の数え上げ（L4）：試走の判定の表は 360 行で前回と同じ（readable 120，all 80，readable_excl_mosei_afew 80，readable_impro_only 80）．本番の見込みは前回どおり 450 行．

## A．小さく流した結果

1. **試走（指定どおり）**：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W always exp/analyze_value_split_e87.py --smoke --runs fuse-audio-video-text --targets IEMOCAP-impro IEMOCAP-script --out checks/E87/code_run_v7/smoke`．壁時計 1 分 59 秒，最大の常駐メモリ 1.07 GB，警告 0 件，例外なし（`smoke.log`）．行の数は前回と同じ：curves 784，curves_rep 560，values 168，values_seed 3,680，units 5,292，corpus 1,764，family 35,280（記録 21,168，L 7,056，U 7,056），direct 3,024，direct_corpus 1,008，direct_family 20,160，judgments 360，judgments_dS 28，judgments_563188d 28，assumptions 2,745．`checks.csv` は `chosen_missing_expected` 12 件だけ．values の NaN・Inf 0，全表で Inf 0．見出しのコミット 45586c4，`dirty` 偽．物差しの項目 16，物差しのコミット 8d73121・36193e5，`code_dirty` はすべて偽．`n_bfix_unchecked`・`n_bfix_fail` の和 0（`chk_struct_v7.log`）．
2. **保存済みの値の再現（`chk_repro_v7.py`）**：curves_rep の UAR と物差しの csv の差の最大 1.7×10⁻¹⁶，C の差 0，C 4 つの UAR の差 0（560 行．K = 100 の 80 行を含む）．values_seed と `results/<wave>_<実行>/<T>.csv` の uar の差の最大 1.7×10⁻¹⁶（3,680 行）．curves の S_raw と反復の平均の差 2.8×10⁻¹⁴，values の S_x と種・反復の平均の差 2.8×10⁻¹⁴．
3. **保存した出力からの作り直し（`chk_struct_v7.py`）**：保存した `units.csv`・`boot_units.npz`・`jk_units.npz` だけから，corpus 1,764 行（区間・限り・区分が一致），family 35,280 行（記録・L・U のすべてで中央値・区間・区間が使えるかが一致），boot_family の主 18,726 配列と直接の差 10,668 配列（保存した 29,394 配列と全部一致），direct_family 20,160 行，judgments 360 行（区分・最終の区分・対比・`rule_sensitive` の列の食い違い 0）を作り直した．点の限りの順序の破れ 0／3,024，読める単位の点の限り ＝ 点 361／361，抽出の限りの順序の破れ 0，印の無い抽出で lo ≠ b ≠ up 0．直接の差の点の限りの差の最大 9.3×10⁻¹⁰，抽出の限りの相対の差の最大 6.0×10⁻⁸．`chk_struct_v7.log` の「手法の印の無い抽出で mlo ≠ mup 1,452」は，検査の台本が `|mask`（定義から 0 の単位では 0）で印を見たためで，`|mmask` で数え直すと 0 だった（`chk_mmask_v7.log`）．
4. **人工のデータの検算（`chk_synth_v7.py`．90 項目のうち 89 項目が通った）**：前回の 66 項目を第 7 版の戻り値（`read_bounds` が 8 個）と直接の差の入力（`mflag_*`，`|mmask`）に合わせて流し，すべて通った．V6 の期待は第 7 版の定義に合わせて「`flag_*_none` は偽」を足し，V6b は手法の印を `|mmask` に置いた．足した 24 項目：V6b′（`|mask` だけにある印は直接の差に入らない），W0（8 番目の戻り値 ＝ none の読みの `bounds_of`．乱数の曲線 3,000 本），W0b（96cdf57 のほどき方は ValueError），W1〜W1g（`method_scores` を人工の値に差し替えて `values_group` を流す．none を選んだ群の手法の側の 7 つの量が，値が none と同じ手法を選んだ群の値とすべての右端・座標で一致．none が上に打ち切られた点で限りが [最初の読み, 10⁶]・読めない・hi_method の印，あいまいの点で [最初, 最後]・amb_method の印，印の無い点で限りが点に縮む，抽出の `mmask` の hi_method のビットが独立に読んだ none の抽出の上の打ち切りと一致（割合 0.336）し，その抽出で `MUB` ＝ 10⁶，n の側は定義から 0 のまま，saerens の向きも同じ枝），W2〜W2h（`aggregate_unit` → `direct_units`．B が none を選んだ群で，点の限りが lo(K'm_A) − 10⁶ と up(K'm_A) − 最初の none の読み，`is_readable` 偽，`flag_hi_method` 真で `flag_*_none` 偽，抽出の印が none の抽出の上の打ち切りと一致して cens_interval，抽出の限りが MLB_A − MUB_none，点の限りと抽出の限りが同じ作り方，あいまいと印の無い場合），W3〜W3c（同じ単位の中の差と ΔS の直接の差．**W3b が通らなかった**．指摘 1），W4〜W4c（包絡の式と順序）．
5. **none を選んだ群の試走（`chk_nonegroup_v7.py`．台本の関数を包んで数えるだけ）**：AV の CREMA-D（`--smoke --runs fuse-audio-video --targets CREMA-D`）と AVT の MELD・MOSEI（`--runs fuse-audio-video-text --targets MELD MOSEI`）．警告 0 件．結果は上の表の 1 行目．
6. **止める条件（`chk_load_v7.py`．20 項目すべて通った）**：SCRIPT_DEPS は 39 本で前回と同じ．`git status` を偽の結果に差し替えて，依存する台本のどれか 1 本に変更があると本番の引数で `--out` を作る前に止まった．

## A．本番の規模の見積もり

- **物差し**：主と K = 100 がそろっている（試走の provenance で 16 項目）．GPU の仕事は起動していない．
- **解析（CPU）**：試走 1 分 59 秒で第 6 版（1 分 53 秒）とほぼ同じ．本番は前回の見込みどおり 15〜30 分．`|mmask`（uint8）が `|mlo` と同じ数だけ増えるので，boot_units は前回の見込み（約 0.9 GB，展開後）に約 0.05 GB を足した量になる見込み．メモリの空きは 56 GB ある．

## B．予想される指摘と，出力から答えられるか

前回の表から変わった行だけを書く．ほかの行は `checks/E87/code_v6.md` の B の表のとおり．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 直接の差・教師の規則 | none を選んだ群の直接の差の扱いは正しいか | 前回の指摘 1，I1 | (a)：`units.csv` の `point_m_lo_bound`・`point_m_up_bound`・`method_readable`・`mflag_*` と `boot_units.npz` の `|mlo`・`|mup`・`|mmask`．試走で作り直しと一致した | — | — |
| 直接の差の表 | none の幅を含めた従来の限り（lo(A) − up(B)）でも結論は同じか | 設計 2.4 | (a)：両方の単位の `|lo`・`|up` の差と `point_lo_bound`・`point_up_bound` | — | — |
| 直接の差の表（同じ単位の中の差，ΔS） | 同じ単位の中の差の直接の差に区間が使えるか，打ち切りの数は | 指摘 1，R1 | 台本のままでは (a) にならない（`interval_ok`・`judgment`・`flag_*_method` が誤る）．両方の単位の `|mask` の手法のビットから作り直せば (a) | 指摘 1 の直し | 無視できる |
| 判定の表 | 包絡の広げ方を変えたら（第 6 版の L ＝ T − (B − LB)） | 設計 2.4 の版の変更 | (a)：保存した B・LB・UB・jackknife から `family_stats` の 1 行を戻して作り直せる | — | CPU で数分 |

計算の後に求められると GPU の学び直しになる項目は前回どおり (c) の 2 行だけである．

## 指摘

### [I1] 1．同じ単位の中の差と ΔS の直接の差で，`flag_*_method` が NaN から真になり，同じ単位の中の差の抽出の印が 0 になる
- 重さ：直してから進む
- 該当箇所：`exp/analyze_value_split_e87.py:774`，`:852`，`:881`，`:891–896`（読んだうえで人工のデータと試走で確かめた）
  ```
  mflags = {...} if q in QUANTS_B + QUANTS_PT else {}                       # :774  DIFFS と dS では空
  "method_readable": bool(m_readable), **{f"mflag_{k}": v for k, v in mflags.items()},   # :852  列が無い行は DataFrame で NaN
  flags = {k: bool(ra.get(f"mflag_{k}", False) or rb.get(f"mflag_{k}", False)) if k.endswith("_method") else False ...}   # :881
  ma, mb = boot_store.get(ra.boot_key + "|mmask"), boot_store.get(rb.boot_key + "|mmask")   # :891  DIFFS の単位には |mmask が無い
  ```
- 問題：(i) 同じ単位の中の差（dir_minus_pos・dir_minus_posloss）と ΔS の単位の行には `mflag_*` の鍵が無いので，`units` の DataFrame でその列は NaN になる．`ra.get(...)` は NaN を返し，`bool(nan or ...)` は真なので，直接の差の `flag_lo_method`・`flag_hi_method`・`flag_amb_method` が 3 つとも真になる．試走で同じ単位の中の差の直接の差 576 行のすべて，ΔS の直接の差 432 行のすべてで真になった（人工の W3b でも ΔS の行で 3 つとも真）．`direct_family.csv` の `n_corp_hi_method` が正の行は，同じ単位の中の差で 3,840／3,840 行になった．(ii) 第 7 版は直接の差の抽出の印を `|mmask` だけから作るが，`|mmask` は QUANTS_B の単位にしか保存しない（A:816–819）．このため同じ単位の中の差の直接の差の抽出の印は常に 0 で，2.5% の基準が当たらない．試走の 576 行は，第 6 版の作り方（両方の単位の `|mask` の手法のビット）なら全部が 2.5% を超えて cens_interval になる行だったが，第 7 版ではすべてが `interval_ok` 真で pos・neg・undet のどれかに区分された．一方，この行の限りと `is_readable` は none を含む作り方（lo(A) − up(B)，none も含めた読めるか）のままなので，1 つの行の中で none と手法の扱いがそろっていない．`judge` は直接の差を `dirq`（dir・dir_bfix・dir_auc）でしか使わないので，`judgments.csv` の値には入らない（読んだうえで，試走の作り直しで判定の列の食い違いが 0 であることを確かめた）．影響は `direct.csv`・`direct_corpus.csv`・`direct_family.csv` の同じ単位の中の差と ΔS の行の `flag_*`・`n_corp_*`・`interval_ok`・`judgment` の列に限られる
- 計算の後で直せるか：直せる．両方の単位の `|mask`（同じ単位の中の差の単位にも保存される）の手法のビットと，`units.csv` の `flag_*_method` から作り直せる
- 求める対応：`direct_units` で，(i) `mflag_*` が NaN のときは偽として扱い，同じ単位の中の差と ΔS の行では flag を `flag_*_method`（ΔS は偽）から作る．(ii) `|mmask` の無い単位では第 6 版と同じく `|mask & 手法の 3 ビット` に戻す（または `aggregate_unit` で同じ単位の中の差の単位にも `|mmask` ＝ dir と pos の `|mmask` の論理和を保存する）．直した後に `chk_synth_v7.py` の W3b と `chk_mmask_v7.py` の (4) を流し，同じ単位の中の差の行で flag が単位の印と一致し，抽出の印が 0 でなくなることを確かめる．直さずに進むなら，これらの行を本文に引かない

### [I1] 2．包絡の値が 10⁶ の何倍にもなる
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:1028`，設計 2.4 節の「包絡の中央値は，コーパスの数が偶数で 10⁶ が中央の 2 つに入ると約 5×10⁵ の値になる」
- 問題：第 7 版は限りのずれにも倍率 r を掛けるので，−10⁶ に置いた抽出は x₀ ＋ r·(−10⁶ − x₀) になり，10⁶ を大きく超える．試走の `boot_family.npz` の L・U の配列のうち絶対値が 1.5×10⁶ を超える値を含むもの 6,751，最大は約 2.5×10⁷（値の向きは不利な側で，判定の向きは変わらない．W4c）．設計の「約 5×10⁵」の文は r ＝ 1 のときだけ正しい
- 計算の後で直せるか：直せる（記述だけの問題）
- 求める対応：設計 2.4 節の文を「約 r·5×10⁵」に直すか，包絡の中央値と端の値は不利な側の極端に置かれたことだけを意味し，大きさは読まないと注に書く

### [I1] 3．provenance の `family` と台本の冒頭の説明が第 7 版の作り方を書いていない
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:12–13` と provenance の `family`（読んだだけ）
- 問題：provenance の `family` は「包絡は広げた後に限りのずれを足す」（第 6 版の式）のままで，第 7 版の L ＝ x₀ ＋ r·(LB − x₀) を書いていない．冒頭の説明は上の打ち切りの上限を「K_all 以上」と書き，最後の読みが上に打ち切られる場合の 10⁶，第 6 版のあいまいの定義，直接の差を手法の読みだけから作ること，none を選んだ群の引き継ぎを書いていない
- 計算の後で直せるか：直せる
- 求める対応：provenance の `family` と冒頭の説明を設計 2.3・2.4 節の文に合わせる

### [I1] 4．96cdf57 の台本は試走で止まる（45586c4 で直った）
- 重さ：記録する
- 該当箇所：96cdf57 の `exp/analyze_value_split_e87.py:663`
- 問題：`read_bounds` の戻り値を 8 個にした変更が，呼ぶ箇所の 1 つ（複製の枝）に入っていなかった．45586c4 で直り，呼ぶ箇所の 2 つとも 8 個でほどいている．人工の検算は関数の戻り値の個数を確かめず，`values_group` を流す項目も無かったので，第 6 版までの検査の台本では見つからない形の誤りだった（W1 の差し替えの道具で今回から見つかる）
- 計算の後で直せるか：該当しない（計算の前に直った）
- 求める対応：要らない．関数の戻り値を変えたら，コミットした台本そのもので試走してから関門に出す

## 目録に無い種類の不備

- **I28（案）「鍵の無い列が NaN になり，`bool(NaN)` が真になる」**：辞書を行にして DataFrame を作るとき，一部の行にだけ鍵を足すと，ほかの行の列は NaN になる．`row.get(列, False)` は NaN を返し，`bool(nan)` と `nan or x` は真になるので，印や読めるかが黙って真になる．台本の確認では，行の種類（座標，差の量，ΔS など）ごとに新しい列が NaN かどうかを数え，その列を真偽として読む箇所で `pd.notna(v) and bool(v)` の形になっているかを確かめる．
- **I29（案）「保存するものの集合と，それを読む側の集合がずれていないか」**：新しい配列（今回は `|mmask`）を保存する条件（QUANTS_B だけ）と，それを読む関数が当てる行（直接の差のすべての量）が違うと，読む側は「無い」を「印なし」として扱う．台本の確認では，読む側が `get(...) is None` で既定値に落ちる行の数を量ごとに数え，既定値に落ちる行が設計で許された行だけであることを確かめる．

---

## 再検査の追記（台本を直した後に使う）

（直した差分を検査したときにここへ書く．`checks/E87/code_run_v7/` の `chk_synth_v7.py`（90 項目．W3b が指摘 1 の確かめ），`chk_mmask_v7.py`（(4) が指摘 1 の試走での確かめ），`chk_struct_v7.py`，`chk_envelope_smoke_v7.py`，`chk_envelope_direct_mag_v7.py`，`chk_repro_v7.py`，`chk_nonegroup_v7.py`（AV の CREMA-D と AVT の MELD・MOSEI で流す），`chk_load_v7.py <一時のディレクトリ>` を流し直せる）

## 計算の後の点検

（計算の後に書く．次を行う：物差しの 34 組 × 2 実行の npz の欠けと `_meta`（code_dirty，コミット），provenance の `e76_check` の項目数（68）と `kidx_equal`，`speakers_crc32` の世界の間の一致，`values.csv` の `n_bfix_unchecked` の和，判定の表の行数（450 の見込み），`|lo`・`|up`・`|mlo`・`|mup`・`|mmask`・`|Kn`・`|Km`・`|Sn`・`|Sm` の数，抽出ごとの L ≤ T ≤ U の破れ（主の族で 0，直接の差の族で 10⁻³ 本以下），none を選んだ 6 群の直接の差の行の限りと印（`chk_nonegroup_v7.py` の数え方），同じ単位の中の差と ΔS の直接の差の flag と抽出の印，結果ファイルのコミットと検査したコミットの差分，ログの例外と警告）
