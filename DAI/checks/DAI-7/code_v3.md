# 関門 1：実装と出力・差分の再検査（DAI-7，2026-11-07，コミット 7df1bd3．設計は第 6 版）

検査したもの：`git diff eb35214 7df1bd3 -- DAI/exp/dai7_probe_v2.py DAI/docs/実験7_下調べ_設計.md` の全部．台本が呼ぶ `dai1.make_G`，`dai1.pass3`（tt の教師の組），`dai_solver.fit`（目的関数の正規化），`dai_solver.balanced_onehot`，`dai_solver.soft_rows`，`dai3.subset_masks`，`dai3.scenario` も読んだ．試走は CPU で行った（`CUDA_VISIBLE_DEVICES=""`）．試走の前に作業木を確かめ，`dai1.git_commit` が見る範囲（DAI/exp，DAI/splits，measure，features，data/metadata）は clean だった．台本は DAI/exp のものをそのまま流した（sha256 2874595f… は 7df1bd3 の版と同じ）．出力は `/home/snics/MER-DA/DAI/checks/DAI-7/code_run_v4/out/` に置き，検算の台本（`check_v4.py`，`check_v4_sum2.py`，`check_keep_main.py`）と結果（`check_v4_out.txt`），ログ（`log_*.txt`）も同じフォルダに置いた．`results/dai7_probe/`，`results/dai7_probe_v2/`，`logs/dai7_probe.log` と tmux の画面は開いていない．本体が `code_run_v3/` に流した試走は eb35214 の dirty の版だったので，コミットした版で流し直した．

本検査が見た値：形，列，NaN と Inf の数，作り直した値と行の値の差，同じ問の 2 組の差，前の版との差，恒等式の差，有効な回の数，所要，ファイルの大きさ．D，ΔD，φ，UAR と criteria の判定の値は画面に出していない．

## 判定
通す．第 6 版で足した量（tt_ownret，変種 cent・keep・scen・chan・sub_*・spk，H1 の書き方 1〜4，H1_AV，全体の数と IEMOCAP を除いた数，H4，tether_weakening）は設計どおりに計算され，作り直しの検算もすべて一致した．残りは記録する 7 件である．

## A．設計と台本の対応表（差分の範囲）

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 3.2 tt_ownret：teacher=ret_m，ρ=1，a=1 | `classifier_sets` の `sets[OWN][m] = pick(z3, key, dict(method="tt", teacher=f"ret_{m}", rho=1.0, a=1))` | 合う．`dai1.pass3` の `tts.update({f"ret_{m}": …})` で生徒自身の ret_m も教師に入り，`make_G` は joint と tt で同じ式（`a·gk + soft_rows(Q, ρ·n)`）を使う．同じ `solve_items` の束で同じ cw・cb・W0・b0 で解かれる．このため joint_a1[ρ=1] との違いが Q だけであることをコードで確かめた．判定は p3 の保存と 0/100,815 本で一致した |
| 3.3 cent：(グループ, 感情) の平均を引き，3 本未満を外し，(グループ, 感情) の中で並べ替える | `run_unit` の `variants["cent"]`，`center_cells`，`d_with_null(..., cells=cellkey[keep])` | 合う．グループは IEMOCAP が speaker_id，ほかが cluster |
| 3.3 keep：cent と同じ発話で中心化しない | `variants["keep"] = (keep_cent, None)` | 合う |
| 3.3 scen（IEMOCAP だけ）：`dai3.scenario` | `variants["scen"]`，fits の `test_scenario` | 合う．外した数の列は無い（[記録 2]） |
| 3.3 sub_*：`dai3.subset_masks` の印．MOSEI は chan | `for nm, mk in dai3.subset_masks(...)` | 合う（AFEW は asr_ok，IEMOCAP は marker と nomarker，MELD は face_single） |
| 3.3 変種の区間：変種の発話を持つ塊だけで抜き直す | `boot_D` の `cl_by_fold`（印で絞る）と `codes` | 合う．試走の塊：MOSEI chan 113/521，MELD face_single 46/66，AFEW asr_ok 11/11 |
| 3.3 spk（IEMOCAP）：話者を塊にした区間 | `boot_D(var="spk")`（clkey=test_group） | 合う（試走の分割 0 で塊 2 つ） |
| 3.3 変種の偶然の水準の乱数は組の間で共有する | `crn(seed, K, ip, VKIND[vname])`．VKIND は 2,5,6,7,8,10,11,12 | 合う．0（テスト），1（候補），3（教師を除く），4（h⁰，対の番号 9）と重ならない |
| 3.5 恒等式と分け方 | `decompose`，`bP`，diffs の `S_*`・`S1_*`・`dLdep_*`・`dLcal_*`・`share_dep_*` | 合う．試走の恒等式の差は 5.6e-16 以下 |
| 3.5 感情ごとの再現率と本数 | uar.csv の `rec_{A,V,T,prod,blk,full}_e*` 24 列と `n_test_e*` | 合う．NaN は 0 |
| 4 H1・H1(joint)・H1′・H1_AV | `count` と `ok_K`（pair="AV" で H1_AV） | 合う |
| 4 書き方 1（φ） | `wd`，`phi_small` | 合う |
| 4 書き方 2（共有の比べ）：D(ρ=1)−D(tt_ownret)，偶然の水準の条件つき．φ の差も見る | `("share", "dD_rho1_ownret", FLOOR_D, True)`，`phi_share` | 合う．D(tt_ownret)−D(none) と φ の差は `dD_ownret_none` と `dphi_ownret_none` で記述として出る |
| 4 書き方 3（MOSEI の chan） | `chan_n_K_robust` | 合う（H1 の条件を満たした K の中で，下端 > 0 かつ全体の点推定の半分以上を数える） |
| 4 書き方 4（cent と keep の比べ） | `cent_n_K_small` | 合う |
| 4 全体の数と IEMOCAP を除いた数 | `*_n_corpora`，`*_n_corpora_excl_IEMOCAP`，`*_depends_on_IEMOCAP` | 合う（H1，H1_joint，H1p，H1_AV，share，H2，H3，additivity_fails） |
| 4 H4（0.5 と 0.25，最も大きい分） | `H4_*`，`H4_rho1_*` | 合う |
| 5 tether_weakening.csv | `tether_table` | 合う．`dai_solver.fit` は ‖W−W0‖²/(2C′) を足して和の重みで割るので，C′ が大きいほど引き戻しが弱い．K 本の重みの和は `balanced_onehot` で 4K，疑似ラベルの項は ρ·u.n である．「C′ の最大 ÷ none の C′」と (4K+ρn)/(4K) の向きは正しい |
| 5 fits の印，筋書き，教師の点数 | `store.update({f"mask_{v}": …})`，`test_scenario` | 合う |
| 5 provenance（pandas，cuda，vkind，r_share 等） | `main` | 合う |

## A．小さく流した結果

- 自己検算（`--selftest`）：12 項目すべてが通った（2.4 秒）．
- 単位の段（CPU，1 単位ずつ続けて流した）：
  | 単位 | 時間 | 最大メモリ |
  |---|---|---|
  | IEMOCAP_AFEW_f2_r0（c=AFEW，テスト 158） | 8 秒 | 0.9 GB |
  | MELD_IEMOCAP_f0_r0（c=IEMOCAP，テスト 731） | 18 秒 | 1.2 GB |
  | AFEW_MELD_f0_r0（c=MELD，テスト 2,105） | 39 秒 | 2.2 GB |
  | AFEW_MOSEI_f1_r0（c=MOSEI，テスト 3,727） | 69 秒 | 3.5 GB |
- 集計の段（4 単位，`--allow-partial`）：28 秒，0.7 GB．
- 形：
  - 1 単位あたり units.csv が 378 行（30 組）．列は IEMOCAP が 101 列，ほかが 73 列である．
  - uar.csv は 65 行 × 51 列（13 組）．
  - 集計の出力：cells.csv 2,056 行，uar_cells.csv 260 行，diffs.csv 2,980 行 × 23 列（main 1,780，cent・keep 各 320，scen・chan・sub_*・spk 各 80，感情ごとの行 160），criteria.csv 4 行 × 96 列，tether_weakening.csv 48 行．
- NaN と Inf：
  - none_tw を除く主の列（D，D_raw，D_null，D_pool，D_partial_t，φ）に NaN は 0 個，Inf も 0 個だった．
  - 区間を作る組（BOOT_SETS の 8 組）の変種の列に NaN は 0 個だった．それ以外の組では，D_cent 以外の変種の列がすべて NaN で，これは設計どおりである．
  - diffs に区間が NaN の行は無い（区間を作らない dDt と G_joint は除く）．
  - criteria の NaN は MOSEI 以外の chan_n_K_robust だけである．
- 有効な回の数：変種の行はすべて 1,000 回だった．main の行の最小は AFEW が 843 回（UAR から作る量），MELD が 995 回である．
- 作り直しの検算：
  - 変種の d_e − d_e⁰ を，fits の点数と印から重み 1 の抜き直しで作り直した．行の値との差の最大は cent 1.2e-8，keep 1.0e-8，scen 1.3e-8，chan 7.7e-9，sub_* 2.5e-8 だった．NaN の形の不一致は 0 個だった．
  - 偶然の水準を `crn(seed, K, 対, VKIND)` で独立に作り直した．差の最大は 1.6e-9 で，感情ごとの本数はすべて一致した．
  - 同じ問の 2 組：joint と joint_a1 では，変種の列すべての差が 0.0 だった．ret と tt[fused_AVT] の D_cent の差の最大は 3.6e-7 だった．
  - 抜き直しの配管：IEMOCAP の単位では keep の印が全部の発話である．keep と main の抜き直しの差（joint_a1−none と ρ=1−tt_ownret）は，回の間の標準偏差が 2.2e-19 以下だった．つまり 2 つは偶然の水準の定数だけ違う．
  - 恒等式：S と (S1 + dLdep + dLcal) の差の最大は 5.6e-16 だった．dLdep_s（分け方の経路）と dLdep_s_none（BL の経路）の点推定と区間の差，および G_s と −S の差の最大は 1.4e-14 だった．
- 保存した値の再現：
  - tt_ownret の判定は，実験 1 の p3 の保存の判定（`p3_<m>_K<K>_pred`）と 100,815 本のうち 0 本しか違わなかった（4 単位 × 5 K × 3 モダリティ）．
  - 前の版（6c4bc26，`code_run_v2/all/`）と共通の 2 単位（AFEW_MELD_f0_r0，IEMOCAP_AFEW_f2_r0）を比べた．units の 363 行の主の列（D，D_raw，D_null，D_pool，D_cent，n_dropped_cent，D_partial_t，φ，d_e，d0_e）の差は 0.0，uar の 60 行の差も 0.0 だった．fits の共通の鍵 872 個にも差は無かった．新しい鍵は 43 個である．
- 中心化で外した発話：MELD 108/2,105，MOSEI 764/3,727，AFEW 22/158，IEMOCAP 0/731（scen では 3/731）．

## A．本番の規模の見積もり

- 単位の段（CPU）：30 ×（AFEW 8 秒 + IEMOCAP 最大 18 秒 + MELD 39〜53 秒 + MOSEI 69 秒）で，約 67〜75 分になる．MELD の分割 0 は 2,105 本で，ほかの分割は最大 2,850 本なので，本数に比例させて上限を出した．
- 集計の段：試走（4 単位）の 28 秒を単位の数に比例させて，約 10〜20 分の見積もりである（読んだだけ．IEMOCAP の変種の数が多いので上限を広めに取った）．
- メモリ：単位の段の最大は 3.5 GB（MOSEI），集計の段は 1 GB 以下の見込み．
- ディスク：`fits/` は 30 ×（1.3 + 4.8 以下 + 13.4〜18 + 26.2）MB で，約 1.4〜1.5 GB になる．rows は約 1 単位 0.3 MB である．
- GPU は要らない．GPU で流すなら，前回と同じく使用量を 6 割に抑え，単位ごとに empty_cache を呼ぶ（`run_unit` の末尾にある）．

## B．予想される指摘と，出力から答えられるか（前回の表に足す行）

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 共有の比べ | 感情ごとに ρ=1 と tt_ownret の差を出せ | a2 | (a)：`boot_D` が返す Be は BOOT_SETS の 8 組すべての感情ごとの値を持つ．`Be[RHO1][:, e] − Be[OWN][:, e]` を `ci` に通す | — | — |
| 共有の比べ | ρ=1 と tt_ownret の差を cent・chan・spk でも出せ | A2，m2 | (a)：DD_VAR に入っている | — | — |
| 共有の比べ・H4 | ρ=1 と tt_ownret について，候補の上の依存の区間と，配備できる結合器（候補で推した判別分析）での L_dep を出せ | R1 | (b)：POOL_SETS に RHO1 と OWN が無い．行の D_pool（点推定）はある．実験 1 の保存の W・b を当て直せば数分で作れるので，学び直しは要らない | POOL_SETS に RHO1 と OWN を足す | 約 130 MB（MOSEI 1 単位で 2.7 MB）．時間はほぼ増えない |
| H1 | none の引き戻しを別の規則で選んだら ΔD はどうなるか | R（C の規則） | (a)：none_tw の 14 点の z が fits にあり（`K{K}_none_tw*_z`），d0 は行にある | — | — |
| H3・H4 | none の引き戻しを別の規則で選んだら L_dep はどうなるか | 同上 | (a)：none_tw の z と fits の `half` から，`crossfit` で blk と full を作り直せる．prod は z の和の argmax に Hᵀ を当てれば同じ判定になる | — | — |
| H1・H3 | IEMOCAP の L_dep・G・S を話者を塊にした区間でも出せ | m2 | (a)：fits の判定と test_group を `boot_uar_unit` に渡す | — | — |
| H1 | 性別ごとに出せ | 過去の例 | IEMOCAP（sample_id の F/M）と AFEW（発話一覧の gender）は (a)．MELD は人物の名前からしか作れず，MOSEI は性別が無いので (c) | 設計 8 節に一文を足す（[記録 4]） | — |
| H1 | 発話の長さで分けよ | 8 節 | (a)：IEMOCAP・MELD は発話一覧の start と end，AFEW は Whisper の書き起こしの duration_sec を sample_id で結べる．MOSEI は確かめていない | — | — |
| H1 | IEMOCAP の Happy を excited とそれ以外に分けよ．MOSEI の Neutral を強さで分けよ | a3〜a7 | (a)：発話一覧の emotion_raw と int_* を sample_id で fits に結ぶ | — | — |
| tether_weakening | 単位ごとに見せよ | m4 | (a)：p3 の tether と固定のファイルから作り直せる | 単位ごとの表も書く（[記録 3]） | 数 KB |
| H2 | ρ の 4 点の依存を cent でも出せ | A2 | (a)：ρ=0.25 は BOOT_SETS に無いが，z は fits にあり，乱数は `crn(seed, K, 対, VKIND)` で作り直せる | — | — |
| ΔD_t | 区間を付けよ | M1 | (a)：fits の teacher_t と z から残りを作り，`crn(…, 3)` で偶然の水準を作り直す | — | — |
| 全部 | 融合の分類器（AVT），第 2 のエンコーダ，自分の none[K] の予測で学ぶ対照 | 8 節 | (c)：範囲の外と書かれている | — | — |

学び直し（実験 1 の分類器の学習）が要る (b) は無い．

## 指摘

### [R1] 候補の上の点数の保存に ρ=1 の組と tt_ownret が無い
- 重さ：記録する
- 該当箇所：`DAI/exp/dai7_probe_v2.py` の `POOL_SETS = ("none", "joint", "joint_a1", "oracle", RHO_SETS[0], RHO_SETS[-1], "ret")`
- 問題：第 6 版で H4 の対照（joint_a1[ρ=1]）と共有の比べ（tt_ownret）が主の量になったが，この 2 組の候補の上の点数が fits に残らない．試走の fits の zpool の組は上の 7 つだけで，`K5_joint_a1_rho1.0_zpool` は無かった．このため，候補の上の依存の区間と配備できる結合器の L_dep を，この 2 組について fits だけからは出せない．
- 計算の後で直せるか：直せる．実験 1 の保存の W・b を候補の特徴に当て直せば数分で出せる．
- 求める対応：本計算の前に POOL_SETS に RHO1 と OWN を足すことを勧める（約 130 MB）．足さない場合は，後で W・b を当て直すことを台帳に書く．

### [記録 2] 筋書きで中心化したときに外した発話の数の列が無い
- 重さ：記録する
- 該当箇所：`run_unit` の行の `n_dropped_cent=int((~keep_cent).sum())`．scen の数を出す列が無い
- 問題：設計 3.3 節は scen についても「外した数を残す」とする．数は fits の `mask_scen` と行の `v_scen_n_e*` から出せる．試走では 3/731 本だった．
- 求める対応：`n_dropped_scen` を行に足すか，設計の文を「fits の印から出す」に直す．

### [記録 3] tether_weakening.csv はコーパス・K・モダリティごとの平均だけである
- 重さ：記録する
- 該当箇所：`summarize` の `TWc = TW.groupby(["c", "K", "mod"])…mean()`
- 問題：単位ごとの値（T ごとに選んだ ρ と C′ の違い）が残らない．作り直しはできる．
- 求める対応：TW もそのまま `tether_weakening_units.csv` に書く．

### [記録 4] 性別の扱いが設計に無い
- 重さ：記録する
- 該当箇所：設計 8 節
- 問題：IEMOCAP と AFEW は性別ごとに後から出せるが，MOSEI には性別の列が無く，MELD は人物の名前からしか作れない．
- 求める対応：8 節に「性別ごとの解析は IEMOCAP と AFEW だけで後から出せる．MOSEI と MELD は性別の記録が無いので範囲の外」と一文足す．

### [記録 5] 中心化の印の一致の確かめは必ず通る形になっている
- 重さ：記録する
- 該当箇所：`run_unit` の `if not (keep == vmask).all()`，`boot_D` の同じ確かめ
- 問題：どちらも，印と中心化の残りを同じグループと同じ `center_cells` から作っているので，必ず一致する．不一致を見つける働きは無い．害は無い．fits と行の対応は，本検査の作り直しで確かめた．
- 求める対応：なし（記録だけ）．

### [記録 6] ΔL_dep の割合の区間は S が 0 に近い回で揺れる
- 重さ：記録する
- 該当箇所：`ci(BP[s]["dLdep"] / BP[s]["S"])`
- 問題：H4 の規則は点推定だけを使うので，判定は変わらない．ただし diffs の `share_dep_*` の lo・hi は，抜き直しの回で S が 0 に近いと極端な値になりうる．
- 求める対応：結果の文では割合の区間を参考とし，S と ΔL_dep の区間を並べて書く．

### [記録 7] 設計の文書の未コミットの変更と，本計算の出力先
- 重さ：記録する
- 該当箇所：`DAI/docs/実験7_下調べ_設計.md` の 0 節 (e)（作業木で 1 行変わっている）．`dai1.git_commit` は DAI/docs を見ない
- 問題：本計算が止まる条件（dirty）は設計の文書の変更を見ない．このため，設計の文書の版が provenance に残らない．また，`results/dai7_probe_v2/` に既に単位の出力があれば `run_unit` が止まり，provenance.json には前の回の記録の後ろに今回の記録が足される．本検査は `results/dai7_probe_v2/` を開いていない．
- 求める対応：本計算の前に設計の文書をコミットする．出力先に前の版の出力が無いことを確かめる．
- 補足（記録）：「全体の数」の share_n_corpora は，H1 と H1′ を満たさないコーパスも数える．書き方 2 は H1 と H1′ を満たしたコーパスだけで使うので，全体の数を書く文では，criteria の share と H1・H1p の列を組にして読む．

当てて問題の無かった項目（差分の範囲）：I1（第 6 版との対応），I5（同じ単位の同じ添字），I7（変種でも共通の乱数で同じ問の組が一致する），I9・I12（集計のコミット 2d1daf6，dirty は False，種 0，1,000 回），I11（有限の確かめ），I14（tt_ownret の判定と前の版の再現），R1 の残り（変種の印と筋書きの保存）．

## 目録に無い種類の不備
1. I18（案）：中心化の印や部分集合の印が「中心化の残り」と一致するかを確かめるときは，2 つを別の源（印は保存した fits，残りは読み込み直した発話一覧）から作る．同じ関数と同じ入力から作った 2 つを比べる確かめは必ず通るので，不一致を見つけない（DAI-7 の gate-code の再検査）．

---
総合の判定：通す（残りは記録する 7 件で，結果の数値を変えない．本計算の前に [R1] の 2 組を POOL_SETS に足すことを勧める）．
