# 整理の後の写しで論文の前半の台本を流し直した検査（2026-11-15，写しのコミット 33cf15a ＝ 整理の作業枝 a06f779 の書き出し）

判定：直す

理由：置き場の変更そのものは数値を変えていない．言葉を含まない種類と世界では，CPU で流した E75・E76 の解析・E79 の delta／select／judge・E80 の kept.csv と段 c の照合が，保存した出力と同じ値になった（多くはバイト単位で同じ，E75 で最大 1.6e-11）．一方で次の 3 点は再現の手順に書かれておらず，そのままでは流し直せない．

- 2026-11-04 の AFEW の言葉の追加（コミット 5ee7278，`measure/data.py`）で，言葉を含む種類の前半の解析が止まるか，保存した値とずれる（元から）
- 整理で台本の sha256 が変わったので，E79 の judge は保存した measure の出力に当てられない．E76 は学び直した新しい出力を解析の台本が受け付けない（整理による部分を含む）
- 5 節の入力の一覧に，E77・E78・E80 の台本が読む表が足りない（文書）

検査した範囲：写し `/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/standalone_front/DAI`．保存した出力 `/home/snics/MER-DA/results/` は読むだけにした（16:15 より新しいファイルが無いことを `find -newermt` で確かめた）．写しの `results/` には，保存した出力から 42 GB を `cp -a` で写した（リンクにはしていない）．GPU は使っていない．全部を `CUDA_VISIBLE_DEVICES=""` を付けて流した．写しの `exp/`・`measure/`・`features/`・`data/metadata/`・`docs/` は変えていない（流した後の `git status --porcelain` は空）．比べた数値の差の大きさだけを書き，値の意味は書かない．

## 0. 整理で台本の中身がどう変わったか（読んだだけ）

- 写しの `exp/` のうち整理の前の `/home/snics/MER-DA/exp/` にもある 44 本を `diff` で比べた．違うのは 38 本で，違う行はどれも `docs/`・`DAI_paper/` のパスを書いた行だった（1 本あたり 2〜14 行）．見た差分は docstring とコメントの行だった．コードの文字列で `docs/`・`DAI_paper/` を指す行は `analyze_e79.py:290` だけで，この行は変わっていない．`measure/` の 7 本，`features/` の 13 本，`data/metadata/` は整理の前と同じだった
- 前半の台本の入出力はすべて `ROOT = Path(__file__).resolve().parent.parent` からの相対パスで決まる．既定の出力先は写しの `results/` の下を指す（`reanalysis_e75.py:1799`，`analyze_e76.py:2010-2014`，`analyze_e79.py:2978,2982`，`explore_source_value.py:40-41`，`explore_source_value30.py:33,56`，`relearn_e76.py:739-750`，`relearn_e76_finish.py:41,99`，`compare_dai.py:953`，`compare_dai2.py:852`，`e80_teacher_j.py:491,494`，`e80_stage_c.py:678,692`）
- 前半の台本とその部品のコードには，写しの外（MER-DA の直下の `docs/`，`DAI_paper/`，`/home/...`）を開く行が無い．例外は `analyze_e79.py:290` の `git grep … HEAD -- docs DAI_paper` で，写しに `DAI_paper/` は無いが，`docs/参照/exploratory-analyses.md` に sha256 の先頭 16 文字が書かれているので通る（3 節）
- 写しの中で開くファイルのうち，写しの外の実体へ向かうのは `data/cache/*.npz`・`.venv`・`weights` のリンクだけである

## 1. 実験ごとの結果

差の数え方：数値の列ごとに「最大の絶対差」と「|差| > 1e-9 の要素の数」を数えた（NaN の位置の食い違いも数に入れた）．比べる道具は `checks/整理_2026-11-15/code_run_repro_early/cmp.py`，比べた結果の全行は同じ場所の `cmp_out/` にある．

「旧群」は検査用の包み `code_run_repro_early/run_old_groups.py` を使った回である．この包みは，読み込んだ `measure/data.py` の定数 `GROUPS_WITH_TEXT` だけを 5ee7278 より前の 4 群（IEMOCAP-impro・IEMOCAP-script・MOSEI・MELD）に戻して台本の `main()` を呼ぶ．ファイルは変えない．

### E75（`exp/reanalysis_e75.py`）

| 流したもの | 比べたもの | 結果 | 所要 |
|---|---|---|---|
| 全部（8 種類 × 項目 1〜13，既定の引数，`--out results/e75`） | 保存した `e75/` の CSV 48 本 | 言葉を含まない 4 種類（A・A(HuBERT)・V・AV）：48 本すべてで \|差\| > 1e-9 の要素は 0．最大の差は 1.64e-11（`e75_11_targets.csv`）．言葉を含む 4 種類：項目 1・6・8 だけが違う．`e75_1_testsets.csv` に AFEW の 4 行が増えた．`gaps.csv` は 16 行から 48 行になった（AFEW の点数が無いという欠けが 36 行増え，4 行が消えた）．`e75_8_delta_halves.csv` は 24 行増え，96 行のうち 384 要素が違い，最大の差は 0.0235．`e75_8_delta_halves_summary.csv` は 8 要素が違い，最大の差は 0.215．`e75_8_shrinkage.csv` は 8 行増え，32 要素が違い，最大の差は 0.00134．`checks.csv` の検算 550 件のうち 16 件が外れた（項目 8 の「Δ の 2 つの半分の平均と保存済みの own_d の差」．保存した回は外れ 0 件）．ほかの項目（2〜5，7，9〜13）は言葉の種類でも差が 0 | 371 秒（保存した回は 394 秒） |
| 旧群で項目 1 と 8（8 種類） | 同じ | 項目 1 は全部が同じ（`e75_1_testsets.csv` も 42 行で同じ）．項目 8 は T が差 0（3.2e-15 以下）．融合の言葉の 3 種類（AT・VT・AVT）だけが，`e75_8_delta_halves.csv` で 287 要素が 1e-9 を超え，最大の差は 1.68e-6（相対で 5.0e-7）．`e75_8_shrinkage.csv` は最大の差が 3.5e-9．検算 82 件の外れは 0 件 | 724 秒（ほかの計算と CPU を分けたので長い．1 回 10 分の決まりを超えた） |

旧群でも融合の言葉の 3 種類に 1e-6 の差が残る理由：`measure/data.py:149` の `_fuse_stats` は，群を `available()`（特徴のファイルがあるか）で選ぶ．このため `text_roberta_AFEW.npz` があると，`GROUPS_WITH_TEXT` を戻しても AFEW が融合の標準化の統計に入る．単一の T では差が 0 なので，差の原因は台本ではなく特徴の側にあると判断した．

### E76 の解析（`exp/analyze_e76.py`．`results/e76/` は保存したものを写した）

| 流したもの | 比べたもの | 結果 | 所要 |
|---|---|---|---|
| `--kinds audio_wavlm video`（B = 20,000，seed 76） | 保存した `e76_analysis/` の CSV 35 本（2 種類の行） | 35 本すべてで差が 0（バイトではなく値で比べた）．検算 112 件の外れは 0 件 | 286 秒 |
| `--kinds fuse-audio-video audio_hubert` | 同じ（A(HuBERT) の E13 を含む 36 本） | 36 本すべてで差が 0．検算 111 件の外れは 0 件 | 377 秒 |
| `--kinds text_roberta fuse-audio-text fuse-video-text fuse-audio-video-text`（今の data.py） | — | 26 秒で止まった：`FileNotFoundError: …/results/e76/text_roberta/AFEW.json`．`groups_for` が AFEW を返すのに，E76 は AFEW の言葉を学んでいないため | 26 秒 |
| 同じ 4 種類を旧群で | 同じ（4 種類の行） | T は 35 本すべてで差が 0．融合の言葉の 3 種類は 2 本だけが違う．`e76a_4_kind.csv` は 213 要素が違い，最大の差は 0.0735．`e76a_4_pairs.csv` は 180 要素が違い（列 G_center_pool・G_crit_c_pool），最大の差は 0.193．検算 145 件のうち 36 件が外れた（「mean_t と特徴の平均の差」と「係数から作り直した点数と保存した点数の差」．どちらも融合の特徴を読み直す検算）．`e76a_judgments.csv` で 3 行，`records.csv` で 12 行の文字の列が変わった．`e76a_3_rank.csv` で 8 列が無くなったのは，言葉の種類だけで流したので値が全部 NaN の列が作られなかったためで，保存した側でもこの 4 種類の行は全部 NaN だった | 250 秒 |

種類ごとの乱数は `zlib.crc32(kind)` から作るので，種類を分けて流しても同じ乱数になる．3 回に分けたのは，全部を 1 回で流すと 668 秒（保存した回）かかり，1 回 10 分の決まりを超えるためである．

### E79（`exp/analyze_e79.py`）

| 流したもの | 比べたもの | 結果 | 所要 |
|---|---|---|---|
| delta，`--runs fuse-audio-video-text fuse-audio-video`（今の data.py） | — | 34 秒で止まった：`欠けか食い違い：{'check': 'ps_missing', 'run': 'fuse-audio-video-text', 'target': 'IEMOCAP-impro', 'pseudo': 'AFEW', 'rep': 0}`．`groups_for` が AVT のソースに AFEW を返すのに，E77 は AFEW を仮のターゲットにしていないため．p4 の既定の実行（T と AVT を含む）も同じ理由で止まる（読んだだけ） | 34 秒 |
| delta，`--runs fuse-audio-video`／`--families p4 --runs audio_wavlm video fuse-audio-video`（今の data.py） | 保存した `e79/delta/`・`e79_p4/delta/` の同じ実行の行 | `delta.csv`・`delta_units.csv`・`delta_reps.csv` は 9／252／1,260 行と 15／456／2,280 行で，差は 0 | 35 秒，39 秒 |
| 旧群で delta → select（e79 の 2 世界と，e79_p4 の既定の 5 実行） | 保存した `delta/`・`select/` | `delta.csv` の sha256 は保存したものと同じ（80e6cd05bc4a7ea7…，fbc4d02298e5f231…）．select の CSV 7 本（candidates・checks・chosen・d8_table・fixed・kept_used・r6_subsets）は 2 組とも sha256 が同じ | delta 21 秒・23 秒，select 477 秒・184 秒（保存した回は 151 秒・70 秒．ほかの計算と CPU を分けた） |
| 旧群で judge（保存した `measure/` を写しの出力の場所に写した） | — | 止まった：`measure の段の台本の版（09f3a345b9455249）が今の台本（1eb98c9fd66f516e）と違う．同じ版で流し直す`（`analyze_e79.py:263-264`） | 数秒 |
| 旧群で judge（写した `measure/provenance.json` の `script_sha256` だけを今の台本の値に書き換えた．元の provenance は `standalone_front/e79*_measure_prov_orig.json` に残した） | 保存した `judge/` の CSV 20 本 × 2 組 | 40 本すべてで sha256 が同じ | 292 秒，306 秒 |
| 旧群で measure を 90 秒だけ（delta・select は写しで作り直したもの） | — | 段の間の照合（δ・chosen.csv・fixed.csv のコミットの確認と sha256）を通り，計算に入ったところで時間切れにした | 90 秒（打ち切り） |

### E32（`exp/explore_source_value.py`，`exp/explore_source_value30.py`）

| 流したもの | 比べたもの | 結果 | 所要 |
|---|---|---|---|
| 両方を既定の引数で | 保存した `explore_relation/e32_*.csv` 7 本 | 6 本は差が 0（`e32_pairs`・`e32_tests`・`e32_constancy`・`e32_allsrc`・`e32_allsrc_tests`・`e32_kinds30`）．`e32_kinds.csv` は 192 行のうち 4 行（T・AT・VT・AVT の K = 25 の B4）の uar と n が違い，最大の差は uar で 0.307，n で 1.13．保存した `e32_kinds.csv` は 2026-10-16 10:30 に書かれた．その後の 2026-10-18 に，入力の `results/compare_<言葉の種類>/` の `*_final*` などが書き足されている（`find -newer` で確かめた）．したがってこの差は整理によるものではなく，保存した出力が入力より古いために生じた（元から）．反復を 30 回にそろえた `e32_kinds30.csv`（2026-10-19）は差が 0 | 556 秒，33 秒 |

### GPU を使う段（引数と入力の読み込みだけ）

| 流したもの | 結果 | 所要 |
|---|---|---|
| `relearn_e76.py`・`compare_dai.py`・`compare_dai2.py`・`e80_teacher_j.py`・`e80_stage_c.py` の `--help` | どれも引数の説明を出して終わった．`relearn_e76_finish.py` は `--help` を受け付けない作りである（`--kinds` は `audio_hubert` だけを受け付ける） | 各数秒 |
| 新しく clone した `results/` の無い作業木で，各台本を小さい引数で（`CUDA_VISIBLE_DEVICES=""`，各 300 秒まで．`code_run_repro_early/gpu_scripts_no_results.txt`） | `compare_dai.py`・`compare_dai2.py`・`e80_teacher_j.py`：特徴を読んだ後，`results/explore_relation/e40_targets.csv` が無いと `FileNotFoundError` で止まる（`geom_e0.py:79` の `settings`．`teach_student39.py:140` も同じ表を読む）．`e80_stage_c.py`：`results/e80/kept.csv` が無いと止まる．`relearn_e76.py`：照合に使う表が無くても止まらず，CPU で学び始めた（300 秒で打ち切った．provenance に clone のコミット 33cf15a と `code_hash` が書かれた）．`relearn_e76_finish.py`：`--kinds は ('audio_hubert',) だけ` で止まる | 各 1〜300 秒 |
| `e80_teacher_j.py --summarize results/e80/avt results/e80/av`（GPU を使わない段） | `kept.csv` の sha256 が保存したものと同じ（83d82f05e1009b74…） | 11 秒 |
| `e80_stage_c.py --world <AV・AVT> --check`（GPU を使わない段） | 照合の食い違いは 0 項目．`check.csv` の sha256 は 2 世界とも保存したものと同じ | 17 秒，13 秒 |
| `run/e77.sh`・`e77b.sh`・`e77h.sh 1〜3`・`e78.sh`・`e78b.sh`・`e78h.sh 1〜3` を，新しい clone で `.venv/bin/python` を引数を表示するだけのものに替えて流した | `logs/` が無い clone では，どれも `logs/e77h_avt.log: No such file or directory` で Python を起動せずに終わった（終了の値 1）．`logs/` を作ると，どれも `cd` が clone の直下になり，呼ぶ台本（`exp/compare_dai.py`・`exp/compare_dai2.py`）が存在した．`e78.sh`・`e78b.sh` は `logs/e77.done` を待ってから動いた | 数秒 |

## 2. 結果の記録（provenance.json）

- E75：`provenance.json` に写しのコミット 33cf15ad…，`dirty: false`，台本の sha256（563111bc…．`sha256sum` の値と同じ），読み込んだ部品 17 本の sha256 と状態（すべて clean）が書かれた．CSV の 1 行目にも同じコミットと sha256 が書かれた．入力のファイルは大きさと更新の時刻だけで，sha256 は書かれない（元からの作り）
- E76 の解析：コミット 33cf15a，台本の sha256（67c764de…．`sha256sum` と同じ），部品 41 本の sha256，入力 136 本の sha256 が書かれた
- E79：delta・select・judge の `provenance.json` にコミット 33cf15a，`dirty: false`，台本の sha256（1eb98c9f…）が書かれた．`require_committed` が見つけた文書の名前は `HEAD:"docs/\345\217\202\347\205\247/exploratory-analyses.md"` のように，日本語の部分が 8 進数に置き換わった形で書かれる（git の `core.quotepath` の既定による）
- E80：`kept_provenance.json` にコミット 33cf15a と台本の sha256（42624dc6…）が書かれた
- E32：2 本とも provenance を書かない．出力の `results/explore_relation/e32_*.csv` を，止まらずに上書きする（元からの作り）

検査用の包みで `GROUPS_WITH_TEXT` を戻したことは，どの provenance にも書かれない．旧群の回の出力は比べるためだけに使い，写しの `results/e75_oldgroups/`・`e76a_Cold/`・`e79/`・`e79_p4/` に置いた．

## 3. `require_committed` が写しで通るか

`analyze_e79.py:290` の `git grep -l -F <先頭 16 文字> HEAD -- docs DAI_paper` は，写しで 7 本（e79 と e79_p4 の delta・chosen・fixed と，E80 の kept.csv）すべてについて `docs/参照/exploratory-analyses.md` を見つけた．select・judge は `--skip-commit-check` を付けずに通った．写しに `DAI_paper/` は無いが，pathspec に存在しない場所があっても git grep は止まらない．

## 4. 直すべき点

### 4.1 AFEW の言葉の追加の後で，言葉を含む種類の前半の解析が流せないか，保存した値とずれる（元から．整理によるものではない）

- ファイル：行：`measure/data.py:59`（`GROUPS_WITH_TEXT` に AFEW を足した．コミット 5ee7278，2026-11-04），`measure/data.py:149`（`_fuse_stats` は `available()` で群を選ぶ），`docs/再現の手順.md` 5 節と 7 節
- 何がおかしいか：前半の計算は 2026-11-01〜02 に，AFEW の言葉が無い状態で行った．今の `data.py` では次のことが起きる．
  - E79 の delta は AVT と T で止まる（`ps_missing … pseudo AFEW`）．このため言葉を含む世界と種類の E79 は，保存した E77・E78 の出力から流し直せない
  - E76 の解析は言葉の 4 種類で止まる（`results/e76/text_roberta/AFEW.json` が無い）
  - E75 は止まらないが，言葉の 4 種類で行が増える．項目 8 の値は最大 0.0235 ずれ，検算が 16 件外れる
  - `GROUPS_WITH_TEXT` を戻しても，融合の言葉の 3 種類では標準化の統計に AFEW が入る．このため E76 の 4 節の 2 本は最大 0.193 ずれ（検算の外れは 36 件），E75 の項目 8 は最大 1.7e-6 ずれる
  - 再現の手順にはこの限界が書かれていない
- どう直すか：`measure/data.py` に，AFEW の言葉を `groups_for` と `_fuse_stats` の両方から外す切り替えを足す．切り替えは既定で切っておき，環境変数で入れる．前半の言葉を含む種類の解析はこの切り替えを入れて流し，台本の provenance にその値を書く．`docs/再現の手順.md` の 5 節と 7 節には，この切り替えを入れないと言葉を含む種類が保存した値にならないことを書く．切り替えを入れたときに保存した値が出るかは，この検査では `GROUPS_WITH_TEXT` についてだけ確かめた．`_fuse_stats` を戻したときの値は確かめていない

### 4.2 E79 の judge を保存した measure の出力に当てられない（整理による）

- ファイル：行：`exp/analyze_e79.py:263-264`（前の段の台本の sha256 が今の台本と同じでないと止まる），`docs/再現の手順.md` 5 節の E79 の行
- 何がおかしいか：整理で docstring の 2 行を書き換えたため，台本の sha256 が 09f3a345… から 1eb98c9f… に変わった．保存した `e79/measure/`・`e79_p4/measure/` に judge を当てると止まる．judge の計算そのものは，provenance の sha256 を書き換えれば保存した値と同じになる（40 本の sha256 が一致）．流し直すには，写しで delta → select → measure → judge の 4 段を全部流す必要がある．measure は GPU を使わないが，保存した回で 1,203 秒と 1,496 秒かかった．依頼文では measure を GPU の段に数えていたが，台本は起動の最初に `CUDA_VISIBLE_DEVICES` を空にする（`analyze_e79.py` の冒頭）
- どう直すか：`docs/再現の手順.md` 5 節の E79 の行に，次の 3 点を書く．4 段を同じ台本の版で続けて流すこと．保存した measure の出力には整理の後の台本で judge を当てられないこと．measure は CPU の段で 20〜25 分かかること

### 4.3 E76 の学び直しから解析までを整理の後の木で通せない（整理による部分と元からの部分）

- ファイル：行：`exp/analyze_e76.py:95,262-267`（json の `code_hash` が E76 の本番の値 9891f066… でないと止まる），`exp/relearn_e76.py:141-146`（`code_hash` は読み込むコード 10 本の中身の sha256）
- 何がおかしいか：写しの `code_hash` は bdadfdac… で，E76 の本番の値と違う．違う理由は 2 つある．整理でコメントを書き換えた `interventions.py`・`logreg.py`・`map.py`・`pair_table.py`・`relearn_e76.py` と，5ee7278 で変わった `measure/data.py` である．このため，手順の表のとおりに `relearn_e76.py → relearn_e76_finish.py → analyze_e76.py` を流すと，`analyze_e76.py` が「受け入れないコードのハッシュ」で止まるはずである．この止まり方は流して確かめていない．`code_hash` を計算して比べただけである
- どう直すか：`docs/再現の手順.md` 5 節の E76 の行に，`analyze_e76.py` が受け付けるのは保存した `results/e76/` だけであり，整理の後に学び直した出力は受け付けないことを書く

### 4.4 5 節の入力の一覧が足りない（文書）

- ファイル：行：`docs/再現の手順.md` 5 節の表（E77・E78 と E80 の行），`exp/geom_e0.py:72-79`，`exp/teach_student39.py:140,233-236`
- 何がおかしいか：E77・E78 の入力は「特徴，`results/multisource_scores_*`」とだけ書かれている．実際には `compare_dai.py`・`compare_dai2.py`・`e80_teacher_j.py` が `results/explore_relation/e40_targets.csv` を読み，無いと止まる．`geom_e0.settings` は言葉の種類で `e39_text_targets.csv`・`e37_targets_text.csv` を，第 2 エンコーダで `e62_targets.csv` を読む．`teach_student39.py` は `e46_targets.csv` も読む（読んだだけ）．E80 の段 c は `results/e80/kept.csv` を読む．また E75 の既定の入力 `results/explore_relation/kstar_cluster.csv`（`reanalysis_e75.py:107`）は，表の「`explore_relation/snapshots/`」に含まれない
- どう直すか：5 節の表の入力の列に，上の表の名前を足す

### 4.5 起動の台本が `logs/` を作らない（整理による．作業枝の 7748bc0 で直っている）

- ファイル：行：写し（a06f779）の `run/e77.sh`・`e77b.sh`・`e77h.sh`・`e78.sh`・`e78b.sh`・`e78h.sh` の `cd "$(dirname "$0")/.."` の次の行
- 何がおかしいか：GitHub から取り出した形では `logs/` が無いので，Python を起動せずに終わる（1 節の最後の行）
- どう直すか：作業枝の 7748bc0 で 6 本すべてに `mkdir -p logs` が入ったことを `git diff a06f779 7748bc0` で確かめた（読んだだけ）．追加の対応は要らない．a06f779 から 7748bc0 の間で `exp/` と `measure/` は変わっていないので，この報告のほかの結果は 7748bc0 にも当てはまる

### 4.6 `results/MANIFEST.tsv` が git に入っていない（文書．main に入れる前に確かめる）

- ファイル：行：`docs/再現の手順.md` 1 節の 5 と 5 節の最初の段落，6 節
- 何がおかしいか：文書は「`results/MANIFEST.tsv`（git の中）」「2 MB 以下の表は `results/` に git で入れてあり」と書く．a06f779 でも 7748bc0 でも，`git ls-files results` は 0 本である．写しの `results/` は空で，前半の台本はどれも MER-DA の `results/` から写さないと流せなかった
- どう直すか：main に入れるときに `tools/collect_results.py` で一覧と小さい表を git に入れる．入れないなら文書の 3 か所を今の状態に書き直す

### 4.7 記録の細かい点（記録する）

- `exp/analyze_e79.py:160-161,290`：`git grep` の出力の日本語のパスが 8 進数に置き換わる．`git("-c", "core.quotepath=off", "grep", …)` にすると読める形で記録される
- `exp/analyze_e79.py:2712`：judge で pandas の `DtypeWarning`（列 `pseudo`・`copied_from` の型が混ざる）が 16 回出る．値は保存したものと同じだった
- `exp/explore_source_value.py:154`：`ConstantInputWarning` が出る．値は保存したものと同じだった
- E32 の 2 本は provenance を書かず，`results/explore_relation/` の出力を止まらずに上書きする（元から）

## 5. 所要と，この検査で使ったもの

- 全体で約 45 分（16:16〜17:00）．ほかの検査役の GPU の計算と並べて，CPU だけで流した
- 1 回 10 分の決まりを超えたのは E75 の旧群の回（724 秒）だけである．原因は，ほかの CPU の計算と並べたことである
- 写しの `results/` に写した 42 GB は検査の後も残してある（`standalone_front/DAI/results/`）．写しの中で書き換えたのは次の 3 か所である．`e79/measure/provenance.json`・`e79_p4/measure/provenance.json` の `script_sha256`（元の値は `standalone_front/e79*_measure_prov_orig.json`），`e80c_*/check.csv`（`--check` が書き直した．中身は保存したものと同じ），`explore_relation/e32_*.csv`（E32 が上書きした）
- 比べる道具・包み・ログ：`checks/整理_2026-11-15/code_run_repro_early/`（`cmp.py`，`run_old_groups.py`，`cmp_out/*.txt`，各ログ）．E32 のログは結果の値を画面に出すので置いていない
