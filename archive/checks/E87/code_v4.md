# 関門 1：実装と出力（E87 第 4 版，2026-11-03，コミット 36193e5）

## 判定
直してから進む．前回の指摘 1〜4 と記録 5〜9 はすべて直った．ただし新しく直した `contrast_allowed` の行（`exp/analyze_value_split_e87.py:1206–1207`）が pandas の `itertuples` の行に `.get` を呼ぶので，P87-1 と P87-2 の最終の区分が両方とも当たりになる行が 1 つでもあると，`judge` が AttributeError で止まる．この行と，差の量の向きの誤り（指摘 2），K'・S_x の jackknife の保存（指摘 3）を，解析の本番（`results/e87/`）の前に直す．物差しの学び直し（主は 34 組できた．K = 100 は e87k100 窓で実行中）は止める必要が無い．

検査の範囲：設計は `docs/exploratory-analyses.md` 10632〜10740 行（E87 の 1〜6 節．第 4 版），台本は `exp/analyze_value_split_e87.py`（36193e5．`git diff ce16aaa 36193e5` を読み，判定の段は全文を読み直した）と `exp/relearn_tgtonly_e87.py`（変更なし），台帳は `docs/seen-data.md` の E87 の行，前回の報告は `checks/E87/code_v3.md` である．検査の台本とログは `checks/E87/code_run_v4/` に置いた．流したものはすべて CPU で流した（`CUDA_VISIBLE_DEVICES=""`．この環境に `nvidia-smi` は無い）．GPU では K = 100 の学び直し（PID 2488777，21:59 起動，`--kgrid 100 --out results/e87_tgtonly_k100`）が動いている．触れていない．`results/` の下は読むだけにした．試走の出力は消し，検査の台本とログ（値を含まない）だけを残した．

## 前回（第 3 版）の指摘の判定

| 前回の指摘 | 判定 | 根拠 |
|---|---|---|
| 1 [I1] `contrast_allowed` を最終の区分で決める | **意図は直った．ただし落ちる**（新しい指摘 1） | `judge` の末尾（A:1205–1207）が，最終の区分を作った後に `P87_1_final`・`P87_2_final` ＝ holds と (A)(B)(C) から作る．readable の行だけが真になりうる．人工の表で，台本の写しの `r.get(...)` を `getattr(r, ..., None)` に置き換えた版では，3 版とも当たりの表で真，P87-1 の最終の区分が「側が変わる」の表で偽になった（`chk_synth_v4.log` の S1・S2）．元の台本は同じ表で AttributeError で止まった．右端の置き方で変わるか（`*_rule_sensitive`）は条件に入っていない（記録 4） |
| 2 [I1] 感度 (a) の座標の組を all・all_extreme の版でも作る | **直った** | A:1146–1148 で，主の比較では座標の組 4 つをすべての版に作る．試走の判定の表では posloss・bfix・dir_auc の組が 5 つの版にそれぞれ 8 行ある．人工の表では posloss の組の最終の区分が当たりになった（「打ち切りの版で変わる」にならない） |
| 3 [I1] 広げない版の判定の行 | **直った** | `judge_one` に `inflate` の引数（A:995，1004）．主の比較・主の座標の組では，readable・all・all_extreme に 6 つの設定（帯 frac・fixed・half_dir，cens_max 0.10，hi_only，広げない）がある（A:1150–1153．試走で 6 つを確かめた）．最終の区分のキーに `jk_inflation` と `mask_kind` が入った（A:1178）．人工の表で広げない行にも独自の最終の区分が付いた |
| 4 [I1・設計] all の版の扱い | **直った** | 設計 2.4 節と 3 節が，all・all_extreme では印の基準を読めるコーパスにだけ当てると書いた．`family_stats`（A:904–956）は `readable_ok` を読めるコーパスの割合だけで決め，all_extreme では読めないコーパスを `true_direction` の向きに ±1e6 本で置き，向きが決まらなければ `unknown_dir` で区間を使わない．検算は下の A の 3 |
| 5 `uar_bfix` の値の無い行の数 | **直った** | `method_scores`（A:520–522）が数え，`values.csv` の `n_bfix_unchecked` に書く．試走では 0．`chk_bfixref_v3.py` を流し直すと，試走の 3,680 行はすべて rows の表に値を持っていた．`checks.csv` には入らないので，本番の後にこの列の和を確かめる |
| 6 帯 `half_dir` のコーパスの集合 | **直った** | 設計 (k) に「向きが読めるコーパスだけで取る」と書き，台本（A:1056）は readable の版で向きが読めるコーパスに限る |
| 7 設計の文の食い違い 2 か所 | **直った** | 2.1 節の I14 の文は全計算の値（4 節と同じ）になった．(a) は「P87-2 は n_posloss の帯で，(C) は n_dir − n_posloss で」になった |
| 8 K 本の不一致で止める | **直った** | `load_one`（A:228–233）が `<T>_e76_check.csv` の `compared` の行の `kidx_equal` に偽があれば止まる．人工の npz で，偽が 1 反復あると止まり，`set_differs` だけのファイル（`kidx_equal` の列が無い）と `no_common_K` のファイルでは止まらないことを確かめた（`chk_load_v4.log`）．ファイルが無いときは照合を飛ばす（記録 5）．今の主の 34 組はすべてファイルがあり，`compared` 240 行の `kidx_equal` はすべて真，`set_differs` 100 行だった |
| 9 物差しの code_dirty，話者の並びの crc32，警告 | **直った** | `code_dirty` が真なら止まる（A:219–220．人工の npz で確かめた．主の 34 本はすべて偽，コミットは 8d73121）．同じコーパスの別の世界と crc32 が違えば止まる（A:414–418．同じ論理を切り出して確かめた）．試走のログに警告は 0 件だった |

## A．設計と台本の対応表（第 4 版で変わった所）

台本 2 は `exp/analyze_value_split_e87.py`（A:行）を指す．第 3 版から変わっていない行は前回の対応表のとおり合っている．

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 2.2 物差しの `code_dirty` で止める．AVT と AV の話者の並びの crc32 が違えば止める | A:219–220，A:414–418 | 合う |
| 2.3 saerens・massbal の向きの座標（dir・dir_auc・dir_bfix・ΔS_dir）を定義から 0 に置く | A:79–80，A:560，A:580，A:655，A:694 | 合う．試走で `values.csv` の該当の列 7 つが 48 行ですべて 0，`units.csv` の「定義から 0」は 504 行（saerens 252・massbal 252．量は dir・dir_auc・dir_bfix 各 144，dS_dir 72） |
| 2.3 `uar_bfix` の値の無い行の数を記録する | A:520–522，A:570–571 | 合う（`values.csv` の `n_bfix_unchecked`） |
| 2.3 (p) K = 100 を外した 6 点の曲線での読み（点だけ） | A:591–597，A:709，A:889，A:944 | 合う（`values.csv` の `n_*_6pt_*`，`units.csv` の `point_6pt`，`family.csv` の `median_6pt`）．`--without-k100` では主の読みと同じ値になることを試走で確かめた．K = 100 ありの経路は読んだだけ（添字は K_GRID の 6 点，右端は主と同じ logk） |
| 2.3 563188d の反復ごとの読み（K = 100 を除く） | A:642–653 | 合う（`sub6`，`logk_rep`） |
| 2.4 抽出の中の印を種類別（下・上・あいまい × none・手法）にビットで束ねる | A:75–78，A:619，A:668–677 | 合う．人工の 200 通りで往復が一致した．試走の 3,780 単位で，保存した mask から 3 つの割合の列が差 0 で再現され，ビットは 64 未満だった |
| 2.4 上の打ち切りかあいまいの抽出の割合が 2.5% を超える単位の区間は使わない．(o) は上だけ | A:723–724，A:760 | 合う．試走で `interval_ok` ＝（main の割合 ≤ 0.025），`interval_ok_hi_only` ＝（上だけの割合 ≤ 0.025），上だけ ≤ main ≤ 全種類が全行で成り立ち，区間が使えないのに正・負・判定できないになった行は 0 |
| 2.4 族の区間は読めるコーパスのそれぞれに基準を当てる（和集合の割合は記録） | A:939–955 | 合う．人工の表で，5 コーパスが別々の抽出で各 2% のとき和集合 10% を記録して区間は使え，1 つが 3% なら使えず，基準 10% なら使えた |
| 2.4 all_extreme：読めないコーパスを向きの極端（±10⁶）に置く．向きは 4 通り，あいまい・両向きは決まらない | A:81，A:830–837，A:919–928 | 合う（n_x の量について）．`true_direction` は 64 通りの印で独立の規則と一致した．差の量（dir_minus_*，直接の差）では向きが誤る（指摘 2） |
| 2.4 all・all_extreme では印の基準を読めるコーパスにだけ当てる | A:939–940，A:953 | 合う．読めないコーパスの抽出の 70% に印があっても all・all_extreme の区間は使えた．読めるコーパスが 0 なら使えない（設計に無い条件．記録 7） |
| 2.5 `boot_units.npz` に K'(none)・K'(手法)（`|Kn`・`|Km`）と主の右端の S_x（`|Sn`・`|Sm`） | A:746–751 | 合う．試走で `|Kn`・`|Km` 各 1,872，`|Sn`・`|Sm` 各 1,008．jackknife の値は保存しない（指摘 3） |
| 3 最終の区分（readable・all・all_extreme の 3 つで同じ側） | A:1177–1204 | 合う．人工の表で 4 通り（同じ側 → holds，all と all_extreme が fails → `undet_side_changed`，all_extreme が向きを決められない → `undet_censoring_direction_unknown`，all が undet → `undet_censored_version_undet`）を通した |
| 3 対比の文の条件（最終の区分が当たりで (A)(B)(C)） | A:1205–1207 | 意図は合う．実装が落ちる（指摘 1） |
| 3 感度 (a)(b) でも 3 つの版で最終の区分 | A:1146–1148 | 合う |
| 3 感度 (l)(n)(o) の行 | A:1150–1153 | 合う（主の比較・主の座標の組．readable・all・all_extreme） |
| 3 (m) ΔS の比の判定．(A) を ΔS_pos で | A:1211–1240 | 合う．人工の表で ΔS_pos が K = 0 で −2，K = 25 で −0.5 なら holds，−1.5 なら undet |
| 3 563188d：区分は区間の上下だけ，読めないのは打ち切りの反復が半数を超えるとき | A:1251–1252 | 合う（`frac_cens_rep <= 0.5` で読める．none を選んだ群は反復ごとの読みが打ち切りにならない） |
| 3 仮定 2 に区間が使えない単位の数（2 つの基準）と種類別の割合の中央値 | A:1299–1302 | 合う |
| 4 解析の台本が `kidx_equal` を読み，違えば止める | A:228–233 | 合う（ファイルが無いときは飛ばす．記録 5） |

計算する組の数え上げ（L4）は前回と同じである．試走の判定の表の 520 行は，比較と族の組 28（none-abl_refb は単モダリティの 3 族，none-abl_ref0 は融合の 1 族）から，主の比較の 4 族 × 右端 2 × 35 行（readable・all・all_extreme は主の座標の組 6 設定と感度の座標の組 3，ほかの 2 版は 1 ＋ 3）＝ 280 と，ほかの比較 24 組 × 右端 2 × 版 5 ＝ 240 の和に一致した．本番は融合 AV の族が加わり 650 行になる見込みである．

## A．小さく流した結果

1. **試走（指定どおり）**：`./.venv/bin/python exp/analyze_value_split_e87.py --smoke --without-k100 --runs fuse-audio-video-text --targets IEMOCAP-impro IEMOCAP-script --out checks/E87/code_run_v4/smoke`．壁時計 1 分 54 秒，最大の常駐メモリ 1.05 GB，警告 0 件（`smoke.log`）．行の数：curves 672，curves_rep 480，values 168，values_seed 3,680，units 5,292，corpus 1,764，family 26,460（版 5 × 広げる／広げない／上だけ の 3 通り），direct 3,024，direct_corpus 1,008，direct_family 15,120，judgments 520，judgments_dS 28，judgments_563188d 28，assumptions 2,745．`checks.csv` は `chosen_missing_expected` 12 件だけ．`values.csv` の数値の列に NaN・Inf は 0．見出しのコミットは 36193e5，`dirty` は偽．1 コーパスなので判定の表の区分は `not_judgeable` だけで，対比の条件 (A)(B) の列は作られなかった（このため指摘 1 は試走では起きない）．構造の検査は `chk_struct_v4.py`．
2. **保存済みの値の再現（`chk_repro_v3.py` を流し直した）**：`curves_rep` の S_uar/100 と物差しの csv の uar の差の最大 1.7e−16，C の差 0．`values_seed` の S_uar/100 と `results/<wave>_<実行>/<T>.csv` の uar の差の最大 1.7e−16（3,680 行）．`values.csv` の S_x と種・反復の平均の差 2.8e−14．d・o と `uar_bfix` の 1e−6 の照合は台本の中で働き，止まらなかった．
3. **退化した場合と新しい部分の検算（`chk_synth_v4.py`．人工のデータ．32 項目のうち 31 項目が通った）**：pack_mask の往復と mask_frac（200 通り）．`true_direction` の 64 通り（0 が 1，＋ が 3，− が 3，? が 57）．`family_stats`：readable は読めるコーパス 4 つの中央値，all は 5 つ，all_extreme で＋のコーパスを +1e6 に置いた中央値と抽出ごとの中央値，−が 2 つなら中央値は読めるコーパスの最小，あいまいと両向きは `unknown_dir` で区間を使わない（all の版は使える），読めないコーパスは広げない，G_eff < 20 の倍率（jk の半幅／百分位の半幅）と広げない版の倍率 1，hi_only ではあいまい 5% のコーパスでも区間が使える．`judge` の最終の区分の 4 通りと `contrast_allowed`（台本の写しを直した版で確かめた）．通らなかった 1 項目は「元の台本の `judge` は P87-1 と P87-2 の最終の区分が当たりの表で落ちない」で，AttributeError（`'Pandas' object has no attribute 'get'`）が出た（指摘 1）．
4. **止める条件（`chk_load_v4.py`．11 項目すべて通った）**：人工の npz で，`code_dirty` が真なら止まる．`kidx_equal` が 1 反復でも偽なら止まる．`set_differs` と `no_common_K` だけのファイルでは止まらない．ファイルが無いと照合を飛ばす．話者の並びの crc32 は，同じコーパスで 1 人違えば止まり，別のコーパスとは比べない．
5. **K = 100 の差し込み（`chk_k100_v3.py` を流し直した．17 項目すべて通った）**：第 4 版の `load_one` でも，並び 1, 5, 10, 25, 50, 100, 全件，C と C 4 つの UAR の差し込み，テストの行の並べ直し，止める 4 つの場合が前回と同じに働いた．
6. **保存した出力だけからの作り直し（`chk_reagg_v4.py`）**：`units.csv`・`boot_units.npz`・`jk_units.npz` だけから，corpus 1,764 行（区間，区分，`true_direction`），family 26,460 行（中央値，区間，区間が使えるか），judgments 520 行（区分と最終の区分のすべての列）が保存した表と一致した．
7. **複製の引き算（`chk_struct_v4.py`）**：1,872 単位で，`|Km` − `|Kn` が n_x の複製と一致した（相対の差の最大 5.7e−6．float32 の丸め）．684 単位で，`|Sm` − `|Sn` が ΔS の複製と一致した（差の最大 7.6e−6）．K'(none) の複製は，同じ（世界，分類器，K，コーパス，右端，量）の比較の間で 288 組すべて同じだった（差 0．抽出を共有している）．
8. **差の量の向き（`chk_diffdir_v4.log`）**：試走の corpus.csv の dir_minus_* の 336 行のうち 18 行，direct_corpus.csv の 1,008 行のうち 38 行が，向き＋か−を持った．この向きは OR でまとめた印から作られている（指摘 2）．

## A．本番の規模の見積もり

- **物差し（主）**：34 組すべてできた（npz 34，`_e76_check.csv` 34，ログに例外・警告なし，DONE あり）．
- **物差し（K = 100）**：21:59 に始まり，ログの最後は AVT の IEMOCAP 台本（1 組 18〜35 秒）．34 組で 30 分前後と見込む．GPU のメモリは測っていない（nvidia-smi が無い）．台本は前回の検査のとおり 6 割に抑え，反復ごとに空きブロックを返す．
- **解析（CPU）**：試走は 1 分 54 秒で第 3 版と同じだった．本番は前回の見込みのとおり 15〜30 分とみる．保存の量は，試走の生の配列が boot_units 2.6 MB・boot_family 3.9 MB で，B が 20 倍，（分類器，コーパス）が約 7 倍，族が 1.25 倍になるので，boot_units は約 0.4 GB，boot_family は約 0.1 GB（圧縮前）と見込む．空きのメモリは 57 GB，ディスクは 587 GB ある．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 単位・コーパスの表 | none 自体と手法がそれぞれ何本ぶんか（K'(none)・K'(手法)）の区間 | R1，D，設計 2.5 | **百分位は (a)**：`units.csv` の `boot_key` に `|Kn`・`|Km` を付けた鍵の配列の 2.5・97.5% 点（試走で 624 単位を計算できた）．コーパスは世界の単位の配列の平均（抽出は共有）．none を選んだ単位は保存されないが，K'(none) は比較の間で同じなので別の比較の鍵から取れる．**G_eff < 20 の単位とコーパスでの jackknife の和集合は (b)** | jackknife の値（`J_n`・`J_m` からの K' と S_x）を `jk_units.npz` に `|Kn`・`|Km`・`|Sn`・`|Sm` で保存する（指摘 3） | 話者の数 × 単位の数の float32．数 MB．時間は変わらない |
| 単位の表 | S_x(none)・S_x(手法) そのものの区間 | R1 | 百分位は (a)：`|Sn`・`|Sm`（主の右端の鍵．試走で 336／336 単位にある）．jackknife の和集合は (b) | 同上 | 同上 |
| 族の表・仮定 2 | 打ち切りの種類ごと（下・上・あいまい × none・手法）の数 | R1，設計 2.4 | (a)：点の印は `units.csv`・`corpus.csv` の `flag_*`，`family.csv` の `n_corp_*`．抽出の中の割合は `units.csv` の `frac_cens_*`（群の平均），`corpus.csv` の `frac_draw_<種類>`（世界の OR），`boot_units.npz` の `|mask` のビット（`mask_frac(m, (種類,))` で再現できることを確かめた） | — | — |
| 判定の表 | 抽出の中の基準を 10% にしたら／上の打ち切りだけを数えたら | 設計 (n)(o) | (a)：主の比較・主の座標の組は `judgments.csv` の `cens_max` ＝ 0.10 と `mask_kind` ＝ hi_only の行（readable・all・all_extreme，最終の区分つき）．ほかの比較と座標の組は保存した出力に `family_stats(cens_max=0.10)`・`family_stats(mask_kind="hi_only")` と `judge_one` を当て直す（試走で 516 組を計算でき，作り直しが保存した表と一致することを確かめた） | — | — |
| 判定の表 | 族を jackknife の幅まで広げない版の当否 | 設計 (l) | (a)：`judgments.csv` の `jk_inflation` ＝ False の行（最終の区分つき），`family.csv` と `boot_family.npz` の `|noinfl|` | — | — |
| 曲線の図・判定の表 | K = 100 を外した 6 点の曲線で読んだら | 設計 (p)，R4 | 点は (a)：`values.csv` の `n_*_6pt_*`，`units.csv` の `point_6pt`，`corpus.csv` の `point_6pt`，`family.csv` の `median_6pt`．区間と当否は (b)（設計は点だけと決めている．抽出ごとの 6 点の読みを保存していない）．`--without-k100` で `results/` の外に CPU で流し直せば出る | 求められたら抽出ごとの 6 点の読みを流し直す | 解析 1 回（15〜30 分） |
| 判定の表 | 読めないコーパスを向きの極端に置いた版（all_extreme） | 設計 3 | (a)：`judgments.csv` の variant ＝ all_extreme の行，`family.csv`，`boot_family.npz`．**ただし dir_minus_* と直接の差（教師に帰す規則の all_extreme の列を含む）の all_extreme は向きが誤る**（指摘 2） | 差の量の向きを直す | 時間 1% 未満 |
| 判定の表 | 右端の置き方で当否が変わるか | 設計 (d) | (a)：`*_rule_sensitive` と右端 min の行（最終の区分も右端ごとにある） | — | — |
| 判定の表 | 対比の文を書けるか | 設計 3 | (a)：`contrast_allowed`（readable の行）と (A)(B)(C) の列．**今の台本はこの列を作る所で落ちうる**（指摘 1） | 指摘 1 を直す | — |
| 判定の表 | 反復ごとに読んでから平均したら | 563188d，R1 | (a)：`values.csv` の `n_*_rep`・`frac_cens_*_rep`，`units.csv` の `point_rep`，`family.csv` の `median_rep`，`judgments_563188d.csv` | — | — |
| 判定の表 | IEMOCAP 台本を除く／MOSEI・AFEW を除く | 設計 (e)(g)，L6 | (a)：版 `readable_impro_only`・`readable_excl_mosei_afew` の行．この 2 つの版には最終の区分が無い（記録 6） | — | — |
| 判定の表 | 候補の数をそろえた対照 | gate-compare，S14 | (a)：`direct*.csv` の abl_refb − abl_ownK・abl_ref0 − abl_ownK と判定の表の `refb_vs_ownK_*`・`ref0_vs_ownK_*`（readable・all の版） | — | — |
| 曲線の図 | 物差しの C を 4 つのどれかに固定したら | R4，C5 | UAR の点は (a)：`curves_rep.csv` の `uar_C*`．向き・位置・区間は (b)：npz に C 4 つの点数があるので CPU で流し直せば出る（学び直しは要らない） | C を固定した読みを感度の行に足す | 解析の時間 1.3〜2 倍 |
| 全体 | 性別ごと | R1，ML Minor-1 | (b)：出力に性別の列が無い．発話ごとの点数（物差しの npz，E77 の logp）があるので CPU で流し直せば出る | 話者群から性別を引き，部分集合で流す引数 | CPU で 1 回 |
| 曲線の図 | K = 200 など点を足せ | R4 | (c)：設計 4 節に，IEMOCAP 即興と AFEW で学習プールが足りないと書かれている．ほかの K は GPU の学び直しが要る | — | — |
| 全体 | 別の学習器・第 2 のエンコーダ・仮のターゲットの版 | R4，R7，L2 | (c)：設計 4 節に範囲と理由が書かれている | — | — |

計算の後に求められると GPU の学び直しになる項目は，設計 4 節が範囲の外とした (c) だけである．(b) のうち，K'・S_x の jackknife の保存（指摘 3）は今回の 1 回に入れることを勧める．流し直すたびに評価のターゲットで新しく判定することになる（L10）．

## 指摘

### [I1] 1．`contrast_allowed` を作る行が，P87-1 と P87-2 の最終の区分が両方とも当たりになる行があると AttributeError で止まる
- 重さ：直してから進む（本番の前に必ず直す）
- 該当箇所：`exp/analyze_value_split_e87.py:1206–1207`
  ```
  J["contrast_allowed"] = [bool(r.variant == "readable" and r.P87_1_final == "holds" and r.P87_2_final == "holds" and r.get("contrast_A_pos_present_then_drops") is True
                                and r.get("contrast_B_counts") is True and bool(r.get("contrast_C_dir_gt_pos"))) for r in J.itertuples(index=False)] if len(J) else []
  ```
- 問題：`J.itertuples(index=False)` が返す行は名前つきのタプルで，`.get` を持たない．`and` が左から評価されるので，readable の行で `P87_1_final` と `P87_2_final` が両方 holds になったときだけ `r.get` が呼ばれ，例外で止まる．流して確かめた：人工の表（5 コーパス，3 つの版で同じ側に当たり）で，元の台本の `judge` は `AttributeError: 'Pandas' object has no attribute 'get'` で止まった．台本の写しで `r.get(...)` を `getattr(r, ..., None)` に置き換えた版は通り，`contrast_allowed` は真になった（`chk_synth_v4.log`）．試走は 1 コーパスで判定がすべて `not_judgeable` だったので，この行を通らなかった．本番でこの例外が起きると，`judgments.csv` と npz は書かれず，`results/e87/` にはそれまでの CSV だけが残る．しかも止まったこと自体が「どこかの族と比較で P87-1 と P87-2 の最終の区分が当たりになった」ことを設計者に知らせる
- 計算の後で直せるか：数値は直せる（流し直す）．ただし止まった事実から結果の一部が分かった後に，評価のターゲットで 2 回目の判定をすることになる（L10）
- 求める対応：3 か所の `r.get("…")` を `getattr(r, "…", None)` に替える（または `J.to_dict("records")` で回す）．直した後，`chk_synth_v4.py` を流して 32 項目すべてが通ることを確かめる

### [I1] 2．差の量（dir_minus_pos・dir_minus_posloss と直接の差）の「本当の本数の向き」を，2 つの量の印の OR から作るので，all_extreme の版と `true_direction` の列が誤る
- 重さ：直してから進む
- 該当箇所：`exp/analyze_value_split_e87.py:632`（`res["flag"][(rule, dq)] = {k: (res["flag"][(rule, "dir")][k] or res["flag"][(rule, pq)][k]) …}`），`:792`（直接の差の印 ＝ 2 つの比較の印の OR），`:892`（`"dir": … true_direction(fl)`），`:919–928`（all_extreme），`:1167`（教師に帰す規則を all_extreme の版でも `family_stats(…, extreme=True)` で計算する）
- 問題：`true_direction` は「手法が上に外れた → 本当の n は大きい」という n_x ＝ K'(手法) − K'(none) の向きの規則である．n_dir − n_pos では，引かれる側（pos）で手法が上に外れると本当の差は小さくなる．OR でまとめた印からは「＋」が出る（人工の印で確かめた）．直接の差 n_a − n_b（例：oursx − abl_ownK）では none が同じモデルなので none の印は差で消えるのに，`lo_none` から「＋」が出る．b の側で手法が上に外れた場合も「＋」になる．試走では corpus.csv の dir_minus_* の 18 行，direct_corpus.csv の 38 行が＋か−を持った．これらの行の all_extreme の族の値（`family.csv`・`direct_family.csv`・`boot_family.npz`）と，判定の表の all_extreme の行の (C) の列と `teacher_*`・`refb_vs_ownK_*`・`ref0_vs_ownK_*` の列は，誤った向きの極端で計算される．最終の区分と `contrast_allowed` はこれらを使わない（(C) は readable の行の値だけを使う）ので，主の判定は変わらない（読んだだけ）
- 計算の後で直せるか：直せる（保存した出力から作り直せる）．ただし誤った数値が表に出る
- 求める対応：差の量では `true_direction` を「?」に置く（all_extreme の版は向きが決まらず判定できない）か，2 つの量の印を別々に持って向きを合成する（引かれる側の＋と−を入れ替え，直接の差では none の印を外す）．設計 2.4 節の all_extreme の文に，差の量での扱いを 1 文足す

### [R1] 3．K'(none)・K'(手法)・S_x の jackknife の値を保存しないので，G_eff < 20 のコーパスでは設計の区間の規則（百分位と jackknife の t 区間の和集合）を後から当てられない
- 重さ：直してから進む
- 該当箇所：`exp/analyze_value_split_e87.py:746–753`（`boot_store` に `|Kn`・`|Km`・`|Sn`・`|Sm` を入れるが，`jk_store` には n_x の jackknife（`th`）しか入れない），`:566–569`・`:623`（`J_n`・`J_m` と K' の jackknife の読みは計算しているが捨てる）
- 問題：設計 2.5 節は「K'(none)・K'(手法) の複製（`|Kn`・`|Km`），主の右端での S_x(none)・S_x(手法) の複製（`|Sn`・`|Sm`）．後から K' や S_x そのものの区間を出せる」とする．百分位の区間は保存した配列から出せる（試走で確かめた）．しかし設計 2.4 節の区間は G_eff < 20 の単位とコーパスで jackknife の t 区間との和集合で，IEMOCAP（試走の G_eff は 10 前後）ではこちらが主の区間になる．保存した出力からは百分位だけの区間しか作れず，主の規則より狭い
- 計算の後で直せるか：解析を CPU で流し直せば出る．学び直しは要らないが，評価のターゲットで 2 回目の計算になる
- 求める対応：`J_n`・`J_m` からの jackknife の K'(none)・K'(手法)（`read_K(cv["J"][q], logk, J_n[q], …)[0]` と手法の側）と S_x（`J_n[q]`・`J_m[q]`）を，単位の鍵に `|Kn`・`|Km`・`|Sn`・`|Sm` を付けて `jk_store` に入れる．セッションの jackknife（S）も同じに入れる

### [I1] 4．`contrast_allowed` は右端の置き方で当否が変わる行でも真になる
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:1206`
- 問題：設計 3 節は「右端の置き方で当否が変わる予言は主張にしない」とし，対比の文の条件には `*_rule_sensitive` を書いていない．台本は設計どおりである．`summary.md` の判定の表は `contrast_allowed` を出すが `*_rule_sensitive` を出さないので，表だけを読むと右端で変わる行でも対比の文を書けると読める（読んだだけ）
- 計算の後で直せるか：直せる（`judgments.csv` に両方の列がある）
- 求める対応：`summary.md` の判定の表に `P87_1_rule_sensitive`・`P87_2_rule_sensitive` を足すか，設計 3 節の対比の文の条件に「どちらも右端で変わらない」を足す

### [I1] 5．`_e76_check.csv` が無いときは K 本の照合を飛ばす
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:228–229`（`if fe.exists():`）
- 問題：設計 4 節は「解析の台本が `_e76_check.csv` の `kidx_equal` を読み，違えば止める」とする．ファイルが無い組は止まらず，provenance の `e76_check` にも残らない．今の主の 34 組はすべてファイルを持ち，`kidx_equal` はすべて真だった
- 計算の後で直せるか：直せる（計算の後の点検で，provenance の `e76_check` の項目の数が 34 × 2 実行あるかを確かめる．この役が行う）
- 求める対応：主の実行ではファイルが無ければ止めるか，provenance に「無い」と記録する

### [I1] 6．感度の版（readable_excl_mosei_afew，readable_impro_only）と帯 half_dir の all_extreme には，最終の区分の比べ相手が無いか，構造的に判定できない
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:1189–1190`（readable 以外の版は final が空），`:1058`（all_extreme で向きの中央値が ±1e6 なら帯は NaN）
- 問題：(e)(g) の版の当否は版の区分であり，主の当否は最終の区分なので，設計 3 節の「主の当否と違えば両方を書く」で比べると，打ち切りの扱いの違いが感度の違いに混ざる．帯 half_dir では，all_extreme で向きの読めないコーパスが極端に置かれて中央値が ±1e6 になると帯が決まらず，最終の区分は「打ち切りの版が判定できない」になる（読んだだけ）．設計 3 節は最終の区分を座標の組（(a)(b)）にだけ求めているので，台本は設計どおりである
- 計算の後で直せるか：直せる
- 求める対応：結果を書くとき，(e)(g) は版の区分どうし（主の readable の区分と）で比べると明記する

### [I1] 7．all・all_extreme の族の区間は，読めるコーパスが 1 つも無いと使えない
- 重さ：記録する
- 該当箇所：`exp/analyze_value_split_e87.py:955`（`"ok": bool(readable_ok and out["n_readable"] >= 1 and not out["unknown_dir"])`）
- 問題：設計 2.4 節にこの条件は無い．readable の版が判定できるときは読めるコーパスが下限以上あるので，最終の区分には効かない．all の版の行だけを読むときに効く（人工の表で確かめた）
- 計算の後で直せるか：直せる
- 求める対応：設計 2.4 節に 1 文足す

## 目録に無い種類の不備

- **I23（案）結果の値によってしか通らない分岐を，人工の表で全部通す．**判定の表の後段（`contrast_allowed` のように，予言が当たったときだけ評価される条件）は，1 コーパスの試走では通らない．そこに実装の誤りがあると，本番で特定の結果が出たときだけ止まり，止まったこと自体が結果を知らせる（指摘 1）．台本の確認では，判定の分岐ごとに「当たり・外れ・判定できないのそれぞれに入る人工の表」を作り，すべての分岐を一度は通すことを求める．
- **I24（案）差の量に，元の量の印をそのまま写さない．**「印から本当の値の向きを決める」規則は，量 x ＝ K'(手法) − K'(none) について書かれる．x の差（x_a − x_b，同じ none を共有する比較の差）に印を OR で写すと，引かれる側の向きが逆になり，共有する none の印は差で消えるのに残る（指摘 2）．派生した量ごとに，印から作る規則がその量で成り立つかを確かめる．

---

## 再検査の追記（台本を直した後に使う）

（直した差分を検査したときにここへ書く．`chk_synth_v4.py`（32 項目），`chk_struct_v4.py`，`chk_reagg_v4.py`，`chk_load_v4.py` をそのまま流し直せる）

## 計算の後の点検

（計算の後に書く．次を行う：物差しの 34 組 × 2 実行の npz の欠けと `_meta`（code_dirty，コミット），provenance の `e76_check` の項目数と `kidx_equal`，`speakers_crc32` の世界の間の一致，`values.csv` の `n_bfix_unchecked` の和，`chk_bfixref_v3.py` による照合の働いた行の数，判定の表の行数（650 の見込み），`|Kn`・`|Km`・`|Sn`・`|Sm` の数，結果ファイルのコミットと 36193e5 以後の差分，ログの例外と警告）
