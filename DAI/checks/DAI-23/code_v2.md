# 関門 1：実装と出力の再検査（DAI-23，2026-11-15，コミット 1fb89ff．差分 1376f64..1fb89ff）

## 判定
判定：通す．前回の指摘 1〜6 はすべて直り，新しい台本で退化の試し 8 単位と集計が包みなしで最後まで流れ，主の量と C′ を固定した版はすべて 0.0 になった．新しく見つけた指摘 1（C′ を固定した版の M3 が R1 の κ を使う）は集計の段だけの直しで，単位の出力から CPU で直せるので，単位の段は今の台本で流してよい．

経緯：検査の時点の HEAD は 9529e23，途中から d21b840 に進んだ（別のセッションの DAI-22 の文書のコミット）．`git diff --stat 1fb89ff d21b840 -- exp features measure docs/設計/実験23_設計.md` は空で，検査した台本と設計は 1fb89ff と同じである．

## 前回の指摘の確かめ

| 前回の指摘 | 直した箇所 | 確かめ | 結果 |
|---|---|---|---|
| 1．途中保存の 0 次元の配列でまとめる段・読む段が落ちる | features/build_audio_lowpass.py:106–110 | `--op lp --fc 500 --corpus MELD --limit 64 --chunk 32` → `--merge` → `common.load_cache` → `dai23.corpus_audio("MELD", "audio_wavlm-lp500_trial")`（`DD.load_corpus` を通る）を流した | 直った．6 列（op，lowpass_fc，lowpass_order，filter，model_revision，code_commit）はすべて長さ 64．`corpus_audio` は 64 行 × 1024 次元を返した．X の NaN 0．試しの 3 ファイル（`data/cache/audio_wavlm-lp500_trial_MELD.npz` と途中保存 2 つ）は消した |
| 2．provenance を JSON に書けず落ちる | exp/dai23.py:566 | 包みなしで units を 4 回，summary を 1 回流した | 直った．`provenance_log.jsonl` 6 行，`provenance.json` が書けた．`kappa_design` の鍵は `IEMOCAP/all` など |
| 3．C′ の端の数え方が実験 14 と違う | exp/dai23.py:422–435 | 退化の試しの L・B の行を，dai14.py:878–884 と同じ数えを実験 14 の A の内側の曲線（同じ 8 単位）に当てた値と照らした（`code_run_v2/verify_v2.py`） | 直った．照らせた 18 行（L・B × 3 コーパス × 目標 single_V・single_T・prop_one）で share_low・share_high・share_low_nonplateau の差の最大 0.0，flag の不一致 0．目標 label と single_A は実験 14 の A に曲線が無いので照らせない（12 行） |
| 4．L・B の C′ を固定した感度の保存が無い | exp/dai23.py:201–206，340–341，383–392 | 単位の npz の格子の予測と，main.csv の `*_fixC` を確かめた | 直った．1 単位に 96 鍵（2 操作 × 3 目標 × 2 版 × 8 点）．C′ の点は `D14.fixed_c_index(c14)` で，実験 14 の全単位の csv から作ると 3 コーパスとも (0.001, 0.001) になり，実験 14 の `convergence.json` の `candidates_clusters_fixC` と一致する．ただし M3 の κ に新しい指摘がある（下の指摘 1） |
| 5(1)．単位ごとの行の数の照合が無い | exp/dai23.py:334–339 | summary を流した | 直った．8 単位（r = 0 で 63 か 99 行，r ≠ 0 で 27 行）で止まらなかった |
| 5(2)．A・V・T の特徴の sha256 とモデル名が無い | exp/dai23.py:560，566 | provenance を読んだ | 直った．`cache_sha256` に A・V・T・L・B × 3 コーパスの 15 鍵があり，空は 0．`wavlm_model` は `microsoft/wavlm-large`（build_audio_lowpass.py:41 の MODEL と同じ） |
| 5(3)．fc.json にコミットと asr_check.csv の sha256 が無い | features/check_lowpass_asr.py:208–211 | `--n 2 --out-dir checks/DAI-23/code_run_v2/asr` を流した | 直った．`code_commit` と `asr_check_sha256` があり，sha256 は `asr_check.csv` を計算し直した値と一致した |
| 5(4)．特徴の npz にコミットが無い | features/build_audio_lowpass.py:49–50，110 | 試しの npz を読んだ | 直った（`code_commit` の列）．作業木の汚れは残さない（指摘 4） |
| 5(5)．退化の試しの置き場 | 設計 8 節 | 設計を読んだ | 設計は `checks/DAI-23/code_run/degenerate/`（1376f64 の台本と包み）を名指す．新しい台本の退化の試しは `checks/DAI-23/code_run_v2/degenerate/` にある（指摘 3） |
| 6．単位ごとに GPU の空きブロックを返さない | exp/dai23.py:242–243 | 読んだ | 直った |

## A．設計と台本の対応表（差分の行だけ）

| 設計の項目 | 台本の箇所（1fb89ff） | 合っているか |
|---|---|---|
| 6 節 C′ の端：単位 × 組ごとに選んだ点を数え，目標ごとに max(上端, 平らでない下端) > 0.25 | dai23.py:422–435 | 合う（上の照合で差 0） |
| 7 節 R1fixC：L・B の J_A・J_V・J_T の真と #shufcl の格子の全点の予測を保存する | dai23.py:201，203–206 | 合う．`fit_r1` の `grid_keys` は最初の分け方だけで解き（dai14.py:393–423），元の分け方の予測になる |
| 7 節 実験 14 の R1fixC と同じ格子の点を A と X に当てた M1・M2・M3（記述．分け方の揺れを足さない） | dai23.py:340–341，383–392 | 点と項は合う．A の格子の予測は実験 14 の npz から，X は dai23 の npz から取り，`load_merged` の鍵の重なりの検査（dai23.py:260–262）を通る．`split_vals` を渡さず，`v_split` は NaN．**M3 の κ は R1 の κ**（指摘 1） |
| 7 節 部分集合 | dai23.py:384 | all と nonshared（MELD）．nomarker は出さない．実験 14 の R1fixC も all と nonshared だけ（dai14.py:757）なので同じ |
| 8 節 出力の記録（A・V・T の特徴の sha256，fc.json の sha256，特徴の npz の濾波の方式と WavLM の revision） | dai23.py:559–569，build_audio_lowpass.py:106–110，check_lowpass_asr.py:208–211 | 合う |
| 12 節 変更の一覧 | — | 差分の 7 項目と一致する |

設計の表の組の数：主の表の組（2 操作 × 3 コーパス × M1・M2・M3・M3d，MELD は all と nonshared）は前回と同じ 32 行で，退化の試しの `main.csv` でも 32 行出た．`main.csv` は前回の 173 行に `*_fixC` の 24 行（2 操作 × 3 量 × 4 組）が足されて 197 行になった．`assumptions.csv` は前回の 78 行から 72 行になった（C_low_nonplateau の 6 行が C_edge の列に入った）．

## A．小さく流した結果

台本は 1fb89ff と同じ（HEAD 9529e23 と d21b840．exp・measure・features は汚れていない）．GPU はほかの処理が約 1.7 GB を使っていた．

| 試し | コマンド | 所要 | 結果 |
|---|---|---|---|
| selftest | `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W default exp/dai23.py --stage selftest` | 約 2 秒 | 「6 項目とも通った」 |
| 退化の試しの単位（IEMOCAP） | `exp/dai23.py --stage units --out-dir checks/DAI-23/code_run_v2/degenerate --targets IEMOCAP --folds 0 1 --seeds 0 1 --l-kind audio_wavlm --b-kind audio_wavlm --verify` | 3 分 45 秒 | f0 r0 65 秒 63 行，f0 r1 48 秒 27 行，f1 r0 66 秒 63 行，f1 r1 43 秒 27 行．照合の一致 4 単位とも J_T・J_V で 1.0．勾配の最大 1.0×10⁻⁶ |
| 退化の試しの単位（MOSEI・MELD） | 同じで `--targets MOSEI` か `MELD`，`--folds 0 --seeds 0` と `--folds 1 --seeds 1` | MOSEI 132 秒・77 秒，MELD 129 秒・71 秒 | 行は 99，27，99，27．照合の一致は 1.0．GPU の使用は全体で最大 3.8 GB（ほかの処理の約 1.7 GB を含む） |
| 退化の試しの集計 | `--stage summary --out-dir checks/DAI-23/code_run_v2/degenerate --allow-partial --l-kind audio_wavlm --b-kind audio_wavlm` | 50 秒 | `main.csv` 197 行 29 列，`descriptive.csv` 172 行，`assumptions.csv` 72 行，criteria・convergence・provenance．effect・lo・hi の NaN 0，Inf 0．NaN があるのは `v_split`・`v_split_indep`（fixC の 24 行．分け方の揺れを足さないため），`df`（MOSEI の 54 行．引き直しの区間），`fixC0`・`fixC1`（fixC でない 173 行）だけ |
| 特徴の試し | `features/build_audio_lowpass.py --op lp --fc 500 --corpus MELD --limit 64 --chunk 32`，続けて `--merge` | 各 10 秒前後 | 上の表の 1 |
| 操作の確かめ | `features/check_lowpass_asr.py --n 2 --out-dir ../checks/DAI-23/code_run_v2/asr` | 75 秒 | `fc.json` に code_commit と asr_check_sha256 が入った |

**退化した場合の検算**：L と B に元の声の特徴を入れた 8 単位で，主の量 32 行（M1・M2・M3・M3d）と `*_fixC` の 24 行の効果・区間の両端・標準誤差の絶対値の最大がすべて 0.0 になった．

**格子の予測の再現**：L の格子の予測を実験 14 の A の格子の予測（single_V・single_T の真と #shufcl，8 点，8 単位の 256 組）と照らすと，一致の割合の最小 0.99955，平均 0.999997 で，1.0 でない組は 3 つ（IEMOCAP f0 r0 の C′ = 10 の single_V/#shufcl，MELD f1 r1 の C′ = 10 の single_V/真，MOSEI f0 r0 の C′ = 0.1 の single_T/真．どれも 1 発話）．罰則の弱い点での最適化の許容の差と考えられる．固定に使う点（3 コーパスの試しで C′ = 0.0001〜0.01，本番では 0.001）では全部一致したので，fixC の効果は 0.0 になった．L と B の格子の予測は全部一致した．

**見た値の申告**：集計の行の数・NaN の数・退化の量が 0 であることと，C′ の端の割合の照合の差だけを画面に出し，割合の値そのものは出していない．操作の確かめの 2 本の試しの要約の表（一致率と超過）が画面に出た．この 2 本は前回の 4 本の試しと同じ sha256 の順の先頭 2 本で，前回見てよいとされた範囲の内側である．選ばれた C′ の点（fixC0・fixC1）を見た．実験 14 の predictions.csv では R1fixC の P3・P3b の行の数（10 行）だけを数え，値は見ていない．

## A．本番の規模の見積もり

- 単位：格子の予測を足した分で 1 単位の所要が 1.1〜1.4 倍になった（IEMOCAP 51 → 65 秒と 30 → 43 秒，MOSEI 119 → 132 秒と 63 → 77 秒，MELD 120 → 129 秒と 62 → 71 秒）．150 単位で約 2.8 時間（IEMOCAP 約 40 分，MOSEI 約 67 分，MELD 約 63 分．本番の r ≠ 0 は照合が無いので少し短い）．GPU は全体で 4 GB 未満．
- 保存：1 単位 1.0〜6.8 MB（前回より約 7%増え）．合計約 0.52 GB．
- 特徴と操作の確かめの見積もりは前回と同じ（特徴 2 操作で約 1〜1.9 時間，操作の確かめ約 1 時間）．
- 集計：全単位で数分．

## B．差分で変わった予想される指摘と，出力から答えられるか

前回の表のうち，変わった行と新しい行だけを書く．ほかの行は前回の判定のまま．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 主の表 | C′ を固定したら結論は変わるか | R4，I18 追記（3） | (a)：`main.csv` の `M1_X_fixC`・`M2_X_fixC`．M3 は指摘 1 の直しが要る．直さなくても，単位の npz の `K5__grid<i>__R1__<X>__<nm>__<ver>` と実験 14 の A の格子の予測から `D14.fixc_cell` で D(A ← T)・D(A ← V) を固定の点で作れば，CPU で κ を作り直した M3 を出せる | — | — |
| 主の表 | C′ を固定した版にも分け方の揺れを入れよ | R4，統計 | (c)：格子の予測は元の分け方だけで保存する．設計 7 節に「分け方の揺れは足さない」とあり，判定に使わない記述なので妥当 | — | — |
| 主の表 | C′ を固定したときの L3（顔の生徒との差） | R4 | (a)：V の格子の予測（`K5__grid<i>__R1__V__single_T__*`）は実験 14 の npz にある．`fixc_cell` で作れる | — | — |
| 仮定の表 | C′ の端の割合を生徒 A と並べよ | C | (a)：実験 14 の `assumptions.csv` の C_edge（K = 5，student A）と，dai23 の C_edge の行は同じ数えになった | — | — |

止めるべき (b) は無い．

## 指摘

### [I1・R4] 1．C′ を固定した版の M3 が，R1 で作った κ を使う
- 重さ：直してから進む（集計の段の前．単位の段には関わらない）
- 該当箇所：exp/dai23.py:387（`terms = quant(X, kp)[f"{q}_{X}"]`），347（`kp = kap[(T, subset)]`）
- 問題：読んだだけ（代数）．`M3_X_fixC = M1_fixC − κ_R1 · [D_fixC(A ← V) − D_fixC(X ← V)]` で，κ_R1 は実験 14 の R1 の P3 ÷ P3b である．同じ割合の帰無を固定の点で書くと，D_fixC(X ← m) = c · D_fixC(A ← m) のとき `M3_X_fixC = (1 − c) · [D_fixC(A ← T) − κ_R1 · D_fixC(A ← V)]` になり，κ_fixC = D_fixC(A ← T) ÷ D_fixC(A ← V) が κ_R1 と違えば 0 にならない．この版を「M3 の C′ の感度」として読むと，κ の違いが感度に混ざる．退化の試しでは X = A なので c = 1 となり，この誤りは 0.0 の中に隠れる．実験 14 の predictions.csv には R1fixC の P3・P3b の行（all と nonshared）があるので，κ_fixC は作れる．
- 計算の後で直せるか：直せる（集計の段だけ．単位の学び直しは要らない）．
- 求める対応：`*_fixC` の M3 に，固定の点で作った κ_fixC を使う（同じ集計で D_fixC(A ← T)・D_fixC(A ← V) を `fixc_cell` で作るか，実験 14 の predictions.csv の R1fixC の P3 ÷ P3b を読む）．κ_fixC と κ_R1 の両方の版を出すか，`main.csv` に `kappa` の列として κ_fixC を残す．設計 7 節の文にどちらの κ かを書く．

### [I9] 2．別のセッションのコミットで，単位の段の再開と集計が止まる
- 重さ：記録する
- 該当箇所：exp/dai23.py:137–141（既にある単位が別のコミットなら止める），331–332（コーパスの単位のコミットが 1 つでなければ止める）
- 問題：この再検査の間に，同じ作業木で別のセッションが 2 回コミットした（9529e23 → 7be91bc → d21b840．台本は変わっていない）．退化の試しの単位は IEMOCAP が 9529e23，MOSEI・MELD が d21b840 になった．本番は 1 回の起動の中ではコミットが 1 つにそろうが，落ちて起動し直す前に誰かがコミットすると，run_unit は「別のコミットで既にある」で止まる．1 つのコーパスの途中で起動し直すと，集計がコミットが 1 つでないとして止まる．誤った数値は出さない．
- 計算の後で直せるか：直せる（単位の csv のコミットは台本の sha256 と照らせる）が，学び直しになりうる．
- 求める対応：実験 22 と同じく，固定のコミットの別の作業木から流す．そうしないなら，本番の間に同じ作業木でコミットしないことを起動の手順に書く．

### [I9] 3．設計 8 節の退化の試しの置き場が古い台本の試しを指す
- 重さ：記録する
- 該当箇所：docs/設計/実験23_設計.md 8 節の順序 (1)
- 問題：設計は `checks/DAI-23/code_run/degenerate/`（1376f64 と包みの台本）を名指す．1376f64 の台本には C′ を固定した版と新しい C′ の端の数えが無い．1fb89ff の台本の退化の試しは `checks/DAI-23/code_run_v2/degenerate/`（8 単位．IEMOCAP は f0・f1 × r0・r1，MOSEI・MELD は f0 r0 と f1 r1）にある．
- 求める対応：結果の文で退化の試しを引くときは `code_run_v2/degenerate/` を名指し，その単位のコミット（9529e23，d21b840．台本は 1fb89ff と同じ）を書く．

### [I9] 4．特徴の台本は作業木の汚れを記録しない
- 重さ：記録する
- 該当箇所：features/build_audio_lowpass.py:50，features/check_lowpass_asr.py:209
- 問題：特徴の npz と fc.json は `git rev-parse HEAD` だけを書き，features の台本が未コミットの変更を持つかを書かない．dai23 の units の段は features の汚れで止まり（dai23.py:541–542），台本の sha256 を provenance に残すので，特徴を作った後に台本を直してコミットしても検出は units の段の時点の sha256 になる．
- 求める対応：特徴を作る前に features がきれいなことを起動の手順で確かめる．台本に足すなら `git status --porcelain features` の結果を `code_dirty` として残す．

### [記録] 5．C′ を固定した版の行の追加で，MOSEI の後ろの行の引き直しの乱数が変わる
- 重さ：記録する
- 該当箇所：exp/dai23.py:344，389
- 問題：MOSEI の区間は 1 つの `rng` から引き直しの重みを順に引く（dai14.py:145）．`*_fixC` の行は all の主の量の後，記述と nomarker の前に入るので，記述と nomarker の行の区間は fixC を足す前の台本と同じ乱数にならない．主の量の行は前と同じ順で引く．本番の前の版の値は無いので，結果には影響しない．
- 求める対応：無し（記録だけ）．

## 目録に無い種類の不備
- 感度の版（C′ を固定するなど）で，主の量に使う係数（κ など）を主の作りの推定値のまま使うと，帰無の下で 0 にならない．退化の試し（同じ特徴を入れる）では係数の違いが消えるので見つからない．感度の版の量を足すときは，係数も感度の版の作りで作り直すか，両方を並べる（I1 か R4 への追記の案）．
- 同じ作業木で別のセッションがコミットすると，コミットの一致で守る台本は再開と集計で止まる．長い計算は固定のコミットの別の作業木で流す（I9 か I10 への追記の案）．

## 確かめた項目
I1，I2，I3，I4，I5，I6，I7，I9，I10，I11，I12，I14，I18，R1，R4，R8，R11．

試しの出力：`checks/DAI-23/code_run_v2/`（degenerate/，asr/，units_*.log，gpumem_*.txt，検算の台本 verify_v2.py）．`results/` と `data/cache/` の本番の名前には書いていない．
