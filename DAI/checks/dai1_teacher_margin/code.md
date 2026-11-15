# 関門 1：実装と出力（計算の後の解析 dai1_teacher_margin，2026-11-16，コミット f874dfd）

## 判定
判定：直してから通す．数値の作り（先生と自分の候補の上の UAR，名前の対応，主の先生の gain，並びの照合）は全部の単位で保存値と一致したが，要約の数えに同じ先生の重複（AFEW の 8 点）と自分と同じ先生（AVT の 8 点）が入り，コーパスをまたいだ順位相関しか無いので，この 3 点を直してから結果を見る．

対象：`tools/dai1_teacher_margin.py`（コミット f874dfdb，作業木に差分なし）．入力：`results/dai1/eval/units/*_p1.npz・_p3.csv`（単位のコミット ba98974），`results/dai14/units/*.npz`，`results/dai22/units/*.npz`，`results/dai1/stats/desc/teacher_types.csv`，`results/dai1/stats/families_flagged.csv`，`results/dai1/select/frozen.json`．設計 `docs/設計/実験1_設計.md` 7 節（予測を付ける側の種類）．`docs/設計/実験21_設計.md` は読んでいない．
試しでは行の数・照合・所要・エラーだけを見た．margin・gain・相関の値は画面に出していない（照合のために差の最大と等号の真偽だけを出した）．

## A．設計と台本の対応表

| 設計（依頼）の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 先生の UAR＝K 本を除いた候補の上の先生の判定の UAR | `exp/dai1.py:449`（`nk = setdiff1d(arange(n), kpos[K])`），`:457`（主の先生 sel/one），`:491`（tt の各先生）．台本 `tools/dai1_teacher_margin.py:90,94,98` が `uar_pool` を読む | 合う．`uar_pool` は `S.uar(u.ypool[nk], Q.argmax(1)[nk])*100`．K 本を除く候補である |
| 自分の UAR＝K 本の基準 none[K] の，同じ候補の上の UAR | 台本 `:49-61`（実験 14 の `none__K{K}__{s}`，実験 22 の `none__K{K}`，`nk__K{K}`） | 合う．実験 14・22 の none は `dai3.Saved.none`→`dai1.none_wb`（固定の引き戻しの強さ）で，実験 1 の B = none と同じ識別器（下の検算で全 200 単位一致） |
| 先生の名前の対応 tt:<名前> → teacher_types.csv の teacher | 台本 `:96-97`（先頭の `tt:` を外す），`:106` | 合う．名前の集合が一致し，落ちた点が 0（736 点＝tt の 704 点＋主の先生 32 点） |
| 主の先生（teacher = one）の gain＝族 1（prop − none） | 台本 `:79,103-104` | K = 5・25 では合う．固定のファイルで 4 コーパス × 4 生徒 × 2 K の全 32 条件が main = one，ρ = 1，a = 1 で，族 1 の prop は `joint, teacher=one, ρ=1, a=1` と同じ問．tt は `make_G` で joint と同じ式（`exp/dai1.py:401-402`） |
| gain の区間の作り | teacher_types と families_flagged | 合う．どちらも `dai1_stats.effect_stats`（AFEW・IEMOCAP・MELD は jk+seed，MOSEI は boot_cl+seed，df も同じ） |
| 単位と候補の並びの照合 | 台本 `:57-58`（実験 14／22 の `sid` と p1 の `sample_pool`） | 合う．ただし `nk` の照合は台本に無い（検算で全単位一致を確かめた．指摘 R5） |
| K = 5, 25．単位 5 分割 × 10 種 | 台本 `:45,82-85,101-102` | 合う．全点 n_units = 50 |
| margin の集約＝単位ごとの差の平均 | 台本 `:111` | gain の推定（分割ごとの種の平均の分割平均，`dai1_stats.theta`）と，種の数が分割で等しいので同じ重み |
| 生徒自身のモダリティを含まない印 | 台本 `:112`（teacher_types の `no_own_modality`，主の先生は偽） | 合う．`dai1_desc.py:641` の規則（AVT の生徒は常に偽，ret・fused・poe・noret・start は含む扱い） |
| 出力の上書きをしない | 台本 `:71-73` | 合う |

設計の表の組の数：生徒 A・V・T は先生 21＋主の先生 1＝22，AVT は 25＋1＝26．4 コーパス × 2 K × (22×3＋26)＝736．台本の出力も 736 行で，(T, 生徒, K) ごとに 22 が 24 組，26 が 8 組．

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tools/dai1_teacher_margin.py --out-dir checks/dai1_teacher_margin/code_run/full`（GPU はほかの処理が 89 % 使っていたので CPU）．所要 11 秒．全 4 コーパスを流した（小さくする必要が無い規模）
- 出力：`checks/dai1_teacher_margin/code_run/full/teacher_margin.csv`（736 行），`teacher_margin_summary.csv`（10 行）．警告・例外なし
- 出力の点検（`code_run/verify.py`）：同じ (T, 生徒, K, 先生) の重複 0，NaN 0，Inf 0，UAR はすべて 0〜100，gain は全行で lo ≤ effect ≤ hi，要約の spearman に NaN 0，commit 列は f874dfd，dirty は偽
- 保存値の再現（全 200 単位 × K 2 つ）：
  - 自分の UAR（実験 14／22 から作った値）と実験 1 の段 2 の `teacher` の行の `none_{s}` の `uar_pool`：差の最大 7.1e-15
  - tt の先生の `uar_pool` と，同じ予測を段 2 が記録した値（single_* ↔ none_*，pair_* ↔ none_AV 等，fused_AVT ↔ none_AVT，ret_*，poe_none）：差の最大 0
  - 主の先生 one の `uar_pool` と段 2 の `product (1,1,1)` の `uar_pool`：差の最大 0．one の行は各 K で 200 単位に 1 行ずつ
  - 実験 14・22 の `nk__K{K}` と，p1 の `sample_K{K}` から作った K 本を除く印：不一致 0 単位
- 保存に無い先生（src_*，start_AVT，noret_sel，one）の候補の上の確率を，学び直さずに作り直せるか（`code_run/rebuild_one.py`．AFEW f0 r0）：`dai3.Saved`＋`dai1.build_tk`＋ソースの W・b から 24 個を作り直し，保存の `uar_pool` との差の最大 7.1e-15，0.4 秒．B の (a) の根拠に使う
- 退化した場合の検算：AVT の生徒の先生 fused_AVT は none_AVT そのもの（`exp/dai1.py:476`）で，自分と同じ識別器なので margin は 0 になるはずである．8 点すべてで |margin| < 1e-9 を確かめた（値の向きは見ていない）

## A．本番の規模の見積もり
CPU で約 11 秒，GPU は使わない．メモリの制約なし．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 点ごとの表 | margin に区間を付けよ | 9.3 節 S，review-ml の区間の要求 | (b) だが学び直し不要 | 単位ごとの長い表（T, f, r, 生徒, K, 先生, teacher_uar_pool, self_uar_pool）を書き出す．区間は gain と同じく分割と種の分散から作る．塊の jackknife／bootstrap にするなら候補の発話ごとの判定が要る（下の行） | 3.7 万行，約 3 MB，数秒 |
| 点ごとの表 | 単位ごとの margin と gain の関係（点の平均どうしでなく） | tac-round1-status の「単位の値を残さず学び直し」 | (b) だが学び直し不要 | 上の長い表に，単位ごとの gain（p3 の tt の行の `uar` − p1 の none の行の `uar`．固定の引き戻しの強さ）を足す | 数秒 |
| 要約 | 「支配的なモダリティ」の定義はどれか．コーパスごとにどのモダリティが強いか | 依頼の問い，9.3 節 D | (a) | `teacher_margin.csv` の `self_uar_pool` を (T, K) ごとに A・V・T で比べる（候補の上）．テストの上で言うなら p1 の csv の `method=none` の行の `uar`（固定の Cw・Cb） | なし |
| 要約 | コーパスをまたいで順位相関を取ると，コーパスの違い（自分の強さ）だけで相関が出る．コーパスの中で見よ | 9.3 節 C（交絡），Simpson | (b) だが学び直し不要 | (T, 生徒, K) の中の順位相関（22 か 26 点），または (T, 生徒, K) ごとに中心化した margin と gain の相関を要約に足す | 数秒 |
| 要約 | 同じ先生を 2 回数えていないか | 9.3 節 L4 | (b)（指摘 R1） | AFEW の K=5・25 は選んだ指数が (1,1,1) で noret_sel と poe_none が同じ予測．1 つだけ残す印を足す | なし |
| 要約 | 自分と同じ先生（AVT の fused_AVT）は自己学習であり，ほかの先生と同列に数えてよいか | 9.3 節 D | (b)（指摘 R2） | `teacher_is_self` の印を足し，要約から外した版も出す | なし |
| 点ごとの表 | margin は候補の上，gain はテストの上である．テストの上の先生の当たり方でも同じか | 9.3 節 D（測る場所の食い違い） | (a)（作り直し） | none_*，ret_*，poe_none，product は段 2 の csv の `uar` にある．src_*，start_AVT，noret_sel は `rebuild_one.py` と同じ手でテストの特徴から作り直す | 200 単位で数分（CPU） |
| 点ごとの表 | margin（判定の UAR）ではなく，先生が正しく自分が誤る発話の割合（補い合い），先生と自分の一致（κ），感情ごとの再現率で説明できるか | review-affect の感情ごとの要求，9.3 節 A | (a)（作り直し） | 自分の候補の上の判定は実験 14／22 の npz にある．先生の判定は段 2 の npz（`p2_K{K}_{name}_pool`，float16）と `rebuild_one.py` の手で作り直す．`y`・`cl`・`sex` も実験 14 の npz にある | 200 単位で数分（CPU） |
| 点ごとの表 | 生徒の伸びしろ（候補の全ラベルで学んだ上限 − none）で gain が決まっているのではないか | 9.3 節 C（別の説明） | (a) | p1 の csv の `method=oracle`（K=-1）と `method=none` の行を，固定の (Cw, Cb) で選び，テストの `uar` の差を単位ごとに平均する | 数秒 |
| 点ごとの表 | 柔らかいラベルの質（真の感情の確率，鋭さ）で説明できるか | 9.3 節 D | (a)＋(作り直し) | p3 の `_teacher_stats` の `maxp`・`entropy`・`neff_*` はすでにある．真の感情の確率は作り直した先生の確率から出す | 数分 |
| 要約 | K=5 と K=25 は別の証拠か | 9.3 節 S（点どうしの依存） | (a)（書き方の注意） | src_* と start_AVT は K に依らない同じ識別器で，K で変わるのは候補の除き方だけ．K をまたいで点を足し合わせない | なし |
| 要約 | 先生の種類の数え上げは検定か | 9.3 節 S | (c) ではなく書き方の注意 | 台本の冒頭（`:20-21`）のとおり検定ではない．結果の文で「数え上げ」とだけ書く | なし |
| 全体 | 別のエンコーダ，別のコーパスでも同じか | 9.3 節 L | (c) | 実験 1 の範囲の外．結果の文で 4 コーパス・この特徴に限ると書く | — |

(b) のうち，学び直しを要するものは無い．すべて保存した単位から CPU で数秒〜数分で出せるので，「止める」にした行は無い．

## 指摘

### [L4] AFEW の K = 5・25 で noret_sel と poe_none が同じ先生で，要約で 2 回数えている
- 重さ：直してから進む
- 該当箇所：`exp/dai1.py:477-478`，`tools/dai1_teacher_margin.py:119-127`．固定のファイル `results/dai1/select/frozen.json` の AFEW の `w` は K=5・25 とも `[1.0, 1.0, 1.0]`
- 問題：`noret_sel = product(tk, wsel[K], 0, src="none")` は指数が (1,1,1) のとき `poe_none = poe(tk, "AVT", 0)`（none の A・V・T の積）と同じ確率になる．流した出力で AFEW の 2 K × 4 生徒の 8 組すべてで teacher_uar_pool と gain が完全に一致した．要約の `sign_agree`・区間の数え・spearman で同じ点を 2 回数える
- 計算の後で直せるか：直せる
- 求める対応：(T, K) ごとに `wsel[K] == (1,1,1)` のとき noret_sel の行に重複の印を付け，要約から外す（点ごとの表には残してよい）

### [D] AVT の生徒の fused_AVT は自分そのもので，margin は定義から 0 になる
- 重さ：直してから進む
- 該当箇所：`exp/dai1.py:476`（`tts["fused_AVT"] = tk["none_AVT"][0]`），実験 22 の `none__K{K}` も同じ none_AVT
- 問題：8 点すべてで |margin| < 1e-9．この点は「ほかの先生」ではなく自分の判定を疑似ラベルにした学び直しで，符号の数えでは 0 なので数えられず，順位相関には 0 の点として入る．生徒 A・V・T には同じ種類の点が無い（自分の single は外してある．`exp/dai1.py:474`）ので，生徒の間で先生の集合がそろわない
- 計算の後で直せるか：直せる
- 求める対応：`teacher_is_self` の印を足し，要約は外した版を主にする．結果の文では自己学習の点として別に書く

### [C] 要約の順位相関がコーパスをまたいで取られ，自分の強さの差が相関を作りうる
- 重さ：直してから進む
- 該当箇所：`tools/dai1_teacher_margin.py:119-120`（`groupby(["K", "student"])` と K ごとの全部）
- 問題：margin＝先生 − 自分で，(T, 生徒, K) の中の 22〜26 点は同じ自分の値を共有する．4 コーパスを混ぜた相関は，コーパスごとの自分の強さと gain の水準の違いだけで正にも負にもなる．依頼の問い（支配的なモダリティでは上がらない）はコーパスごとに支配的なモダリティが違いうるので，コーパスの中の比べが要る
- 計算の後で直せるか：直せる
- 求める対応：(T, 生徒, K) の中の順位相関と符号の数えを要約に足す．コーパスをまたぐ値は (T, 生徒, K) ごとに中心化した版も出す

### [S] 点どうしの依存の書き方
- 重さ：記録する
- 該当箇所：`tools/dai1_teacher_margin.py:20-21`
- 問題：同じ先生の確率（fused_AVT，pair_*，poe_none，src_*_AVT，start_AVT，主の先生など）を 4 生徒が受け取るので，生徒をまたいだ点の gain は同じ疑似ラベルを共有する．src_* と start_AVT は K に依らない同じ識別器で，K=5 と K=25 の点は候補の除き方だけが違う．主の先生は (T, 生徒, K) に 1 点ずつで重複は無い（流して確かめた）
- 計算の後で直せるか：直せる
- 求める対応：結果の文で「点は独立でない．数え上げと順位相関は記述である」と書き，K をまたいで点を合計しない

### [R] 単位ごとの値を書き出していない
- 重さ：記録する
- 該当箇所：`tools/dai1_teacher_margin.py:98,110-113`（`per` の単位ごとの値を平均してから捨てる）
- 問題：margin の区間，単位ごとの margin と gain の関係を後で求められると，台本を書き換えて流し直すことになる．流し直しは 11 秒で済むので学び直しは要らない
- 計算の後で直せるか：直せる
- 求める対応：`teacher_margin_units.csv`（T, fold, seed, student, K, teacher, teacher_uar_pool, self_uar_pool）を同時に書く．単位ごとの gain も足すなら p3 の tt の行の `uar` と p1 の none の行の `uar` から作る

### [I9] 記録の不足
- 重さ：記録する
- 該当箇所：`tools/dai1_teacher_margin.py:116-117`
- 問題：`dai1.git_commit()` の dirty は `exp splits measure features data/metadata` だけを見て `tools/` を見ないので，この台本の未コミットの変更は dirty に出ない．`units_commit="ba98974"` は単位の保存から読まずに書き込んでいる（今回は p3 の commit 列が ba98974583… で一致）．実験 14・22 の単位のコミットと，入力の csv・frozen.json の sha256 を残していない．DAI の取り決め（`results/<実験>/provenance.json`）にも書かない
- 計算の後で直せるか：直せる
- 求める対応：台本と入力 3 本の sha256，実験 14・22 の単位の commit を出力の列か provenance に足す．`units_commit` は p3 の commit 列から読む

### [I] 黙って点を落とす分岐と，照合の不足
- 重さ：記録する
- 該当箇所：`tools/dai1_teacher_margin.py:107-108`（`len(g) != 1` で `continue`），`:57-59`（`nk` を照合しない）
- 問題：teacher_types.csv に行が無い先生や 2 行ある先生を黙って落とす．今回は 736 点＝見込みの数で，落ちた点は 0 だった．`nk__K{K}` は実験 14・22 の保存をそのまま使い，p1 の `sample_K{K}` と照らさない（今回は全単位一致）
- 計算の後で直せるか：直せる
- 求める対応：最後に点の数が (21×3＋25＋4)×2×4＝736 であることを確かめて止める．`nk` を `~isin(sample_pool, sample_K{K})` と照らす

### [記録] 冒頭の文の誤り
- 重さ：記録する
- 該当箇所：`tools/dai1_teacher_margin.py:16-17`
- 問題：「族 4 の既定の設定 default − none ではなく」とあるが，族 4 は prop − default で，default − none という族は無い．「48 条件のうち 44 で同じ」は K の全体の話で，この台本が扱う K=5・25 の 32 条件では全部 prop と default が同じ問である
- 求める対応：「族 1 の prop − none．K=5・25 の 32 条件で prop は ρ=1，a=1，teacher=one の問で，tt と同じ式」と書き直す

## 結果を書くときの注意（依頼の (4)）
- margin は候補の上（K 本を除く）の判定の UAR の差，gain はテストの上の UAR の差である．測る発話の集合が違うので，`gain_gt_margin`（gain > margin の数）は同じ物差しの比べではない
- 生徒の学び直しは出発点（ソースの重みの平均）への引き戻しつきで，K 本も入る（ρ=1，a=1）．margin が正でも，引き戻しの強さ（固定の Cw は 0.001〜0.01）で生徒が先生に追いつかないことがある．伸びしろ（上限 − none）を並べると区別できる（B の表）
- `no_own_modality` は生徒 A・V・T だけで真になり（各 72 点），AVT の生徒と主の先生は常に偽である．「自分のモダリティを含まない先生」の比べは単一の生徒の中だけで行う
- 主の先生は 3 つの学び直した識別器の積で，その学び直しの疑似ラベルは none_AVT から作る（`exp/dai1.py:279-290` の段 2．`:284` の `F = tk["none_AVT"][0]`）．AVT の生徒の主の先生は自分（none_AVT）を間接に含む

## 目録に無い種類の不備
- 「選んだつまみが既定の値と一致すると，別の名前の 2 つの条件が同じ計算になる」（今回の noret_sel と poe_none）．9.3 節 L4 の下に「条件の名前が違っても，固定した値によって同じ予測・同じ問になる組を，表と数えの前に検出して印を付ける」を足す案
- 「先生の候補に，生徒の比較の基準そのものが入る」（AVT の fused_AVT）．9.3 節 D に「疑似ラベルの作り手の種類を比べるとき，基準の識別器と同じ作り手を含むかを確かめ，含むなら自己学習として別に扱う」を足す案
