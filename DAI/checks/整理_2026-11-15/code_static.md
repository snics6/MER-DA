# 整理の検査：置き場の差分と DAI 単独での再現（2026-11-15，コミット 405b810・a06f779，比べた元 8fd53fd）

判定：直す

## 判定
直す．計算の式・定数・既定の値・分岐を変えた台本は 1 本も無く，`frozen/eval_gate.json` の作り直しも正しいが，評価の回と仮のターゲットの回の起動の台本 3 本（`run/dai_eval.sh`，`run/dai1_pseudo.sh`，`run/dai2_pseudo.sh`）が DAI の外へ `cd` するので動かず，DAI だけを git の根にした写しでは検査の台本 271 本が落ちるからである．

GPU は使っていない．結果の値は開いていない（ファイルの有無と sha256 の先頭だけを見た）．検算の台本と出力は `checks/整理_2026-11-15/code_run/` にある（`.gitignore` の `checks/*/code_run*/` により git の外）．

## 1．置き場の行だけが変わったか（確かめたこと 1）

`check_rewrite.py` と `check_inplace.py` で，a06f779 の `exp/`・`run/`・`tools/`・`measure/`・`features/` の全ファイルについて，元（対応表の old_path，対応表に無いものは同じパスの 8fd53fd，`copy_untracked_log` は `/home/snics/MER-DA/logs/`）に `rewrite()` を当て直し，tools の `sys.path` の置き換えも足して，新しいファイルと比べた．

| 区分 | 本数 | 結果 |
|---|---|---|
| その場で書き換えた台本（DAI/exp・run の元からある分．対応表に行が無い） | 40 | 39 本が規則と一致，1 本は変化なし |
| copy_front（MER-DA の exp・run から写した台本） | 51 | 42 本が規則と一致，9 本は規則が何も変えず元と同じ |
| move（tools_pending → tools） | 14 | 6 本が規則と一致．4 本は「元に戻した」（extract_dai18/20/21_text.py，rebuild_explain_pdf_2026-11-14.py．元と同じ）．4 本は規則と食い違う（下） |
| copy_untracked_log（dai14_run.sh，dai15_run.sh） | 2 | 規則と一致 |
| copy_common（measure 7 本，features 14 本） | 21 | 元と sha256 が同じ |
| 新しいファイル（run/dai16_probe.sh，tools/collect_results.py，tools/remap_eval_gate.py，tools/reorg_2026-11-15.py，tools/README.md） | 5 | 元が無い（下の指摘 9・12） |

規則と食い違う 4 本（latex_tables_dai16_17.py，md_tables_dai16_17.py，md_tables_dai18.py，pdf_figs_proposal.py）は，差分を 1 行ずつ読んだ．すべて `/home/snics/MER-DA/...` の絶対パスを `_ROOT + "/..."`（`_ROOT` は `Path(__file__).resolve().parents[1]`＝DAI）に置き換えた行と，`OUT = ROOT / "docs" / "説明"` の 1 行で，記録の tsv の manual の行と合う．

さらに，`all_diffs.sh` で元と新の変わった行を全部書き出し（93 本，900 行），1 行ずつ読んだ．変わった行は，パスの文字列（説明の文，使い方の文，`sys.exit` の文，provenance の `script`・`rule` の文字列），`ROOT`・`HERE` の決め方，`sys.path.insert`，`cd` の行，`git status`・`git diff` に渡すパス指定，`git show <commit>:./exp/...`，`ROOT / "frozen"` などのパスの並びだけだった．計算の式・定数・格子・引数の既定の値（パス以外）・分岐が変わった行は無い．

## 2．書き換えたパスが実在するか（確かめたこと 2）

`check_paths.py` で，`exp/`・`run/`・`tools/` の台本の中のパス（文字列と `ROOT / "a" / "b"` の並び）を a06f779 の DAI の木と照らした（`results/`・`logs/`・`data/cache/` と書式の穴を含むものは除く）．実在しないと出た 74 件の内訳は次のとおりである．

- 関数の参照（`measure/data.load_any` など 5 件），文の中の語（`exp/・measure` など 3 件），旧版の台本の名前を引いた説明の文（`exp/check_e60_refuse.py`，`exp/merged_scores.py`，`exp/mech.py`，`exp/test_map.py`，`paper/reviews/round1/review-ml.md`．どれも説明の文だけで，読み込みも実行もしない）
- 一度も git に入っていなかった検査の台本の名前を引いた説明の文（`exp/dai17_posthoc.py:53` の `checks/DAI-17/code_run_post/dependency_table.py`）
- 整理の前の置き場のままにした道具 4 本（extract_dai*_text.py，rebuild_explain_pdf_2026-11-14.py）．指摘 13
- 指摘 7・8 の 3 件（`tools/dai21_posthoc_cremad_sentence.py:105` の `KSHOT_REF`，`tools/latex_tables_dai16_17.py:2・99`）

見張りと sha256 の一覧は，元の一覧と 1 対 1 に対応し，取りこぼしは無い．

- `dai1.EVAL_REQUIRED`（`exp/dai1.py:588-594`）：20 本．元の 20 本を対応表で置き換えたものと同じ．20 本とも a06f779 にある
- `dai20_eval.WATCH`（`exp/dai20_eval.py:79`），`dai21_eval5.WATCH`・`FIXED`（`exp/dai21_eval5.py:78-80`）：元の `DAI/exp/…`・`DAI/frozen/…`・`DAI/splits`・`DAI/docs/実験2x_設計.md` が `exp/…`・`frozen/…`・`splits`・`docs/設計/…` に変わり，`measure`・`features`・`data/metadata`・`requirements.txt` は DAI の中の写しを指すようになった．`dai_data.py` が `sys.path` に入れるのも DAI の中の `measure/`・`features/` なので，見張る対象と読み込む対象は一致している
- 起動の台本の sha256 の確かめ（`run/dai11〜18・20・21・k10` の `sha256sum exp/<台本>`）：すべて `exp/` を指す．ただし指摘 1 の 3 本は `cd` の先が違う

git のパス指定の意味：台本は `git` を `cwd=ROOT`（＝DAI の直下）で呼び，パス指定はすべて相対である．git は相対のパス指定を cwd から解くので，MER-DA の中の `DAI/` で流しても（`exp` は `DAI/exp` になる），DAI だけを根にしたリポジトリで流しても，同じファイルを指す．`git show <commit>:./exp/…`（`exp/dai1_desc.py:1377`）の `./` も cwd から解かれる．`check_eval_gate` と `git_commit` を両方の置き方で呼んで確かめた（4 節）．例外はパス指定の無い `git status` を使う 2 か所で，指摘 10 に書いた．

## 3．文書と固定のファイルの中身（確かめたこと 3）

`archive` 735 行のうち 431 行，`move` の文書 69 行，`copy_snapshot` 6 行，`copy_front` の設計と検査の報告は，元と sha256 が同じである．`archive` の残り 304 行は，`archive/checks/*/code_run*/` の下の試走の出力で，対応表の後ろの `untrack_checks_output` の行で git から外してある（304 本すべてに外した行がある）．同じパスのまま中身が変わった DAI のファイルは `CLAUDE.md`，`docs/実験の地図.md`，`frozen/eval_gate.json` の 3 本だけで，どれも整理で直す予定のものである．`frozen/` の固定のファイル（dai1・2・3・15・20・21_frozen.json，subsets.csv）は変わっていない．

## 4．frozen/eval_gate.json（確かめたこと 4）

`check_gate.py` で次を確かめた．

- `required` の 40 本の鍵は，すべて a06f779 にあり，sha256 が一致する
- `_meta.before_reorg.required` と `approvals` は，8fd53fd の eval_gate.json と一致する．元の `_meta.rule` も残っている
- 元の 40 本の鍵を対応表で置き換えると，新しい 40 本の鍵にちょうど対応する（足りない鍵も余分な鍵も無い）
- `_meta.changed_after_gate` の 16 本は，鍵の sha256 が元と違うものを計算し直した一覧と一致する．行ごとの old_sha256・new_sha256 も合う．中身は `docs/設計/実験1_設計.md` と，置き場の行を書き換えた 15 本（exp 13 本，`run/dai_eval.sh`，`run/eval_procedure.md`）である
- `docs/設計/実験1_設計.md` は，8fd53fd の時点で既に元の gate の sha256 と違っていた（評価の回の後の 1 行の直し）．`_meta.reorg_note` の説明と合う．このほかに，整理の前から gate と食い違っていた鍵は無い
- `approvals` の 21 個は，元の報告を対応表で置き換えた新しい置き場を指す．報告の中身は元と同じ sha256 である

`dai1.check_eval_gate("frozen/dai1_frozen.json")` を，評価の段を流さずに関数として呼んだ．DAI だけを根にした写し（ROOT は写しの DAI）と，`/home/snics/MER-DA-reorg/DAI`（MER-DA の中のフォルダ）の両方で通り，戻り値は同じ `92378fa0…` だった．`dai1.git_commit()` は両方で `dirty=False` を返した．

## 5．入口の台本の import（確かめたこと 5）

`check_imports.py` で，地図の 2 節の入口の台本 37 本と `tools/` の全部から AST で import をたどった．たどった台本は 89 本で，すべて `exp/`・`tools/`・`features/` にある．`measure/` の `data`・`split`・`shift`・`ablate` は，各台本が `sys.path.insert(0, str(ROOT / "measure"))` で入れる DAI の中の写しから読まれる．`archive/` の台本を読み込む台本は無い．`importlib.import_module`・`spec_from_file_location` による読み込みも無い．

DAI だけを根にした写しで，構文の検査（108 本），`bash -n`（run の 22 本），後半の 22 個のモジュールと前半の入口のモジュール（map，relearn_e76，relearn_e76_finish，analyze_e76，reanalysis_e75，compare_dai，compare_dai2，e80_teacher_j ほか）の import を流し，すべて例外なしで通った．`ROOT` はすべて写しの DAI を指した．起動の台本と文書が呼ぶ `exp/`・`tools/`・`features/`・`measure/` の台本はすべて実在する．`--stage` の名前（dai1：source・pseudo・select・eval，dai2：pseudo・select・eval，dai3：validate・detect・eval，dai20：freeze・units・summary，dai21：freeze・check3・check18・students4・cremad・summary，analyze_e79：delta・select・measure・judge），`--world`，環境変数（`DAI11_SHA`，`DAI12_SHA`，`DAI13_SHA`，`SHA16`，`DAI17_SHA`，`SHA18`，`SHA20`，`SHA21`，`MERDA_GPU_FRACTION`），`run/build_data.sh` の段（manifest・check・feat1・feat2）は，再現の手順の表と合う．

## 6．直すべき点

### 1．（止める）評価の回と仮のターゲットの回の起動の台本が DAI の外へ cd する
- ファイル：行：`run/dai_eval.sh:19`，`run/dai1_pseudo.sh:7`，`run/dai2_pseudo.sh:7`
- 何がおかしいか：3 本とも `cd "$(dirname "$0")/../.."` のままである．整理の前は `DAI/run` から 2 つ上がって MER-DA の直下に行き，`DAI/exp/dai1.py` を呼んでいた．`rewrite()` は `cd /home/snics/MER-DA\n` だけを置き換えるのでこの行を直さず，後ろのパスだけを `exp/dai1.py` に変えた．流すと MER-DA の直下（DAI だけのリポジトリでは DAI の親）に移り，`exp/dai1.py` が無いので止まる．`logs/` を DAI の外に作り，`git status --porcelain exp splits run measure features data/metadata` と `sha256sum … splits/* frozen/*.json` も DAI の外を見る．3 本から `cd` の先を計算すると，どれも `/home/snics/MER-DA-reorg` になり，そこに `exp/dai1.py` は無いことを確かめた（読んで計算しただけで，台本は流していない）
- どう直すか：3 本の行を `cd "$(dirname "$0")/.."` にする．`rewrite()` に `cd "$(dirname "$0")/../.."` → `cd "$(dirname "$0")/.."` の規則を足すか，手で直した行として書き換えの記録の tsv に足す．`run/dai_eval.sh` は `eval_gate.json` の required に入っているので，直した後に gate を作り直す．`tools/remap_eval_gate.py` は `before_reorg` があると止まるので，`_meta.before_reorg` から作り直す道を足す

### 2．（止める）DAI だけを根にした写しで，git に入っている検査の台本 271 本が落ちる
- ファイル：行：`.gitignore:30-36`（`checks/*/code_run*/`，`checks/*/review_*_run*/`，`checks/*/review_*_sim*/`，`checks/*/post_run*/`，`checks/*/post_tools_run*/`，`archive/checks/*/code_run*/`）
- 何がおかしいか：a06f779 では，これらのフォルダの下の `.py`・`.md`・`.sh` が 271 本 git に入っている（MER-DA の中では，一度追跡したファイルは `.gitignore` に当たっても残る）．渡された写し（`standalone_front/DAI`）にはこの 271 本が無い．この中には，`run/eval_procedure.md` の 4 で流す `checks/DAI-eval/code_run/m3_frozen_keys.py`，9 で流す `checks/DAI-1/code_run_post/c9_predcheck.py`，DAI-20・21 の計算の後の点検の台本（`checks/DAI-20/post_run/check_*.py`，`checks/DAI-21/post_run/check_*.py`，`post_tools_run*/verify_*.py`）が入る．GitHub に DAI だけを上げるときに同じ作り方をすると，これらが失われる
- どう直すか：写しを作るときに追跡しているファイルの一覧（`git ls-files DAI`）から作る（`git subtree split --prefix=DAI` など）か，`.gitignore` に `!checks/*/code_run*/**/*.py` などの例外を足して `.py`・`.md`・`.sh` を残す．どちらにしたかを `docs/再現の手順.md` 1 節に書く

### 3．（直す）論文の表の台本が DAI の外の results/ を読む
- ファイル：行：`paper/scripts/tab_mixing.py:12`
- 何がおかしいか：`ROOT = Path(__file__).resolve().parents[3]` は `DAI/paper/scripts/` から 3 つ上の MER-DA の直下を指し，`ROOT / "results/e76_analysis/e76a_1_kind.csv"` を読む．整理の後は `results/` を DAI の直下に置くので，DAI だけのリポジトリでは DAI の親を読んで止まる．`paper/` は整理の書き換えの対象から外れていたので残った（読んだだけ）
- どう直すか：`parents[2]` にする（DAI の直下）．`paper/` は論文の相談のセッションが書く置き場なので，そのセッションに伝えて直してもらう

### 4．（直す）results/MANIFEST.tsv が git に入っていると書いてあるが，a06f779 には results/ のファイルが 1 本も無い
- ファイル：行：`README.md:31`，`docs/再現の手順.md:12・67・82`，`CLAUDE.md:85`
- 何がおかしいか：`git ls-tree a06f779 DAI/results` は空である．渡された写しにも `results/MANIFEST*.tsv` は無い．「2 MB 以下の表は `results/` に git で入れてあり」（`docs/再現の手順.md:67`）は今のコミットでは成り立たない
- どう直すか：`tools/collect_results.py` を流して `results/MANIFEST.tsv`・`MANIFEST_outside_git.tsv` と選んだ表を `git add -f` してからコミットするか，文書を「この後に入れる」と直す

### 5．（直す）値の再現を確かめたと書いてあるが，その記録が無い
- ファイル：行：`docs/再現の手順.md:86`
- 何がおかしいか：「整理の後の台本が保存した値を再現することは，実験ごとに少数の単位で確かめた（`checks/整理_2026-11-15/`）」とあるが，a06f779 にも作業木にも，その確かめの報告は無い（この報告は読むことと静的な検算だけで，値の再現は確かめていない）
- どう直すか：少数の単位で流し直して値を比べ，報告を `checks/整理_2026-11-15/` に置いてから文を残すか，文を消す

### 6．（直す）研究室の PC のリンクの説明が今の状態と合わない
- ファイル：行：`docs/再現の手順.md:14`
- 何がおかしいか：「`.venv`・`weights`・`logs` は MER-DA の直下のものへのリンク，`data/cache/*.npz` は … リンク，… MER-DA の `results/` の同じ名前はこちらへのリンク」とあるが，`/home/snics/MER-DA-reorg/DAI` には `.venv`・`weights`・`logs`・`results` が無く，`data/cache/` には `faces_*.csv.gz` しか無い．リンクがあるのは渡された写しだけである．MER-DA の `results/` を DAI に移してリンクにする作業も，まだ行われていない
- どう直すか：main に入れるときにリンクを作る手順として書き直す（「〜にする」）か，作った後にこの文を残す

### 7．（直す）logs/ を作らない起動の台本が 8 本ある
- ファイル：行：`run/e77.sh`，`e77b.sh`，`e77h.sh`，`e78.sh`，`e78b.sh`，`e78h.sh`，`run/dai14_run.sh`，`run/dai15_run.sh`（どれも `mkdir -p logs` が無い）
- 何がおかしいか：`docs/再現の手順.md:12` は「`logs/`（起動の台本が作る）」と書くが，この 8 本は作らない．`logs/` は git の外なので，取り出したばかりの作業木では `>> logs/e77_avt.log` の向け先が開けず，python が起動されない．`|| echo … >> logs/e77_fail.log` も同じ理由で書けないので，失敗が記録に残らない（読んだだけ）
- どう直すか：8 本の `cd` の次の行に `mkdir -p logs` を足すか，再現の手順に「先に `mkdir -p logs`」と書く

### 8．（直す）保存した単位から集計し直せない範囲の説明が足りない
- ファイル：行：`docs/再現の手順.md:62`
- 何がおかしいか：実験 20・21 だけを挙げているが，実験 1 の記述の集計も同じである．`exp/dai1_desc.py:1361` は今のコミットが評価の回の単位のコミットと同じことを求め，`:1365` は `git show <単位のコミット>:./exp/dai1_desc.py` が今のファイルと同じことを求める．整理で `dai1_desc.py`・`dai1_stats.py` の中身が変わったので，保存した評価の回の単位を整理の後のコミットで集計し直すことはできない（DAI だけのリポジトリでは単位のコミットそのものが無い）．`exp/dai2_stats.py` も評価の回のコミットの作業木で流すことを求める
- どう直すか：4 節に「実験 1〜3 の評価の回の集計（`dai1_stats.py eval`，`dai1_desc.py eval`，`dai2_stats.py eval`，`dai3.py --stage eval`）は，保存した単位からは整理の前のコミット（MER-DA の 8fd53fd まで）の作業木でだけ流し直せる．整理の後の台本では単位から流し直す」と足す

### 9．（記録する）整理の台本と対応表が，実際に行った操作の全部を写していない
- ファイル：行：`tools/reorg_2026-11-15.py:255-263`，`docs/整理の対応表_2026-11-15.tsv`
- 何がおかしいか：
  - 台本の「外す」規則（リンク・拡張子が keep_ext の外・1 MB 超）を 8fd53fd に当てると，`DAI/checks/` の下で 407 本を選ぶ．対応表の `untrack_checks_output` は `DAI/checks/` の下が 465 本，`DAI/archive/checks/` の下が 304 本で，合計 769 本は `.gitignore` の試走の出力の型に当たるもの全部である．差の 362 本（小さい csv・json など）を外した手順は台本に無い
  - 対応表の new_sha256 が今のファイルと合わない行が 9 行ある（元に戻した道具 4 本，手で直した道具 4 本，`docs/実験の地図.md`）．台本を流した直後に計算したまま，後で直した分を足していない
  - 新しく作ったファイル（`run/dai16_probe.sh`，`tools/collect_results.py`，`tools/remap_eval_gate.py`，各 README）の行が無い
- どう直すか：対応表の 9 行の new_sha256 を a06f779 の値に直し，kind `new` の行を足す．書き換えの記録の tsv に「`.gitignore` の型に当たる試走の出力 362 本を追加で外した」行を足すか，台本の規則を実際の操作に合わせる

### 10．（記録する）パス指定の無い git status は，置き方によって見る範囲が違う
- ファイル：行：`exp/compare_dai.py:963`，`exp/compare_dai2.py:862`
- 何がおかしいか：`git status --porcelain --untracked-files=no` にパス指定が無いので，MER-DA の中の `DAI/` で流すと MER-DA 全体の変更を数え，DAI だけのリポジトリでは DAI の中だけを数える．provenance の `dirty` の意味が置き方で変わる．計算は止めない（記録の値だけ）
- どう直すか：直すなら `"--", "."` を足して DAI の中に限る．直さないなら再現の手順の限界に書く

### 11．（記録する）検査役が数えた K 本の本数の表を git から外したので，道具の照合が黙って飛ぶ
- ファイル：行：`tools/dai21_posthoc_cremad_sentence.py:105・555`
- 何がおかしいか：`KSHOT_REF = ROOT / "checks" / "DAI-21" / "post_tools_run" / "kshot_ieo_counts.csv"` があれば照らす作りである．この csv（4,708 バイト）は 8fd53fd では git に入っていたが，整理で外した．取り出したばかりの作業木では `reference_compared=False` になり，照合をしないまま表を書く
- どう直すか：この csv を `git add -f` で残すか，再現の手順に「照合の表は git の外」と書く

### 12．（記録する）新しく書いた起動の台本 run/dai16_probe.sh は検査を通していない
- ファイル：行：`run/dai16_probe.sh:1-20`
- 何がおかしいか：冒頭の文のとおり，2026-11-13 の本計算は手で流しており，この台本はその手順を整理のときに書き起こしたものである．units → summary → `dai16_posthoc.py` を既定の引数で流す．元の手順と同じかは，ログ（`logs/dai16_probe_units.log` など）との照合をしていない（読んだだけ）
- どう直すか：再現の手順の表で「整理のときに書いた」と断るか，gate-code に通す

### 13．（記録する）整理の前の置き場のままの道具は，git の履歴のほかに今のファイルも古い置き場で読む
- ファイル：行：`tools/extract_dai20_text.py:44-57・119-127`（extract_dai18_text.py，extract_dai21_text.py，rebuild_explain_pdf_2026-11-14.py も同じ）
- 何がおかしいか：`docs/再現の手順.md:56` と `tools/README.md:8` は「git の履歴を整理の前の置き場で読む」と書くが，これらは `ROOT = Path("/home/snics/MER-DA")` に固定し，`ROOT / "DAI" / "docs" / "実験20_設計.md"` などの今のファイルを `read_text()` で読み，`DAI/docs/tools_pending` から `md_tables_dai20` を読み込む．DAI-18・20・21 の結果の文の下書きは，DAI だけでは作り直せない
- どう直すか：文書の理由を「置き場を MER-DA の 8fd53fd の作業木に固定している」と正確に書き，結果の文の下書きが DAI だけでは作り直せないことを再現の手順の限界に足す

### 14．（記録する）地図の中に DAI の外のパスがある
- ファイル：行：`docs/実験の地図.md:6・38`
- 何がおかしいか：3 行目で「パスはすべて DAI の直下からの相対パス」と書いたが，`research_notes/解析論文のストーリー_2026-11-08/…` は MER-DA の直下にあり，DAI の中に無い
- どう直すか：`../research_notes/…`（MER-DA の中でだけ読める）と書く

### 15．（記録する）起動の台本が .venv の python を呼ぶとは限らない
- ファイル：行：`docs/再現の手順.md:9`，`run/dai_eval.sh:21`，`run/dai1_pseudo.sh:10`，`run/dai2_pseudo.sh:12`
- 何がおかしいか：この 3 本は `PY=${PY:-python}` で，PATH の `python` を呼ぶ．手順の「起動の台本は `.venv/bin/python` を呼ぶ」と合わない
- どう直すか：手順に「この 3 本は `PY=.venv/bin/python` を渡す」と書く

### 16．（記録する）そのほかの細かい点
- `tools/latex_tables_dai16_17.py:2・99`：説明の文と，書き出す tex の中の文が `docs/実験16・17_下調べ_結果の表_K5・10・25_2026-11-14.md`（正しくは `docs/結果/…`）を指す．`\\_` の書き方のため規則が当たらなかった．値には影響しない
- `exp/analyze_e79.py:290`：`git grep … -- docs DAI_paper` の `DAI_paper` は DAI の中に無い．E79 の入力の sha256 の先頭 3 つ（`80e6cd05…`，`5778c4b7…`，`3d748088…`）は，どちらの置き方でも `docs/参照/exploratory-analyses.md` で見つかることを確かめた
- 選ぶ段を一から流し直すと，固定のファイルの中の `rule` の文字列（`exp/dai1_select.py:161`，`exp/dai2_select.py:183`，`exp/dai3.py:720・1020`）が「docs/設計/…」になり，コミット済みの固定のファイル（「DAI/docs/…」）と sha256 が合わない．流し直した固定のファイルは値で比べる必要がある．再現の手順の 4 節に足すとよい
- `CLAUDE.md` は `../CLAUDE.md`，`../docs/methods.md`，`../docs/seen-data.md`，`../.claude/agents`（関門の役の定義）に依る．DAI だけのリポジトリではこれらが無い
- `measure/audit_fold_overlap.py` は写していない．DAI の中で引く台本は無い
- 整理の前からのこと：`dai20_eval.SCRIPTS`・`WATCH` に，import の閉包に入る `dai1_select.py` が無い（`exp/dai20_eval.py:76-79`）．整理で変えたものではない
- `slides/2026-11-10-解析論文/scripts/export_figdata.py:16` は MER-DA の直下を `ROOT` にする．資料は再現の対象にしないと README に書いてある

## 検算の台本と出力（checks/整理_2026-11-15/code_run/）
- `check_rewrite.py` → `rewrite_check.tsv`，`rewrite_mismatch.diff`：対応表の行ごとの照合
- `check_inplace.py` → `inplace_check.tsv`，`inplace_mismatch.diff`：exp・run・tools・measure・features の全ファイルへの規則の当て直し
- `all_diffs.sh` → `all_diffs.txt`：元と新の変わった行の全部（93 本，900 行）
- `check_paths.py` → `paths_missing.tsv`：台本の中のパスの実在
- `check_gate.py`：eval_gate.json の照合
- `check_imports.py`：入口からの import のたどり（引数に DAI の置き場を渡す）
- `check_docs.py` → `docs_missing.tsv`：新しい文書の中のパスの実在

## 追記（報告の後）
写しの上で流していた前半の入口の import は，残りの explore_source_value，explore_source_value30，dai16_posthoc，dai17_posthoc を含めてすべて例外なしで終わった（終了コード 0）．どれも ROOT は写しの DAI を指した（dai16_posthoc は ROOT を持たない）．判定は変わらない．
