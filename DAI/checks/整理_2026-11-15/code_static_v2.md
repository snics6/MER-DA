# 整理の直しの再検査（2026-11-15，a06f779 → 7866565：7748bc0・89ef337・7866565）

判定：通す

## 判定
通す．台本の変更は置き場の行と `logs/` の作成と固定のファイルの鍵の読み替え 1 行だけである．前半の台本 44 本は整理の前とバイトまで同じで，`frozen/eval_gate.json` は正しく作り直され，MER-DA の中でも DAI だけのリポジトリでも `dai1.check_eval_gate` が通る．`.gitignore` の書き換え後に DAI だけを根にしても，追跡中のファイルは 1 本も落ちない．残る点（7 節）は文の数の食い違いなど記録の水準で，計算と再現の手順を止めない．ただし `paper/scripts/tab_mixing.py:12`（前の指摘 3）は直っていないので，論文のセッションに伝える必要がある．

GPU は使っていない．結果の値は開いていない．評価の回のコミットを確かめるために，MER-DA の `results/dai1・dai2・dai3` の `provenance_log*.jsonl` の commit の欄だけを数えた．検算の台本は `checks/整理_2026-11-15/code_run/`（`check_gate_v2.py` ほか）にある．

## 1．台本の変更が置き場の行と logs の作成だけか
`git diff a06f779 HEAD -- DAI` の台本の差分を全部読んだ．

- `run/dai_eval.sh:19`，`run/dai1_pseudo.sh:7`，`run/dai2_pseudo.sh:7`：`cd "$(dirname "$0")/../.."` を `cd "$(dirname "$0")/.."` に直した．DAI だけのリポジトリで 3 本の `cd` の先を計算すると DAI の直下になり，そこに `exp/dai1.py` がある
- `run/e77.sh`・`e77b.sh`・`e77h.sh`・`e78.sh`・`e78b.sh`・`e78h.sh`・`dai14_run.sh`・`dai15_run.sh`：`cd` の次に `mkdir -p logs` を 1 行足しただけである
- `tools/latex_tables_dai16_17.py:2`：説明の文のパスだけを直した
- `tools/remap_eval_gate.py:23-25`：`before_reorg` があるときは止まらずに，元の鍵（`_meta.before_reorg`）から作り直すようにした．計算の台本ではない
- `exp/dai21_eval5.py:333`：`load_frozen` で鍵の頭の `DAI/` を外す 1 行を足した．`frozen/dai21_frozen.json` の `source_sha256` の鍵 5 個のうち，頭に `DAI/` があるのは `DAI/frozen/dai20_frozen.json` だけである．直す前は，整理後の DAI に `ROOT / "DAI/frozen/dai20_frozen.json"` が無いので，sha256 の照合が黙って飛んでいた．直した後は `frozen/dai20_frozen.json` を指し，照合が元のとおり働く．ほかの 4 個（`results/…`）は変わらない．`dai20_eval.load_frozen`（`exp/dai20_eval.py:170`）の鍵は `results/…` だけで，同じ直しは要らない
- 前半の台本 38 本の差分は，整理で書き換えた説明の文を元に戻したものである（2 節）

計算の式・定数・格子・引数の既定の値・分岐が変わった行は無い．DAI だけのリポジトリで `bash -n`（run の 22 本）と `exp/dai21_eval5.py` の構文の検査が通った．

## 2．前半の台本 44 本が 8fd53fd の exp/ とバイトまで同じか
`reorg_2026-11-15.py` の `FRONT_EXP` の 44 本について，MER-DA の `git show 8fd53fd:exp/<名前>.py`，`git show HEAD:DAI/exp/<名前>.py`，作業木のファイルの 3 つの sha256 を比べた．44 本すべてで一致した．`exp/` の差分 39 本は，前半の 38 本と `dai21_eval5.py` である（前半の残り 6 本は整理で何も変わっていなかった）．書き換えの記録の tsv では，前半の 44 本の規則ごとの行を消し，「restored byte-identical」の 1 行にまとめてある．

## 3．frozen/eval_gate.json と dai1.check_eval_gate
`check_gate_v2.py`（前回の `check_gate.py` の比べる先を HEAD にしたもの）で確かめた．

- `required` の 40 本の鍵は，すべて HEAD にあり sha256 が一致する
- `_meta.before_reorg` は 8fd53fd の eval_gate.json の `required`・`approvals` と一致する．元の `_meta.rule` も残る
- 元の鍵と新しい鍵は 1 対 1 に対応する
- `_meta.changed_after_gate` は 14 本で，計算し直した一覧と一致する．`docs/設計/実験1_設計.md` と，書き換えた台本 13 本（exp 11 本，`run/dai_eval.sh`，`run/eval_procedure.md`）である．`exp/compare_dai.py` と `compare_dai2.py` は元に戻したので，一覧から消えた
- `approvals` の 21 個は新しい置き場を指し，報告の中身は元と同じである

`dai1.check_eval_gate("frozen/dai1_frozen.json")` を関数として呼んだ．`/home/snics/MER-DA-reorg/DAI`（MER-DA の中のフォルダ）と，HEAD の DAI を `git archive` で取り出して DAI の `.gitignore` だけで新しいリポジトリにした写し（`scratchpad/sa_v2/DAI`）の両方で通り，戻り値は同じ `a6ba9623…` だった．`dai1.git_commit()` はどちらでも `dirty=False` を返した．評価の段そのものは流していない．

## 4．DAI の .gitignore だけを当てたとき追跡中のファイルが落ちないか
- HEAD の DAI を `git archive` で取り出し，`git init` → `git add -A` をした．追跡するファイルは 1,498 本で，HEAD の `DAI/` の 1,498 本と一覧が完全に一致した．落ちたものも増えたものも無い．前回落ちていた検査の台本 271 本（`checks/DAI-eval/code_run/m3_frozen_keys.py`，`checks/DAI-1/code_run_post/c9_predcheck.py` を含む）も残る
- `checks/DAI-21/post_tools_run/kshot_ieo_counts.csv` は，`!` の例外の行で追跡に入った（前の指摘 11 が直った）
- 逆の向きも見た．MER-DA の main の作業木で `DAI/` の下にある追跡外のファイル 6,513 本に，新しい `.gitignore` を当てた．除外されないのは 24 本で，どれも DAI-20・21 の検査役の台本（.py・.sh），表の下書き（.md），`.done`・`failed` の印のファイルである．データ（npz・csv・json）は除外される
- 整理で git から外した 769 本に新しい `.gitignore` を当てると，465 本が除外され，304 本は除外されない（7 節の 2）

## 5．手順書・README・tools/README の文が台本と合うか
- `docs/再現の手順.md` 1 節の 2：`dai1_pseudo.sh`・`dai2_pseudo.sh`・`dai_eval.sh` が `PY`（既定は PATH の `python`）を呼ぶという文は，3 本の `PY=${PY:-python}` と合う
- 1 節の研究室の PC のリンクの文：「main に入れた後」と断ってあり，前回の食い違いは解消した
- 4 節の実験 2・3 の評価の段が止まる理由の文：`exp/dai2.py:182`（固定のファイルの `script_sha256` と今の台本を比べる）と `exp/dai3.py:1877`（`fz3["_meta"]["script_sha256"]` と今の台本を比べる）と合う．評価の回のコミット `ba98974` も確かめた．MER-DA の `results/dai2/provenance_log_eval_*.jsonl` の 40 行と，`results/dai3/provenance_log_eval_*.jsonl` の 4 行はすべてこのコミットだった．`results/dai1/provenance_log_eval_*` では，各ファイルの 51 行のうち 5 行がこのコミットで，残りは 896136c（45 行）と 922446c（1 行）である．どの行が評価の段の行かは見分けていない
- 4 節の集計の文：`exp/dai1_desc.py:1361・1365` と合う
- 5 節の前半の文：「中身は整理の前と同じ（バイトまで同じ）」は 2 節で確かめたとおりである．言葉の種類の値がずれることと E76 の `code_hash` の話は code_repro_early.md の範囲で，この検査では確かめていない
- 7 節の計算機の違いによるずれの数値は code_repro_late.md の範囲で，この検査では確かめていない
- `tools/README.md` の extract 系の道具の文：作業の根を `/home/snics/MER-DA` に固定し，今のファイルも整理の前の置き場で読むという説明が，台本（`tools/extract_dai20_text.py:44-57・119-127`）と合う．`run/dai16_probe.sh` を整理のときに書いたという断りも入った
- `docs/実験の地図.md:6`：research_notes が DAI の外にあることを書き足した（`:38` は 7 節の 5）
- 食い違いが 1 つある（7 節の 1）

## 6．前の指摘の状態
| 前の番号 | 状態 |
|---|---|
| 1 cd の先 | 直った（1 節） |
| 2 写しで台本が落ちる | 直った（4 節） |
| 3 `paper/scripts/tab_mixing.py:12` の `parents[3]` | **残っている**（HEAD でも `parents[3]`） |
| 4 results/MANIFEST.tsv | 依頼のとおり指摘から外す（main に入れるときに行う） |
| 5 再現を確かめたという文 | 直った（7 節の文を消し，code_repro_early・late の結果を書いた） |
| 6 リンクの説明 | 直った |
| 7 logs/ を作らない台本 | 直った（8 本） |
| 8 集計し直せない範囲 | 直った（4 節に実験 1〜3 を足した） |
| 9 対応表と記録 | おおむね直った（kind `new` の 14 行，元に戻した道具と手で直した道具の sha256）．7 節の 3 |
| 10 パス指定の無い git status | 残っている（`exp/compare_dai.py:963`，`compare_dai2.py:862`）．前半の台本はバイトまで同じに保つと決めたので，直さないのが筋である．再現の手順の限界に 1 行あるとよい |
| 11 kshot_ieo_counts.csv | 直った |
| 12 run/dai16_probe.sh | 断りを tools/README に書いた |
| 13 extract 系の道具の説明 | 直った |
| 14 地図の research_notes | `:6` は直った．`:38` は残る（7 節の 5） |
| 15 PY の文 | 直った |
| 16 細かい点 | `tools/latex_tables_dai16_17.py:99` の書き出す文は残る（7 節の 4）．analyze_e79 の `DAI_paper` のパス指定，CLAUDE.md の `../` への依存，`dai20_eval.WATCH` の `dai1_select.py` は前回のまま（記録） |

## 7．残る点（どれも記録する）

### 1．手順書の本数が gate と合わない
- ファイル：行：`docs/再現の手順.md:60`
- 何がおかしいか：「置き場の行を書き換えた台本 15 本」とあるが，前半の台本を元に戻したので `_meta.changed_after_gate` の台本は 13 本（exp 11 本，`run/dai_eval.sh`，`run/eval_procedure.md`）である．`run/eval_procedure.md` は手順の文書なので，「台本と手順の文書 13 本」と書くのが正確である
- どう直すか：「15 本」を「13 本（exp の 11 本，`run/dai_eval.sh`，`run/eval_procedure.md`）」にする

### 2．archive の下の試走のデータが除外されない
- ファイル：行：`.gitignore:30-52`
- 何がおかしいか：`archive/checks/**` について除外するのは npz・npy・pkl だけで，`archive/checks/*/code_run*/` の下の csv・json・txt・log は除外されない．整理で git から外した 304 本（`archive/checks/DAI-4/code_run/gate_test/*.json` など）は，この型に当たる．main に入れたときに，これらが手元に残っていると追跡外のファイルとして現れ，`git add -A` で入る．reorg-dai の作業木には無いので，今の写しには影響しない
- どう直すか：`archive/checks/*/code_run*/**/*.csv`・`*.json`・`*.txt`・`*.log` の 4 行を足す

### 3．対応表の sha256 が 1 行古い
- ファイル：行：`docs/整理の対応表_2026-11-15.tsv` の `DAI/docs/再現の手順.md` の行（kind `new`）
- 何がおかしいか：new_sha256 が HEAD の `再現の手順.md` と合わない．最後のコミットで 4 節・7 節を直した後に更新していない．書き換えの記録の tsv の行も合わないが，表が自分自身を含むためで，避けられない
- どう直すか：`再現の手順.md` の行を HEAD の sha256 に直す（直すと対応表の中身が変わるが，対応表は自分の行を持たないので循環しない）

### 4．LaTeX の表に書き出す文のパス
- ファイル：行：`tools/latex_tables_dai16_17.py:99`
- 何がおかしいか：書き出す tex の中の `\texttt{docs/実験16・17\_下調べ\_結果の表\_K5・10・25\_2026-11-14.md}` に `結果/` が無い．値には影響しない
- どう直すか：`docs/結果/` にする

### 5．地図の 38 行目
- ファイル：行：`docs/実験の地図.md:38`
- 何がおかしいか：`research_notes/解析論文のストーリー_2026-11-08/inventory_D_DAI.md` に，6 行目と同じ「DAI の外」の断りが無い
- どう直すか：同じ断りを足す

### 6．論文の表の台本（前の指摘 3．DAI の整理の枝の外の担当）
- ファイル：行：`paper/scripts/tab_mixing.py:12`
- 何がおかしいか：`ROOT = Path(__file__).resolve().parents[3]` は MER-DA の直下を指し，そこの `results/e76_analysis/e76a_1_kind.csv` を読む．main に入れた後は MER-DA の `results/` からのリンクで読めるが，DAI だけのリポジトリでは読めない
- どう直すか：`parents[2]` にする．`paper/` は論文の相談のセッションが書く置き場なので，そちらに伝えてから直してもらう
