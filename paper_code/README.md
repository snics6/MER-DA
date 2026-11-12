# paper_code：DAI に出す解析論文の実験コード（2026-11-13 に写した）

論文に使う実験のコードだけを，元のリポジトリから写したフォルダである．後で別のリポジトリとして GitHub に上げることを想定している．

- `src/` の中は，元のリポジトリと同じ相対パスで写した（`src/exp/...`，`src/DAI/exp/...`，`src/measure/...`，`src/features/...`）．台本は自分の位置から ROOT を決め，`sys.path` に同じフォルダや `measure/` を足して import するので，置き場を変えずに写すと import がそのまま働く．
- `MANIFEST.tsv`：ファイルごとに，論文のどの節で使うか，写した時の sha256，結果を計算したコミット，そのコミットから台本が変わったかを書いた．
- `copy_info.json`：写した時の元のリポジトリのコミット．
- 論文の節と実験の対応は，元のリポジトリの `DAI/docs/実験の地図.md`．

## 入っていないもの

- データ（特徴 `data/cache/*.npz`，発話一覧 `data/metadata/`，分割 `DAI/splits/`）．コーパス（IEMOCAP，MOSEI，MELD，AFEW など）の利用条件があるので，公開するかは別に決める．作り方は元のリポジトリの `data/README.md` と `run/build_data.sh`．
- 結果（`results/`）．主張の根拠の控えは元のリポジトリの `DAI/evidence/`．

## 公開する前に決めること

- 使用許諾（ライセンス）．
- 発話一覧と分割の扱い（コーパスの利用条件）．
- コメントと文書が日本語であること（英語の README を足すか）．
- 台本に残る研究室の PC の絶対パスや，検査役の報告への参照．

## 作り直し

元のリポジトリの直下で次を流すと，`src/` と `MANIFEST.tsv` を作り直す．

```
.venv/bin/python DAI/evidence/make_paper_code.py
```

入口の台本の一覧は `DAI/evidence/make_paper_code.py` の `ENTRIES`．
