# 検査の報告

実験ごとに，計算の前の関門（`gate-stats`・`gate-leak`・`gate-compare`・`gate-define`・`gate-probe`・`gate-code`），
査読想定の役（`dai-review-ml`・`dai-review-affect`），計算の後の点検（`code_post*`），結果の文の照合（`claim*`）の報告がある．
手順は `docs/参照/methods.md` 9 節と `CLAUDE.md` の「実験の進め方」．版は名前の末尾の `_v<N>` で，数の大きいものが新しい．

- `E75`〜`E80` は論文の前半の実験の報告で，2026-11-15 に MER-DA の `checks/` から `.md` だけを写した．
- `DAI-1`〜`DAI-3`，`DAI-2_3`，`DAI-eval` は実験 1〜3 と評価の回の報告である．`frozen/eval_gate.json` が一部の報告の sha256 を持ち，評価の段は「その役の最新の版」であることを確かめる．このフォルダに新しい版の報告を足すと，実験 1〜3 の評価の段は止まる．
- `整理_2026-11-15/` は DAI の整理（置き場の変更）の検査の報告である．

**報告の中のパスは，書いた時点の置き場（MER-DA の直下からの相対パス．`DAI/docs/実験20_設計.md` など）のままである．**
報告は書き換えない．今の置き場への読み替えは `docs/整理の対応表_2026-11-15.tsv` で行う．

試走の出力（`code_run*/`，`*_sim*/`，`post_run*/` などの下のデータ）は 2026-11-15 に git から外した（MER-DA の研究室の PC の作業木には残っている）．外したファイルの名前と sha256 は対応表に `untrack_checks_output` として並べた．
