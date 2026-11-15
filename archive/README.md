# 使わなくなった旧版のもの（2026-11-15 に MER-DA の直下から移した）

研究者の指示（2026-11-15）：「今の想定ではMER-DAの直下にDAIがあり，それ以外の直下ファイルはあんまり使っていない？？使っていなければアーカイブに入れてほしい．見づらいので．ただし後で困ったことにならないように注意．」

論文の実験に要るものは，移す前に `DAI/` に写してある（台本 `DAI/exp/`，部品 `DAI/measure/`・`DAI/features/`，発話一覧 `DAI/data/metadata/`，`DAI/requirements.txt`，論文の前半の検査の報告 `DAI/checks/E75`〜`E80`）．
ここに移したものは，DAI の台本も論文のセッションも引いていないことを確かめた．

| 置き場 | 中身 |
|---|---|
| `exp/`，`run/` | 旧版（TAC 向け，E1〜E87）の実験の台本と起動の台本．論文の前半の 44 本は `DAI/exp/` にバイトまで同じ写しがある |
| `measure/`，`features/` | 読み込み・分割・ずれの指標の部品と，特徴を作る台本．`DAI/measure/`・`DAI/features/` に同じ写しがある（`audit_fold_overlap.py` だけは写していない） |
| `data/` | `metadata/`・`README.md`・`weights.sha256`・顔の座標（`DAI/data/` に同じ写し）と，DAI が使わない特徴（`cache/wave_*.npy`・`wave_*.csv`，`cache/parts/`）．論文に使う特徴の npz は `DAI/data/cache/` に移した |
| `checks/` | 旧版の実験の検査の報告（E1〜E87）．git に入っていなかった E84・E86・E87 の報告もこのときに入れた．試走の出力は git に入れていない |
| `paper/` | TAC 向けの旧版の原稿（2026-10-27〜10-28．2026-10-31 からは従わない） |
| `slides/` | 旧版の発表資料（2026-10-27〜）．DAI の発表資料は `DAI/slides/` |
| `paper_code/` | 2026-11-13 に作った論文のコードの写し．DAI に一本化したので使わない |
| `reports/` | 2026-10-27 の調べ直しの報告 |
| `requirements.txt` | `DAI/requirements.txt` と同じ中身 |
| `A theory of learning from different domains.pdf` | 中核の理論の論文（Ben-David ほか 2010） |
| `tmp_dai18_selftest_2026-11-14/` | 2026-11-14 11:09 に実験 18 の試走が MER-DA の直下に残した一時フォルダ（git の外） |

ここの台本の中のパスは MER-DA の直下からの相対パスのままなので，そのままでは動かない．流し直すときは，移す前のコミット（`8c297ed`）を取り出して使う．
git に入れるのは，移したときに git に入っていたファイルだけである（`archive/.gitignore`）．
