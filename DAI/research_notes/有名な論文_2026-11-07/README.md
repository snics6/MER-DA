# 有名な論文の参照ノート（4 分野．2026-11-07）

研究者の依頼（2026-11-07）「半教師ありDA,蒸留，MER,メタ学習の有名な論文をもう一度調べて，その知識をあなたの内部に蓄えて」に応えて作ったノートである．モデルの重みは変えられないので，原典で確かめた内容をこのフォルダに置き，記憶の索引（`~/.claude/projects/-home-snics-MER-DA/memory/famous-papers-4fields.md`）からここを指す．論文について聞かれたら，まずこのフォルダのノートを読む．

4 分野を 1 体ずつのサブエージェントが調べた．各ノートの構成は同じで，0 節が調べ方と確認の水準，1 節が分野の流れ，2 節が論文ごとの要点（問い，方法と核の式，分かったこと，前提と限界，MER-DA との接点，確認の水準と出典），3 節が分野の標準と著者が未解決と書いたこと，4 節が出典の一覧である．

| ノート | 本数 | 確認の水準 | 主な中身 |
|---|---|---|---|
| `半教師ありDA.md` | 39 本（見出し 34） | ◎ 38，○ 1 | co-training・EM・エントロピー最小化・tri-training・Pseudo-Label・Mean Teacher・MixMatch・FixMatch，Ben-David 2010・Daumé 2007・Office，DAN・DANN・CORAL・ADDA・CDAN・MCD・CBST・SHOT・Tent，MME・APE・CDAC・DECOTA・CLDA・AdaMatch・FADA・CCSA・Donahue 2013・SLA，自己学習の理論（Kumar 2020，Chen 2020，Wei 2021，Cai 2021） |
| `蒸留.md` | 35 本＋短く 5 本 | ◎ 34，○ 1 | Bucilă 2006・Ba & Caruana・Hinton 2015，FitNets・attention transfer・CRD，LUPI・一般化蒸留，Born-Again・DML・Yuan 2020・Müller 2019・Mobahi 2020・Stanton 2021・Cho & Hariharan・Beyer 2022，理論（Phuong & Lampert，Menon 2021，Allen-Zhu & Li，Nagarajan 2023），Noisy Student・Meta Pseudo Labels，weak-to-strong（Burns，Lang，Charikar），モダリティをまたぐ蒸留（Gupta 2016，SoundNet，Hoffman 2016，Albanie 2018，xMUDA，Xue 2023，Du 2023，C2KD），ずれのもとの蒸留（Li ほか 2026） |
| `MER.md` | 38 本 | ◎ 33，○ 5 | データセット 9（IEMOCAP，MOSI，MOSEI，MELD，AFEW，CREMA-D，RAVDESS，MSP-Podcast，CH-SIMS），総説 3，融合 8（TFN，LMF，MFN，MulT，MAG-BERT，MISA，Self-MM，UniMSE），会話 3，モダリティの釣り合い 4（gradient blending，OGM-GE，modality competition，UMT/UME），欠けたモダリティ 1，音声エンコーダ 4（wav2vec 2.0，HuBERT，WavLM，SUPERB），感情の表し方 2，コーパス間 2．3.2 節に論文ごとの評価の仕方（感情の数，話者の分け方，WA と UA）の表 |
| `メタ学習.md` | 35 本＋表で 15 本 | 本項目 ◎ 33，○ 2 | Baxter 2000・Maurer，Siamese・Matching・ProtoNet・Relation，MAML・Reptile・iMAML・ANIL・MetaOptNet・R2D2，ベイズの見方（Grant 2018 ほか），基準線の批判（Chen 2019，Tian 2020，Dhillon 2020，Meta-Dataset，BSCD-FSL，Hu 2022），偏った正則化の理論（Denevi 2018・2019，ARUBA，Konobeev 2021），MLDG・MetaReg，二段の最適化と重みの学習，感情認識でのメタ学習 |

確認の水準の記号：◎＝本文を読んだ，○＝抄録か書誌の頁で確かめた，△＝書誌のみ．各ノートの「MER-DA との接点」と 3 節末の「読み」は，サブエージェントの推測で検定していない．論文や研究者への説明で事実として使う前に，原典の該当箇所を確かめる．既に精読したノート（`DAI_paper/参照論文/要点_問い1〜3`，`research_notes/半教師あり DA の先例/ssda.md`，`docs/related-work.md`）と重なる論文は，各ノートで要点を縮めて元のノートを指している．

## 4 分野をまたいで MER-DA に関わる点（本体が 4 つのノートの 1 節と 3 節を読んでまとめた．推測を含む）

1. **単独の分類器の確率の積は，つないだ特徴の上の線形の softmax と同じ形である．**co-training の P(c|x) ∝ P(c|x₁)P(c|x₂)（`半教師ありDA.md` 2 節）と，積を正規化した式（`蒸留.md` 3.5 節）．このため融合の分類器は積の疑似ラベルの関数そのものを表せる．weak-to-strong の理論（Charikar 2024，Lang 2024）は，学ぶ側が疑似ラベルの誤りを表せないことを得の源とするので，融合の分類器が疑似ラベルから得る分は小さいと読める（`蒸留.md` 3.5 節．検定していない）．
2. **疑似ラベルの理論はどれも，疑似ラベルを作るモデルがそれなりに正確なことか，クラスの割合が変わらないことを仮定する**（Kumar 2020，Chen 2020，Wei 2021，Cai 2021．`半教師ありDA.md` 3.3 節）．感情の割合がコーパスで違う MER-DA では，後者が成り立たない．
3. **蒸留で何が伝わるかは決着していない．**クラスの似かたの情報（Hinton，Müller 2019），軟らかい目標による正則化（Yuan 2020，Mobahi 2020），両方のモダリティに共通の特徴（Xue 2023，xMUDA）などの説明が並ぶ（`蒸留.md` 3.2 節）．蒸留元の正解率は蒸留先の成績を決めない．
4. **融合の得が最良の単モダリティより小さい例は多い**（DialogueRNN，MMGCN の MELD，OGM-GE，Wang 2020．`MER.md` 3.1 節）．ただし釣り合いの研究の多くはエンコーダも学ぶ設定で，凍結した特徴の MER-DA にはそのまま当てはまらない．コーパスをまたぐずれのもとで融合の得を扱った研究は，このノートの範囲では見つからなかった（`MER.md` 3.3 節）．
5. **MER の論文は評価の仕方がばらばらである．**IEMOCAP だけでも，4 感情の 2 値，6 クラス，4 クラスの WA と UA の系統があり，話者を分けない論文（OGM-GE の CREMA-D，Xue 2023 の RAVDESS）もある（`MER.md` 3.2 節）．比べる前に条件をそろえる．
6. **MER-DA のソースの平均への L2 の引き戻しは，メタ学習の理論の「偏った正則化」と同じ形である**（Denevi 2018・2019，iMAML，Grant 2018 の MAP，Konobeev 2021．`メタ学習.md` 3.5 節）．ただし課題の数 T = 3 で意味のある保証を出す理論は無い（同 3.3 節の表）．
7. **凍結した特徴の上の単純な基準線は強い**（Chen 2019，Raghu 2020，Tian 2020，Guo 2020．`メタ学習.md` 3.2 節）．DomainBed では leave-one-domain-out による設定の選択が学習ドメインの検証による選択より低かった．感情認識でソースのコーパスを仮のターゲットにして K ごとに設定を選ぶ研究は見つからなかった（`メタ学習.md` 1.8 節）．
8. **SSDA の強い手法の多くはデータ拡張と特徴抽出器の学習を前提にする**ので，凍結したエンコーダの上の線形の分類器に移せる部品は限られる（エントロピー最小化，閾値つきの疑似ラベル，クラスごとの閾値，多様性の項，分布合わせ，CORAL，ソースの重みへの罰則．`半教師ありDA.md` 3.3 節）．

## 確かめられなかったこと（各ノートの 0 節と終わりの報告から）

- 半教師あり DA：van Engelen & Hoos 2020 は抄録と書誌だけ．CORAL の AAAI 版の頁，CBST と Deep CORAL の LNCS の巻，Oliver 2018 の頁，Tent と Wei 2021 の OpenReview の頁は確かめていない．Lee 2013 は公開の写しで読んだ．BiAT，MCL，UniSSDA は書誌だけで外した．基盤モデルの凍結した特徴を使う SSDA で広く引かれる論文は見つからなかった．
- 蒸留：Vapnik & Vashist 2009 の本文は取れず，式は同じ著者の JMLR 2015 で確かめた．Li，Han，Yang，Tong 2026 の arXiv の番号は確かめていない．Meta Pseudo Labels は CVF と arXiv で著者の数が違う．どの論文も証明と付録の全体は読んでいない．
- MER：Dhall 2012（AFEW の原論文）の本文は読めず，EmotiW 2013 で書いた．MSP-Podcast は 2025 年の最終版で代えた．Schuller 2010 の TAC の本文は未読．EmotiW 2017 版の AFEW の分割（773/383/653）と MER-DA の 1,156 行の対応は未確認．
- メタ学習：Ravi & Larochelle 2017 と Chopra 2021 の本文は読めなかった．Schmidhuber 1987 は書誌のみ．
