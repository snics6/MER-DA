# 投稿先 Discover Artificial Intelligence の要項と掲載論文の形（2026-10-31 調べ）

この文書は，投稿先 Discover Artificial Intelligence（Springer Nature，ISSN 2731-0809．以下 DAI）の要項と，掲載論文の章立て・長さをまとめたものである．
誌の公式ページ（投稿規定，目的と範囲）は研究者がブラウザから貼った本文を使った．公式ページと他の情報源が食い違うときは公式ページを採る．

出典（`sources/2026-10-31/`）：

| ファイル | 中身 |
|---|---|
| `submission-guidelines.txt` | 投稿規定の本文（https://link.springer.com/journal/44163/submission-guidelines ．研究者が貼ったもの） |
| `aims-and-scope.txt` | 目的と範囲の本文（https://link.springer.com/journal/44163/aims-and-scope ．研究者が貼ったもの） |
| `doaj_journal.json` | DOAJ の登録情報（DOAJ の API．DOAJ の更新日 2026-09-17） |
| `openalex_source.json` | OpenAlex の誌の情報 |
| `openalex_topic_articles.tsv` | 題名で探した DAI の掲載論文 90 本の一覧（OpenAlex） |

---

## 0. この論文に関わる要点

1. **査読は研究の重要さを評価しない．**査読者には「結果が正確に報告されているかを確かめ，研究に根本的な欠陥があれば不採択にする」ことを求め，「研究の重要さ（significance）は評価させない」と書いてある（投稿規定 Step 3）．TAC のように新しさや影響の大きさで落とす誌ではない．その代わり，数値の報告の正確さと，設計の欠陥（漏れ，比較の不公平，統計の誤り）が審査の中心になる
2. **論文の種類は Research か Methodology が候補である．**Methodology は「新しい実験の方法・計算の方法・検定・手順，または既存の方法の改良版」で，「今あるものより明らかに進んでいること」と「十分に試されていること」を求める．どちらにするかは主張が決まってから決める
3. **長さの制限は書かれていない．**要旨は 250 語未満である．Review だけは「どの長さでも受け付ける」と明記している
4. **原稿に入れる申告**：資金の申告は必須（無いと差し戻される）．人のデータを使う研究は倫理の申告が要る．データの入手先の申告（Data availability）も全員に必須である
5. **書式**：Word か LaTeX（Springer Nature の雛形を推奨．ZIP で送り，誌が PDF を作る）．文字は 12 pt 以上，見出しは 3 段まで，引用は角括弧の番号，参考文献は NLM の書き方（Vancouver 方式の元）を推奨，DOI は完全な URL で書く

## 1. 誌の基本

| 項目 | 値 | 出典 |
|---|---|---|
| 出版社 | Springer Nature の Discover シリーズ．2021 年から完全なオープンアクセス | DOAJ |
| 査読 | 査読者は著者の名前と所属を見る．著者には査読者の名前を伏せる（single-anonymous）．2 人以上が査読する | 投稿規定 Step 3 |
| 著作権 | 著者が保持し，Springer Nature に出版の許諾を与える．掲載時に Creative Commons の使用許諾が付く | 投稿規定 Step 4（DOAJ の「著者は著作権を保持しない」は公式ページと食い違うので採らない） |
| 使用許諾 | CC BY（ほかに CC BY-NC-ND） | DOAJ |
| 掲載料（上限） | 1,390 ユーロ，1,690 米ドル，1,190 英ポンド．国による免除の制度あり | DOAJ（2026-09-17 更新．OpenAlex の 1,590 米ドルは古い値と見て採らない．最終の値は公式の Open access publishing のページで確かめる） |
| 投稿から掲載まで | 8 週（出版社が DOAJ に申告した平均） | DOAJ |
| 投稿の仕組み | Snapp | 投稿規定 |
| 投稿に要るもの | 原稿，250 語未満の要旨，カバーレター（研究の背景と重要さ，この誌に合う理由を短く書く）．図・表・補足資料は任意 | 投稿規定 Introduction |

## 2. 目的と範囲（`aims-and-scope.txt`）

- 「AI の理論・方法論・応用のすべての面」を扱う分野横断の誌である
- 理論と方法論の項目に「machine learning and deep learning」がある．応用の項目に「AI in perception and multimedia sensing」「AI in image analysis and video processing」「human-robot/machine interaction」がある
- 感情認識（emotion recognition），感情計算（affective computing），転移学習，ドメイン適応は項目に名前が出ていない．項目は「含むがこれに限らない」と書いてあるので，範囲から外れるわけではない
- 歓迎する研究として「AI の問題と応用の新しい見方を出し，その価値と有効性を示す研究」を挙げている

## 3. 論文の種類（投稿規定 Step 1）

| 種類 | 定義（要約） | この研究との関係 |
|---|---|---|
| Research | 未発表の新しい科学的な結果 | 候補 |
| Methodology | 新しい実験・計算の方法，検定，手順，または既存の方法の改良版．今あるものより明らかに進んでいること，十分に試されていることを求める | 候補（手法を主張の中心にするとき） |
| Brief Report | 重要な新しい観察を短く報告する．Research ほど作り込んでいない結果でもよい | — |
| Registered Report | データを集めて解析する前に，問いと方法の質を査読する | — |
| Case Study，Comment，Data Note，Matters Arising，Perspective，Review | 事例，論評（多くは編集者の依頼），データの記述，掲載論文への意見，展望，総説 | 対象外 |

- 長さの制限はどの種類にも書かれていない．Springer Nature 共通の案内（[支援ページ](https://support.springernature.com/en/support/solutions/articles/6000271469-is-there-a-word-limit-for-springer-nature-journal-articles-)，2026-09-24 更新）も「誌が定めない限り制限しない．簡潔に書く」としている

## 4. 査読の流れ（投稿規定 Step 3）

1. 投稿時に技術的な検査（ファイルの形，申告の有無など）を受ける．図の不備が査読の前に差し戻される主な理由だと書いてある（Step 6）
2. 誌の内部の編集者が適合性を検査する
3. 外部の編集委員か内部の編集者が，2 人以上の査読者を呼ぶ
4. 大きな修正は元の査読者が見直し，小さな修正は編集者だけが見ることが多い
5. 査読者には重要さを評価させず，結果が正確に報告されているか，研究に根本的な欠陥がないかを見させる
6. 査読者が見つからないときは，Springer Nature が外部の業者に査読者探しを頼むことがある．査読者が見つからなければ投稿を取り下げることがある

## 5. 原稿に入れる申告と，Snapp で同意する方針（投稿規定 Step 2）

| 項目 | 扱い |
|---|---|
| 資金（Funding statement） | **必須**．受けていない場合もその旨を書く．無いと差し戻されうる |
| 倫理（Ethics statement） | 人や動物，そのデータを扱う研究で必要．詳細は Discover シリーズの編集方針の Research ethics の節（未読） |
| データの入手先（Data availability） | **全員に必須**．解析したデータの入手先，公開されていないデータの入手方法を書く |
| 著者の貢献，利益相反，二重投稿，著者資格 | Snapp で申告・同意する |
| 他者の図・表・文の使用 | 既発表のものは著作権者の許可が要る |
| 謝辞 | 任意 |

## 6. 書式（投稿規定 Step 6）

- **ファイル**：Word（図表は本文中．式は数式エディタか MathType）か，LaTeX（図表とまとめて ZIP．誌が PDF に組む）．Springer Nature の LaTeX 雛形を推奨する
- **雛形**：Springer Nature LaTeX Template 第 3.1 版（2024 年 12 月．[Overleaf](https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd)）．クラスは `sn-jnl`，主ファイルは `sn-article.tex`．参考文献の様式は `sn-mathphys-num`（既定）のほかに `sn-vancouver-num` などがある．誌が NLM の書き方を推奨するので，`sn-vancouver-num` が合う見込みである（雛形を置くときに確かめる）
- **文字**：全体で同じ書体，12 pt 以上．Helvetica か Arial を推奨する．雛形の既定の大きさで 12 pt 以上になるかは雛形を置くときに確かめる
- **見出し**：3 段まで
- **略語**：初出で定義し，以後同じ形で使う
- **脚注**：ページの下に置く．参考文献の引用だけの脚注，書誌の詳細を書く脚注，図表を含む脚注は使わない
- **引用と参考文献**：本文では角括弧の番号（例：[1-3, 7]）．一覧は番号順で，発表済みか採録済みの文献だけを載せる（私信と未発表の文献は本文だけで触れる）．DOI は `https://doi.org/...` の完全な形で書く．採録後に誌の様式へ直されることがある
- **表**：アラビア数字の番号を付け，本文で番号順に引く．表題を付ける．表の脚注は小文字の上付き（有意の印は星）
- **図**：本文中に置く．線画は EPS，写真は TIFF を推奨（MS Office の形も可）．フォントを埋め込む．ファイル名は `Fig1.eps` の形．図の中の文字は Helvetica か Arial の 8〜12 pt で大きさをそろえる．図の中に題を書かない．図の説明は「Fig. 1」で始め，番号の後と説明の末尾に句読点を付けない．線は 0.1 mm 以上．解像度は線画 1200 DPI，写真 300 DPI，組み合わせ 600 DPI 以上．付録の図は本文の番号を続け，補足資料の図は別の番号にする
- **見やすさ**：色だけでなく模様でも区別する．図の文字のコントラスト比は 4.5:1 以上
- **補足資料**：査読の対象になり，変換や編集なしに公開される．`ESM_1.pdf` のように名前を付け，本文で「Online Resource 1」「ESM 2」の形で引く
- **校正**：採録後の校正で，新しい結果や値の訂正などの大きな変更は編集者の承認なしにできない

## 7. 掲載論文の形（Europe PMC に全文がある研究論文 6 本）

DAI の研究論文のうち Europe PMC で全文の XML を読めた 6 本を数えた．分野はばらばらで，MER や DA の論文は含まない．

| DOI | 題名（略） | 要旨の語数 | 本文の章（上位の節） | 本文の語数 | 図 | 表 | 参考文献 | 受理までの期間 |
|---|---|---|---|---|---|---|---|---|
| 10.1007/s44163-026-01546-z | LLM 推薦の頑健さの評価 | 181 | Introduction / Background and motivation / Automated red teaming method / Experimental setup / Results / Discussion / Conclusion | 5,989 | 11 | 13 | 44 | 2026-02-26 受付 → 05-22 受理（約 3 か月） |
| 10.1007/s44163-026-01768-1 | 医用画像の前処理の体系的な調査 | 209 | Introduction / Materials and methods / Results / Discussion / Conclusion | 7,032 | 8 | 3 | 33 | 2026-03-30 → 07-07（約 3 か月） |
| 10.1007/s44163-026-02600-6 | 代理モデルと分位点の補正 | 149 | Introduction / Case study / Methodologies / Results and benchmark / Conclusion | 5,240 | 11 | 1 | 31 | 記載なし |
| 10.1007/s44163-025-00713-y | 画像の圧縮表現と検索 | 219 | Introduction / Literature review / Methodology / Implementation and validation / Conclusion and future work | 7,092 | 14 | 6 | 48 | 記載なし |
| 10.1007/s44163-025-00503-6 | 行動認識のクラスタリング | 233 | Introduction / Related work / Methodology / Model implementation / Validation and results / Visualization / Conclusion | 7,953 | 15 | 4 | 66 | 2024-09-07 → 2025-08-25（約 11.5 か月） |
| 10.1007/s44163-024-00199-0 | 再標本化が臨床予測に与える影響 | 278（Purpose / Methods / Results / Conclusion の構造化要旨） | Introduction / Materials and methods / Results / Discussion / Strengths and limitations / Conclusion | 3,511 | 3 | 11 | 39 | 記載なし |

- 本文の語数は，参考文献，申告の節，図と表と式を除いて数えた概数である
- 要旨は 149〜278 語で，今の規定の 250 語未満を超える例（278 語，2024 年）がある．規定が変わった可能性があるので，今の規定に従う
- 章立ては「Introduction → 方法 → 結果 → (Discussion) → Conclusion」の形が多い．関連研究を独立の章にする論文と，しない論文がある
- 申告の節は本文の後に置く．見出しは Author contributions，Funding，Data availability，Code availability，Declarations（Ethics approval，Consent to participate，Consent for publication，Competing interests）である．公開のデータだけを使う論文は，Ethics approval に「公開のデータセットの二次的な解析で，元の提供者が倫理の承認と同意を得ている」と書いている（10.1007/s44163-026-01768-1）

## 8. DAI に載った近い主題の論文（OpenAlex の題名検索．一覧は `sources/2026-10-31/openalex_topic_articles.tsv`）

題名に次の語を含む DAI の論文の数：emotion 24 本，multimodal 107 本，transfer learning 36 本，distillation 7 本，affective 7 本，domain adaptation 3 本，semi-supervised 1 本，cross-corpus 0 本．

- 感情認識の論文は応用が中心である（EEG，音楽，教育，ホテルの口コミ，顔の熱画像など）．コーパスをまたぐ音声・映像・言葉の感情認識の論文は見つからなかった
- DA の論文は 3 本で，どれも応用の枠組みである（CORAL による学生の成績予測 10.1007/s44163-026-02002-8，眼底画像の教師ありの DA 10.1007/s44163-026-02067-5，電力設備の画像 10.1007/s44163-026-01646-w）
- 近い題材：感情の状態の認識で異なる表現を融合する論文（10.1007/s44163-026-02285-x，2026-09），文化をまたぐ感情分析で敵対的学習と LLM からの蒸留を使う論文（10.1007/s44163-026-01879-9，2026-08）
- 中身は題名しか見ていない．引用するかどうかは本文を読んでから決める

## 9. 誌の規模（OpenAlex．2026-10-31 取得）

- 掲載数は年ごとに急に増えている：2021 年 16 本，2022 年 27 本，2023 年 45 本，2024 年 116 本，2025 年 496 本，2026 年（10 月まで）1,709 本
- 2 年平均の被引用数（OpenAlex の値．インパクトファクターとは計算の範囲が違う）は 3.08，h 指数は 41

## 10. まだ確かめていないこと

1. Discover シリーズの編集方針（投稿規定が「editorial policies of the Discover Series」として参照するページ）の Research ethics の節：公開のコーパスの二次的な解析で何を書くか
2. 同じ編集方針の，生成 AI（LLM）を原稿の作成に使ったときの申告の決まり．投稿規定の本文には書かれていない
3. 掲載料の最新の値（Open access publishing のページ）
4. キーワードの数（投稿規定に書かれていない．Snapp の入力欄で決まる見込み）
5. 雛形の既定の文字の大きさと，`sn-vancouver-num` が NLM の書き方に合うか（雛形を置くときに確かめる）

## 11. この研究に当てはめるときの注意

- **査読の基準と主張の書き方**：査読は重要さを評価せず，報告の正確さと根本的な欠陥の有無を見る（0 節の 1）．このため，新しさの主張を強めることよりも，数値の正確さ，設計の漏れの無さ，比較の公平さ，統計の正しさを確実にすることが採否に直結する．`docs/methods.md` 9 節の関門の目録は，この基準にそのまま対応する
- **倫理の申告**：IEMOCAP・MOSEI・MELD・AFEW・RAVDESS・CREMA-D は人の録音と映像であり，「人のデータ」に当たるので倫理の申告が要る．公開のデータの二次的な解析として書く（7 節の例）．各コーパスの利用条件はまだ確かめていない
- **データとコードの公開**：Data availability は必須である．各コーパスの再配布の可否と，非公開の GitHub リポジトリ（snics6/MER-DA）のコードを公開するかを，投稿の前に決める
- **資金の申告**：必須である．研究者に資金の有無と番号を確かめる
- **読者**：DAI の掲載論文は応用の分野が広く，DA や MER の専門家でない読者が多いと見込まれる．定義と記号は初出で説明する
