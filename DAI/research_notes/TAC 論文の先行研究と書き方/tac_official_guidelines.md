# IEEE Transactions on Affective Computing（TAC）の公式の投稿要項・査読基準・編集方針（2024〜2026 年時点）

調査日：2026-10-27．確かめた水準の記号は次のとおりである．

- ◎：公式ページの本文を読んで確かめた．computer.org の誌面ページは JavaScript で描画されるため，同じサイトの記事管理 API（`https://wpcms.computer.org/wp-json/wp/v2/posts/<番号>`）から本文を取り出して読んだ．表示用の URL と API の URL を両方書く．
- ○：公式ページを要約する道具（WebFetch）の抜き書き，または公式の書誌データベース（OpenAlex）の要旨の抜き書きで確かめた．原文の一字一句までは保証しない．
- △：非公式の情報（第三者の集計サイト，検索結果の要約，古い雛形の注記）．

各ページの最終更新日は API の `modified` 欄から読んだ．TAC の投稿要項（Author Information）は 2023-12-28，超過頁の方針は 2025-07-17，編集委員の名簿は 2026-10-26 が最終更新である．

## 1. 守備範囲（aims & scope）と総説の制限

### Takeaway
TAC は，感情を認識・解釈・模擬するシステムの設計に関する研究を載せる学際的な誌である．総説は「この分野の歴史と展望に新しい見方を出すもの」に限って掲載を検討する．多モーダルの感情認識，音声の韻律の分析，感情コーパスの注釈，文化間・言語間の感情表出の違いは，いずれも題目の一覧に明記されている．

### Cited Findings
- ◎ 守備範囲の原文："IEEE Transactions on Affective Computing is a cross-disciplinary and international archive journal aimed at disseminating results of research on the design of systems that can recognize, interpret, and simulate human emotions and related affective phenomena. The journal publishes original research on the principles and theories explaining why and how affective factors condition interaction between humans and technology, on how affective sensing and simulation techniques can inform our understanding of human affective processes, and on the design, implementation and evaluation of systems that carefully consider affect among the factors that influence their usability." — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)；同じ文が [About TAC（API, 2026-07-23 更新）](https://wpcms.computer.org/wp-json/wp/v2/posts/14383) にもある
- ◎ 総説の制限の原文："Surveys of existing work are considered for publication when they propose a new viewpoint on the history and the perspective on this domain." — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)
- ◎ 題目は 3 群に分かれる．Sensing & Analysis 群には "Analysis of text and spoken language for emotion recognition"，"Analysis of prosody and voice quality of affective speech"，"Methods for multi-modal recognition of affective state"，"Tools and methods of annotation for provision of emotional corpora" がある．(Cyber)Psychology & Behavior 群には "Studies on cross-cultural, group and cross-language differences in emotional expression" と "Clarification of concepts related to affective computing (e.g., emotion, mood, personality, attitude) in ways that facilitate their use in computing." がある．3 群目は Behavior Generation & User Interaction である．一覧の前置きは "The journal covers but is not limited to the following topics" — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)
- ◎ 投稿は随時受け付ける："TAC seeks submissions on an ongoing basis" — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)
- ◎ 共催の学会：資金を出す共催は IEEE Systems, Man, and Cybernetics Society と IEEE Computational Intelligence Society，技術面の共催は IEEE Robotics and Automation Society，IEEE Signal Processing Society など 5 学会である — [About TAC（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/14383)
- ◎ 影響度：About TAC に "TAC's 2025 impact factor is 11.3." とある．IEEE Computer Society の 2026-07-24 の発表は，TAC を同学会の誌のうち Impact Factor が高い上位 10 誌に挙げている — [About TAC（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/14383)；[IEEE CS 2026 Journal Impact 発表（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/538522)．△ 第三者の集計サイトは 2025 年の Impact Factor を 9.8 としており，公式の 11.3 と食い違う — [journalmetrics.org](https://www.journalmetrics.org/journal/ieee-transactions-on-affective-computing)（どの年の JCR を指すかが不明である）
- ◎ 2026 年に募集中の特集号：(1) "When Affective Computing Meets Multimodal Large And Reasoning Language Models"（締切 2026-12-30，最終判定の通知 2027-05-01．題目に "Datasets and benchmarks for evaluating LLMs' emotion recognition performance"，"Multimodal and conversational emotion recognition"，"Fairness and bias mitigation in AI for affective computing" がある）— [CFP（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/538248)．(2) "AEGIS: Affective Experience-centered GenAI for Social Good"（締切 2027-01-30）— [CFP（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/546133)

### Inferences
- コーパスをまたいだ多モーダル感情認識の測定は，題目の "Methods for multi-modal recognition of affective state" に当たる．コーパスの違いを文化・言語・収録条件の違いとして論じれば，"Studies on cross-cultural, group and cross-language differences in emotional expression" にも当たる．
- 守備範囲の 2 文目は，感情の検出の技術が「人の感情の過程の理解に役立つこと」を研究の対象に挙げている．機械学習の手法の比較だけで終わらせず，結果を感情の現象（コーパスごとのラベルの付け方の違い，演技と自然発話の違いなど）と結びつけて述べると，守備範囲の文に沿う．これは公式の文からの推測であり，編集方針として明言された要件ではない．

### Gaps
- 範囲外として査読前に返された原稿の割合は，公式の資料に見つからなかった．
- Impact Factor の 11.3 と 9.8 の食い違いの原因（JCR の版の違いか）を確かめられなかった．

## 2. 原稿の種類・頁数の上限・超過料金・補足資料

### Takeaway
通常論文（regular paper）は 2 段組で 12 頁が基準であり，この 12 頁には参考文献と著者略歴が含まれる．投稿時は 18 頁まで出せるが，採択後の最終組版で 12 頁を超えた分は 1 頁（端数も 1 頁）ごとに 220 ドルを払う．補足資料は本文とは別のファイルで出し，頁数の上限はない．付録は補足資料として扱われる．

### Cited Findings
- ◎ TAC の原稿の種類と頁数の原文："Regular paper – 12 double column pages (Submissions may be up to 18 pages in length, subject to MOPC. All regular paper page limits include references and author biographies.)"，"Brief contribution – 8 double column"，"Comments paper – 2 double column"，"Survey paper – 20 double column pages [NOTE: Authors are to include in the cover letter a list of recent relevant survey articles on the same topic (or a claim that none exist), and a brief explanation of why this article makes a significant contribution above these.]" — [TAC Author Information](https://www.computer.org/csdl/journal/ta/write-for-us/15060)（[API, 2023-12-28 更新](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)）
- ◎ 頁の定義と上限の前提："A double column page is defined as a 7.875″ x 10.75″ page with 9.5-point type and 11.5-point vertical spacing. These length limits are taking into account reasonably-sized figures and references." — [TAC Author Information（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)
- ◎ 超過料金の対象と補足資料："All page limits include references and author biographies. For regular papers, pages in excess of these limits after final layout of the accepted manuscript is complete are subject to Mandatory Overlength Page Charges (MOPC). Note: All supplemental material must be submitted as separate files and must not be included within the same PDF file as the main paper submission. There is no page limit on supplemental files." — [TAC Author Information（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)
- ◎ 超過料金の額（現行）："The regular paper page length limit is defined at 12 formatted pages for Transactions and four formatted pages for Letters, including references and author biographies. Any pages or fraction thereof exceeding this limit are charged $220 per page. Submission page limits may differ from overlength page limits." — [TAC Overlength Submission Policy](https://www.computer.org/csdl/journal/ta/write-for-us/201981)（[API, 2025-07-17 更新](https://wpcms.computer.org/wp-json/wp/v2/posts/201981)）；同じ額が [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors) にもある
- ◎ 食い違い：2018-02-18 発効の MOPC 方針の PDF は "Any pages or fraction thereof exceeding this limit are charged $200 per page. Regular paper submissions to Transactions may not exceed 18 formatted pages." と書く．TAC はこの PDF の対象誌の一覧に載っている．額は現行の 2 ページ（220 ドル）が新しい — [MOPC policy PDF](https://ieeecs-media.computer.org/assets/pdf/MOPC_policy.pdf)
- ◎ 請求の時期と未払いの扱い："Authors will be notified of any assessed charges when galley proofs are sent for review. Payment must be sent at the time galley proofs are approved by the author."，"Authors who do not pay Mandatory APCs without a waiver shall be barred from submitting future articles to IEEE publications until such time as their previous outstanding charges are paid."．免除の申請先は pubs-waivers@computer.org で，"A waiver will be considered only prior to authors approving the final version of the proof of a manuscript." — [TAC Overlength Submission Policy（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/201981)
- ◎ 学会全体の規則："Page length and reference limits vary by journal. All page limits include abstracts, references, and author biographies. Special-issue papers must adhere to the same page limit as regular and short papers." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 割引は超過料金に使えない："Discounts cannot be combined or applied to any other fees such as overlength or color page charges." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 補足資料の扱い："All supplemental material must be submitted as separate files and must not be included within the same PDF file as the main paper submission. When submitting your paper in ScholarOne, upload these files as "supplemental material." Files cannot exceed 350 MB."，"All appendices in journal articles are considered supplemental material. If a paper's appendix is not initially designated as supplemental material, you will be asked to remove the appendix from the main paper file and resubmit as separate files."，"If your article is accepted for publication, any supplemental files will appear alongside the article in IEEE Xplore® and the Computer Society Digital Library." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 補足資料に載せてよいもの："We accept and encourage submissions of supplemental materials for review, including short videos, animations, audio files, extended code samples, and extra text and graphics that do not fit in the article." 採択後の最終ファイルでは補足資料を "log #-supp" と名付ける — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ Comments 論文：過去に同誌に載った論文の誤りや異論を述べるためのもので，"Authors are limited to one Comment paper on a previously published article." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)

### Inferences
- 18 頁で採択されて最終組版でも 18 頁なら，超過は 6 頁で，6 × 220 = 1,320 ドルになる．
- 雛形は最終の体裁を表さない（3 節）．投稿時の PDF で 12 頁ちょうどでも，最終組版で 12 頁を超えるおそれがある．超過料金を避けたいなら，投稿時の PDF を 12 頁より少し短くしておく．
- 付録を本文の PDF に入れると差し戻される．証明・追加の表・ハイパーパラメータの一覧は，最初から別ファイルの補足資料として出す．補足資料の頁数に上限はないが，査読者が読む保証はないので，主張を支える根拠は本文に置く．

### Gaps
- 総説（20 頁）と短報（brief contribution，8 頁）が上限を超えたときに超過料金がかかるか，投稿時の上限が何頁かは，TAC の文書に書かれていない（TAC の文は "For regular papers" と通常論文だけを挙げる）．
- 「投稿時の上限 18 頁」が TAC の文書では "may be up to 18 pages" と書かれているが，19 頁以上の原稿を事務的に返すかどうかの運用は確かめられなかった．

## 3. 様式・雛形・ORCID・匿名性・提出の手続き

### Takeaway
雛形の使用は必須であり，IEEE Template Selector で選ぶ．査読は既定で単盲（査読者は著者の名前を知る）であり，二重匿名は理由を添えて申請し，編集長が認めた場合に限る．ORCID は必須である．通常の投稿先は ScholarOne（https://mc.manuscriptcentral.com/taffc-cs）だが，2026 年の特集号の募集は IEEE Author Portal を投稿先に挙げている．

### Cited Findings
- ◎ 雛形："Journals: Using an article template is required for journal submissions."，"You can find the appropriate template using the IEEE Template Selector. However, please note that templates do not reflect the final format; the published paper will appear as formatted by the publication staff."，"We accept Microsoft Word, plain text, and LaTeX files. If you will be submitting LaTeX files, please use the IEEE LaTeX Analyzer before submitting to avoid delays and ensure all requirements are met." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)；TAC の要項も "Please use the IEEE Template Selector to find the correct template." と書く — [TAC Author Information（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)
- ◎ 2 段組："Please note that double-column papers will translate more readily into the final publication format." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ○ LaTeX の雛形：IEEEtran の README は `bare_jrnl_compsoc.tex` を "A bare bones starter file for IEEE Computer Society journal papers" と説明する — [IEEEtran README（CTAN）](https://tug.ctan.org/macros/latex/contrib/IEEEtran/README)．△ 古い版の `bare_jrnl_compsoc.tex` の注記には "The Computer Society usually requires 12pt for submissions" とあるが，現行の Author Guidelines にこの規則は見つからない — [雛形の写し（第三者のサイト）](https://faculty.cs.olemiss.edu/~hcc/researchMethods/notes/IEEEtran/bare_jrnl_compsoc.tex)
- ◎ 図表："When creating your journal article in the template, please embed the figures and tables where they should appear in the article (as close as possible to the first callout) and ensure that they are sized as intended for publication. Incorrectly sized figures will be returned to the author for reformatting."，"Figures and graphs will not be reformatted."，"All images and graphs should be high resolution and adhere to a minimum of 300 dpi at the intended display size." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 参考文献："all articles should have numbered references that are listed in the order in which they appear in the body of the paper"，学術誌では本文の番号を角括弧で示す．"Citations must include full publication information—for articles, this means listing all authors, the article name, the publication's full name, the volume, issue number, month, year of publication, and pages cited." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 文体："Please consult the IEEE Editorial Style Manual."．採択後に題名が書き換えられることがある："Article titles may be reworded for clarity or to align with title word limits." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 著者略歴："Journals: Author biographies are not required, and there is not a specific style format." ただし採択後の最終ファイルの一覧には "photos of authors (for full-length journal articles)" がある — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ ORCID："ScholarOne will prompt you to enter an Open Researcher and Contributor ID (ORCID), which is required by all IEEE publications." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 投稿先：TAC の募集文は "To submit a paper, go to https://mc.manuscriptcentral.com/taffc-cs." と書く — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)．2026 年の特集号の募集文は "Please submit only full papers intended for review, not abstracts, to the IEEE Author Portal submission system" と書く — [MLLM 特集号の CFP（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/538248)．学会の要項は "the IEEE Computer Society is in the process of migrating its periodicals to the IEEE Author Portal submission platform. The Author Portal ... still connects to the publication's peer review database on ScholarOne Manuscripts. We expect to have all of our journals and magazines available on the IEEE Author Portal submission platform by the end of December 2024." と書く — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 投稿の手続き：原稿の種類を "Type" 欄で選ぶ（例 "regular paper"）．キーワードは ScholarOne の ACM の分類表から選び，"We recommend that you enter a minimum of three keywords." 推薦・除外する査読者を挙げる場合は利害関係（IEEE Policies 9.9 節）を避け，求められれば理由を書く．ファイルは 350 MB 以内 — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 匿名性の既定："Unless a double-anonymous review is requested, each article undergoes a single-anonymous peer review process, where the authors do not know the identities of the reviewers, but the reviewers know the identities of the authors. We solicit peer reviews from at least three independent reviewers." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 二重匿名の申請："You can request a review in which your identity is also kept from the reviewers."，"All double-anonymous review requests must be accompanied by an explanation as to why the double-anonymous review is being requested."，"Requests for double-anonymous reviews will be honored at the discretion of the editor in chief based on the reasons behind the requests."，"Note that author anonymity for papers with preliminary versions cannot be guaranteed due to the possibility that reviewers may search for those versions online." 二重匿名を選べない誌として TAC 以外の 5 誌（Cloud Computing，Computers，Software Engineering，Dependable and Secure Computing，Emerging Topics in Computing）が挙がっている — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 採択後の最終ファイル：最終の PDF と一致する編集可能な原稿（.doc または .tex）を出す．その中身は "abstract"，"keywords/index terms"，"author affiliation data"，本文，図表，脚注などである．"No substantial and unauthorized changes can be made to the accepted article during this final stage of the publication process." 採択版は Early Access として公開され，"No changes can be made to an Early Access paper." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)

### Inferences
- LaTeX では `\documentclass[journal,compsoc]{IEEEtran}` の 2 段組で書き，Template Selector で TAC 向けの雛形を確かめるのが安全である．12pt の 1 段組で出すと頁数が増え，18 頁の上限に触れやすい．
- TAC は二重匿名を選べない誌の一覧に載っていないので，申請すれば二重匿名を選べる．arXiv に原稿を載せている場合は匿名が保てないと学会の要項自身が書いている．

### Gaps
- TAC の投稿の入口が現在 ScholarOne と IEEE Author Portal のどちらか（両方が動いているのか）は，公式の文書どうしで食い違う．投稿の直前に TAC の Author Information のページで確かめる必要がある（要確認）．
- TAC の Template Selector が示す雛形の版と，投稿時の文字の大きさの指定は確かめていない（要確認）．
- 通常論文に cover letter が必須かどうかは，TAC の文書に書かれていない．総説と，会議版がある原稿では cover letter に書く事項が決まっている（2 節，7 節）．

## 4. 査読の評価項目

### Takeaway
TAC の査読票（ScholarOne の評価項目）は公開されておらず，見つからなかった．公開されている IEEE Computer Society の査読者向けの指針は，方法・分析・結果・結論の妥当さ，技術的な重要さと正確さ，参考文献の使い方を評価するよう求めている．査読の前に，様式・方針の遵守，読める文章か，守備範囲内か，誌が定める技術的な中身の最低水準を満たすか，の予備審査がある．

### Cited Findings
- ◎ 査読者への指示："Start by identifying the type of manuscript (such as research, tutorial, survey, or case study). Recognize the publication's reference number limit and word or length limit"，"Comment on the appropriateness of methods, analyses, results, and conclusions, as well as on the technical significance and accuracy of the work. You may also assess whether the use of references and footnotes is appropriate. Suggest specific improvements to the paper. Describe how to revise the manuscript to give it focus and organization." — [IEEE CS Reviewer Resources](https://www.computer.org/publications/reviewer-resources)
- ◎ 頁数を守らせる指示："please suggest ways to manage the length of a manuscript should you recommend that the author add new content." — [IEEE CS Reviewer Resources](https://www.computer.org/publications/reviewer-resources)
- ◎ 引用の強要の禁止："it is unacceptable for a reviewer to require additional citations for the sole purpose of influencing bibliometric measures of either an individual or a periodical" — [IEEE CS Reviewer Resources](https://www.computer.org/publications/reviewer-resources)
- ◎ 自分の既発表の論文との差："please check to ensure that they have cited the previous work(s) and have clearly indicated how the new submission offers substantively novel or different contributions beyond those of the previously published work(s)." — [IEEE CS Reviewer Resources](https://www.computer.org/publications/reviewer-resources)
- ◎ 予備審査の基準："Such criteria include that the authors have followed applicable guidelines for style; that the authors have adhered to all relevant policies; that the article is comprehensible and well written, so that readers can understand its contents; that the article is within the scope of the publication; and that the article meets the minimum criterion for technical substance as established by the publication." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 担当編集者の推薦："The editor/GE recommendation must provide clear indication and explanation of the changes the authors must make or must address if a revision recommendation is made. If the editor/GE recommendation conflicts with the reviews, or if the reviews conflict with one another, the editor/GE must explain the basis of the recommendation" — [How to Make Peer Review Recommendations and Decisions](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ 査読者の数と期間："We assign three or more reviewers per submission"，"most publications will ask you to finish the review within three weeks." — [IEEE CS Reviewer Resources](https://www.computer.org/publications/reviewer-resources)
- ◎ 前編集長 Schuller の 2019 年の論説の結び："the bright future of Affective Computing, which will soon improve our lives in many ways as long as we keep an eye also on the ethical side and social implications given by future Artificial Intelligence empowered by Artificial Emotional Intelligence ensuring auditability, benchmarking, confidence, data-reliance, and explainability." — [Schuller 2019, TAC 10(1)（Augsburg 大学の機関リポジトリの写し）](https://opus.bibliothek.uni-augsburg.de/opus4/files/71751/71751.pdf)
- △ ほかの IEEE 誌の査読票は，守備範囲への適合，新しさ，技術的な正しさ，書き方を共通に問う（IEEE Access，IEEE Trans. Vehicular Technology など）．TAC の票と同じとは限らない — [IEEE Access Reviewer Guidelines](https://ieeeaccess.ieee.org/reviewers/reviewer-guidelines/)；[IEEE TVT Guidelines for Reviewers](https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-reviewers/reviewing-technical-papers)

### Inferences
- 公式の指示から読み取れる評価の軸は，(1) 守備範囲に入るか，(2) 既存研究（著者自身の会議版を含む）より実質的に新しいか，(3) 方法と分析が主張に見合うか，(4) 技術的に重要で正確か，(5) 読んで分かるか，(6) 頁数の上限を守っているか，の 6 つである．
- 査読者は追加の実験を求めるとき，代わりに削る箇所も示すよう指示されている．改訂で頁数が増えるときは，補足資料へ移す箇所を著者から提案すると，指示と合う．

### Gaps
- TAC の ScholarOne の査読票の項目（novelty / significance / technical soundness / clarity / relevance などの名前と段階）は公開されていない．査読を経験した人に見せてもらう以外に確かめる手段が見つからなかった．

## 5. 決定の種類・改訂の回数と期限・採択率・判定までの期間

### Takeaway
決定は Accept，Minor Revision，Major Revision，Revise and Resubmit，Reject，Administrative Reject，Editorial Reject の 7 種類で，最終の決定は編集長が下す．2 回の査読を経た原稿に 2 回目の Major Revision は出せない．学会の標準の日程では，投稿から最初の判定まで 8〜11 週ほどになる．採択率は，学会全体で約 25%，TAC については 2018 年の論説が「約 30%，低いときは 25%」と書いている．

### Cited Findings
- ◎ 決定の選択肢："Accept | Minor Revision | Major Revision | Revise and Resubmit"，"Reject | Administrative Reject | Editorial Reject"．"Reviewers, editors, and guest editors (GEs) make a recommendation about a submission, whereas the editor-in-chief (EIC) makes the decision that determines its final outcome."，"All decisions are final." — [How to Make Peer Review Recommendations and Decisions](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ Minor Revision："The submission requires a few changes. ... The revised manuscript might not go back to the reviewers if the editor/GE determines that the revisions are sufficient and appropriate." — [同上](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ Major Revision："The submission requires significant changes. Any revision in length by more than 10% should be a major revision."，"Major revisions go through a second round of review and are assigned to the same editor/GE. We strongly recommend that the editor/GE uses the same reviewers to evaluate the revised manuscript."，"If a paper has already gone through two rounds of reviews, the option of a second major revision is not available because the authors were given specific instructions and did not fulfill the requirements." — [同上](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ Revise and Resubmit："The submission is not suitable for publication, but the authors are encouraged to revise and resubmit upon rejection. Authors can be instructed to request that the same editor/GE be assigned to their new submission"．Reject："future submissions of the article will not be considered for review, even after further revisions." — [同上](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ 査読者を付けずに返す 2 種類：Administrative Reject は "Does not follow the IEEE guidelines for style."，"Obviously violates IEEE policies."，"Is incomprehensible (in other words, so poorly written that it is unreadable)."，"Does not fall within the scope of the publication or SI." のどれかに当たるもの．Editorial Reject は "The submission does not meet the minimum criterion for technical substance established for the periodical (reviewers will not be assigned)."．Editorial Reject に著者が異議を申し立てると，"the EIC must consult with at least two members of the editorial board" — [同上](https://www.computer.org/publications/making-peer-review-recommendations)
- ◎ 通常論文の標準の日程："The EIC has 1 week to assign an editor."，"Editors have 1 week to invite reviewers—at least three reviewers must be assigned."，"Reviewers are given 3-6 weeks to review the submission."，"Editors are given 2 weeks to submit a recommendation once the required number of reviews are in."，"The EIC has 1 week to make a decision once the editor's recommendation has been submitted." — [Peer Review Schedules](https://www.computer.org/publications/peer-review-schedules)
- ◎ 改訂の期限："Depending upon the publication, authors typically have 4-12 weeks to submit their major revisions and 1-4 weeks to submit their minor revisions. If the author does not submit the revised version on time, the manuscript is withdrawn and rendered inactive"．期限切れの場合は "such papers must be submitted as a new paper with a new log number and set of dates" — [Peer Review Schedules](https://www.computer.org/publications/peer-review-schedules)
- ◎ 特集号の日程："In general, SIs allow 5 months to complete the peer review process"，改訂の期限は "generally 2 to 4 weeks" — [Peer Review Schedules](https://www.computer.org/publications/peer-review-schedules)
- ◎ 学会全体の採択率："The editor in chief makes the final decision regarding acceptance. Only about 25 percent of all manuscripts submitted are accepted." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ TAC の採択率（2018 年の論説）：IEEE の 6 年ごとの定期審査の結果として "continuously increasing annual submission rates for the journal that has roughly doubled since its start while maintaining a low acceptance rate around 30 percent or as low as 25 percent" — [Schuller 2018, TAC 9(1)（Augsburg 大学の機関リポジトリの写し）](https://opus.bibliothek.uni-augsburg.de/opus4/files/71819/71819.pdf)
- ◎ 同じ論説の著者アンケート："review time took too long" と答えた著者は，その年の初めの約 4 分の 1 から年末には 0 になった — [Schuller 2018](https://opus.bibliothek.uni-augsburg.de/opus4/files/71819/71819.pdf)
- ○ 2023 年の論説：2021 年に出た 12 巻 1〜4 号の 82 本から最優秀論文を選んだ（"The winners were selected from 82 papers published in issues 12(1)-12(4) based on a vote by the Associate Editors."）— [André 2023, TAC 14(1) の OpenAlex の要旨](https://api.openalex.org/works/doi:10.1109/taffc.2023.3236419)
- ◎ 盗用の検査："Articles will be screened for plagiarism before acceptance." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)

### Inferences
- 標準の日程を足すと，編集者の割り当て 1 週，査読者の依頼 1 週，査読 3〜6 週，推薦 2 週，決定 1 週で，最初の判定まで 8〜11 週になる．これは学会の目安の合計であり，TAC の実績の値ではない．
- 2 回目の査読の後に Major Revision は出ないので，1 回目の Major Revision の回答で指摘をすべて解消できないと，Reject か Revise and Resubmit になる．
- 1 回目の改訂で本文の長さが 10% を超えて変わると，Major Revision の扱いになる．

### Gaps
- TAC の実際の「投稿から最初の判定までの日数」「採択までの日数」「直近の採択率」は，公式の資料に見つからなかった．IEEE Xplore の誌の頁の指標は，取得が自動の bot 検査で止められたため読めていない（要確認）．
- TAC が Major Revision に与える期限の週数（4〜12 週のどれか）は，TAC の文書に書かれていない（要確認）．

## 6. 編集長の論説（2019〜2026）と，現在の編集長・編集委員の構成

### Takeaway
現在の編集長は Jesse Hoey（University of Waterloo）で，任期は 2023〜2026 年（2025 年に再任）である．2027〜2028 年の編集長は公募中であり，2026-10-27 時点で後任の発表は見つからなかった．副編集長は 5 人，Associate Editor は 110 人で，110 人のうち 45 人が中国の機関に所属する．2019〜2026 年の編集長の論説は 5 本見つかったが，求める論文の性質や不採録の理由を正面から述べた論説は見つからなかった．

### Cited Findings
- ◎ 編集長と副編集長：Editor-in-Chief は "Jesse Hoey | University of Waterloo, Canada"．Associate Editors-in-Chief は Celso M. de Melo（DEVCOM Army Research Laboratory），Javier Hernandez（Microsoft Research），Hayley Hung（Delft University of Technology），Chi-Chun Lee (Jeremy)（National Tsing Hua University），Dan Zhang（Tsinghua University）の 5 人 — [TAC Editorial Board](https://www.computer.org/csdl/journal/ta/about/107327)（[API, 2026-10-26 更新](https://wpcms.computer.org/wp-json/wp/v2/posts/107327)）
- ◎ Associate Editor は 110 人である．所属の国を名簿から数えると，中国 45（国名のない北京郵電大学の 1 人を含む），イタリア 10，オランダ 8，米国 7，シンガポール 6，カナダ 5，オーストラリア 4，インド 4，英国 4，日本 3（NTT の Shiro Kumano，奈良先端大の Monica Perusquia-Hernandez，岐阜大の Kazunori Terada），スペイン 3，そのほか 11 か国から 1〜2 人である — [TAC Editorial Board（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/107327)（数は名簿から本調査で数えた）
- ◎ 編集長の任期：2023-03-03 の発表は "Jesse Hoey, from the University of Waterloo (Canada), is the new EIC of IEEE Transactions on Affective Computing" と書く — [EICs Confirmed（2023）](https://www.computer.org/press-room/editors-in-chief-confirmed-for-ieee-computer-society-publications)．2025 年の発表は再任を "reappointed to a second term, starting on 1 January 2025 and ending on 31 December 2026" と書く — [EICs Confirmed（2025, API）](https://wpcms.computer.org/wp-json/wp/v2/posts/438802)．食い違い：2023 年の発表は新任の編集長の任期を "began on 1 January 2023 and end on 31 December 2025" と書いており，2025 年の再任の発表と合わない
- ◎ 次の編集長の公募："EIC terms begin on 1 January 2027 and are for two years"，一覧に "IEEE Transactions on Affective Computing (2027-2028) (application deadline extended to 10 April 2026)" — [Seeking 2027 Editors in Chief（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/490296)
- ◎ 歴代の編集長：創刊の編集長は Jonathan Gratch，2 代目は Björn Schuller（2018 年末に 2 期目を終えた），3 代目は Elisabeth André（2019 年から．2021〜2022 年に再任）— [Schuller 2019](https://opus.bibliothek.uni-augsburg.de/opus4/files/71751/71751.pdf)；[Incoming EICs（2020, API）](https://wpcms.computer.org/wp-json/wp/v2/posts/119167)
- ○ 2019〜2026 年の編集長の論説（OpenAlex で type が editorial のもの）：Schuller 2019 "IEEE Transactions on Affective Computing–On Novelty and Valence"（10(1)）；André 2021 "Affective Computing in the Times of Pandemics"（12(1)）；André 2022 "Another Year in the Shade of Covid-19"（13(1)）；André 2023 "News on the Journal"（14(1)）；Hoey ほか 2026 "Editorial: The Rise of Affective Computing: Challenges and Opportunities"（17(2), pp. 1210–1213）— [OpenAlex の検索結果](https://api.openalex.org/works?filter=primary_location.source.issn:1949-3045,publication_year:2019-2026,type:editorial)；Hoey の論説の書誌は [Hoey の業績ページ](https://cs.uwaterloo.ca/~jhoey/publications/genbibtex.php?pid=221) でも確かめた
- ◎ Schuller 2019 の論説の中身は，編集長の交代，新しい Associate Editor（音声の感情認識の Fabien Ringeval など），Impact Factor が 4.585 に上がったこと，ACII Asia 2018 の報告である．題名の "Novelty" は論文の新しさの要件ではなく，「最も引用された論文の賞」を新しく始めることなどを指す（"In terms of novelty, it will also be up to her to award the first annual most cited article of the year award"）— [Schuller 2019](https://opus.bibliothek.uni-augsburg.de/opus4/files/71751/71751.pdf)
- ◎ Schuller 2018 の論説は分野の動きとして，深層学習を使う論文の急増と，"one increasingly finds weakly supervised learning exploiting big social multimedia and other big data sources for training Affective Computing systems" を挙げる — [Schuller 2018](https://opus.bibliothek.uni-augsburg.de/opus4/files/71819/71819.pdf)
- ○ André 2022・2023 の論説の冒頭は，旗艦の会議 ACII との連携（ACII で TAC の最優秀論文を発表する会）を報告している — [André 2022 の OpenAlex の要旨](https://api.openalex.org/works/doi:10.1109/taffc.2022.3148681)；[André 2023 の OpenAlex の要旨](https://api.openalex.org/works/doi:10.1109/taffc.2023.3236419)
- ○ 倫理の特集の論説："Guest Editorial: Ethics in Affective Computing"（Gratch, Greene, Picard, Urquhart, 2024）— [OpenAlex の検索結果](https://api.openalex.org/works?filter=primary_location.source.issn:1949-3045,publication_year:2019-2026,type:editorial)
- ◎ 賞：About TAC は Best Paper Award を "selected by the board's associate editors"，Most Influential Paper Award を "based on number of citations" と説明する — [About TAC（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/14383)
- ◎ 編集長に求められる役割として，学会の公募文は "actively soliciting novel, high-quality manuscripts from potential authors" と "upholding the quality of the publication ... ensuring adherence to IEEE Principles of Ethical Publishing" を挙げる — [Seeking 2027 Editors in Chief（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/490296)

### Inferences
- 2026 年 11 月に投稿すると，査読の途中の 2027-01-01 に編集長が替わる．最終の決定は新しい編集長が下す可能性が高い．
- 名簿の名前から判断すると，音声・多モーダルの感情認識を専門とする Associate Editor が複数いる（例：Chi-Chun Lee，Zheng Lian，Rui Liu，Heysem Kaya，Berrak Sisman，Zixing Zhang，Xinzhou Xu，Kun Qian，Ting Dang，Dimitrios Kollias，Guoying Zhao）．この分類は本調査者の知識による推測で，名簿に専門分野は書かれていない．
- 2026 年の特集号の題目は，多モーダルの大規模言語モデル，生成 AI，公平さと偏りの軽減，評価の基準となるデータセットに寄っている．

### Gaps
- Hoey の 2026 年の論説の本文を読めなかった（IEEE Xplore と CSDL の本文に届かなかった）．「分野の課題」「求める論文」について何を述べているかは未確認である．
- André の 2021〜2023 年の論説は要旨の冒頭しか読めていない．
- 編集長が不採録の理由，再現性，データ公開を正面から論じた論説は，読めた範囲では見つからなかった．
- 2027〜2028 年の編集長は 2026-10-27 時点で発表が見つからなかった（例年 12 月ごろに発表される）．

## 7. 著作権・プレプリント・倫理・データとコードの公開

### Takeaway
arXiv と TechRxiv へのプレプリントの掲載は，IEEE は既発表とみなさない．投稿したら掲載中のプレプリントに投稿した旨の注記を足し，採択されたら DOI と著作権表示を付けた採択版か書誌情報に差し替える．出版版（IEEE が組んだ PDF）は載せられない．人を対象とする研究は，倫理審査の承認の有無を投稿時に申告し，論文にも書く．AI が生成した内容を使った場合は謝辞で開示する．データとコードの公開は推奨であり，義務とする文書は見つからなかった．

### Cited Findings
- ○ プレプリント："Authors who have submitted or plan to submit their articles to IEEE may retain their posted preprints" on "arXiv.org, TechRxiv.org, or any not-for-profit preprint server approved by the Publication Services and Products Board (PSPB)"，"IEEE does not consider this to be a form of prior publication." — [IEEE Author Center, Post-Publication Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
- ○ 投稿時の注記："Upon submitting an article to the IEEE for review and possible publication, the author must add the following notice to the first screen of any of his/her previously posted electronic preprint versions of this paper:" 注記の文は "This work has been submitted to the IEEE for possible publication." で始まる — [IEEE Author Center, Submission and Peer Review Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- ○ 採択時："Upon acceptance, the previously posted article must be replaced with either (1) the full citation to the IEEE work with the DOI, or (2) the accepted version of the article with the DOI and an appropriate copyright notice"，"Once accepted by IEEE, the posted article must be removed from any other third-party servers."（arXiv，TechRxiv，著者と所属機関のサイト，助成機関の保管庫は除く）— [IEEE Author Center, Post-Publication Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
- ○ 著作権表示の文："© 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works." — [同上](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
- ○ 出版版：標準の IEEE 著作権書式では "the author may not post the VOR online"．"The article proof that the author receives for approval between acceptance and publication may not be posted online."．助成機関の保管庫に載せる場合の待機期間は 24 か月 — [同上](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
- ◎ オープンアクセス：TAC は "This journal is a hybrid publication, allowing either traditional manuscript submission or author-paid Open Access (OA) manuscript submission." — [TAC Author Information（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)．IEEE 会員は掲載料の 5%，学会の会員は 20% の割引を受ける — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 人を対象とする研究："Authors of articles reporting on research involving human subjects or animals must indicate upon article submission whether an approval was obtained from a relevant review board (or equivalent local/regional review). If such an approval was obtained, the original source and reference must be provided to the editor at submission and appear in the article itself." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ○ IEEE 全体の規則：論文に "a statement in the article that the research was performed under the oversight of an institutional review board or equivalent local/regional body" を入れ，委員会の正式名称を書くか，審査を受けなかった理由を説明する．"authors shall also report that consent from the human subjects in the research was obtained" か，得なかった理由を書く — [IEEE Author Center, Submission and Peer Review Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- ○ AI が生成した内容："The use of content generated by artificial intelligence (AI) in an article (including but not limited to text, figures, images, and code) shall be disclosed in the acknowledgments section of any article submitted to an IEEE publication. The AI system used shall be identified, and specific sections of the article that use AI-generated content shall be identified"．編集と文法の修正に使った場合の開示は "not required, but recommended" — [同上](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- ◎ データとコード：学会の要項は "Share Your Code and Data" の節で Code Ocean と IEEE DataPort を紹介し，"Following best practices of reproducible research by archiving your article's underlying data." を利点に挙げる — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)．TAC の特集号の募集文は "you are also encouraged to upload the data related to your paper to IEEE DataPort ... Uploading your dataset to IEEE DataPort will strengthen your paper and support research reproducibility." と書く — [MLLM 特集号の CFP（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/538248)
- ◎ 二重投稿と会議版：投稿中の会議論文を含め，他で審査中の原稿は受け付けない（"We don't accept articles under review by another publication, including conference proceedings."）．自分の既発表の論文を土台にした場合は "you are required to cite the previous work(s) and clearly indicate how the new submission offers substantively novel or different contributions"，既発表の論文の写しを出し，"Authors must provide a brief description of the differences between the submitted manuscript and the preliminary version(s) or planned conference submission(s)." TAC に投稿した後に会議版を出す予定があれば，cover letter にその旨を書く — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 図と第三者の素材："All text and figures must be owned and created by the author, except possibly for material such as text passages, figures, and data that clearly identify the original source, with permission notices from the copyright owners where required." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)

### Inferences
- 公開コーパス（IEMOCAP，CMU-MOSEI，MELD など）だけを二次利用する研究が「人を対象とする研究」に当たるかは，規則の文からは決まらない．投稿時の申告欄では「新たに人から収録していない．公開コーパスを各配布元の利用許諾に従って使った」と書き，本文にも同じ趣旨の 1 文を置くと，どちらに解釈されても問題が起きにくい．
- 文章の推敲に生成 AI を使った場合の開示は推奨であり，義務ではない．本文・図・コードの生成に使った場合は謝辞での開示が義務である．
- 会議版やワークショップ版がある場合は，差分の説明と写しの提出が必須で，満たさない原稿は "will be rejected" と書かれている．

### Gaps
- 投稿時に足す注記の全文（"This work has been submitted to the IEEE for possible publication." の後に続く文）は，要約の道具を通した抜き書きで冒頭しか確かめていない（要確認）．IEEE Author Center のページを curl で直接取ると 403 が返った．
- TAC がデータとコードの公開，またはその可否の記述（data availability statement）を義務とする文書は見つからなかった．
- 二次利用の公開コーパスについて，TAC が倫理の記述を求めた実例は確かめていない．

## 8. 論文の構成についての学会の指針（要旨の長さ・索引語・構成）

### Takeaway
要旨は通常論文と特集号の論文で 100〜200 語，短い論文で 50 語であり，数式と参考文献を入れてはならない．キーワードは投稿時に ScholarOne で ACM の分類表から 3 つ以上選び，最終ファイルにも索引語（index terms）を入れる．学会の文書が示す「書き方の推奨」（能動態，節番号を付けない，など）は雑誌（magazine）向けであり，TAC のような学術誌には当てはまらない．学術誌の論文の章立てを指定する公式の文書は見つからなかった．

### Cited Findings
- ◎ 要旨："An abstract must clearly state the nature and significance of the paper and is required at the head of the manuscript. Abstracts must not include mathematical expressions or bibliographic references. Word limits for journal abstracts are as follows: Regular/special-issue paper – 100 to 200 words / Short paper – 50 words" — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ キーワード："Keywords should be selected from the ACM taxonomy provided in ScholarOne, and they should be specific and closely aligned with your article's topic."，"We recommend that you enter a minimum of three keywords." 最終ファイルの中身として "keywords/index terms" を挙げる — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)；分類表は [ACM taxonomy PDF](https://ieeecs-media.computer.org/assets/pdf/taxonomy.pdf)
- ◎ 雑誌向けの推奨（学術誌には当てはまらない）：見出しは "Best Practices for Magazines" で，"Use active voice"，"Do not number section headings."，"If you must use acronyms, define each one on its first appearance."，"Conclusions should not summarize but instead outline future goals or lessons learned." などを挙げる — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 付録は補足資料として別ファイルで出す（2 節）— [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- ◎ 英文校正："English language editing services can help refine the language of your article and reduce the risk of rejection without review." — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)

### Inferences
- 学会の要項の要旨の語数の規則は "Short paper" と書き，TAC の原稿の種類は "Brief contribution" と書く．TAC の短報に 50 語の規則が当てはまるかは文書の上で対応が取れていない．通常論文なら 100〜200 語で問題ない．
- 予備審査は「読んで分かる文章か」を問い，Administrative Reject の理由に "incomprehensible" がある．英文の質が低いと査読に回らないおそれがあると学会自身が書いている．

### Gaps
- TAC の論文の章立て，貢献の書き方，限界の節の要否を指定する公式の文書は見つからなかった．
- 索引語の語数の上限と，IEEE の分類表（IEEE Thesaurus）を使うか ACM の分類表を使うかの指定は，最終ファイルの段階については書かれていない（要確認）．

## 9. 投稿前の点検表（守るべき項目）

### Takeaway
以下は，1〜8 節で確かめた公式の規則を投稿前に点検する項目として並べたものである．各項目の根拠は該当する節と URL にある．「要確認」は公式の文書どうしが食い違うか，文書に書かれていない項目である．

### Cited Findings
- 守備範囲：題目の一覧（"Methods for multi-modal recognition of affective state" など）のどれに当たるかを cover letter で 1 文で示せるか．総説なら "new viewpoint on the history and the perspective" を示し，cover letter に既存の総説の一覧と差を書いたか — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)；[TAC Author Information（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/15060)
- 頁数：通常論文は 2 段組で 18 頁以内か．12 頁を超えた分は採択後の最終組版で 1 頁（端数も 1 頁）220 ドルになることを共著者と確認したか．12 頁には要旨・参考文献・著者略歴が入る — [TAC Overlength Submission Policy（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/201981)；[IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 付録と補足資料：付録を本文の PDF から外し，別ファイルの補足資料にしたか．1 ファイル 350 MB 以内か．補足資料にも著作権の問題がないか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 雛形：IEEE Template Selector で選んだ雛形を使ったか（使用は必須）．LaTeX なら IEEE LaTeX Analyzer を通したか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 要旨：100〜200 語か．数式と参考文献を入れていないか．論文の性質と重要さを書いているか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 図表：初めて言及する箇所の近くに，出版時の大きさで埋め込んだか．300 dpi 以上か．他人の図の掲載許可を得たか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 参考文献：本文に出る順の番号か．全著者名・誌名の正式名・巻・号・月・年・頁を書いたか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 投稿の手続き：全著者の ORCID を用意したか．"Type" 欄で正しい原稿の種類（特集号なら特集号の名前）を選んだか．ACM の分類表からキーワードを 3 つ以上選んだか．推薦・除外する査読者に利害関係がないか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 投稿先（要確認）：ScholarOne（https://mc.manuscriptcentral.com/taffc-cs）か IEEE Author Portal かを，投稿の直前に TAC の Author Information のページで確かめたか — [TAC General Call for Papers](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)；[MLLM 特集号の CFP（API）](https://wpcms.computer.org/wp-json/wp/v2/posts/538248)
- 匿名性：単盲（既定）で出すか，理由を添えて二重匿名を申請するかを決めたか．二重匿名なら本文・補足資料・ファイルの属性から著者が分からないようにしたか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 二重投稿と会議版：同じ内容を会議を含むほかの場で審査中にしていないか．会議版があれば，引用し，写しを出し，差分の説明を付けたか．投稿後に会議に出す予定があれば cover letter に書いたか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- プレプリント：arXiv に載せているなら，投稿した旨の注記を足したか（採択後は DOI と著作権表示を付けた採択版に差し替える．出版版と校正刷りは載せない）— [IEEE Author Center, Post-Publication Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
- 倫理：人を対象とする研究なら，倫理審査の承認の有無を投稿時に申告し，委員会の正式名称と同意の取得を本文に書いたか．公開コーパスの二次利用なら，その旨と利用許諾に従ったことを書いたか（後者は推測による推奨）— [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)；[IEEE Author Center, Submission and Peer Review Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- 生成 AI：本文・図・コードの生成に AI を使ったなら，謝辞で道具の名前と使った節を開示したか — [IEEE Author Center, Submission and Peer Review Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- データとコード（推奨）：コードとデータの置き場（Code Ocean，IEEE DataPort，GitHub など）を本文に書いたか — [IEEE CS Author Guidelines](https://www.computer.org/publications/author-resources/authors)
- 改訂の準備：Major Revision の期限（学会の目安で 4〜12 週）に間に合わないと原稿は取り下げ扱いになる．2 回目の Major Revision はないので，1 回目の回答で指摘をすべて解消する計画を立てたか．改訂で長さが 10% を超えて変わると Major の扱いになる — [Peer Review Schedules](https://www.computer.org/publications/peer-review-schedules)；[How to Make Peer Review Recommendations and Decisions](https://www.computer.org/publications/making-peer-review-recommendations)

### Inferences
- MER-DA の論文 1 を TAC に出す場合，12 頁に収めることが最も大きな制約になる．地図の全体の表や理論の導出の詳細は補足資料に移し，本文の主張は本文の図表だけで支える構成にする必要がある．
- 2026 年 11 月以降に投稿すると，最終の決定は 2027 年からの新しい編集長が下す可能性が高い．

### Gaps
- 点検表の「要確認」の項目（投稿先，雛形の文字の大きさ，短報の要旨の語数，総説の超過料金，投稿時の注記の全文）は，投稿の直前に公式のページで確かめ直す必要がある．
