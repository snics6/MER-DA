# 投稿先 IEEE Transactions on Affective Computing（TAC）の要項（2026-10-27 に公式の文書で確かめた）

出典と原文の引用は `../../research_notes/TAC 論文の先行研究と書き方/tac_official_guidelines.md` と `../../reports/TAC 論文の先行研究と書き方.md`．要点は `../../docs/related-work.md` 1.6 節．

## 要点

- 守備範囲：「人の感情を認識・解釈・模倣する系の設計の研究」．題目の一覧に多モーダルの感情認識と，文化・言語をまたぐ感情表現の違いの研究が明記されている
- 分量：通常論文は 2 段組 12 頁（要旨・参考文献・著者略歴込み）．投稿時は 18 頁まで．採択後の最終組版で 12 頁を超えた分は 1 頁（端数も 1 頁）220 ドル（Overlength Submission Policy，2025-07-17 更新）．雛形は最終の体裁を表さないので，投稿時の PDF は 11.5 頁を目安にする
- 付録：本文の PDF に入れない．すべて別ファイルの補足資料（頁数の上限なし，1 ファイル 350 MB）．査読者が読む保証はないので主張の根拠は本文に置く
- 要旨：100〜200 語．数式と参考文献を入れない．キーワードは ACM の分類表から 3 つ以上．雛形（IEEEtran の journal, compsoc）の使用は必須．ORCID 必須
- 査読：既定は単盲．3 人以上．最初の判定まで 8〜11 週（目安）．決定は 7 種類．2 回目の Major Revision は無い．長さが 10% を超えて変わる改訂は Major の扱い．Major の期限は 4〜12 週
- 編集長 Jesse Hoey の任期は 2026-12-31 まで．2026 年 11 月以降の投稿は 2027 年からの新しい編集長が最終の判定を下す見込み
- 倫理と開示：人を対象とする研究の申告．公開コーパスの二次利用でも「新たに収録していない．各配布元の利用許諾に従った」と書く．生成 AI で本文・図・コードを作った場合は謝辞で開示．arXiv へのプレプリントは可
- 載った近い 13 本の形（`../../research_notes/TAC 論文の先行研究と書き方/tac_paper_structures.md`）：掲載版の中央値 15 頁．分析の後に手法を置く 2 部構成に前例 3 本．本文を読めた 8 本に定理・命題は 0 本，検定は 3 本，補正・効果量・信頼区間は 0 本．限界の独立の節は 1 本（約 300 語）

## 投稿前の点検表（公式の規則から）

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
