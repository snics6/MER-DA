"""説明の PDF の原稿を組み直す（1 回だけ使う道具）．手順の説明（1〜8 節）は前の版（git の 26bdd0d 以降の HEAD）から写して語を直し，
結果の節（9〜12 節）と用語は書き直す．図は bigfig で大きくする．"""
import subprocess
from pathlib import Path

ROOT = Path("/home/snics/MER-DA")
TEX = ROOT / "DAI/docs/説明_実験16の案_図解_2026-11-13.tex"
SRC = "eee1f0a:DAI/docs/説明_実験16の案_図解_2026-11-13.tex"       # 前の版（34 ページ）
old = subprocess.check_output(["git", "show", SRC], cwd=ROOT).decode().split("\n")
assert old[91].startswith("\\section{何を解きたいか}") and old[653].startswith("\\section{何を比べるか}") and old[641].startswith("\\begin{figure}")
body = "\n".join(l for l in old[91:641] if not l.startswith("\\ex{"))


def rep(o, n, cnt=1):
    global body
    assert body.count(o) == cnt, (body.count(o), o[:60])
    body = body.replace(o, n)


rep("\\begin{tikzpicture}", "\\begin{adjustbox}{scale=1.22,max width=\\linewidth}\\begin{tikzpicture}", body.count("\\begin{tikzpicture}"))
rep("\\end{tikzpicture}", "\\end{tikzpicture}\\end{adjustbox}", body.count("\\end{tikzpicture}"))
# 2 節
rep("今の方法（基準）は，微調整した 3 つの分類器の確率を同じ強さで掛け合わせる．", "今までの方法は，微調整した 3 つの分類器の確率を同じ強さで掛け合わせる．これを\\textbf{積}と呼ぶ．")
rep("\\subsection*{分類器のつながり（今の提案の手順．実験 1 で作ったもの）}", "\\subsection*{分類器のつながり}")
rep("（ソース 2 つ × 声・顔・言葉，\\\\と融合）", "（ソースごと × 声・顔・言葉，\\\\と融合）")
rep("ソース 2 つの分類器を\\\\平均する", "ソースの分類器を\\\\平均する")
rep("{\\textbf{今の提案}\\\\3 つの確率を同じ重みで掛ける（積）}", "{\\textbf{積（今までの方法）}\\\\3 つの確率を同じ重みで掛ける}")
rep("{\\textbf{実験 16}\\\\表から作った重みで\\\\3 つの点を足す（8 節）}", "{\\textbf{提案の手法}\\\\信頼度の表から作った重みで\\\\3 つの点を足す（8 節）}")
rep("{今の提案は，さらに積を疑似ラベルにして\\\\融合の分類器を学び直す（「提案の融合の生徒」）．\\\\実験 16 はその前の段階，「3 つをどうまとめるか」\\\\だけを変えて，推定時の答えで比べる．}",
    "{提案の手法が変えるのは，\\\\「3 つの答えをどうまとめるか」だけである．\\\\まとめた答えは，融合の分類器を学び直すときの\\\\疑似ラベルにも使える（10 節）．}")
rep("ソースの分類器は (2) の出発点として (4) に入る．R2 の「⑦ 両方の組」だけは，ソースの分類器も (4) と並べてまとめる．}", "ソースの分類器は (2) の出発点として (4) に入る．}")
# 3 節
rep("相性が本当に持ち越せるかは，この実験の最初に確かめる．", "相性がどこまで持ち越せるかは，7.3 節と 11 節で数を示す．")
# 4 節
rep("①〜④のどの段でも，テストの発話の正解は使わない．", "①〜④のどのステップでも，テストの発話の正解は使わない．")
# 5 節
rep("ソースで見込むターゲットでの信頼度にする（研究者の 2 段の案）．", "ソースで見込むターゲットでの信頼度にする．")
rep("この表をそのまま使うのが R2 の「④ ソースだけ」である．", "この表をそのまま使う版を「ソースだけで測った表」と呼び，9 節で提案と並べる．")
# 6 節
rep("（n0 ＝ 0 では戻し先が効かないので，中身が違うのは 14 通り）．R2 の「⑦ 両方の組」では，ソースの分類器の重さ（微調整した分類器の 0.25 倍か 1 倍）も一緒に決める．", "（n0 ＝ 0 では戻し先が効かないので，中身が違うのは 14 通り）．")
# 7 節
rep("下の表は，K = 5 で微調整した分類器の UAR（実験 16 の記録）である．", "下の表は，K = 5 で微調整した分類器の UAR（下調べの記録）である．")
rep("次のようになった（K = 5．下の R0 の表の「最強のモダリティが合う」）．", "次のようになった（K = 5）．")
rep("顔の分類器が比較的よく当てた感情は 4 コーパスとも Happy．R0 では，ソースで測った相性の形と正解の形の相関が 0.49〜0.77）", "顔の分類器が比較的よく当てた感情は 4 コーパスとも Happy．ソースで測った相性の形と，全ラベルで測った形の相関は 0.49〜0.77）")
rep("感情ごとの足し算の数（切片）をずらす．K 本で切片を決める版と，全ラベルで決める版（参照）も別に記録した．", "感情ごとの足し算の数（切片）をずらす．")
rep("\\caption{R2 の条件と，表の作り方の対応．}", "\\caption{9 節で並べる表の測り方．}")
rep("条件 & 表の作り方 \\\\", "呼び名 & 表の作り方 \\\\")
rep("① 本体の形・水準は K 本＋相性はソース & 7.1〜7.3 節のとおり．", "提案（水準は K 本＋相性はソース） & 7.1〜7.3 節のとおり．")
rep("③ 本体の形・K 本だけ & ", "K 本だけで測った表 & ")
rep("④ 本体の形・ソースだけ & ", "ソースだけで測った表 & ")
rep("参照：全ラベル & ", "全ラベルで測った表（参照） & ")
# 8 節
rep("\\caption{感情を順に仮定して比べる形（本体の案）．", "\\caption{感情を順に仮定して比べる．")
rep("\\multicolumn{6}{l}{\\textbf{今の方法}：重みは全部 1．点を足す（確率の積と同じ順になる）}", "\\multicolumn{6}{l}{\\textbf{積}：重みは全部 1．点を足す（確率を掛けるのと同じ順になる）}")
rep("\\multicolumn{6}{l}{\\textbf{実験 16}：感情ごとに，その列の重みを掛けて足す}", "\\multicolumn{6}{l}{\\textbf{提案の手法}：感情ごとに，その列の重みを掛けて足す}")
rep("今の方法では Sad（$+0.73$）が答えになる．実験 16 では，", "積では Sad（$+0.73$）が答えになる．提案の手法では，")
rep("今の方法", "積", body.count("今の方法"))
rep("8 節の 2 つのまとめ方は，この重みを使う．", "8 節のまとめ方は，この重みを使う．")
rep("答えが特定の感情に寄りやすい．主の比べでは，ラベルを使わずに，", "答えが特定の感情に片寄りやすい．そこで，ラベルを使わずに，")
rep("感情ごとの足し算の数（切片）をずらす．", "感情ごとの足し算の数（切片）をずらす．9 節より後の結果は，この切片をずらした答えの値である．")
rep("\\scriptsize\n\\setlength{\\tabcolsep}{4pt}", "\\footnotesize\n\\setlength{\\tabcolsep}{5pt}")
for bad in ("R2", "R0", "実験 16", "実験 17", "研究者の", "本体の"):
    assert bad not in body, (bad, body[max(0, body.find(bad) - 40):body.find(bad) + 40])

iw = next(i for i, l in enumerate(old) if l.startswith("\\newtcolorbox{waku}"))
pre = "\n".join(old[2:iw])          # \documentclass から色の定義まで（\newtcolorbox の前の行まで）
head = r"""% 提案の手法（感情ごとの信頼度の表で声・顔・言葉の答えをまとめる）の図解と結果．2026-11-14 に組み直した（前の版は git の eee1f0a）．
% 作り方：.venv/bin/python DAI/docs/tools_pending/pdf_figs_proposal.py（結果の図表を結果ファイルから書き出す）→ cd DAI/docs && lualatex 説明_実験16の案_図解_2026-11-13.tex（2 回）
""" + pre + r"""
\usepackage{graphicx}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\newtcolorbox{waku}[1]{colback=black!3,colframe=black!55,boxrule=0.6pt,arc=1mm,title={#1},fonttitle=\bfseries}
\newtcolorbox{chui}{colback=cTest!6,colframe=cTest!70,boxrule=0.6pt,arc=1mm}
\newcommand{\chip}[2]{\tikz[baseline=-0.6ex]\node[fill=#1,text=white,rounded corners=1.5pt,inner sep=2.2pt,font=\scriptsize\bfseries]{#2};}
\newcommand{\chipk}[2]{\tikz[baseline=-0.6ex]\node[fill=#1,text=black,rounded corners=1.5pt,inner sep=2.2pt,font=\scriptsize\bfseries]{#2};}
% 図を大きくする：1.22 倍にし，本文の幅を超えるなら幅いっぱいに収める
\usepackage{adjustbox}
\newcommand{\bigfigopts}{scale=1.22,max width=\linewidth}

\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,inner sep=4pt,font=\small},
  clf/.style={draw=cClf!70!black,fill=cClf!18,rounded corners=2pt,align=center,inner sep=3pt,font=\scriptsize,minimum width=13mm,minimum height=7mm},
  arr/.style={-{Stealth[length=2.2mm]},thick},
  corp/.style={draw,rounded corners=3pt,minimum width=30mm,minimum height=12mm,align=center,font=\small},
}

\title{提案の手法の図解と結果\\[4pt]
\large 感情ごとの信頼度の表で，声・顔・言葉の分類器の答えをまとめる\\
\large （水準はターゲットの K 本で，相性はソースで測る）}
\author{MER-DA（DAI の作業場）}
\date{2026 年 11 月 14 日 更新}

\begin{document}
\maketitle

\begin{waku}{この資料の読み方}
\begin{center}
\small
\begin{tabular}{>{\raggedright}p{2.2cm}p{12.6cm}}
\toprule
節 & 中身 \\
\midrule
1〜3 & 何を解きたいか，使う部品，信頼度とは何か \\
4〜8 & 提案の手法の手順（4 つのステップ） \\
9 & 下調べの結果（4 コーパス） \\
10 & コーパスを 6 つに増やした結果 \\
11 & 分かったことと，まだ分からないこと \\
12 & いま計算している評価の回 \\
13・14 & 用語，結果ファイルの名前との対応 \\
\bottomrule
\end{tabular}
\end{center}
\begin{itemize}
  \item 「例」と書いた数は説明用に作った数である．9〜11 節の数は，結果ファイルから台本で書き出した．
  \item 9 節の結果は照合を通している．10 節の結果は出力の点検を通したが，結果の文の照合はこれからである．
  \item この資料は提案の手法だけを扱う．下調べで比べたほかの版と，脇に置いた案（近いソースの発話を引く案）は外した．全部の版の表は \texttt{DAI/docs/実験16・17\_下調べ\_結果の表\_K5・10・25\_2026-11-14.md} と \texttt{実験18\_下調べ\_結果の表\_2026-11-14.md} にある．
\end{itemize}
\end{waku}

\noindent 色の意味：
\chip{cSrc}{ソースのラベル付きデータ}\
\chip{cK}{ターゲットの K 本}\
\chipk{cU}{ターゲットのラベル無し}\
\chip{cTest}{テスト}\
\chip{cClf}{分類器}\
\chipk{cRel}{信頼度の表}\
\chip{cMeta}{別のコーパスで決める数}

%======================================================================
"""
tail = r"""
%======================================================================
\section{下調べの結果（4 コーパス）}

\textbf{下調べ}は，評価に使うコーパスのテストを開かずに，手法の形と設定を決めるための計算である．
4 つのコーパスのうち 1 つを触らずに残し，残り 3 つのうち 1 つを仮のターゲット，2 つをソースにする（5.1 節）．
成績は仮のターゲットのテストで測る．コーパスの役の割り当て 12 通り × 話者の分け方 5 通り × K 本の引き方 2 通りで，計 120 回計算した．

並べるのは，信頼度の表の測り方 3 通り（7.3 節の表）と，参照である．どれも，積と同じ 3 つの分類器の答えをまとめる．
\begin{itemize}
  \item \textbf{提案}：水準は K 本，相性はソース．
  \item \textbf{K 本だけで測った表}：12 個すべてを K 本で測る．
  \item \textbf{ソースだけで測った表}：ターゲットのラベルを表に使わない．
  \item \textbf{全ラベルで測った表（参照）}：仮のターゲットの全ラベルで測る．実際には使えない．表を正しく測れたときの上限の目安である．
\end{itemize}
\textbf{基準}：区間の下端が 0 より上で，差が 1 点以上のとき，「上回った」と数える．

\input{説明_提案_図表_下調べ}

%======================================================================
\section{コーパスを 6 つに増やした結果}

CREMA-D（声と顔）と RAVDESS（声だけ）を足して，同じ下調べを 6 コーパスで流した（300 回）．どちらも，俳優が決まった文を演技で読むコーパスである．
\begin{itemize}
  \item 元の 4 コーパスの値も，9 節の値と同じにはならない．ソースに CREMA-D と RAVDESS が入り，出発点と微調整した分類器が変わるためである．
  \item RAVDESS は声しか無いので，3 つの答えをまとめる提案は使えない．図と表には入れていない．
  \item この計算では，提案の答えを疑似ラベルにして融合の分類器を学び直すステップも流した（最後の表）．
  \item 出力の欠けと重複の点検は通った．結果の文の照合はこれからである．
\end{itemize}

\input{説明_提案_図表_6コーパス}

%======================================================================
\section{分かったことと，まだ分からないこと}

\subsection*{分かったこと（9 節と 10 節の数から）}

\input{説明_提案_図表_まとめ}

\subsection*{まだ分からないこと}
\begin{itemize}
  \item 下調べはソースが少なく（4 コーパスの下調べでは 2 つ），K 本の引き方は 2 通りである．ソースが 3 つで，引き方が 10 通りの評価の回で同じ向きになるかは，12 節の計算で測る．
  \item 既にある方法（確率の平均，温度の較正，重みの表を乱数で探す方法など）と比べてどうかは，まだ測っていない．12 節の計算で測る．
  \item 提案の形は，下調べの結果を見て選んだ．評価の回のテストの発話は，下調べで仮のターゲットにしたときのテストと同じ発話である．評価の回は，見ていないデータでの確かめにはならない．
\end{itemize}

%======================================================================
\section{いま計算している評価の回}

\begin{waku}{状態}
2026 年 11 月 14 日 21 時 25 分に研究室の PC で起動した．単位の計算が 5.3〜5.8 時間，集計が 10〜30 分の見込みである．結果が出たら，この節を結果に書き換える．
\end{waku}

\subsection*{何を測るか}
\begin{itemize}
  \item 4 つのコーパスのそれぞれをターゲットにし，残り 3 つをソースにする．話者の分け方 5 通り × K 本の引き方 10 通りで，1 コーパスあたり 50 回，計 200 回計算する．K は 5・10・25．
  \item 設定（戻す強さ n0，戻し先，相性を使うか）は，下調べが選んだ値に固定した（下の表）．評価の回では選び直さない．
  \item 主にする K は置かない．上回った・上回らないの判定の規則も置かず，結果を表で見る．
\end{itemize}
\begin{center}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llll}
\toprule
ターゲット & K = 5 & K = 10 & K = 25 \\
\midrule
AFEW & n0 ＝ 20，戻し先は 3 つの平均 & n0 ＝ 0 & n0 ＝ 20，戻し先は 3 つの平均 \\
IEMOCAP & n0 ＝ 0 & n0 ＝ 0 & n0 ＝ 0 \\
MOSEI & n0 ＝ 0 & n0 ＝ 0 & n0 ＝ 0 \\
MELD & n0 ＝ 20，戻し先はソースの水準 & n0 ＝ 20，戻し先はソースの水準 & n0 ＝ 0 \\
\bottomrule
\end{tabular}
\end{adjustbox}

\smallskip
どの条件でも相性は使う．n0 ＝ 0 は，K 本で測った水準をそのまま使う設定である．
\end{center}

\subsection*{何と比べるか}
\begin{center}
\small
\begin{tabular}{>{\raggedright}p{3.9cm}>{\raggedright\arraybackslash}p{11.2cm}}
\toprule
組 & 相手 \\
\midrule
同じ 3 つの分類器の答えの，別のまとめ方 & 積／確率の平均／発話ごとに自信のある分類器を重くする平均／水準だけの重み／K 本だけで測った表／取り違えの表で確率を読み替えて掛ける方法／分類器ごとの温度の較正／K 本で合わせたモダリティごとの重み／重みの表を乱数で探す方法（Kaya ほか 2017 の型．候補の数は 10・100・1,000）／別のコーパスで選んだ指数の積 \\
学び直した 1 つの分類器 & K 本だけで学んだ融合の分類器／積を疑似ラベルにして学び直した融合の分類器／温度つき蒸留／一致した発話だけの疑似ラベル／ラベルと疑似ラベルを混ぜる方法（Zhang ほか 2021 の形）／共訓練／共正則化．同じ 200 回の上の差と区間を出す \\
標準の手法 & DANN，CORAL，MME，自己学習など（前に計算した結果の UAR を並べる） \\
\bottomrule
\end{tabular}
\end{center}
提案の部品を 1 つずつ外した版も測る．外す部品は，相性，分類器どうしの水準の違い，水準を戻す操作，残り方，「どの感情でどのモダリティが強いか」の成分である．差がどの部品から来たかを分けて読むために使う．

%======================================================================
\section{用語}
\begin{center}
\small
\begin{tabular}{>{\raggedright}p{3.6cm}p{11.8cm}}
\toprule
用語 & 意味 \\
\midrule
ソース／ターゲット & ラベルが全部ある既存のコーパス／ラベルが少ない新しいコーパス \\
K 本 & ターゲットで感情ごとに付いているラベルの本数（5，10 か 25） \\
分類器 & 発話の特徴から 4 つの感情の確率を出す線形の式 \\
微調整 & ソースの分類器の平均から始めて，K 本で学び直すこと \\
積 & 微調整した声・顔・言葉の確率を同じ強さで掛ける，今までのまとめ方 \\
提案の手法 & 信頼度の表から作った重みで，3 つの分類器の点を感情ごとに足すまとめ方．水準は K 本，相性はソースで測る \\
信頼度 & 分類器ごと・感情ごとの数．本当がその感情のとき，その分類器をどれだけ頼ってよいか \\
水準／相性 & 信頼度の行の平均／平均からの感情ごとの差 \\
残り方 & ソースの外で測った信頼度 ÷ ソースの中で測った信頼度 \\
戻す強さ n0 & K 本で測った水準を，戻し先へどれだけ近づけるか．ラベルの本数で数える \\
切片 & 感情ごとに点数へ足す数．答えが特定の感情に片寄るのを直す \\
仮のターゲット & ソースの 1 つをターゲットに見立てたもの．全ラベルがあるので答え合わせができる \\
下調べ／評価の回 & 仮のターゲットで手法の形と設定を決める計算／決めた手法を本来のターゲットで 1 回だけ測る計算 \\
疑似ラベル & 人ではなく分類器が付けたラベル \\
UAR & 感情ごとの正解率の平均．でたらめなら 25 点 \\
基準 & 区間の下端が 0 より上で，差が 1 点以上．満たした条件に $\uparrow$ を付ける \\
\bottomrule
\end{tabular}
\end{center}

\section{結果ファイルの名前との対応}
\begin{center}
\small
\begin{tabular}{>{\raggedright}p{4.2cm}p{11.2cm}}
\toprule
名前 & 意味 \\
\midrule
\texttt{prod} & 積 \\
\texttt{M\_F6\_mixsel} & 提案（水準は K 本＋相性はソース．設定は別のコーパスで選ぶ） \\
\texttt{M\_F6\_K} & K 本だけで測った表 \\
\texttt{M\_F6\_ext} & ソースだけで測った表 \\
\texttt{M\_F6\_orc} & 全ラベルで測った表（参照） \\
\texttt{M\_F6n\_mixsel} & 提案から「どの感情でどのモダリティが強いか」の成分を消した版 \\
\texttt{none\_AVT}，\texttt{none\_AV} & K 本だけで学んだ融合の分類器（声・顔・言葉／声・顔） \\
\texttt{adj} & ラベルを使わずに切片をずらした答え（この資料の数はすべてこの答えの値） \\
\bottomrule
\end{tabular}
\end{center}
結果の置き場：4 コーパスの下調べは \texttt{results/dai16\_probe/}・\texttt{results/dai16\_probe\_k10/}，6 コーパスは \texttt{results/dai18\_probe/}，評価の回は \texttt{results/dai20\_eval/}．

\end{document}
"""
body = body.replace("\\chip{cMeta}{組み立て方の数 3 つ}", "\\chip{cMeta}{組み立て方の数 3 つ}")
TEX.write_text(head + body + "\n" + tail)
print("ok", len((head + body + tail).split("\n")), "lines")
