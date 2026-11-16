"""Explanatory figures (2026-11-16, researcher's request: show the experimental setting, the proposed fusion and the
compared fusion methods as figures).

fig_setup_loco      : leave-one-corpus-out evaluation (4 runs) and the steps inside one run (Sec. 4.3, 5.1)
fig_method_flow     : the proposed fusion, step by step (Sec. 7.2)
fig_fusion_glyphs   : where each compared fusion method (F1-F10, Table 10) takes its weights from, drawn as a
                      3 x 4 weight table (rows: audio, visual, text classifiers; columns: N, H, S, A)

All three are schematic: no measured values. Japanese versions (suffix _ja) are for the co-researcher slides.
Output: ../figures/<name>{,_ja}.pdf and .png (300 dpi), width 174 mm.
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_explain.py
"""
import numpy as np

from make_figs_overview import (C_K, C_PROP, C_SRC, C_U, FS, GREY, MOD, W, arrow, box, canvas, mark, save, setup,
                                txt)

CORP = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
C_EQ = "#BDBDBD"     # equal / fixed weights
C_UTT = "#9E9E9E"    # per-utterance weights


def circ(i, lang="ja"):
    return "①②③④⑤⑥"[i] if lang == "ja" else f"({i + 1})"


def en_digits(d):
    """Arial has no circled digits: write (1)...(5) in the English texts."""
    def f(v):
        if isinstance(v, str):
            for i, c in enumerate("①②③④⑤"):
                v = v.replace(c, f"({i + 1})")
            return v
        if isinstance(v, list):
            return [f(x) for x in v]
        if isinstance(v, dict):
            return {k: f(x) for k, x in v.items()}
        return v
    return f(d)


# ------------------------------------------------------------------ setting
TS = {
    "ja": dict(
        left="LOCO 評価：4 コーパスを 1 つずつターゲットにする",
        tgt="ターゲット", src="ソース", run="回",
        uar="→ UAR",
        legend_t="ターゲット（感情ごとに K 本のラベル，\nラベル無しの発話，テストの発話）",
        legend_s="ソース（全部の発話にラベル）",
        crema="Q3 では CREMA-D（声と顔）をターゲットにした回も\n加えた（ソースは 4 コーパス）",
        right="1 回の中の手順",
        steps=[
            "ソースのコーパスごとに，声・顔・言葉・AVT の線形分類器を学ぶ．\n同じモダリティの分類器の重みと切片を平均して「アンカー」にする",
            "ターゲットを話者で 5 つに分け，1 つをテストにする\n（MOSEI はチャンネルか動画で分ける）．残り（候補）から感情ごとに\nK 本を抜いてラベルを使い，ほかの候補はラベル無しとして使う",
            "K 本の基準の分類器：アンカーから K 本だけで学び直す",
            "疑似ラベルによる学び直し：声・顔・言葉の分類器の答えをまとめた\n疑似ラベルを候補に付け，K 本と疑似ラベルで学び直す（Q1 は積，\nQ3 は積と提案．F1〜F10 は学び直す前のまとめた答えで比べた）",
            "テストの発話の UAR を比べる（Q1 の gain ＝ ④ − ③）．\n5 分割 × K 本の抜き方 10 ＝ 50 反復の平均（K = 0 は 5 反復）",
        ],
        hp="ハイパーパラメータは，ソースのコーパスを 1 つずつ\n仮のターゲットにした計算で選んで固定した",
        note="模式図",
    ),
    "en": dict(
        left="Leave-one-corpus-out: each corpus is the target once",
        tgt="target", src="source", run="run",
        uar="→ UAR",
        legend_t="Target (K labels per emotion,\nunlabelled and test utterances)",
        legend_s="Source (all utterances labelled)",
        crema="Q3 adds a run with CREMA-D (audio and visual)\nas the target (sources: the 4 corpora)",
        right="Steps inside one run",
        steps=[
            "Train linear audio, visual, text and AVT classifiers on each source\ncorpus. Average the weights and biases per modality: the anchor",
            "Split the target into 5 speaker-disjoint folds; one is the test fold\n(MOSEI: by channel or video). From the rest (candidates), use the labels\nof K utterances per emotion; the other candidates stay unlabelled",
            "K-label baseline: retrain from the anchor on the K labels only",
            "Pseudo-label retraining: label the candidates with the fused answer\nof the audio, visual and text classifiers; retrain on K labels + pseudo-\nlabels (Q1: product; Q3: product and proposed; F1–F10 compared as fused answers)",
            "Compare the test UAR (Q1 gain = ④ − ③). Mean of 50 repetitions\n(5 folds × 10 draws of the K labels; K = 0: 5 repetitions)",
        ],
        hp="Hyperparameters were chosen on runs that used each source corpus\nas a provisional target, then fixed",
        note="Schematic",
    ),
}


def fig_setup(lang):
    setup(lang)
    t = TS[lang] if lang == "ja" else en_digits(TS[lang])
    H = 76
    fig, ax = canvas(H)
    # left: LOCO rows
    txt(ax, 2, H - 2, t["left"], weight="bold")
    bw, bh, gap = 12.6, 6.5, 1.0
    x0, ytop = 9, H - 10
    for r, tc in enumerate(CORP):
        y = ytop - r * (bh + 3.2) - bh
        txt(ax, 2, y + bh / 2, f"{t['run']} {r + 1}" if lang == "en" else f"{r + 1} 回目", va="center", size=FS - 0.8, color=GREY)
        for j, c in enumerate(CORP):
            x = x0 + j * (bw + gap)
            if c == tc:
                box(ax, x, y, bw, bh, fc="white", ec="black", lw=1.3, z=2)
                txt(ax, x + bw / 2, y + bh / 2, c, ha="center", va="center", size=FS - 0.8, weight="bold")
            else:
                box(ax, x, y, bw, bh, fc="#EDE7F3", ec=C_SRC, lw=0.6, z=2)
                txt(ax, x + bw / 2, y + bh / 2, c, ha="center", va="center", size=FS - 0.8, color=C_SRC)
        xe = x0 + 4 * (bw + gap)
        txt(ax, xe - 0.3, y + bh / 2, t["uar"], va="center", size=FS - 0.9)
    yl = ytop - 4 * (bh + 3.2) - 1.5
    box(ax, 2, yl - 3.2, 4.5, 3.2, fc="white", ec="black", lw=1.3, z=2)
    txt(ax, 7.5, yl - 1.6, t["legend_t"], va="center", size=FS - 1.0, ls=1.1)
    box(ax, 2, yl - 10.0, 4.5, 3.2, fc="#EDE7F3", ec=C_SRC, lw=0.6, z=2)
    txt(ax, 7.5, yl - 8.4, t["legend_s"], va="center", size=FS - 1.0)
    txt(ax, 2, yl - 13.0, t["crema"], size=FS - 1.0, color=GREY, ls=1.1)
    # right: steps
    xr = 76
    txt(ax, xr, H - 2, t["right"], weight="bold")
    y = H - 8
    hs = [8.6, 11.6, 5.0, 11.6, 8.6]
    for i, (st, h) in enumerate(zip(t["steps"], hs)):
        ec = C_PROP if i == 3 else ("black" if i != 2 else C_K)
        box(ax, xr, y - h, W - xr - 1, h, fc="white" if i != 3 else "#FDF0E8", ec=ec, lw=0.7, z=2)
        txt(ax, xr + 1.2, y - 1.0, circ(i, lang), size=FS + 0.6 if lang == "ja" else FS - 0.6, weight="bold")
        txt(ax, xr + 5.5, y - 1.2, st, size=FS - 1.15, ls=1.15)
        if i < 4:
            arrow(ax, (xr + 3, y - h), (xr + 3, y - h - 1.6), lw=0.6, head=5)
        y -= h + 1.6
    txt(ax, xr, y - 0.6, t["hp"], size=FS - 1.15, color=GREY)
    txt(ax, W - 1, 0.8, t["note"], ha="right", va="bottom", size=FS - 1.2, color=GREY)
    save(fig, "fig_setup_loco" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ weight-table glyph
EMO = ["N", "H", "S", "A"]


def glyph(ax, x, y, kind, cw=3.4, ch=2.9, show_labels=True, seed=0):
    """3 x 4 weight table. kind: equal | utter | src_row | k_row | k_all | prop."""
    rng = np.random.default_rng(seed)
    strip = 2.2
    gx = x + strip + 0.6
    for r, m in enumerate("AVT"):
        yy = y + (2 - r) * ch
        if show_labels:
            mark(ax, x - 1.6, yy + ch / 2, m, size=3.0)
        # row strip: where the per-classifier number comes from
        if kind in ("src_row", "k_row", "prop"):
            col = C_SRC if kind == "src_row" else C_K
            box(ax, x, yy + 0.2, strip, ch - 0.4, fc=col, ec=col, lw=0.3, r=0.3, z=3)
        for c in range(4):
            if kind == "equal":
                fc = C_EQ
            elif kind == "utter":
                g = 0.45 + 0.4 * rng.random()
                fc = (g, g, g)
            elif kind == "src_row":
                fc = "#D9CDE6"
            elif kind == "k_row":
                fc = "#C6DBEF"
            elif kind == "k_all":
                a = 0.25 + 0.7 * rng.random()
                fc = tuple((1 - a) + a * v for v in (0.12, 0.47, 0.71))
            else:  # prop: shape from the source inside each row
                v = rng.random()
                fc = (0.42 + 0.45 * v, 0.24 + 0.55 * v, 0.6 + 0.3 * v)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((gx + c * cw, yy), cw, ch, fc=fc, ec="white", lw=0.6, zorder=3))
            if kind == "utter" and r == 1 and c == 0:
                pass
    if show_labels:
        for c, e in enumerate(EMO):
            txt(ax, gx + c * cw + cw / 2, y + 3 * ch + 0.3, e, ha="center", va="bottom", size=FS - 2.0, color=GREY)
    return gx + 4 * cw


TG = {
    "ja": dict(
        rows=["ラベルを使わない\n（同じ重みか確信度）", "別のコーパスで\n決める", "ターゲットの K 本で\n3 つの数を決める", "ターゲットの K 本で\n表を測る・探す"],
        F={"F1": "F1 積\n（同じ重みで掛ける）", "F2": "F2 確率の平均", "F4": "F4 確信度で重みづける\n平均（発話ごとに変わる）",
           "F3": "F3 別のコーパスで\n選んだ指数の積",
           "F5": "F5 水準だけの重み", "F8": "F8 分類器ごとの\n温度の較正", "F9": "F9 K 本で合わせた\nモダリティごとの重み",
           "F6": "F6 全部を K 本で\n測った表（12 個）", "F7": "F7 取り違えの表で\n読み替えて掛ける", "F10": "F10 1,001 個の表から\nK 本で選ぶ"},
        utter="発話ごとに変わる",
        prop_t="提案",
        prop="行の強さ（3 つの数）← ターゲットの K 本\n行の中の感情ごとの形 ← ソースのコーパス",
        kcount="K 本で決める数",
        kc=["0", "0", "3", "12 か表 1 つ"],
        legend=["同じ重み", "発話ごと", "別のコーパスで決める", "ターゲットの K 本で決める"],
        axis="表の行は分類器（声・顔・言葉），列は感情（N，H，S，A）．左の細い帯は分類器ごとの 1 つの数",
        note="模式図．色の濃さは実験の値ではない",
    ),
    "en": dict(
        rows=["No labels used\n(equal or confidence)", "Set on other\ncorpora", "3 numbers set on\nthe K target labels", "Table measured or\nsearched on K labels"],
        F={"F1": "F1 Product\n(equal weights)", "F2": "F2 Mean of\nprobabilities", "F4": "F4 Confidence-weighted\nmean (per utterance)",
           "F3": "F3 Product with exponents\nchosen on other corpora",
           "F5": "F5 Level-only\nweights", "F8": "F8 Per-classifier\ntemperature scaling", "F9": "F9 Modality weights\nfitted on K labels",
           "F6": "F6 All 12 reliabilities\nfrom K labels", "F7": "F7 Confusion-table\nre-reading", "F10": "F10 Best of 1,001\ntables on K labels"},
        utter="varies per utterance",
        prop_t="Proposed",
        prop="Row strength (3 numbers) ← K target labels\nPer-emotion shape within a row ← source corpora",
        kcount="Numbers set on K labels",
        kc=["0", "0", "3", "12 or 1 table"],
        legend=["equal", "per utterance", "set on other corpora", "set on the K target labels"],
        axis="Rows: audio, visual, text classifiers; columns: emotions N, H, S, A. Thin strip: one number per classifier",
        note="Schematic; shades are not measured values",
    ),
}


def fig_glyphs(lang):
    setup(lang)
    t = TG[lang]
    H = 92
    fig, ax = canvas(H)
    rows = [["F1", "F2", "F4"], ["F3"], ["F5", "F8", "F9"], ["F6", "F7", "F10"]]
    kinds = {"F1": "equal", "F2": "equal", "F4": "utter", "F3": "src_row", "F5": "k_row", "F8": "k_row",
             "F9": "k_row", "F6": "k_all", "F7": "k_all", "F10": "k_all"}
    rh = 20.5
    ytop = H - 4
    xl, cardw = 30, 30
    for r, (lab, fs, kc) in enumerate(zip(t["rows"], rows, t["kc"])):
        y = ytop - (r + 1) * rh
        box(ax, 1, y + 0.6, W - 2 - 52, rh - 1.2, fc="#FAFAFA", ec="#DDDDDD", lw=0.5, z=1)
        txt(ax, 3, y + rh - 2.2, lab, size=FS - 0.6, weight="bold", ls=1.1)
        txt(ax, 3, y + rh - 9.6, f"{t['kcount']}：\n{kc}" if lang == "ja" else f"{t['kcount']}:\n{kc}", size=FS - 1.4, color=C_K if r >= 2 else GREY, ls=1.1)
        for j, f in enumerate(fs):
            x = xl + j * cardw
            gy = y + 8.2
            glyph(ax, x + 3, gy, kinds[f], seed=j + 3 * r)
            txt(ax, x + 1.2, gy - 0.6, t["F"][f], size=FS - 1.3, ls=1.05)
    # proposed card (right)
    xp, wp = W - 50, 49
    yp0 = ytop - 4 * rh + 0.6
    box(ax, xp, yp0, wp, 4 * rh - 1.2, fc="#FDF0E8", ec=C_PROP, lw=1.2, z=1)
    txt(ax, xp + 2, ytop - 2.0, t["prop_t"], weight="bold", color=C_PROP, size=FS + 1.0)
    gx_end = glyph(ax, xp + 14, ytop - 22, "prop", cw=5.2, ch=4.4, seed=11)
    txt(ax, xp + 2, ytop - 25, t["prop"], size=FS - 1.0, ls=1.25)
    txt(ax, xp + 2, ytop - 33, f"{t['kcount']}：3" if lang == "ja" else f"{t['kcount']}: 3", size=FS - 1.0, color=C_K)
    # legend
    ly = ytop - 42
    for i, (lab, col) in enumerate(zip(t["legend"], [C_EQ, C_UTT, C_SRC, C_K])):
        yy = ly - i * 4.6
        box(ax, xp + 2, yy - 2.6, 4, 2.6, fc=col, ec=col, lw=0.3, r=0.3, z=3)
        txt(ax, xp + 7.5, yy - 1.3, lab, va="center", size=FS - 1.1)
    txt(ax, xp + 2, ly - 19.5, t["axis"], size=FS - 1.5, color=GREY, ls=1.15) if lang == "en" else \
        txt(ax, xp + 2, ly - 19.5, "表の行は分類器（声・顔・言葉），\n列は感情（N，H，S，A）．\n左の細い帯は分類器ごとの 1 つの数", size=FS - 1.5, color=GREY, ls=1.15)
    txt(ax, W - 1, 0.6, t["note"], ha="right", va="bottom", size=FS - 1.4, color=GREY)
    save(fig, "fig_fusion_glyphs" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ proposed fusion, step by step
TM = {
    "ja": dict(
        x="ラベル無しの発話 x\n（候補）",
        clf="積を作る分類器の点\nz_k(i | x)（中心化した対数確率）",
        s1="① 感情ごとの形（相性）\nソースのコーパスで，分類器 × 感情の\n信頼度を測り，行の平均を引く",
        s2="② 強さ（水準）\nターゲットの K 本で，\n分類器ごとに 1 つ測る",
        s3="③ 重みの表 W(k, i)\n＝ 強さ ＋ 形",
        s4="④ 感情ごとに重みを掛けて足す\ns(i | x) ＝ Σ_k W(k, i) · z_k(i | x)\nラベルを使わずに切片をずらし，\n答えが一方の感情に寄らないようにする",
        s5="⑤ まとめた答えを疑似ラベル（soft label）にし，\nAVT の分類器を K 本と疑似ラベルで学び直す",
        note="模式図．棒と色の濃さは実験の値ではない",
    ),
    "en": dict(
        x="Unlabelled utterance x\n(candidate)",
        clf="Scores z_k(i | x) of the\nclassifiers that form the product\n(centred log-probabilities)",
        s1="① Per-emotion shape (profile)\nreliability of classifier × emotion on\nthe source corpora, row mean removed",
        s2="② Strength (level):\none number per\nclassifier, from the\nK target labels",
        s3="③ Weight table W(k, i)\n= strength + shape",
        s4="④ Weight and add per emotion\ns(i | x) = Σ_k W(k, i) · z_k(i | x)\nshift the intercepts without labels so that\nthe answers do not drift to one emotion",
        s5="⑤ Use the fused answer as soft pseudo-labels and\nretrain the AVT classifier on K labels + pseudo-labels",
        note="Schematic; bars and shades are not measured values",
    ),
}


def fig_method(lang):
    setup(lang)
    t = TM[lang] if lang == "ja" else en_digits(TM[lang])
    H = 74
    fig, ax = canvas(H)
    # input utterance and three classifiers with score bars
    box(ax, 2, H - 13, 34, 10, fc="white", ec=C_U, lw=0.8)
    txt(ax, 19, H - 4.5, t["x"], ha="center", color=C_U, size=FS - 0.6)
    vals = {"A": [0.2, -0.6, 0.1, 0.4], "V": [-0.3, 0.9, -0.2, -0.4], "T": [0.5, -0.1, -0.5, 0.1]}
    for r, m in enumerate("AVT"):
        y = H - 27 - r * 13
        box(ax, 2, y - 1, 34, 11, fc="white", ec=MOD[m][0], lw=0.8)
        mark(ax, 5, y + 7.5, m, size=4.0)
        base = y + 3.8
        for c, (e, v) in enumerate(zip(EMO, vals[m])):
            bx = 12 + c * 5.5
            ax.add_patch(__import__("matplotlib").patches.Rectangle((bx, base), 3.6, v * 4.2, fc=MOD[m][0], ec="none", zorder=3))
            txt(ax, bx + 1.8, y + 0.0, e, ha="center", va="bottom", size=FS - 2.0, color=GREY)
        ax.plot([11, 34], [base, base], color="black", lw=0.4, zorder=4)
        arrow(ax, (19, H - 13), (19, H - 15.2), lw=0.5, head=4) if r == 0 else None
    txt(ax, 2, H - 63, t["clf"], size=FS - 1.4, color=GREY, ls=1.1)
    # table in the middle
    gx0, gy0 = 58, H - 46
    cwm = 4.6
    box(ax, 42, H - 16, 54, 13, fc="#F4F0F8", ec=C_SRC, lw=0.8)
    txt(ax, 44, H - 4.6, t["s1"], size=FS - 1.0, color=C_SRC, ls=1.15)
    box(ax, 44, H - 72, 27, 15, fc="#EAF2FA", ec=C_K, lw=0.8)
    txt(ax, 45.5, H - 58.5, t["s2"], size=FS - 1.2, color=C_K, ls=1.15)
    glyph(ax, gx0, gy0, "prop", cw=cwm, ch=cwm, seed=11, show_labels=True)
    xs = gx0 + 2.8  # first column of the cells
    txt(ax, 70, gy0 - 3.0, t["s3"], size=FS - 0.6, weight="bold", color=C_PROP, ls=1.1)
    arrow(ax, (xs + 2 * cwm, H - 16), (xs + 2 * cwm, gy0 + 3 * cwm + 3.2), color=C_SRC, lw=0.8)
    arrow(ax, (57, H - 57), (gx0 + 1.1, gy0 - 0.4), color=C_K, lw=0.8)
    # from classifiers to the table
    arrow(ax, (36, H - 35), (gx0 - 3.2, gy0 + 1.5 * cwm), lw=0.7)
    # right column
    box(ax, 100, H - 25, 72, 19, fc="#FDF0E8", ec=C_PROP, lw=1.0)
    txt(ax, 102, H - 7.6, t["s4"], size=FS - 0.9, ls=1.25)
    arrow(ax, (xs + 4 * cwm + 0.8, gy0 + 1.5 * cwm), (100, H - 18), lw=0.8)
    box(ax, 100, H - 42, 72, 11, fc="white", ec="black", lw=0.8)
    txt(ax, 102, H - 33.2, t["s5"], size=FS - 0.9, ls=1.25)
    arrow(ax, (136, H - 25), (136, H - 31), lw=0.8)
    txt(ax, W - 1, 0.8, t["note"], ha="right", va="bottom", size=FS - 1.3, color=GREY)
    save(fig, "fig_method_flow" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ worked example: why the fusion matters
TX = {
    "ja": dict(
        head="例：正解が Happy の 1 つの発話（顔の分類器が最も強いコーパス）",
        clf=["声の分類器", "顔の分類器（最も強い）", "言葉の分類器"],
        eq="同じ重みで掛ける（積．Q1 の疑似ラベル）",
        wt="顔の重みを大きくして掛ける（提案の考え方）\n重み：声 0.87，顔 1.32，言葉 0.81",
        ans_eq="答え：Neutral（外れ）", ans_wt="答え：Happy（当たり）",
        true="正解",
        note="模式図．確率は説明のための例．重みは AFEW の K = 10 の提案の重み（4 感情の平均）を，感情に共通の重みとして使った",
    ),
    "en": dict(
        head="Example: one utterance whose true emotion is Happy (the visual classifier is the strongest)",
        clf=["Audio classifier", "Visual classifier (strongest)", "Text classifier"],
        eq="Multiply with equal weights (product; Q1 pseudo-labels)",
        wt="Multiply with a larger visual weight (idea of the proposal)\nweights: audio 0.87, visual 1.32, text 0.81",
        ans_eq="Answer: Neutral (wrong)", ans_wt="Answer: Happy (correct)",
        true="true",
        note="Schematic. Probabilities are illustrative; weights are the AFEW K = 10 proposed weights (mean over emotions) used as emotion-common weights",
    ),
}
PX = {"A": [0.50, 0.25, 0.15, 0.10], "V": [0.15, 0.55, 0.15, 0.15], "T": [0.50, 0.25, 0.15, 0.10]}
WX = {"A": 0.87, "V": 1.32, "T": 0.81}


def bars(fig, x, y, w, h, vals, color, title, true_i=1, hl=None, ylim=0.7):
    ax = fig.add_axes([x / W, y / HX, w / W, h / HX])
    ax.patch.set_alpha(0)
    cols = [color] * 4
    if hl is not None:
        cols = ["#CCCCCC"] * 4
        cols[hl] = color
    ax.bar(range(4), vals, color=cols, width=0.65, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=FS - 1.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(EMO, fontsize=FS - 1.2)
    ax.set_ylim(0, ylim)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(length=0, pad=1.5)
    ax.get_xticklabels()[true_i].set_fontweight("bold")
    ax.set_title(title, fontsize=FS - 0.8, pad=2)
    return ax


HX = 88


def fig_example(lang):
    setup(lang)
    t = TX[lang]
    fig, ax = canvas(HX)
    txt(ax, 2, HX - 2, t["head"], weight="bold")
    cw = 46
    for j, m in enumerate("AVT"):
        x = 8 + j * (cw + 10)
        bars(fig, x, HX - 36, cw, 24, PX[m], MOD[m][0], t["clf"][j])
    pa = np.array(PX["A"]) * np.array(PX["V"]) * np.array(PX["T"])
    pa = pa / pa.sum()
    sc = sum(WX[m] * np.log(PX[m]) for m in "AVT")
    pw = np.exp(sc - sc.max())
    pw = pw / pw.sum()
    box(ax, 4, 6, 80, 40, fc="white", ec="#444444", lw=0.8)
    box(ax, 90, 6, 82, 40, fc="#FDF0E8", ec=C_PROP, lw=1.0)
    txt(ax, 6, 44.5, t["eq"], size=FS - 0.6, weight="bold")
    txt(ax, 92, 44.5, t["wt"], size=FS - 0.6, weight="bold", color=C_PROP, ls=1.15)
    bars(fig, 18, 17, 50, 18, list(pa), "#555555", "", hl=int(np.argmax(pa)))
    bars(fig, 104, 17, 50, 18, list(pw), C_PROP, "", hl=int(np.argmax(pw)))
    txt(ax, 44, 11.0, t["ans_eq"], ha="center", size=FS, weight="bold", color="#B2182B")
    txt(ax, 129, 11.0, t["ans_wt"], ha="center", size=FS, weight="bold", color="#2166AC")
    arrow(ax, (60, HX - 40), (44, 46.5), lw=0.8)
    arrow(ax, (110, HX - 40), (129, 46.5), lw=0.8, color=C_PROP)
    txt(ax, W - 1, 0.6, t["note"], ha="right", va="bottom", size=FS - 1.8, color=GREY)
    save(fig, "fig_fusion_example" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ Q1 model: what is learned where, and how it is retrained
TQ = {
    "ja": dict(
        src_t="① ソース：残る 3 コーパス（全部にラベル）",
        srcs=["ソース 1（例：IEMOCAP）", "ソース 2（例：MOSEI）", "ソース 3（例：MELD）"],
        src_note="コーパスごとに，凍結した特徴の上の\n線形分類器を 4 つ学ぶ",
        anc_t="② アンカー",
        anc_note="3 つのソースの分類器の重みと切片を\nモダリティごとに平均する",
        ancs=["声のアンカー", "顔のアンカー", "言葉のアンカー", "AVT のアンカー"],
        tgt_t="③④ ターゲット（例：AFEW）で学び直す",
        klab="感情ごとに K 本のラベル",
        pl="候補（ラベル無し）の疑似ラベル\n＝ 積（次のページ）",
        base="③ K 本の基準の分類器\nアンカーから，K 本\nだけで学び直す",
        retr="④ 疑似ラベルで学び\n直した分類器\nアンカーから，K 本 ＋\n疑似ラベルで学び直す",
        reg="③④ とも同じアンカーから出発し，アンカーとの差の 2 乗を罰則にして学ぶ．違うのは疑似ラベルを使うかどうかだけ",
        test="⑤ テストの発話の UAR を比べる\ngain ＝ ④ − ③",
        each="声・顔・言葉・AVT の 4 つの分類器で，それぞれ自分のアンカーから同じ手順で学び直す（図は声の例）",
        note="模式図",
    ),
    "en": dict(
        src_t="(1) Sources: the other 3 corpora (all labelled)",
        srcs=["Source 1 (e.g. IEMOCAP)", "Source 2 (e.g. MOSEI)", "Source 3 (e.g. MELD)"],
        src_note="Per corpus, train 4 linear classifiers\non frozen features",
        anc_t="(2) Anchor",
        anc_note="Average the weights and biases of\nthe 3 source classifiers per modality",
        ancs=["Audio anchor", "Visual anchor", "Text anchor", "AVT anchor"],
        tgt_t="(3)(4) Retrain on the target (e.g. AFEW)",
        klab="K labels per emotion",
        pl="Pseudo-labels on the unlabelled\ncandidates = product (next page)",
        base="(3) K-label baseline:\nretrain from the\nanchor on K labels",
        retr="(4) Pseudo-label\nretraining: from the\nanchor on K labels\n+ pseudo-labels",
        reg="(3) and (4) start from the same anchor with the same squared-distance penalty; only the pseudo-labels differ",
        test="(5) Compare the test UAR\ngain = (4) − (3)",
        each="The audio, visual, text and AVT classifiers are each retrained from their own anchor in the same way (audio shown)",
        note="Schematic",
    ),
}


def clf_icons(ax, x, y, gap=6.0, size=3.6):
    for j, m in enumerate("AVT"):
        mark(ax, x + j * gap, y, m, size=size)
    txt(ax, x + 3 * gap - 1.2, y, "AVT", va="center", size=FS - 1.6, weight="bold")


def fig_q1_model(lang):
    setup(lang)
    t = TQ[lang]
    H = 80
    fig, ax = canvas(H)
    # sources
    txt(ax, 2, H - 2, t["src_t"], weight="bold", color=C_SRC, size=FS - 0.4)
    for i, nm in enumerate(t["srcs"]):
        y = H - 17 - i * 14
        box(ax, 2, y, 44, 11, fc="#EDE7F3", ec=C_SRC, lw=0.7)
        txt(ax, 4, y + 9.5, nm, size=FS - 1.1, color=C_SRC)
        clf_icons(ax, 8, y + 3.4)
        arrow(ax, (46, y + 5.5), (56, H - 40 + 5), lw=0.5, color=C_SRC, head=5)
    txt(ax, 2, H - 61, t["src_note"], size=FS - 1.3, color=GREY, ls=1.1)
    # anchors
    txt(ax, 57, H - 2, t["anc_t"], weight="bold", size=FS - 0.4)
    box(ax, 56, H - 58, 30, 50, fc="#F5F5F5", ec="black", lw=0.7)
    for j, (m, nm) in enumerate(zip(["A", "V", "T", None], t["ancs"])):
        yy = H - 15 - j * 11
        box(ax, 58, yy - 3.5, 26, 8, fc="white", ec=MOD[m][0] if m else "black", lw=0.9, z=2)
        if m:
            mark(ax, 61, yy + 0.5, m, size=3.6)
        txt(ax, 64 if m else 60, yy + 0.5, nm, va="center", size=FS - 1.1, weight="bold" if j == 0 else "normal")
    txt(ax, 56, H - 61, t["anc_note"], size=FS - 1.3, color=GREY, ls=1.1)
    # target: audio example
    xt = 92
    txt(ax, xt, H - 2, t["tgt_t"], weight="bold", size=FS - 0.4)
    box(ax, xt, H - 15, 36, 8, fc="white", ec=C_K, lw=0.9)
    txt(ax, xt + 18, H - 11, t["klab"], ha="center", va="center", size=FS - 1.1, color=C_K)
    box(ax, xt + 40, H - 17, 40, 10, fc="white", ec=C_U, lw=0.9)
    txt(ax, xt + 60, H - 12, t["pl"], ha="center", va="center", size=FS - 1.2, color=GREY, ls=1.1)
    box(ax, xt, H - 38, 38, 15, fc="#EAF2FA", ec=C_K, lw=0.9)
    mark(ax, xt + 3, H - 26, "A", size=3.6)
    txt(ax, xt + 6, H - 24.2, t["base"], size=FS - 1.15, ls=1.15)
    box(ax, xt + 42, H - 38, 38, 15, fc="#FDF0E8", ec=C_PROP, lw=0.9)
    mark(ax, xt + 45, H - 26, "A", size=3.6)
    txt(ax, xt + 48, H - 24.2, t["retr"], size=FS - 1.15, ls=1.15)
    # arrows into the two classifiers
    arrow(ax, (xt + 10, H - 15), (xt + 10, H - 23), lw=0.7, color=C_K)
    arrow(ax, (xt + 30, H - 15), (xt + 52, H - 23), lw=0.7, color=C_K)
    arrow(ax, (xt + 62, H - 17), (xt + 62, H - 23), lw=0.7, color=GREY)
    # from the audio anchor
    ax.plot([84, 89], [H - 14.5, H - 14.5], color=MOD["A"][0], lw=1.0, zorder=3)
    ax.plot([89, 89], [H - 14.5, H - 42], color=MOD["A"][0], lw=1.0, zorder=3)
    ax.plot([89, xt + 70], [H - 42, H - 42], color=MOD["A"][0], lw=1.0, zorder=3)
    arrow(ax, (xt + 28, H - 42), (xt + 28, H - 38), lw=1.0, color=MOD["A"][0])
    arrow(ax, (xt + 70, H - 42), (xt + 70, H - 38), lw=1.0, color=MOD["A"][0])

    # test
    box(ax, xt + 20, H - 64, 42, 11, fc="white", ec="black", lw=0.9)
    txt(ax, xt + 41, H - 58.5, t["test"], ha="center", va="center", size=FS - 1.0, ls=1.15)
    arrow(ax, (xt + 12, H - 38), (xt + 26, H - 53), lw=0.7)
    arrow(ax, (xt + 56, H - 38), (xt + 56, H - 53), lw=0.7)
    txt(ax, 2, 10.0, t["each"], size=FS - 0.9, weight="bold")
    txt(ax, 2, 5.0, t["reg"], size=FS - 0.9)
    txt(ax, W - 1, 0.6, t["note"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_q1_model" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ how the product pseudo-labels are made (Sec. 5.1)
TP = {
    "ja": dict(
        s1="1 段目：AVT の K 本の基準の分類器で，\n候補（ラベル無しの発話）に確率を付ける",
        s2="2 段目：声・顔・言葉の分類器を，それぞれのアンカーから\nK 本のラベル ＋ 1 段目の確率で学び直す",
        clf=["声の分類器", "顔の分類器", "言葉の分類器"],
        mult="3 つの確率を発話ごとに掛けて，\n合計が 1 になるように割る",
        q="積 q(x)\n＝ Q1・Q2 の疑似ラベル",
        use="q を疑似ラベルにして，声・顔・言葉・AVT の\n各分類器をアンカーから学び直す（前のページの ④）",
        own="q には，学び直す分類器自身のモダリティの答えも入っている",
        note="模式図．棒は説明のための例",
    ),
    "en": dict(
        s1="Stage 1: the AVT K-label baseline gives\nprobabilities on the candidates (unlabelled)",
        s2="Stage 2: retrain the audio, visual and text classifiers from\ntheir anchors on K labels + the stage-1 probabilities",
        clf=["Audio classifier", "Visual classifier", "Text classifier"],
        mult="Multiply the 3 probabilities per\nutterance and renormalise",
        q="Product q(x)\n= Q1–Q2 pseudo-labels",
        use="Retrain each of the audio, visual, text and AVT\nclassifiers from its anchor on q (step (4))",
        own="q contains the answer of the retrained classifier's own modality",
        note="Schematic; bars are illustrative",
    ),
}


def fig_product_pl(lang):
    setup(lang)
    t = TP[lang]
    H = 66
    fig, ax = canvas(H)
    box(ax, 2, H - 13, 62, 10, fc="#EAF2FA", ec=C_K, lw=0.9)
    txt(ax, 4, H - 5, t["s1"], size=FS - 0.9, ls=1.15)
    box(ax, 2, H - 31, 86, 12, fc="white", ec="black", lw=0.8)
    txt(ax, 4, H - 21.5, t["s2"], size=FS - 0.9, ls=1.15)
    arrow(ax, (30, H - 13), (30, H - 19), lw=0.7)
    vals = {"A": [0.45, 0.25, 0.2, 0.1], "V": [0.2, 0.5, 0.15, 0.15], "T": [0.4, 0.35, 0.15, 0.1]}
    for j, m in enumerate("AVT"):
        x = 4 + j * 29
        bx = x + 2
        box(ax, x, 6, 26, 21, fc="white", ec=MOD[m][0], lw=0.8)
        mark(ax, x + 3, 23.5, m, size=3.6)
        txt(ax, x + 6, 23.5, t["clf"][j], va="center", size=FS - 1.3)
        for c, v in enumerate(vals[m]):
            ax.add_patch(__import__("matplotlib").patches.Rectangle((bx + 1 + c * 5.5, 10), 3.6, v * 18, fc=MOD[m][0], ec="none", zorder=3))
            txt(ax, bx + 2.8 + c * 5.5, 9.6, EMO[c], ha="center", size=FS - 2.0, color=GREY)
        ax.plot([bx, bx + 22], [10, 10], color="black", lw=0.4, zorder=4)
        arrow(ax, (x + 13, H - 31), (x + 13, 27), lw=0.6)
    box(ax, 96, 30, 36, 12, fc="white", ec="black", lw=0.8)
    txt(ax, 98, 39.5, t["mult"], size=FS - 1.0, ls=1.15)
    for j in range(3):
        ax.plot([4 + j * 29 + 13, 4 + j * 29 + 13], [6, 2.5], color="black", lw=0.6, zorder=3)
    ax.plot([17, 92], [2.5, 2.5], color="black", lw=0.6, zorder=3)
    ax.plot([92, 92], [2.5, 34], color="black", lw=0.6, zorder=3)
    arrow(ax, (92, 34), (96, 34), lw=0.6)
    pa = np.array(vals["A"]) * np.array(vals["V"]) * np.array(vals["T"])
    pa = pa / pa.sum()
    box(ax, 138, 22, 34, 22, fc="#F5F5F5", ec="black", lw=1.0)
    txt(ax, 140, 41.5, t["q"], size=FS - 0.8, weight="bold", ls=1.15)
    for c, v in enumerate(pa):
        ax.add_patch(__import__("matplotlib").patches.Rectangle((143 + c * 6.5, 25), 4.2, v * 10, fc="#555555", ec="none", zorder=3))
        txt(ax, 145.1 + c * 6.5, 24.6, EMO[c], ha="center", size=FS - 2.0, color=GREY)
    arrow(ax, (132, 36), (138, 36), lw=0.8)
    box(ax, 96, 5, 76, 12, fc="#FDF0E8", ec=C_PROP, lw=0.9)
    txt(ax, 98, 14.5, t["use"], size=FS - 0.9, ls=1.15)
    arrow(ax, (155, 22), (155, 17), lw=0.8)
    txt(ax, 96, H - 5, t["own"], size=FS - 1.2, color=GREY)
    txt(ax, W - 1, 0.4, t["note"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_product_pl" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ the three settings (Sec. 4.3, Table 2)
T3 = {
    "ja": dict(
        cols=[("単一ソース設定", "6.1〜6.3 節"), ("全ソース設定", "6.4 節"), ("LOCO 設定", "5 節，6.5〜6.7 節，7 節")],
        src=["ソース：1 つの群\n（全発話にラベル）", "ソース：ターゲット以外の\n全部の群", "ソース：残る 3 コーパス"],
        tgt=["ターゲット：1 つの群", "ターゲット：1 つの群", "ターゲット：AFEW，IEMOCAP（即興），\nMOSEI，MELD の 1 つ（7 節は CREMA-D も）"],
        how=["ソースの全発話と K 本を\nまとめて学ぶ（合併の分類器）",
             "全部の群から作ったアンカーを\nK 本で学び直す",
             "3 つのソースの分類器の平均を\nアンカーにし，K 本で学び直す"],
        clfs=["合併の分類器\nターゲットだけの分類器（K 本だけ）\nソースだけの分類器（ソースだけ）",
              "K 本の基準の分類器\n疑似ラベルによる学び直し\nラベル無しの発話を使う 8 手法",
              "K 本の基準の分類器\n疑似ラベルによる学び直し"],
        ks=["K = 1，5，10，25，50，全件", "K = 0，5，25", "K = 0，5，10，25"],
        ask=["ほかのコーパスと K 本が\n誤りのどの部分を直すか", "疑似ラベルと同じモダリティの\nラベル無しの発話が何を直すか", "疑似ラベルで UAR が上がるか（Q1），\nなぜか（Q2-c），まとめ方（Q3）"],
        note="模式図．原稿の表 2 を図にした",
    ),
    "en": dict(
        cols=[("Single-source", "Sec. 6.1–6.3"), ("All-source", "Sec. 6.4"), ("LOCO", "Sec. 5, 6.5–6.7, 7")],
        src=["Source: one group\n(all labelled)", "Source: all groups\nexcept the target", "Source: the other 3 corpora"],
        tgt=["Target: one group", "Target: one group", "Target: one of AFEW, IEMOCAP (impro.),\nMOSEI, MELD (Sec. 7 adds CREMA-D)"],
        how=["train on the source + K labels\ntogether (pooled classifier)",
             "retrain an anchor built from\nall groups on K labels",
             "average of the 3 source classifiers\nas anchor; retrain on K labels"],
        clfs=["pooled classifier\ntarget-only (K labels)\nsource-only",
              "K-label baseline\npseudo-label retraining\n8 methods using unlabelled data",
              "K-label baseline\npseudo-label retraining"],
        ks=["K = 1, 5, 10, 25, 50, all", "K = 0, 5, 25", "K = 0, 5, 10, 25"],
        ask=["which part of the error do other\ncorpora and K labels correct?", "what do pseudo-labels and same-modality\nunlabelled data correct?", "does UAR rise (Q1), why (Q2-c),\nhow to fuse (Q3)?"],
        note="Schematic of Table 2",
    ),
}


def fig_three_settings(lang):
    """Slide-readable version: few words, large type (the figure is shown about 0.85 times its size)."""
    setup(lang)
    big = {"ja": dict(
        cols=[("単一ソース設定", "6.1〜6.3 節"), ("全ソース設定", "6.4 節"), ("LOCO 設定", "5，6.5〜7 節")],
        src=["ソース：1 つの群", "ソース：ターゲット\n以外の全部の群", "ソース：残る 3 コーパス"],
        tgt=["ターゲット：1 つの群", "ターゲット：1 つの群", "ターゲット：4 コーパスの 1 つ\n（7 節は CREMA-D も．ソース 4 つ）"],
        how=["ソースの全発話と K 本を\nまとめて学ぶ", "全部の群のアンカーを\nK 本で学び直す", "3 つのソースの平均を\nアンカーに K 本で学び直す"],
        clfs=["合併の分類器\nターゲットだけの分類器\nソースだけの分類器", "K 本の基準の分類器\n疑似ラベルで学び直し\n同じモダリティの 8 手法", "K 本の基準の分類器\n疑似ラベルで学び直し"],
        ks=["K = 1〜全件", "K = 0，5，25", "K = 0，1，5，10，25"],
        note="模式図．原稿の表 2 を図にした（6.7 節も LOCO 設定の出力を使う）"),
        "en": dict(
        cols=[("Single-source", "Sec. 6.1–6.3"), ("All-source", "Sec. 6.4"), ("LOCO", "Sec. 5, 6.5–7")],
        src=["Source: one group", "Source: all groups\nexcept the target", "Source: the other 3 corpora"],
        tgt=["Target: one group", "Target: one group", "Target: one of 4 corpora\n(Sec. 7 adds CREMA-D)"],
        how=["train on source + K labels\ntogether", "retrain the all-group\nanchor on K labels", "anchor = mean of 3 sources;\nretrain on K labels"],
        clfs=["pooled classifier\ntarget-only\nsource-only", "K-label baseline\npseudo-label retraining\n8 same-modality methods", "K-label baseline\npseudo-label retraining"],
        ks=["K = 1 to all", "K = 0, 5, 25", "K = 0, 5, 10, 25"],
        note="Schematic of Table 2")}[lang]
    H = 69
    fig, ax = canvas(H)
    cw = 56
    fs = FS + 1.6
    for i in range(3):
        x = 1 + i * (cw + 2)
        box(ax, x, 4, cw, H - 5, fc="#FAFAFA", ec="black", lw=0.9)
        txt(ax, x + 2, H - 3.2, big["cols"][i][0], weight="bold", size=fs + 1.2)
        txt(ax, x + cw - 2, H - 3.6, big["cols"][i][1], ha="right", size=fs - 2.2, color=GREY)
        box(ax, x + 2, H - 20, cw - 4, 10, fc="#EDE7F3", ec=C_SRC, lw=0.8, z=2)
        txt(ax, x + 3.5, H - 11.3, big["src"][i], size=fs, color=C_SRC, ls=1.1)
        box(ax, x + 2, H - 32, cw - 4, 10, fc="white", ec=C_K, lw=1.0, z=2)
        txt(ax, x + 3.5, H - 23.3, big["tgt"][i], size=fs, color=C_K, ls=1.1)
        txt(ax, x + 2, H - 34.3, big["how"][i], size=fs - 0.4, ls=1.1)
        box(ax, x + 2, H - 56, cw - 4, 13.5, fc="white", ec="black", lw=0.8, z=2)
        txt(ax, x + 3.5, H - 44.3, big["clfs"][i], size=fs - 0.4, ls=1.15)
        txt(ax, x + 2, H - 57.5, big["ks"][i], size=fs, weight="bold")
    txt(ax, W - 1, 0.4, big["note"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_three_settings" + ("_ja" if lang == "ja" else ""))


# ------------------------------------------------------------------ pipeline of the experiments (Ryumina et al. 2025, Fig. 1 style)
TPL = {
    "ja": dict(
        heads=["コーパス（LOCO）", "モダリティの切り出し", "凍結した特徴（発話で平均）", "線形分類器", "ターゲットで学び直す"],
        corp_note="1 つをターゲット，\n残る 3 つをソースに\nする（4 回）",
        mods=["声", "顔", "言葉"],
        cut=["16 kHz の波形\n（先頭 60 秒）", "顔検出（YOLOv8n-face）\n1 秒に 5〜6 枚，最も大きい顔", "書き起こし\n（AFEW は Whisper large-v3）"],
        enc=["WavLM-large 最終層\n→ 1,024 次元", "ResNet-50 表情モデル\n→ 512 次元", "RoBERTa-base 最終層\n→ 768 次元"],
        avt="AVT：3 つをつなぐ（2,304 次元）",
        clf_note="ソースごとに学び，3 つの\n平均をアンカーにする",
        clfs=["声の分類器", "顔の分類器", "言葉の分類器", "AVT の分類器"],
        tgt=["K 本の基準の分類器\nアンカーから K 本だけで", "疑似ラベルによる学び直し\nK 本 ＋ 候補の疑似ラベル\n（6 節は積，7 節は提案など）", "テストの発話の UAR\n（話者かクラスタで分ける．50 反復）"],
        note="実験の流れ（模式図）．CREMA-D（7 節）は声と顔だけで，ソースは 4 コーパス",
    ),
    "en": dict(
        heads=["Corpora (LOCO)", "Modality separation", "Frozen features (utterance mean)", "Linear classifiers", "Retraining on the target"],
        corp_note="one corpus is the target,\nthe other 3 are sources\n(4 runs)",
        mods=["Audio", "Visual", "Text"],
        cut=["16 kHz waveform\n(first 60 s)", "face detection (YOLOv8n-face),\n5–6 frames/s, largest face", "transcript\n(AFEW: Whisper large-v3)"],
        enc=["WavLM-large last layer\n→ 1,024 dims", "ResNet-50 emotion model\n→ 512 dims", "RoBERTa-base last layer\n→ 768 dims"],
        avt="AVT: concatenation (2,304 dims)",
        clf_note="trained per source; the\nmean of the 3 is the anchor",
        clfs=["Audio classifier", "Visual classifier", "Text classifier", "AVT classifier"],
        tgt=["K-label baseline\nfrom the anchor on K labels", "Pseudo-label retraining\nK labels + pseudo-labels\n(Sec. 6: product;\nSec. 7: proposed, etc.)", "Test UAR\n(speaker/cluster-disjoint, 50 reps)"],
        note="Pipeline of the experiments (schematic). CREMA-D (Sec. 7): audio and visual only; 4 source corpora",
    ),
}


def fig_pipeline(lang):
    setup(lang)
    t = TPL[lang]
    H = 80
    fig, ax = canvas(H)
    xs = [1, 27, 63, 103, 136]
    ws = [24, 34, 38, 31, 37]
    hcol = ["#EEEEEE", "#FCEFD9", "#E3EEF7", "#ECE6F2", "#FDF0E8"]
    ecol = ["#777777", "#C98A1B", "#2F6FA6", C_SRC, C_PROP]
    top = H - 2
    for x, w, h, fc, ec in zip(xs, ws, t["heads"], hcol, ecol):
        box(ax, x, 8, w, top - 8, fc=fc, ec=ec, lw=0.9, r=1.2)
        box(ax, x, top - 6, w, 6, fc=ec, ec=ec, lw=0.9, r=1.0, z=2)
        txt(ax, x + w / 2, top - 3, h, ha="center", va="center", size=FS - 1.4, color="white", weight="bold")
    # corpora
    cs = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
    for i, c in enumerate(cs):
        y = top - 15 - i * 9
        tgt = i == 0
        box(ax, xs[0] + 2, y, ws[0] - 4, 7, fc="white" if tgt else "#EDE7F3", ec="black" if tgt else C_SRC, lw=1.2 if tgt else 0.6, z=3)
        txt(ax, xs[0] + ws[0] / 2, y + 3.5, c, ha="center", va="center", size=FS - 1.2, weight="bold" if tgt else "normal", color="black" if tgt else C_SRC)
    txt(ax, xs[0] + 2, top - 52, t["corp_note"], size=FS - 1.9, color=GREY, ls=1.1)
    # rows: audio, visual, text
    rows_y = [top - 18, top - 33, top - 48]
    for r, m in enumerate("AVT"):
        y = rows_y[r]
        col = MOD[m][0]
        box(ax, xs[1] + 2, y - 5.5, ws[1] - 4, 11, fc="white", ec=col, lw=1.0, z=3)
        mark(ax, xs[1] + 4.5, y + 3, m, size=3.4)
        txt(ax, xs[1] + 7, y + 3, t["mods"][r], va="center", size=FS - 1.0, weight="bold", color=col)
        txt(ax, xs[1] + 4, y + 0.3, t["cut"][r], size=FS - 2.1, ls=1.1)
        box(ax, xs[2] + 2, y - 5.5, ws[2] - 4, 11, fc="white", ec=col, lw=1.0, z=3)
        txt(ax, xs[2] + 4, y + 3.6, t["enc"][r], size=FS - 1.5, ls=1.15)
        arrow(ax, (xs[0] + ws[0], y), (xs[1] + 2, y), lw=0.6, head=5)
        arrow(ax, (xs[1] + ws[1] - 2, y), (xs[2] + 2, y), lw=0.7, head=5, color=col)
        arrow(ax, (xs[2] + ws[2] - 2, y), (xs[3] + 2, y), lw=0.7, head=5, color=col)
    # AVT concat
    ya = top - 61
    box(ax, xs[2] + 2, ya - 3.5, ws[2] - 4, 7, fc="#F5F5F5", ec="black", lw=0.8, z=3)
    txt(ax, xs[2] + ws[2] / 2, ya, t["avt"], ha="center", va="center", size=FS - 1.6)
    arrow(ax, (xs[2] + ws[2] - 2, ya), (xs[3] + 2, ya), lw=0.7, head=5)
    # classifiers
    for r, (yy, lab) in enumerate(zip(rows_y + [ya], t["clfs"])):
        col = MOD["AVT"[r]][0] if r < 3 else "black"
        box(ax, xs[3] + 2, yy - 3.2, ws[3] - 4, 6.4, fc="white", ec=col, lw=1.0, z=3)
        txt(ax, xs[3] + ws[3] / 2, yy, lab, ha="center", va="center", size=FS - 1.4)
        ax.plot([xs[3] + ws[3] - 2, xs[3] + ws[3] + 0.6], [yy, yy], color=GREY, lw=0.6, zorder=2)
    ax.plot([xs[3] + ws[3] + 0.6, xs[3] + ws[3] + 0.6], [ya, rows_y[0]], color=GREY, lw=0.6, zorder=2)
    arrow(ax, (xs[3] + ws[3] + 0.6, top - 30), (xs[4] + 2, top - 30), lw=1.0, head=6, color=GREY)
    txt(ax, xs[3] + 2, top - 7.5, t["clf_note"], size=FS - 2.0, color=C_SRC, ls=1.1)
    # target retraining
    hs = [11, 15, 10]
    y = top - 10
    for i, (lab, h) in enumerate(zip(t["tgt"], hs)):
        ec = C_K if i == 0 else (C_PROP if i == 1 else "black")
        box(ax, xs[4] + 2, y - h, ws[4] - 4, h, fc="white", ec=ec, lw=1.0, z=3)
        txt(ax, xs[4] + 3.5, y - 1.2, lab, size=FS - 1.8, ls=1.15)
        y -= h + 3
    txt(ax, W - 1, 1.0, t["note"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_pipeline" + ("_ja" if lang == "ja" else ""))


if __name__ == "__main__":
    for lang in ("ja", "en"):
        fig_setup(lang)
        fig_glyphs(lang)
        fig_method(lang)
        fig_example(lang)
        fig_q1_model(lang)
        fig_product_pl(lang)
        fig_three_settings(lang)
        fig_pipeline(lang)
