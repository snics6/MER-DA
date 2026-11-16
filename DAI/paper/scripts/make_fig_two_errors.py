"""Slide figure (2026-11-16 night, researcher: "p6の前に，二つの誤りの説明が必要"):

fig_two_errors_ja : the two parts of a linear classifier's error, drawn in a 2-D feature space.
  (a) a classifier whose boundary separates the two emotions,
  (b) answer bias (intercept shift): the boundary has the right direction but is shifted;
      moving it in parallel (changing only the intercepts) removes this error,
  (c) remaining error (direction of the weights): the boundary is tilted; no parallel shift removes it,
      only changing the weights (the direction) does.
Illustration only: the points are drawn from two made-up Gaussians (seed 0), not from the experiments.
Output: ../figures/fig_two_errors_ja.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_two_errors.py
"""
import matplotlib.pyplot as plt
import numpy as np

from make_figs_overview import FS, GREY, MM, W, save, setup

C1, C2 = "#0072B2", "#D55E00"   # emotion k (circles), emotion l (triangles)


def panel(ax, pts1, pts2, w, b, title, note, good_b=None, good_w=None):
    ax.scatter(pts1[:, 0], pts1[:, 1], s=7, marker="o", fc="none", ec=C1, lw=0.6)
    ax.scatter(pts2[:, 0], pts2[:, 1], s=8, marker="^", fc="none", ec=C2, lw=0.6)
    def line(bb, **kw):   # boundary w.x + bb = 0, w = (w0, w1)
        if abs(w[1]) < 1e-9:
            ax.plot([-bb / w[0]] * 2, [-3.2, 3.2], **kw)
        else:
            xs = np.linspace(-4, 4, 50)
            ax.plot(xs, -(w[0] * xs + bb) / w[1], **kw)
    line(b, color="black", lw=1.4)
    if good_b is not None:
        line(good_b, color="black", lw=0.9, ls="--")
    if good_w is not None:
        ax.plot([0, 0], [-3.2, 3.2], color="black", lw=0.9, ls="--")
    # count errors
    s1 = pts1 @ w + b
    s2 = pts2 @ w + b
    e1 = (s1 < 0)
    e2 = (s2 > 0)
    ax.scatter(pts1[e1, 0], pts1[e1, 1], s=7, marker="o", fc=C1, ec=C1, lw=0.6)
    ax.scatter(pts2[e2, 0], pts2[e2, 1], s=8, marker="^", fc=C2, ec=C2, lw=0.6)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")
        s.set_linewidth(0.5)
    ax.set_title(title, fontsize=FS + 1.2, loc="left", pad=3)
    ax.text(0.0, -0.05, note, transform=ax.transAxes, ha="left", va="top", fontsize=FS - 0.2, linespacing=1.35)
    return int(e1.sum() + e2.sum())


def main():
    setup("ja")
    rng = np.random.default_rng(0)
    n = 60
    cov = np.array([[0.55, 0.0], [0.0, 1.1]])
    p1 = rng.multivariate_normal([1.3, 0.0], cov, n)    # emotion k
    p2 = rng.multivariate_normal([-1.3, 0.0], cov, n)   # emotion l
    fig = plt.figure(figsize=(W * MM, 70 * MM))
    axs = [fig.add_axes([0.02 + i * 0.335, 0.27, 0.30, 0.60]) for i in range(3)]
    w_good = np.array([1.0, 0.0])
    panel(axs[0], p1, p2, w_good, 0.0, "(a) 境目が 2 つの感情を分けている",
          "線（境目）の右を感情 k，左を感情 l と\n答える．塗りつぶした印が誤り")
    panel(axs[1], p1, p2, w_good, -1.25, "(b) 答えの偏り（切片のずれ）",
          "線の向きは正しいが平行にずれ，答えが\n感情 l に偏る．切片だけを直して線を\n戻す（破線）と，(a) と同じ誤りに戻る",
          good_b=0.0)
    th = np.deg2rad(62)
    w_tilt = np.array([np.cos(th), np.sin(th)])
    panel(axs[2], p1, p2, w_tilt, 0.0, "(c) 残りの誤り（重みの向きのずれ）",
          "線が傾き，線の近くで 2 つの感情が混ざる．\n平行にずらしても (a) の数までは減らない．\n重み（線の向き）を直す（破線）必要がある",
          good_w=True)
    for ax in axs:
        ax.text(3.8, 2.9, "感情 k ○", ha="right", va="top", fontsize=FS - 0.4, color=C1)
        ax.text(-3.8, 2.9, "△ 感情 l", ha="left", va="top", fontsize=FS - 0.4, color=C2)
    fig.text(0.02, 0.955, "2 次元の特徴で描いた模式図．点は説明のために作った例で，実験の値ではない",
             fontsize=FS - 0.6, color=GREY)
    save(fig, "fig_two_errors_ja")


if __name__ == "__main__":
    main()
