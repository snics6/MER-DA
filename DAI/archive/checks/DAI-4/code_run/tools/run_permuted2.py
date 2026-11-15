"""学習の経路がラベルを読まないことの検査：K 本の外の候補のラベルとテストのラベルを並べ替えて同じ単位を解き，W・b・判定・印・重みが変わらないかを見る。"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai1, dai4 as D4
orig = dai1.Unit
class PUnit(orig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rng = np.random.default_rng(123)
        out = np.setdiff1d(np.arange(self.n), self.kpos[25])
        yp = self.ypool.copy(); yp[out] = rng.permutation(yp[out]); self.ypool = yp
        self.ytest = rng.permutation(self.ytest)
        print("permuted: n_out", len(out), "labels moved frac", float((yp != orig_y(self)).mean()), flush=True)
def orig_y(u): return u.cp.y[u.pool]
dai1.Unit = PUnit
sys.argv = ["dai4.py", "--stage", "pseudo", "--out-dir", "/home/snics/MER-DA/DAI/checks/DAI-4/code_run/out2/perm", "--allow-dirty",
            "--targets", "AFEW", "--cs", "MELD", "--folds", "1", "--seeds", "0"]
D4.main()
