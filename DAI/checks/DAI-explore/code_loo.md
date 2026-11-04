（検査役は Sonnet のサブエージェント．報告の .md を書けなかったので，本体が記録から取り出して保存した．対象は `DAI/exp/explore_reliability_loo.py` の第 1 版．2026-11-05）

判定：直してから進む．台本は設計どおりの数を計算し，保存済みの値も再現できたが，LOO が発話単位であり話者（塊）を分けていないこと，順位の同点で A が勝つこと，事前分布だけの対照が無いことを，本番の前に直す．

# 関門 1：実装と出力（DAI 探索 `explore_reliability_loo.py`，2026-11-05，コミット b6493db．台本は未追跡）

## 判定
直してから進む．確認 1〜3・5 は問題なし．確認 4 に直す点が 3 つある（P1：話者を分けた LOO の追加，P2：同点の扱い，P3：事前分布だけの対照）．AFEW の 50 単位の試走は，発話単位 LOO の列だけなら今の台本のままで有効である．塊ごとの LOO の列を足すなら AFEW は流し直す（所要は約 5 分）．

## A．設計と台本の対応表
| 設計（docstring）の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 単位 (T,f,r)，K∈{5,25}，m∈{A,V,T} | :35, :79-93 | 合っている．p3 が無い単位は :84 で黙って飛ばす．AFEW の 50 単位は p3 がそろっている |
| LOO：1 本抜いた 4K−1 本で none と同じ形（出発点＝ソース平均，引き戻し＝固定の値，釣り合い）を学ぶ | :95-98 | 合っている．同じ設定の none の保存値（p1 の W・b）を P=1 で再現した（下の検算 a） |
| 4K 個の問を 1 束で解く | :95-98 | 合っている．1 本ずつ解いた結果と一致した（検算 b） |
| loo_acc・loo_uar（抜いた発話の判定） | :99-101 | 合っている．`Z[arange,arange]` は問 i の発話 i を取り出している（検算 b） |
| none[K] のテスト UAR | :103 | 値は合っている．p2 の確率は half 保存で，その argmax と float の argmax の食い違いは 3727 本中 0〜1 本（記録のみ） |
| 直した単独の識別器の確率・判定（explore_fusion_grid と同じ） | :104-109 | 合っている．AVT の生徒（`avt_joint`）と `prod111` が，保存済みの `results/explore_fusion/rule_units.csv` と最大差 0.0 で一致した（AFEW の 1 単位） |
| 順位の一致（最強モダリティ，Spearman） | :120-123 | LOO の順位と直した単独の識別器の順位だけが計算される．docstring 8 行目の none[K] の強さ（`none_test_m`）は列に保存されるが，順位の一致には使われない（P2・P3 を参照） |
| 積の融合 (a) rule_loo | :53-55, :117 | 合っている |
| 同 (b) rule_loo_prior：post=(n_loo·loo+n0·prior)/(n_loo+n0)，n0=20，n_loo=4K | :36, :115 | docstring と一致．`source_prior` は `cv_uar` 列（% 単位）を選び，LOO の UAR（% 単位）と尺度が一致した |
| (c) (1,1,1)，上限，AVT の生徒 | :117-118, :74-77 | 合っている．ただし `<out>/grid_units.csv` が無いと :74 で最初に落ちる．上限は 50 単位平均で，一部の単位だけを流すと他の列と母集団が食い違う |
| 出力 loo_units.csv，loo_cells.csv | :126-134 | 合っている．`stalled` の集計は `loo_stalled_A` だけである（:131）．V・T の分が落ちる |

設計の表に載る世界は T=4 × K=2 の 8 セルで，台本はすべて計算する．計算しない組は無い．

## A．小さく流した結果
小さい版は `checks/DAI-explore/code_run_loo/loo_small.py`（台本の複製．単位の数だけ環境変数で絞る）を使った．パスは `DAI/checks/DAI-explore/code_run_loo/`．`results/` には何も書いていない．`MERDA_GPU_FRACTION=0.2` で流した．

- AFEW 1 単位は 16.7 秒で完走した．NaN・Inf は 0，収束しない問は 0 だった．最後の 1 つ（`loo_gradmax`）は 1e-6 以下．
- 検算 a（保存値の再現）：K=5・25，A・V・T，AFEW と MOSEI の 1 単位ずつについて，G=gK(K) を P=1 で解いた W・b と，p1 の none の保存値を比べた．差は W で 3.6e-9 以下，b で 1.6e-8 以下で，出発点・引き戻し・目的関数は pass1 と同じである．
- 検算 b（束と単独）：問 i=0，nK/2，nK−1 を 1 問ずつ解いた．抜いた発話の点数との差は 3e-12 以下で，判定も一致した．また，問 i の G の和は 4K−1 であり，G[kp[i]] の和は 0 だった（assert 済み）．
- 検算 c（突き合わせ）：`avt_joint` と `prod111` の差は 0.0．
- 全問で `ok=True`（勾配が 1e-5 以下かつ有限）だった．
- 退化した場合の検算（項の重みを 0 にする版）は，この台本には対応する項が無い．代わりに，指数 (1,1,1) の積が保存済みの explore_fusion_grid の値と一致することで，積の式を確かめた．
- `prod` の奇妙な式は，Python の条件式の優先順位により `float(ei)+0.0 if False else float(ei)*logP[m]` が常に else 側になる．そのため意図どおり Σ e_m log P_m になる（上の検算 c で確認済み）．

## A．本番の規模の見積もり
- メモリ：(100 × n × d) の積は作られない．`einsum("nd,pkd->pnk")` は X（n×d）を共有し，出力は (P,n,4) の float64 である．MOSEI（n=14907）K=25 の P=100 で実測した GPU の最大は 2.9 GiB だった．このうち 2.6 GiB は常駐のデータで，LOO の分は 0.3 GiB 程度である．上限（0.2×24=4.8 GiB，0.3 なら 7.2 GiB）に収まる．問を分ける必要は無い．
- 注意：複数の T を 1 つのプロセスで続けて流すと，`dai1.Corpus.cache` が 4 コーパス分（1.0 + 1.3 GiB 程度）を保持し続ける．MOSEI を最後にすると 0.2 では収まらないおそれがある（私の検算用の台本は 3 コーパスを続けて読んで OOM になった）．T ごとにプロセスを分けるか，T ごとの終わりに `dai1.Corpus.cache.clear(); torch.cuda.empty_cache()` を呼ぶ．0.3 なら 4 つを続けても収まる見込みだが，未確認である．
- 時間（実測，1 単位あたり）：AFEW 約 8 秒，IEMOCAP 約 8 秒，MELD 約 12 秒，MOSEI 約 20 秒（K=25 の A で 6.9 秒，V・T で約 3 秒）．50 単位 × 4 コーパスで合計およそ 40 分．AFEW の 50 単位は 5〜7 分で，「約 20 分」より短い（GPU を共有するとさらに遅くなる）．

## B．予想される指摘と，出力から答えられるか
| 表・図 | 予想される指摘 | もとにした項目か例 | (a)(b)(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| loo_cells（LOO の順位の一致） | LOO は発話単位で，同じ話者の発話が学習側にもあり，K 本が話者で偏る．音声は話者の情報が強いので音声だけ水増しになる | S・L（データの取り決め 5），過去の指摘「話者を単位に」 | (b) | 塊ごとに抜く LOO の列 `cl_uar_m`（P1） | 問の数は塊の数（≤4K）で，LOO より少ない．秒単位 |
| 同上 | 同点で A が選ばれる偏り，Spearman が NaN になる | A（結果の読み方） | (b) | 同点の記録と，連続な量（抜いた発話の対数損失）の列（P2） | 無視できる |
| rule_loo_prior | 事前分布だけで同じ結果が出るのではないか（K=5 は重みが事前 1/2） | C（対照），過去の指摘「部品を外せ」 | (b) | `rule_prior`（n_loo=0）の列（P3） | 積の計算が 1 回増えるだけ |
| 順位の一致 | none[K] の強さの順位との一致も出せ | docstring 8 行目 | (a) | `loo_units.csv` の `loo_uar_m` と `none_test_m` から，heavy と Spearman を単位ごとに計算できる | 0 |
| loo_cells の差（rule − avt_joint など） | 差の信頼区間を話者（塊）を単位に作り直せ | 過去の指摘（review-ml） | (b) | 単位ごとのテストの判定（int8）を各規則について保存し，塊の添字は p1 の `sample_test` と `DAI/splits/<T>.csv` から作る | 8 MB 程度（200 単位 × 規則 5 つ × 約 4 KB） |
| 同上 | 感情ごとの再現率，ターゲットごとの内訳 | 過去の指摘 | ターゲットごとは (a)．感情ごとは (b) | 上の判定の保存で足りる | 同上 |
| LOO の信頼度 | 信頼度の校正，抜いた発話の点数を見せよ | A | (b) | 抜いた発話の点数 (nK,4) を m ごとに保存（`loo_held.npz`，`sample_id` つき） | 1 単位 5 KB 程度 |
| 全体 | C の別の規則，新しいコーパス，エンコーダの微調整 | 過去の指摘 | (c) | 設計の節に「範囲の外」と書くか，限界に書く（`tether` は仮のターゲットで固定した値のみ使うので，C の規則の感度はこの探索の範囲の外．台本に書かれていない） | 0 |
| 上限 | 上限はテストで選んだので楽観的である | S・L | (a) | docstring に「上限」と書かれている．報告でもそう書けば足りる | 0 |

## 指摘
### [S／L，話者の取り決め 5] LOO が発話単位で，話者（塊）を分けていない
- 重さ：直してから進む
- 該当箇所：`explore_reliability_loo.py:95-101`
- 問題：K 本の中の 1 本 i だけを抜くので，同じ話者・同じ塊の別の発話が学習側に残る．テストは塊で分かれているので，LOO の信頼度はテストの状況を表さない．実測した K 本の塊の構成は次のとおりである．
  - AFEW の K=25 は，塊の数が 24〜38 で，100 本のうち 80〜90 本が同じ塊の別の 1 本と並ぶ．最大の塊は 11〜21 本ある．
  - IEMOCAP は塊が 4（セッション）しかなく，K=25 では 100 本すべてが別の 1 本と塊を共有する．
  - MELD の K=25 も 80〜91 本が塊を共有する．
  - MOSEI だけは塊がほぼ全部別である．
  - 発話単位 LOO と塊ごとの LOO を，4 単位 × K 2 つ × A・V・T で比べた．IEMOCAP と AFEW の 16 セルのうち 2 セルで，モダリティの順位が変わった．UAR の差は，AFEW の K=25 で 1〜6 ポイント，IEMOCAP では最大 15 ポイントあった．音声が水増しされる傾向があるが，符号は一定ではない．
- 計算の後で直せるか：直せる．ただし LOO を学び直す必要があるので，今の台本のままでは出力から出せない．
- 求める対応：発話単位の LOO は残し，塊ごとの LOO を足す（`:95` の for m の中の，`rec[f"loo_stalled_..."]` の後に挿入）．私が小さく流して，動くことと `ok=True` を確かめた形は次のとおりである（`code_run_loo/cluster_loo.py` に実装がある）．

```
cl = u.cp.df["cluster"].to_numpy()[u.pool][kp]; uc = pd.unique(cl)      # m のループの外，:90 の前でよい
...
Gc = torch.stack([S.balanced_onehot(u.ypool, kp[cl != c], u.n) for c in uc])
rc = S.fit(u.Xp[m], Gc, W0, b0, Cw=cw, Cb=cb)
Zc = S.logits(u.Xp[m][kp], rc["W"], rc["b"])                             # (塊の数, 4K, 4)
heldc = np.zeros(nK, int)
for p, c in enumerate(uc):
    h = np.flatnonzero(cl == c)
    heldc[h] = Zc[p][torch.as_tensor(h, device=Zc.device)].argmax(1).cpu().numpy()
rec[f"cl_uar_{m}"] = uar(yk, heldc); rec[f"cl_stalled_{m}"] = int(rc["stalled"]); rec["n_clusters"] = len(uc)
```

  `len(uc) < 2` のときは NaN を入れる．実測では K=5 の最小でも 4 塊あった．規則も `rule_cl`・`rule_cl_prior`・`heavy_cl`・`spearman_cl` として並べる．同じ式を `:114-123` に足す．

### [A，結果の読み方] 順位の同点で A が選ばれる．Spearman が NaN になりうる
- 重さ：直してから進む
- 該当箇所：`:114-123`（`np.argmax(loo)`，`np.argmax(post)`），`:123`
- 問題：K=5 では 20 本なので LOO の UAR は 1/20 刻みで，同点が出る．AFEW で 4 単位 × 2 つの K の 8 セルを流したところ 1 セルで同点になり，MELD の K=5 の試走でも V と T が同じ値だった．`np.argmax` は先頭の A を返すので，heavy_loo は A に偏る．3 つとも同点なら Spearman は NaN になり，`.mean()` が黙って落とす．
- 計算の後で直せるか：直せる（`loo_uar_m` から出せる）が，連続な量の列は学び直しが要る．
- 求める対応：`rec[f"loo_nll_{m}"] = float(-torch.log_softmax(Z[torch.arange(nK), torch.arange(nK)], 1)[torch.arange(nK), torch.as_tensor(yk, device=Z.device)].mean())`（`:101` の後）を足す．同点のセルを数える列 `heavy_loo_tie` を足し，`heavy_loo_ok` は同点を 1/（同点の数）の得点にするか，同点を除いて報告する．あわせて `:131` の `stalled` は `loo_stalled_A + V + T` の合計に直す．

### [C（対照）] 事前分布だけの対照が無い
- 重さ：直してから進む
- 該当箇所：`:115-118`
- 問題：K=5 では n_loo=20，n0=20 なので，`rule_loo_prior` の指数の半分は事前（3 ソースの交差検証の UAR）で決まる．LOO が何を足したのかを，`rule_prior` なしでは分けられない．事前は，最良の C を選んだ交差検証の値で，尺度が LOO（K 本だけの学習）より高い（IEMOCAP の A は 70 台）ことも，読み方に書く．
- 計算の後で直せるか：直せる（`prior_m` は保存されるが，積の判定が保存されない）が，指数から積を計算し直す必要がある．
- 求める対応：`:116` に `e_prior = exponents([prior[m] for m in "AVT"])` を足し，`rule_prior=uar(u.ytest, prod(logT, e_prior))` と `heavy_prior` を足す．

### [記録する]
- `:50` の `float(ei) + 0.0 if False else` を削り，`sum(float(ei) * logP[m] for ei, m in zip(e, "AVT")).argmax(1)` にする（動作は同じ）．
- `:74` は，出力先の下に `grid_units.csv` が無いと最初に落ちる．上限は 50 単位の平均なので，単位を絞ったときは `oracle` を `first` で取る意味が他の列と違うことを docstring に書く．`results/explore_fusion/grid_units.csv` を `<out>` に置くか，パスを引数にする．
- 出力に，単位ごとのテストの判定（int8）と，塊ごとの LOO・発話ごとの LOO の点数（`sample_id` つき）の保存を足す（上の B の表）．足すと，話者単位の区間と感情ごとの再現率が学び直しなしで出せる．
- `:84` の欠けた単位の飛ばしは，期待の単位の数を assert するか，欠けの数を print する．
- 複数の T を 1 つのプロセスで流すときは，T の終わりに `Corpus.cache.clear()` と `torch.cuda.empty_cache()` を呼ぶ（上の見積もりの注意）．

## 確認項目への答え（まとめ）
1. LOO の問の作り方：`np.delete(kp,i)` の釣り合いの one-hot は正しい．束の渡し方，出発点，引き戻し，目的関数は dai1 の pass1 の none と同じで，W で 3.6e-9 以下，b で 1.6e-8 以下の差で再現した．取り出しも正しい．
2. 積の式と判定の取り出し：正しい．`avt_joint` と `prod111` は explore_fusion_grid の保存値と一致した（差 0.0）．
3. `source_prior`：`cv_uar` 列（先頭の「uar」を含む列）を選び，% 単位で LOO と同じ尺度になる．`corpus`・`kind` の名前も合う．n0=20 と n_loo=4K は docstring と一致する．
4. 順位の一致と上限：指摘 P1〜P3 のとおり．上限は `results/explore_fusion/grid_units.csv`（4 コーパス × 3 つの K × 63 点 × 50 単位）から読まれ，`oracle` は 50 単位平均になる．
5. 所要とメモリ：上のとおり．2.9 GiB で収まり，問を分ける必要は無い．

## 目録に無い種類の不備（9.3 節への追加案）
- 「保持した内部の見積もり（LOO・交差検証）が，最終の評価と同じ単位（話者・塊）で分けられているか」：評価は塊で分けるのに，信頼度の見積もりが発話単位だと，塊の情報が強いモダリティ（音声）が水増しされる．S6・L3 に近いが，探索の内側の見積もりを単独の項目にする．
- 「複数のコーパスを 1 プロセスで続けて流すときの GPU のキャッシュの積み上がり」：`Corpus.cache` が 4 コーパス分を保持するため，メモリの上限が小さいと最後の T で落ちる．

## 流したファイル
`/home/snics/MER-DA/DAI/checks/DAI-explore/code_run_loo/`（`loo_small.py`，`verify.py`，`clusters.py`，`cluster_loo.py`，`out/`）．`results/` の下は読み取りのみで，書き換えていない．