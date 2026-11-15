# Step 7 (G3 and S4-3): facts for design

Read-only survey, 2026-11-02, repo /home/snics/MER-DA at HEAD 40e58f8 (working tree has an unrelated change in exp/explore_view_fusion.py).
No evaluation values (UAR, accuracy, deltas, J) are reported here. Structure counts were computed on CPU with
/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/g3_counts.py
(output: /tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/g3_counts_all.csv).
That script replicates `map.test_mask` and `map.kshot_order` exactly and reads only ids, labels, speakers and manifest columns.

---------------------------------------------------------------------------------------------------

## 1. How units are built (E77, E78, E80 stage c)

### 1.1 The common split object: `compare_b1.Ctx` (exp/compare_b1.py:63-80)

All three scripts build units from `CB.Ctx(group, reps, KS_LAB_ALL, center="ind")`:

- `te = M.test_mask(tgt, r)` (compare_b1.py:71) -> exp/map.py:121-125: `speaker_folds(g.speaker, k=5, seed=r)`, test = fold `r % 5`.
  So the test side is one of 5 speaker folds of the target (all utterances of the test speakers).
- `order = M.kshot_order(tgt, np.flatnonzero(~te), r)` (compare_b1.py:72) -> map.py:128-132: per class, a permutation of
  the non-test rows with `rng_for("map-kshot:<group>", r)`. `take_k` (map.py:135-140) takes the first K per class, so K sets are nested,
  and it raises ValueError if any class has fewer than K rows in the pool.
- `ct[r]` = mean of the non-test rows (center "ind", compare_b1.py:74).
- `idx[(r, K)]` for K in KS_LAB_ALL = ("1","5","10","25") always (compare_dai.py:71; `--ks` only restricts which K are solved).

The scripts then set, in every loop, `pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])`, i.e.
**the unlabeled candidate pool is every non-test utterance of the target**, and
`BD.label_sets_of(ctx, r, pool)` (exp/baselines_distill.py:389-391) maps `ctx.idx[(r,K)]` to positions inside `pool`
(KeyError if a labeled row is not in `pool`).

**Labeled examples are drawn out of the pool and stay in it.** In `Prob` (baselines_distill.py:144-166) the problem has 4n rows over all
n pool rows; true-label weights are added at the K labeled positions (`wt[K]`), and weak-label rows (`weak_rows`, `temp_rows`,
`hard_rows`; distill_belief.py:61-65, baselines_distill.py:95-112) and the entmin/IM entropy terms run over all n rows, including the
labeled ones. Exceptions that drop labeled rows from the unlabeled term: T3A (`t3a_probs`, compare_dai.py:298-300) and LP (`lp_rows`,
compare_dai.py:368-370).

Everything downstream depends on the pool, not only the weak-label term:
- `make_students` (exp/multimodal_distill.py:78-91) and `Student` (exp/teach_student39.py:62-93) build each student on `Xu = X[pool] - mean(pool)`:
  scale calibration `pair_scale(Xu ...)` (teach_student39.py:70-81), whitening metric from the pool covariance (83-92, depends on rho_M),
  `L0 = Xu @ A.T + a0` = the starting point / teacher pF on the pool (93).
- Test features are centred with the pool mean (`Zte`, multimodal_distill.py:90; compare_dai.py:879).
- m_eff = "n" means total weight = pool size n (baselines_distill.py:87-88), so reused settings with meff "n" scale with the new pool.

Sources: `M5.pool_info(data, S, C, cache)` (exp/meta5.py:64-90) uses only source groups; cache key `(tuple(sorted(sources)), C)`
(meta5.py:66), in-memory, independent of the target split.

### 1.2 compare_dai.py (E77)

- World run: `run_world` (compare_dai.py:808-855).
  - Pseudo-target units: loop 820-836. For c in S (S = groups not in T's corpus, 809), `pool_c` = S minus c's corpus (821);
    `ctxc = CB.Ctx(evc, pseudo_reps, ...)` (826); `pool, te` from `ctxc` (828); students `make_students(world, c, ..., pool_c, pool, te, ...)` (829),
    `world_unit(...)` (832). Sources use C_FIT = 0.01.
  - Evaluation units: 838-851. `info` from all S (838); `ctx = CB.Ctx(ev, reps, ...)` (840); `pool, te` (843); `make_students(world, T, ..., S, pool, te, ...)` (844);
    `world_unit(...)` (847); `rec.save` (848).
  - `world_unit` (637-685): reference rows (`ref_rows`, 593-619, eval only), group A/B for the fused student (`run_ab`, 386-450),
    teachers built on the pool: pF (student L0), poe0, pFK (fused none_K), ref/ref0 (2nd-stage solves, 651-661), saved in
    `<T>_r<r>_teachers.npz` (669-672, eval only); group A/B for each modality student and group C (`run_c`, 477-498: kd, kdh, dist, own),
    cotrain (`cotrain_rec`, 501-542), coreg (`coreg_rec`, 545-588).
- Kind run: `run_kind` (869-924). Pseudo units 887-904 (C_FIT), eval units 906-920 with `C_T` read from
  `results/multisource_scores_<kind>/<T>.csv` (873). `student_of` (876-879) builds `Student` on `Xu = X[pool] - ct[r]`. `kind_unit` (688-694) = ref rows + `run_ab`.
- Options: `--pseudo-targets none` skips all pseudo units (934, 962), `--ks` (937, `set_ks` 710-718), `--copy-pseudo` (IEMOCAP-script copies impro, 742-794),
  `--coreg-eval-only`, `--nc-seeds` (entmin/IM restarts on eval only, 419-427).

### 1.3 compare_dai2.py (E78)

Same Ctx, pool, label sets:
- World: `run_world` (compare_dai2.py:676-744); inner `unit` (691-705) calls `make_students` and `world_unit2` (606-654).
  Pseudo loop 708-721 (pool 714), then `chosen = choose_seeded(prows)` (722, function 554-570) and `<T>_chosen.csv` (723-724);
  eval loop 728-741 (pool 733). Evaluation runs the full seed-0 grid plus seeds 1-2 of the pseudo-chosen settings (`problems_for`, 573-588),
  so the E78 evaluation depends on its own pseudo pass unless `chosen` is supplied otherwise.
- Kind: `run_kind` (751-824); `unit` (760-781) builds `Student` on `Xu = X[pool] - ct` (762-772); pseudo 787-804 (pool 798), eval 810-820 (pool 814).
- CORAL (`run_coral`, 478-548): SVD of `[source; Xu]` per source (481-486); r dropped if `min(len(Xu), source sizes) < 5r` (499).
  After the G3 exclusion the smallest pool is 1461 (> 640), so no r is lost on IEMOCAP/MELD.

### 1.4 e80_stage_c.py (E80 stage c, oursx teachers)

- Pseudo units: `pseudo_pass_a` (e80_stage_c.py:308-336; pool 323) computes per (c, r) the students and teachers with
  `EA.unit_pass1` (e80_teacher_j.py:211-235); `pseudo_pass_b` (339-370; pool from `u["pool"]`, 347) rebuilds students and solves none/dist/kd/kdh.
- Evaluation units: `run_target` (431-525). Before the eval loop, `agg_T` (poolC and mlp) is fitted on **all pseudo units of T** (482-487,
  `fit_agg` 206-217; seed `crc32(world|T|eval)`), which forces `pseudo_pass_a` to run (or the unit cache from impro for script, 456-462)
  even when only eval rows are wanted. Eval loop 488-516: `ctx = CB.Ctx(...)` (491), pool (495), `make_students` (496),
  `unit_pass1` (497: none_K, pF, poe0, agree, other_*, ds = Dawid-Skene on the pool votes, pFK, ref, ref0, all on the pool),
  `none_rows` (231-241), `solve_unit` (244-292: kd/kdh/dist with the kept teachers; teacher rows).
- Teacher names per student role: `teachers_for` (e80_teacher_j.py:256-262).
- Which teachers are computed: `make_plan` (e80_stage_c.py:152-187) from `results/e80/kept.csv` (keep == True), restricted to teachers not
  already in E77 (`E77_DIST`, `E77_KD`, lines 85-86).

### 1.5 Summary of the train side

| unit | test | unlabeled pool | K labeled | sources |
|---|---|---|---|---|
| eval (T, r) | speaker fold r%5 of T (seed r) | all other T utterances | first K per class of a per-class permutation of the pool (seed name "map-kshot:T", r); subset of the pool | all groups outside T's corpus; kinds use C_T, worlds C_FIT |
| pseudo (T; c, r) | speaker fold r%5 of c | all other c utterances | same rule on c | S minus c's corpus, C_FIT |

---------------------------------------------------------------------------------------------------

## 2. Where to insert the G3 filter; ids; counts

### 2.1 Minimal insertion point

The pool is not stored in Ctx; each script recomputes it from `~ctx.te[r]` at these lines (eval units only matter for G3):

- compare_dai.py:843 (world eval), 911 (kind eval) [pseudo: 828, 895]
- compare_dai2.py:733 (world eval), 814 (kind eval) [pseudo: 714, 798]
- e80_stage_c.py:495 (eval) [pseudo: 323, and 347 via `u["pool"]`]

`Ctx` also uses `~te` for the K-shot order (compare_b1.py:72) and the centre (74). A G3 variant therefore has to change three things in one place:
`keep = ~te & ~same_block_as_test`, `kshot_order` over `flatnonzero(keep)`, `ct` over `keep`, and expose `ctx.pool[r]`.
Smallest footprint: a Ctx subclass in a new driver script plus replacing the eval-loop pool line by `ctx.pool[r]`.
`label_sets_of` will raise KeyError if any labeled row is outside the new pool, which is a useful fail-fast check.

Design choice to make explicitly (K-shot): with a new pool, `rng.permutation(pool[y==c])` gives an entirely different draw.
Alternative: compute `kshot_order` on the current pool, then delete excluded rows from each per-class order (keeps nesting and keeps the current
labeled examples whenever they survive). Counts of current labeled examples that fall in excluded blocks are in 2.3.

Do not edit compare_dai.py / compare_dai2.py / e80_stage_c.py in place while E77/E78/E80c run (memory: no-edit-running-script), and note
`check_resume` (compare_dai.py:699-707) requires `<T>.csv` to carry the current commit; write G3 outputs to a new `--out-dir`.

The E79 measure stage assumes the candidate set is the complement of the test rows (`pool = np.setdiff1d(..., te)`, analyze_e79.py:1530)
for D and candidate-based quantities. G3 outputs should save the pool rows explicitly (e.g. `<T>_r<r>_pool.npy`), and an E79-style reader
must use them.

### 2.2 Getting sample_id / session / dialogue alongside features

- `measure/data.py` `load_group` (data.py:70-128 approx.; sort at the end of the function) returns `(df, X)` with df columns `sample_id, speaker_id, dialog_type, label_4, speaker_group`,
  rows sorted by `sample_id` (end of `load_group`). `load_fused` (data.py:157-178) returns the first kind's df restricted to the intersection of sample_ids,
  sorted by sample_id. `load_any` (data.py:192-200) dispatches.
- `map.load` (map.py:102-107) keeps only `speaker_group` and `person`; it drops `sample_id`, but its row order equals `load_any`'s df.
- `compare_dai.samples_table(name, T)` (compare_dai.py:797-799) returns `sample_id, speaker_id, speaker_group` aligned to the rows used by the
  scripts; E77/E80c save it as `<T>_samples.csv` (compare_dai.py:980; e80_stage_c.py:444).
- Manifests: `data/metadata/iemocap_utterance_manifest.csv` has `session` (1-5) and `dialog_id`; `meld_utterance_manifest.csv` has `split`
  and `dialogue_id`; MELD dialogue ids restart per split (1423 unique ids vs 1432 unique (split, dialogue_id)), so the block key must be
  `split|dialogue_id`. `analyze_e79.py` already does this: `CLUSTER_COLS` (93-94) and `cluster_labels(T, sample_ids, ...)` (1396-1414).
  MOSEI manifest has `video_id`.

### 2.3 Counts (structure only)

Definitions: pool = non-test rows of T for rep r; "excluded" = pool rows whose IEMOCAP session (or MELD split|dialogue_id) contains at least one
test utterance; K5/K25 in excluded = how many of the current K=5 (20 rows) / K=25 (100 rows) labeled examples are in excluded rows;
"test sharing" = test utterances whose block also has pool utterances (current split).

**IEMOCAP: are the 2 test speakers from the same session?** Mostly no.
- IEMOCAP-impro (all 5 runs identical): always 2 speakers from 2 different sessions. Reps r and r+5 have identical test sets
  (all 10 speaker sizes differ, so the greedy fold assignment ignores the seed); only the K-shot draws differ between r and r+5.
  r0/r5 Ses01_M+Ses05_F, r1/r6 Ses02_M+Ses05_M, r2/r7 Ses03_M+Ses04_M, r3/r8 Ses03_F+Ses04_F, r4/r9 Ses01_F+Ses02_F.
- IEMOCAP-script, AVT/AV/V: r3 and r8 = Ses04_F+Ses04_M (same session, nothing excluded); all others 2 different sessions;
  r1 (Ses02_F+Ses03_M) differs from r6 (Ses01_M+Ses02_F).
- IEMOCAP-script, A/T (`audio_wavlm`, `text_roberta`): r2 and r7 = Ses05_F+Ses05_M (same session, nothing excluded); others different sessions.
- Consequence: excluding the 2 test sessions removes the 2 dialogue partners, so the pool goes from 8 to 6 speakers (3 sessions).

AVT world (`fuse-audio-video-text`), per rep:

| T | rep | n_test | n_pool | excluded | frac | pool after | min class after | K5 in excl | K25 in excl | test sharing |
|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP-impro | 0 | 616 | 2327 | 636 | 0.273 | 1691 | 196 | 7 | 27 | 616 |
| IEMOCAP-impro | 1 | 622 | 2321 | 639 | 0.275 | 1682 | 236 | 5 | 26 | 622 |
| IEMOCAP-impro | 2 | 588 | 2355 | 573 | 0.243 | 1782 | 115 | 7 | 32 | 588 |
| IEMOCAP-impro | 3 | 573 | 2370 | 588 | 0.248 | 1782 | 115 | 4 | 30 | 573 |
| IEMOCAP-impro | 4 | 544 | 2399 | 507 | 0.211 | 1892 | 205 | 5 | 24 | 544 |
| IEMOCAP-impro | 5 | 616 | 2327 | 636 | 0.273 | 1691 | 196 | 3 | 30 | 616 |
| IEMOCAP-impro | 6 | 622 | 2321 | 639 | 0.275 | 1682 | 236 | 5 | 32 | 622 |
| IEMOCAP-impro | 7 | 588 | 2355 | 573 | 0.243 | 1782 | 115 | 5 | 25 | 588 |
| IEMOCAP-impro | 8 | 573 | 2370 | 588 | 0.248 | 1782 | 115 | 7 | 29 | 573 |
| IEMOCAP-impro | 9 | 544 | 2399 | 507 | 0.211 | 1892 | 205 | 6 | 17 | 544 |
| IEMOCAP-script | 0 | 524 | 2025 | 564 | 0.279 | 1461 | 268 | 6 | 30 | 524 |
| IEMOCAP-script | 1 | 515 | 2034 | 502 | 0.247 | 1532 | 261 | 5 | 21 | 515 |
| IEMOCAP-script | 2 | 509 | 2040 | 455 | 0.223 | 1585 | 267 | 9 | 28 | 509 |
| IEMOCAP-script | 3 | 497 | 2052 | 0 | 0.000 | 2052 | 411 | 0 | 0 | 0 |
| IEMOCAP-script | 4 | 504 | 2045 | 531 | 0.260 | 1514 | 274 | 3 | 24 | 504 |
| IEMOCAP-script | 5 | 524 | 2025 | 564 | 0.279 | 1461 | 268 | 6 | 27 | 524 |
| IEMOCAP-script | 6 | 515 | 2034 | 542 | 0.266 | 1492 | 286 | 9 | 27 | 515 |
| IEMOCAP-script | 7 | 509 | 2040 | 455 | 0.223 | 1585 | 267 | 4 | 15 | 509 |
| IEMOCAP-script | 8 | 497 | 2052 | 0 | 0.000 | 2052 | 411 | 0 | 0 | 0 |
| IEMOCAP-script | 9 | 504 | 2045 | 531 | 0.260 | 1514 | 274 | 7 | 25 | 504 |
| MELD | 0 | 2115 | 9193 | 3213 | 0.350 | 5980 | 518 | 3 | 29 | 1965 |
| MELD | 1 | 2115 | 9193 | 3391 | 0.369 | 5802 | 593 | 8 | 34 | 1975 |
| MELD | 2 | 2114 | 9194 | 3460 | 0.376 | 5734 | 474 | 13 | 44 | 1950 |
| MELD | 3 | 2114 | 9194 | 3358 | 0.365 | 5836 | 477 | 7 | 33 | 1924 |
| MELD | 4 | 2850 | 8458 | 3570 | 0.422 | 4888 | 465 | 7 | 37 | 2385 |
| MELD | 5 | 2115 | 9193 | 3236 | 0.352 | 5957 | 531 | 7 | 31 | 1949 |
| MELD | 6 | 2115 | 9193 | 3312 | 0.360 | 5881 | 592 | 9 | 38 | 1960 |
| MELD | 7 | 2114 | 9194 | 3353 | 0.365 | 5841 | 478 | 6 | 40 | 1951 |
| MELD | 8 | 2114 | 9194 | 3289 | 0.358 | 5905 | 496 | 9 | 39 | 1921 |
| MELD | 9 | 2850 | 8458 | 3570 | 0.422 | 4888 | 465 | 8 | 45 | 2385 |

MELD details (AVT): test reps 0-3, 5-8 contain 68-73 speakers and 595-628 dialogues; reps 4 and 9 contain only 2 speakers (2850 utterances, 742 dialogues).
Pool speakers drop from 210-215 to 145-173 (reps 0-3, 5-8) and from 281 to 194 (reps 4, 9).

IEMOCAP exclusion by `dialog_id` instead of session removes the same number in most reps, fewer in some
(impro r2/r7: 558 vs 573; script r4/r9: 505 vs 531).

Ranges for the other runs (full per-rep rows in g3_counts_all.csv):

| run | T | excluded (min-max) | frac | pool after (min) | min class after | K5 in excl (max) | K25 in excl (max) |
|---|---|---|---|---|---|---|---|
| fuse-audio-video, video | IEMOCAP-impro | 507-639 | 0.211-0.275 | 1682 | 115 | 7 | 32 |
| fuse-audio-video, video | IEMOCAP-script | 0-564 | 0-0.279 | 1461 | 261 | 9 | 30 |
| fuse-audio-video, video | MELD | 3213-3570 | 0.350-0.422 | 4888 | 465 | 13 | 45 |
| audio_wavlm | IEMOCAP-impro | 507-639 | 0.211-0.275 | 1682 | 115 | 7 | 32 |
| audio_wavlm | IEMOCAP-script | 0-564 | 0-0.273 | 1500 | 264 | 8 | 34 |
| audio_wavlm | MELD | 3224-3579 | 0.350-0.422 | 4904 | 458 | 12 | 48 |
| text_roberta | IEMOCAP-impro | 507-639 | 0.211-0.275 | 1682 | 115 | 7 | 32 |
| text_roberta | IEMOCAP-script | 0-564 | 0-0.273 | 1500 | 264 | 8 | 34 |
| text_roberta | MELD | 3283-3579 | 0.356-0.422 | 4905 | 465 | 11 | 44 |

- After exclusion every class keeps >= 115 pool rows, so `take_k` for K = 1, 5, 10, 25 (always drawn) does not fail.
- MOSEI: 0 pool utterances share a `video_id` with a test utterance in every run and rep (speaker ids are video ids or channel ids, so the speaker
  split already separates videos). G3 as defined does not change MOSEI units, and the review's MOSEI concern (one person across videos) cannot be
  addressed with the manifest.
- Units affected: IEMOCAP-impro, IEMOCAP-script, MELD x 10 reps = 30 eval units per run. With session exclusion, IEMOCAP-script reps whose
  2 test speakers share a session (AVT/AV/V r3, r8; A/T r2, r7) are unchanged and serve as an identity check.

---------------------------------------------------------------------------------------------------

## 3. E79 selected settings: storage and reuse

- E79 has not been run on the full outputs yet (needs E77, E78, E80c to finish on RunPod). There is no `results/e79` on the lab PC.
  The lab PC has `results/e80/kept.csv` (+ `kept_provenance.json`) and an old A5000 copy `results/e77_fuse-audio-video-text` (only IEMOCAP-impro
  complete, all 5 K, commit 69c5bd5 dirty; not the production run).
- `--stage select` writes `<out>/select/chosen.csv` (analyze_e79.py:1011), `candidates.csv`, `fixed.csv`, `kept_used.csv`, `r6_subsets.csv`,
  `d8_table.csv`, `checks.csv`, provenance with sha256 (1015-1027).
- chosen.csv columns (rows built in `pick_one`, 844-918): `run, target, student, role, K, selector, variant, none_score, copied_from, wave, method,
  param, meff, score, n_cand, n_dropped, n_tie, chose_none`. `param` is the normalized param `pnorm` (seed suffix `s<n>` stripped for seeded methods,
  617-618, `strip_seed` 132-133, `seed_of` 136-143). `variant` in {incl, excl, ns_incl, ns_excl} (MOSEI/MELD get ns_*). `wave` in {e77, e78, e80c}.
  `method == "__none__"` means no candidate; incl rows may say `method == "none"`.
  (`none_score` and `score` are pseudo-target UAR means; do not print them when checking.)
- Selectors (`sel_defs`, 502-558): entmin, self, im (meff in MEFFS only, so TIM is excluded), t3a, t3a_b, lp, coral, coralctl, dann/mme/mapctl
  (not for fused students), groupB (= union of the 8 group-B methods, 531), and for worlds kd, oursx, oursx_noJ, kd_noJ, ours, abl_* , xmuda, cotrain.
- **oursx R4**: `D["oursx"] = dist(kept)` (543): dist rows of waves e77 and e80c whose teacher is in the kept set for (run, T, student, K),
  any weighting in WGTS (5) and any meff in MEFFS (4) (`dist`, 512-515). kept sets from `load_kept` (667-698).
  **kd/kdh**: `D["kd"] = kdf(kept)` (542; `kdf` 517-519): kd and kdh rows of waves e77/e80c whose teacher (param without `T<temp>`) is kept.
  Selection rule: `choose` (570-583) inside `pick_one`: mean pseudo-target UAR over all (pseudo, rep) units, only candidates present in every unit,
  ties by the string `method|pnorm|meff`; incl picks none if best <= none. Rows per (run, T, student, K, selector, variant) in `stage_select` (834-1029).
  P5 uses incl (oursx, kd, groupB; judge 2235-2240); P4 uses excl.
- E78 also writes its own seed-0 choice per (student, method, K) to `<T>_chosen.csv` (compare_dai2.py:722-724, 805-808) to run seeds 1-2 on evaluation;
  E79 cross-checks it (analyze_e79.py:913-923).
- **Can a new script evaluate only the chosen settings?** Yes, the information is sufficient: (wave, method, param, meff) maps to solver arguments:
  dist `teacher:wgt` + meff -> `weak_rows`/`temp_rows`/`hard_rows` (compare_dai.py:486-495; e80_stage_c.py `rows_of` 119-126);
  kd/kdh `teacherT{2|4}` + meff (479-484, `kdh_solve` 455-474); entmin `-` + meff (411-415, plus seeds 1-2 via `shaken`, 249-257);
  im `a{alpha}` + meff (`im_solve` 260-283); self tau + meff (397-410); t3a `M{M}`; lp `k{k}a{alpha}{form}` + meff; coral `r{r}rho{rho}cp{cp}cb{cb}`
  (compare_dai2.py:532); dann/mme/mapctl `map_param` without seed (compare_dai2.py:138-142) with seeds 0-2.
  The batched L-BFGS in `logreg.fit_batch` keeps a separate active mask and line search per problem (logreg.py, `active` around 262-321), so
  solving a subset gives the same solution up to floating point. But no existing entry point takes "only these settings": `run_ab`, `run_c`,
  `world_unit`, `world_unit2`, `run_coral`, `solve_unit` loop over full grids. Options: (a) run the existing per-unit functions on new-split units
  (full grids at `--ks 0 5`, no pseudo units) and pick the chosen rows afterwards; (b) a new driver that calls the primitives above per chosen key.
  For oursx/kd, the teacher arrays must be rebuilt on the new pool in either case (1.4).
- E79 measure reads `<T>.csv`, `<T>_samples.csv`, `<T>_diag.csv` (1485), `_te.npy`, `_logp.npy`, `_pred.npy` per rep (1527-1540), and requires
  E77/E78/E80c `_te.npy` to match and full tables; a G3 output directory with only a subset of rows will not pass E79 as is. A separate small
  comparison (method - none per unit, paired by rep, with the same interval rules) is the simpler reader.

---------------------------------------------------------------------------------------------------

## 4. Timing (timing columns only)

### 4.1 Lab copy, A5000 (old run, commit 69c5bd5, AVT world, T = IEMOCAP-impro, all 5 K, GPU shared with E76)

Source: results/e77_fuse-audio-video-text/IEMOCAP-impro_conv.csv (columns pseudo, rep, student, what, sec, n_prob) and file mtimes.

Per evaluation unit (sum over the 4 students), seconds, min / median / max over 10 reps:

| what | min | median | max |
|---|---|---|---|
| none | 2.0 | 2.6 | 6.9 |
| self (3 rounds) | 28.4 | 36.6 | 63.3 |
| entmin (seed 0 only; that commit had no nc seeds) | 18.2 | 26.2 | 59.7 |
| lp | 29.2 | 32.5 | 50.0 |
| dist | 47.2 | 52.9 | 79.7 |
| kd | 13.0 | 14.5 | 20.8 |
| own | 4.4 | 5.5 | 15.6 |
| ref (ref_rows: srconly/merged/tgtonly/wstar) | 61.1 | 101.7 | 160.3 |
| wstar | 2.7 | 4.7 | 8.6 |
| "-" (solves without a tag: 2nd-stage ref teachers and cotrain inner solves) | 25.1 | 37.7 | 68.8 |
| cotrain_total | 27.6 | 44.0 | 87.3 |
| coreg_total | 175.9 | 439.7 | 724.1 |

Not recorded in conv: IM (`im_solve` bypasses `Prob2.stats`), kdh (`kdh_solve`), T3A, saerens/massbal, student construction.
Per-call (one student, one batched solve): entmin 2.5-11.3 s for 20 problems; self 1.8-6.4 s per round for 60 problems; dist 6.1-28.6 s for 400 problems;
kd 1.8-7.0 s; none 0.5-1.4 s (largest is the fused student).

Wall clock per eval unit (gaps between `_logp.npy` mtimes): 634, 829, 528, 1087, 1150, 583, 1098, 1048, 965 s for r1-r9, and 670 s for r0
(includes `pool_info` for S). Total for the target 24,524 s (logs/e77b_fuse-audio-video-text.log), of which pseudo units 10 x about 20-35 min.
Pseudo unit (median per unit): MELD/MOSEI as pseudo-target: self 108/145 s, entmin 130/158 s, dist 208/258 s, lp 138/173 s, kd 61/82 s.

### 4.2 H100 (RunPod) totals per target (from the main session's watcher, scratchpad watch_state.txt; the per-unit CSVs are on RunPod only)

E77 (`--ks 0 5 25`, copy-pseudo, coreg-eval-only, nc seeds 1-2):
AVT IEMOCAP-impro 5516 s (10 pseudo + 10 eval units), AVT IEMOCAP-script 2748 s (eval only, pseudo copied: about 275 s per eval unit with all E77 methods);
AV IEMOCAP-impro 4193 s, AV IEMOCAP-script 2054 s (about 205 s per eval unit), AV MELD 8008 s.
E78: AV IEMOCAP-impro 6568 s, AV MELD 7930 s, AVT IEMOCAP-impro 6799 s.
E80 stage a per target: AVT 192-242 s, AV 331-392 s.
H100 is about 3.5x faster than the A5000 (checks/E80/code.md:234).

### 4.3 E78 and E80c estimates already in the gate reports (A5000, shared GPU)

- E78 (checks/E78/code.md:324-331): DANN/MME/mapctl seed-0 full grid 407 s per unit (audio); CORAL 235 s per student-unit (r 32, 64 + control);
  per world unit about 32 min (AV) and 48 min (AVT) with all E78 methods.
- E80 stage c (checks/E80/code.md:228, 234): kept-subset trial eval 20 s (impro) and 29 s (script) per unit; dist/KD solves for all eval units
  AVT 1.5-1.9 h, AV 1.3-1.7 h (A5000); pass-1 per pseudo unit 16-22 s (AVT) and 4-13 s (AV) (checks/E80/code.md:36, 42).

### 4.4 Implication for G3 (no new values, only arithmetic on the above)

30 eval units per run (IEMOCAP-impro, IEMOCAP-script, MELD x 10). On H100 a full E77 eval unit for a world costs about 3.5-4.5 min at K 0/5/25;
restricted to K 0/5 and to none + teachers + chosen rows, and with 21-42 % smaller pools, it is far less. The extra fixed cost is rebuilding
agg_T for oursx (1.4) per (world, T), and E78 methods (coral, dann, mme) only for the chosen settings with seeds 0-2.

---------------------------------------------------------------------------------------------------

## 5. S4-3 (C' for entmin, IM, self-training)

### 5.1 How C', C_b, rho_M are fixed

- `settings(kind, T)` (exp/geom_e0.py:68-80) returns (gamma, rho_s, rho_M, C_b, C', step). Worlds call it for each modality student and the fused
  student (compare_dai.py:810); kinds at 872. Source files: results/explore_relation/e40_targets.csv (A, A-HuBERT, V, AV), e39_text_targets.csv and
  e37_targets_text.csv (text and AVT), e62_targets.csv (2nd encoders).
- Students carry them: `make_students` (multimodal_distill.py:85-90) and `student_of` (compare_dai.py:876-879) set `rho_m`, `cb`, `cp`;
  `Z = Xstd @ A_m` with `A_m = st.mats[rho_m][0]`, so rho_M fixes the coordinates, not only the penalty.
- Used in `Prob2.solve` (compare_dai.py:174-208): `cp = self.s["cp"]` read at call time (175), `fit_batch(..., (cp,), anchor=st.anchor(rho_m), anchor_c=st.C,
  Cb=[s["cb"]]*n, ...)` (200-202); in `im_solve` (277-280) and `kdh_solve` (469-471) the same way. entmin = `pr.solve(..., u=mass/n, inits=none[K])`
  (411-415); IM = `im_solve` (416-418); self = 3 rounds of `pr.solve` with `hard_rows(cur, tau, meff)` starting from the none probabilities (397-410).
- Objective (logreg.py:242-246): `(sum w*nll)/S + ||W - W_anchor||^2/(2 C' S) + extra/S`, with S = sum of weights + extra mass; intercept penalty
  `||b - b_anchor||^2/(2 C_b S)` (logreg.py:142). The penalty centre is the E39 draft (`st.anchor`), not the none solution.
- E39 values (hyperparameters, from `settings`):
  - C' = 0.003 for every A (audio_wavlm), V (video) and AV (fuse-audio-video) target; text_roberta and AVT fused: 0.01 (IEMOCAP impro/script), 0.03 (MOSEI, MELD).
    In world runs the modality students use their own kind's value (e.g. AVT world audio student 0.003).
  - C_b: 0.5 or 2.0 (A), 0.5 (V), 2.0 (T), 0.5 (AV), 2.0 (AVT IEMOCAP) / 8.0 (AVT MOSEI, MELD).
  - rho_M: 0.9-1.0.
  - The grid edge 0.003 matches `CPS = (0.003, 0.01, 0.03, 0.1, 0.3)` in exp/teach_joint.py:47 (likely the E40 grid; not verified further).
  - 3x and 10x: 0.009 and 0.03 (A, V, AV); 0.03 and 0.1 (C' 0.01); 0.09 and 0.3 (C' 0.03).
- CORAL already selects C' from (0.01, 0.03, 0.1) plus the E39 value (compare_dai2.py:72, 487), which is why the review exempts it.

### 5.2 Current grids (compare_dai.py:78-83, 386-427)

- entmin: meff in {100, 500, 2000, n} -> 4 settings per K (seed 0 on pseudo units; seeds 1-2 = shaken restarts on eval units only).
- IM: alpha in {0.1, 0.3, 1} x meff 4 = 12 per K, plus TIM (alpha 0.1, mass 40K) for K >= 1, which E79 does not select (meff not in MEFFS).
  Seeds 1-2 on eval units.
- self: tau in {0.5, 0.7, 0.9} x meff 4 = 12 per K, 3 rounds.
- With C' x {1, 3, 10}: entmin 12, IM 36, self 36 candidates per K.

### 5.3 Where 3x/10x C' can be added

- `fit_batch` accepts several C values in one call, but with `extra` (entmin, IM) the extra mass is indexed per mask, so in practice one C per call
  (docstring logreg.py:165-167; mass at 207). Separate calls per C' are needed for entmin/IM; self could batch but `Prob2.solve` hard-codes `(cp,)`
  and reads `f.W[cp]`.
- Lowest-change route: because `Prob2.solve` and `im_solve` read `pr.s["cp"]` at call time, a new driver can (1) solve none with the E39 C',
  (2) set `pr.s["cp"] = m * cp0` for m in {3, 10}, (3) rerun the entmin / IM / self blocks of `run_ab` (and the shaken restarts on eval units), and
  (4) tag rows with a distinct param (e.g. `-C3x`, `a0.1C3x`, `0.5C3x`) so they do not collide with E77 keys; keep the seed suffix last so
  `strip_seed`/`seed_of` still work (analyze_e79.py:132-143). none, P0 for self, and the entmin/IM starting points stay at the E39 C'.
- Prob construction cost: `Prob.__init__` makes `X = Z.repeat(4, 1)` (4n x d, float64) per student and sets `st.Xstd = None` (baselines_distill.py:152, 166);
  reusing one Prob per student and switching `s["cp"]` avoids a second copy.

### 5.4 What the P4 judgment uses (`--families p4`)

- Constants: `GROUP_B = (entmin, self, im, t3a, lp, coral, dann, mme)` (analyze_e79.py:80), `P4_KS = ("0","5")` (81), `P4_SELS` (102).
- select: `--families p4` keeps only selectors in P4_SELS (980-981), does not read kept.csv (670-677) or build D8/R6 rows; reads `<T>_pseudo.csv`
  from E77 and E78 (`load_pseudo`, 586-630). Variants excl (main), ns_excl (MOSEI/MELD reselected without IEMOCAP-script), incl.
- judge (2173-2230): for every run and student, family = "P4" for kinds (audio_wavlm, video, text_roberta), "P4'" for the fused student of
  fuse-audio-video-text and fuse-audio-video, "P4_uni_world" (descriptive) for world modality students; K in P4_KS is main, K = 25 goes to "_otherK".
  Comparison `none-<method>` on metric "dir" (direction loss), variant excl, plus sensitivity rows ns_excl (`noscript_reselect`), `nc_seed0`
  (entmin/IM read with seed 0 only), `rev_order`, `iemocap_impro_only` (text and AVT); descriptive `P4_desc_{uar,pos,bfix,ofix}`.
  Fused students skip dann/mme/mapctl. Units are per (run, student, K, corpus) (IEMOCAP impro+script pooled as one corpus).
- measure reads per eval target: `<T>.csv`, `_samples.csv`, `_diag.csv`, `_r<r>_te.npy`, `_logp.npy`, `_pred.npy` from E77 and E78 dirs
  (1485-1540); delta reads `<T>_ps_<c>_r<r>_none.npz` + `<T>_pseudo.csv`.
- For S4-3 only entmin, im, self rows change. The scope that feeds P4 and P4': kinds A, V, T (all their targets) and the fused students of AV and AVT.
- Unit counts that a reselection would touch (pseudo units computed, IEMOCAP-script copied from impro; eval units = 10 x targets):
  audio_wavlm 175 / 70, video 120 / 60, text_roberta 40 / 40, fuse-audio-video 120 / 60, fuse-audio-video-text 40 / 40 (pseudo / eval).
- analyze_e79.py recognizes waves e77, e78, e80c only (`sel_defs` 507-531, `load_pseudo` 589, `eval_table` 1131); e80c rows are restricted to
  dist/kd/kdh/teacher (`E80C_METHODS`, 583). New C' rows therefore need either a new wave in analyze_e79.py (parse_dirs, sel_defs, eval tables,
  overlap checks) or a separate reader.

---------------------------------------------------------------------------------------------------

## 6. Things that make the re-computations costly or tricky

1. **Pool-dependent students and teachers (G3).** Student scale, whitening metric (rho_M), centre, starting point pF/L0, pFK, ref/ref0, poe0, agree,
   other_*, ds, ownK are all computed on the pool. Nothing in `<T>_r<r>_teachers.npz` (E77) or the E80c npz can be reused under the new split; the
   whole unit must be rebuilt (only source-side `pool_info` is reusable).
2. **oursx agg_T needs pseudo units.** e80_stage_c's evaluation `agg_T` (poolC/mlp) is fitted on all of T's pseudo units (e80_stage_c.py:482-487) and is not
   saved (only lambda/seed/loss in `<T>_pool.csv`). It does not depend on T's split, but rebuilding it needs `pseudo_pass_a` (GPU, all pseudo units of T)
   or the stage-a unit files `results/e80/{avt,av}/units/<world>/<T>/<c>_r<r>.npz` (on RunPod; `load_unit`, e80_teacher_j.py:392-404; `choose_lam` needs
   `person`, recoverable from `samples_<c>.csv` + `cand`). Stage a did not compute T = IEMOCAP-script units (same as impro).
   The stage-c check tolerance against stage-a arrays is 0, and stage a and stage c ran on different H100s (checks/E80/code.md:313).
3. **E78 evaluation depends on its pseudo pass.** Seeds 1-2 for DANN/MME/mapctl/xMUDA are run only for settings chosen in the same run's pseudo pass
   (compare_dai2.py:573-588); `--pseudo-targets none` yields an empty `chosen` and only the seed-0 grid. A G3 run must inject the chosen settings.
4. **K-shot redraw.** Up to 13 of 20 (MELD) and 9 of 20 (IEMOCAP) current K=5 labeled examples fall in excluded blocks; the rule for redrawing
   (fresh permutation vs. filtered current permutation) changes how comparable the two splits are and must be fixed before computing.
5. **m_eff = n** changes with the pool size (35-42 % smaller for MELD), so the same selected setting means a smaller total weight in G3.
6. **E79 readers assume pool = complement of te** (analyze_e79.py:1530) and full method tables; G3 needs saved pool rows and a separate comparison.
7. **IEMOCAP reps r and r+5 share the test set** (impro always; script except r1/r6), so the 10 reps are 5 distinct test sets with 2 K-draws each.
   MELD reps 4 and 9 have 2 test speakers with 2850 utterances.
8. **Some reps are unchanged by G3** (script reps with both test speakers in one session), useful as identity checks; MOSEI is unchanged entirely.
9. **S4-3 pseudo-target cost dominates.** Reselection needs entmin/IM/self at 2 more C' values on all 495 pseudo units of the five P4 runs (plus eval
   units with seeds 1-2 for entmin/IM). On A5000 a pseudo unit's entmin and self alone took 130-158 s and 108-145 s at 5 K (AVT, all students); IM time
   is not recorded. Two extra C' values roughly triple that block. Restricting to students that feed P4/P4' (kinds and fused students) skips world
   modality students, but `make_students` builds all students of a world.
10. **Commit and resume checks.** `check_resume` (compare_dai.py:699-707) and E79's commit/sha256 chain (analyze_e79.py:252-283) stop on mismatched
    commits; new outputs need new directories and their own provenance.
11. **Settings files must exist where the job runs**: results/explore_relation/{e40_targets, e39_text_targets, e37_targets_text}.csv and
    results/multisource_scores_<kind>/<T>.csv (C_T for kind evaluation).
12. **GPU memory**: E80c batches up to 480 problems (9-10 GB on A5000 estimate; `--gpu-fraction 0.3` on H100). Smaller pools reduce memory.
