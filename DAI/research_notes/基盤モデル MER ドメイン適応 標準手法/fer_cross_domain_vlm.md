# Cross-domain facial expression recognition (static and dynamic) built on vision / vision-language foundation models, 2022–2026

Scope note (how these notes were made, 2026-11-08). Full texts of 31 papers were downloaded from arXiv / IJCAI and read as extracted PDF text (methods, implementation details, result tables). Three papers could be accessed only as an abstract or search snippet, and they are marked "[abstract only]". Numbers are copied from the papers' own tables. "Older" marks pre-2022 work, which is included only as context. Venue labels marked "(venue per X)" are taken from another paper's citation table, not from the paper itself.

Our setting, used in the "difference" column: frozen ResNet-50 face-expression features (an FER model applied to detected faces, averaged per utterance), WavLM audio, RoBERTa text; linear softmax classifiers; target corpus with K labeled utterances per class plus unlabeled utterances; 2–3 source corpora drawn from AFEW / IEMOCAP / MOSEI / MELD; 4 classes; metric UAR.

## Q1. Which vision foundation models are standard for FER and dynamic FER, and are they frozen?

### Takeaway
CLIP (OpenAI weights, ViT-B/32 most often, then ViT-B/16 and ViT-L/14) is the dominant foundation model in 2023–2026 FER/DFER papers. Since 2024 most CLIP-based FER methods keep both CLIP encoders frozen and train only prompts, adapters, a projection head, or a separate small network. The face-specific self-supervised video models (MAE-DFER, SVFAP, S4D on VoxCeleb2; MARLIN on YouTubeFace; FMAE on Face9M) are fully fine-tuned, apart from a few parameter-efficient variants. A frozen backbone with only a linear classifier on top appears in these papers only as a baseline, and it is clearly weaker than fine-tuning on in-domain FER benchmarks.

### Cited Findings
**CLIP-based FER (static and video)**
- CAFE / "Generalizable FER" (Zhang et al., ECCV 2024) uses only the visual encoder of CLIP ViT-B/32. It states "we fix F [the CLIP features] during the training process to prevent the FER model to optimize face features to overfit the given train set". A trainable ResNet-18 produces sigmoid masks over the fixed CLIP features. The masked features are split into 7 channel groups that give the class logits directly (no FC layer), and a channel-diverse loss is added. Adam, lr 2e-4 — [arXiv 2408.10614](https://arxiv.org/abs/2408.10614)
- Exp-CLIP (Zhao, Cao, Gong, Patras; WACV'25 per the citation table of [TTA-CaP](https://arxiv.org/abs/2603.21309)): "only the learnable projection head … will be updated, and the rest of the models are all frozen". The head is a single linear matrix (512×4096 for ViT-B/32 and B/16, 768×4096 for ViT-L/14), shared by the image and text encoders. It is trained 5 epochs by contrastive alignment to Flan-T5-XXL (11B) LLM features, using 45K unlabeled CAER-S face images. A linear head beat a 2-layer MLP, a 3-layer MLP and a Transformer encoder head in zero-shot accuracy — [arXiv 2405.19100](https://arxiv.org/abs/2405.19100)
- EmoCLIP (Foteinopoulou & Patras; FG'24 per [FineCLIPER Table 1](https://arxiv.org/abs/2407.02157)): CLIP ViT-B/32 image and text encoders are initialised from CLIP and fine-tuned on MAFW video–caption pairs, with an added temporal layer. Ablation on MAFW 11-class (UAR/WAR): frozen CLIP backbone 16.97–20.40/17.61–21.69; "EmoCLIP (Frozen backbone)" 23.60/31.36; fine-tuned EmoCLIP 25.86/33.49 — [arXiv 2310.16640](https://arxiv.org/abs/2310.16640)
- CLIPER (Li et al., arXiv 2023): CLIP ViT-B/16 for static FER, ViT-B/32 for DFER. Stage 1 learns multiple expression text descriptors with both encoders frozen; stage 2 fine-tunes the image encoder (lr 5e-6) with the text fixed. RAF-DB WAR: zero-shot CLIP 40.16, linear probe 84.29, full fine-tuning 90.38, CLIPER two-stage 90.91. FERV39k UAR/WAR: zero-shot 21.20/16.20, linear probe 33.06/44.57, full fine-tuning 38.68/50.13 — [arXiv 2303.00193](https://arxiv.org/abs/2303.00193)
- DFER-CLIP (Zhao & Patras, BMVC 2023): "the CLIP text encoder is fixed and we fine-tune the CLIP image encoder"; the temporal model, class token and learnable context are trained from scratch with cross-entropy. DFEW / FERV39k / MAFW UAR/WAR: linear probe 45.46/57.40, 32.47/43.72, 30.74/42.95; full fine-tuning with temporal model 58.28/70.27, 40.55/51.22, 38.39/50.92; DFER-CLIP 59.61/71.25, 41.27/51.65, 39.89/52.55; zero-shot CLIP 23.34/20.07, 20.99/17.09, 18.42/19.16; zero-shot FaRL (ViT-B/16) 23.14/31.54, 21.67/25.65, 14.18/11.78 — [arXiv 2308.13382](https://arxiv.org/abs/2308.13382)
- FineCLIPER (ACM MM 2024): CLIP ViT-B/16 with frozen pre-trained layers plus trainable lightweight adapters (13M tunable parameters). DFEW 62.81/72.86, FERV39k 42.88/52.01, MAFW 42.19/53.12 UAR/WAR. Its Table 1 lists tunable parameters of other methods: CLIPER 88M, DFER-CLIP 90M, MAE-DFER 85M, S2D 9M — [arXiv 2407.02157](https://arxiv.org/abs/2407.02157)
- A3lign-DFER (arXiv 2024): CLIP image and text encoders are frozen in all three training steps. Only label-text embeddings and text/video "dynamic encoders" are trained — [arXiv 2403.04294](https://arxiv.org/abs/2403.04294). Backbone CLIP ViT-L/14; DFEW 64.09/74.20 (per [FineCLIPER Table 1](https://arxiv.org/abs/2407.02157))
- PE-CLIP (arXiv 2025): frozen CLIP ViT-B/16 visual and text encoders with a GRU-based Temporal Dynamic Adapter, a Shared Adapter and prompt learning (9M trainable parameters). DFEW 5-fold: UAR 62.82, WAR 74.04 — [arXiv 2503.16945](https://arxiv.org/abs/2503.16945)
- MPA-FER (ICCV 2025): "the CLIP backbone remains frozen"; learnable visual prompts (8) and textual prompts (10); ChatGPT-generated class descriptions are used as hard prompts; frozen class prototypes. Accuracy with ViT-B/16: RAF-DB 92.51, AffectNet-7 67.85, FERPlus 91.15; ViT-L/14: 93.74 / 68.89 / 91.81. Its CoOp baseline gets RAF-DB 86.15, AffectNet-7 62.27 — [arXiv 2506.21017](https://arxiv.org/abs/2506.21017)
- EmoCapCLIP (Sun et al., arXiv 2025) trains CLIP on EmoCap100K (more than 100K facial-emotion captions). Zero-shot UAR/WAR on RAF-DB with ViT-B/32: OpenAI CLIP 37.68/30.80, SigLIP2 49.46/48.24, Exp-CLIP 39.56/42.14, FLIP 45.14/49.32, EmoCapCLIP 62.89/67.03. With ViT-B/16, zero-shot FaRL gets 30.38/40.74 on RAF-DB — [arXiv 2507.21015](https://arxiv.org/abs/2507.21015)

**Face-specific self-supervised and supervised models (mostly fine-tuned)**
- FaRL (CVPR 2022): ViT-B/16 pre-trained on LAION-FACE (20M face image–text pairs). The paper evaluates it with frozen backbones on face parsing, alignment and attribute recognition. No FER benchmark appears in the main paper (by full-text search for "expression"/"emotion") — [arXiv 2112.03109](https://arxiv.org/abs/2112.03109)
- MAE-DFER (ACM MM 2023): self-supervised pre-training on the VoxCeleb2 dev set (1,092,009 clips), then fine-tuning on six DFER datasets — [arXiv 2307.02227](https://arxiv.org/abs/2307.02227). DFEW 63.41/74.43, FERV39k 43.12/52.07, MAFW 41.62/54.31; 85M tunable parameters ([FineCLIPER Table 1](https://arxiv.org/abs/2407.02157))
- SVFAP (arXiv 2024): masked video autoencoding on VoxCeleb2 (100 epochs, "about 5–6 days" on 4 RTX 3090), followed by downstream fine-tuning — [arXiv 2401.00416](https://arxiv.org/abs/2401.00416)
- S2D (arXiv 2023): a ViT-B/16 static FER model first trained on AffectNet, then frozen and extended with Temporal-Modeling Adapters and landmark-aware prompters ("<10% tunable"). DFEW 65.45/74.81, FERV39k 43.97/46.21, MAFW 43.40/52.55 — [arXiv 2312.05447](https://arxiv.org/abs/2312.05447); numbers from [FineCLIPER Table 1](https://arxiv.org/abs/2407.02157)
- S4D "Static for Dynamic" (arXiv 2024, v3 2025): dual-modal (image + video) self-supervised pre-training of a shared ViT, then joint fine-tuning on static and dynamic FER data — [arXiv 2409.06154](https://arxiv.org/abs/2409.06154)
- MMA-DFER (arXiv 2024): frozen unimodal SSL encoders, MAE-Face (pre-trained on static face image datasets) and AudioMAE (AudioSet), joined by a fusion bottleneck and prompt tokens (for example 1.3M parameters for one temporal-adaptation configuration). It reports +1.5% UAR/WAR over S2D on DFEW and +1% on MAFW — [arXiv 2404.09010](https://arxiv.org/abs/2404.09010)
- MARLIN (arXiv 2022/2023): ViT masked autoencoder pre-trained on YouTubeFace videos. With linear probing on CMU-MOSEI emotion it reports accuracy 80.60 (VideoMAE 80.39; supervised MViTv1 80.45; the trimodal GMF 81.14). With 1% of the training labels, linear-probe emotion accuracy is 78.61 — [arXiv 2211.06627](https://arxiv.org/abs/2211.06627)
- FMAE / FMAE-IAT (arXiv 2024): MAE pre-trained on Face9M (about 9M face images). Fine-tuned FER accuracy: AffectNet-8 64.79, RAF-DB 93.45 — [arXiv 2407.11243](https://arxiv.org/abs/2407.11243)
- Bulat et al. (ECCV 2022) compare face pre-training strategies in few-shot settings and report that "linear adaptation results in significant performance degradation" compared with fine-tuning — [arXiv 2103.16554](https://arxiv.org/abs/2103.16554)
- DINOv2: no FER paper that evaluates frozen DINOv2 features across datasets was found. A 2026 audio-visual paper uses DINOv2 ViT-L/14 but adapts the backbone ([arXiv 2603.12221](https://arxiv.org/abs/2603.12221), seen only as a search snippet).

### Inferences
- Linear probe versus fine-tuning on frozen CLIP, as reported: about 6 points of RAF-DB WAR (84.29 vs 90.38, CLIPER); 5.6 UAR on FERV39k (33.06 vs 38.68, CLIPER); 12.8 UAR on DFEW (45.46 vs 58.28, DFER-CLIP). Generic CLIP features therefore leave a sizeable gap on in-domain FER. Our frozen features come instead from a ResNet-50 already trained for expressions, so these gaps do not transfer directly to our setting.
- The move from fine-tuned CLIP (2023: CLIPER, DFER-CLIP, EmoCLIP) to frozen CLIP with prompts or adapters (2024–2025: A3lign, FineCLIPER, PE-CLIP, MPA-FER, Exp-CLIP, CAFE) is motivated in these papers by efficiency and by overfitting or generalization (CAFE and MPA-FER say so explicitly).

### Gaps
- No paper was found that compares frozen DINOv2, CLIP, FaRL and MAE-Face features with the same linear probe across FER datasets.
- Whether EmoCapCLIP updates all CLIP layers during its caption training was not checked.
- The venues of CLIPER, SVFAP, S2D, MMA-DFER, A3lign-DFER, PE-CLIP, FMAE-IAT and MARLIN were not confirmed from the PDFs (arXiv versions were read).

## Q2. Cross-domain / cross-dataset FER methods (UDA, source-free DA, TTA, prompt learning, zero-shot, few-shot) and reported gains over source-only or zero-shot

### Takeaway
Across the papers read, the foundation-model papers and the domain-adaptation papers form two separate lines of work.
- CLIP-based papers do cross-dataset work almost only as zero-shot transfer or single-source domain generalization, with no target data (CAFE, Exp-CLIP, EmoCLIP, FineCLIPER, EmoCapCLIP), or as subject-level test-time adaptation on pain and stress videos (TTA-CaP, CLIP-AUTT).
- Classic UDA, multi-source UDA and source-free DA for FER still use ResNet-18/50 or MobileNet backbones initialised from ImageNet or MS-Celeb-1M and train the whole network.
- No paper was found that combines a frozen foundation model, a few labeled target samples, unlabeled target data and multiple sources, which is our setting.
- Reported gains: UDA/DA methods gain about 6–14 points of mean accuracy over source-only. TTA gains are up to 11.34 points in one transfer pair but negative in others. Zero-shot CLIP improvements are 3–8 points of average UAR (Exp-CLIP) and up to about 17 UAR on DFEW (EmoCLIP).

### Cited Findings
**Zero-shot / single-source domain generalization with foundation models (no target data)**
- CAFE (ECCV 2024): train on one dataset, test on five test sets (RAF-DB, FERPlus, AffectNet, SFEW2.0, MMA), 7 classes, accuracy. Trained on RAF-DB: CAFE 88.72 / 73.16 / 45.86 / 52.86 / 56.80 (mean 63.48). EAC 89.54 / 54.38 / 43.91 / 43.39 / 37.27 (mean 53.70). "CLIP+Finetune" (fixed CLIP, a trainable FER model and an FC layer on concatenated features) 88.69 / 59.71 / 44.31 / 41.30 / 44.62 (mean 55.73), so CAFE adds +13.45 on FERPlus and +11.56 on SFEW2.0. CAFE also has the best mean when the source is FERPlus (62.28), AffectNet (60.27), SFEW2.0 (45.04) or MMA (61.76). The abstract motivates the setting: "Recent domain adaptation FER methods need to acquire labeled or unlabeled samples of target domains … which might be infeasible" — [arXiv 2408.10614](https://arxiv.org/abs/2408.10614)
- Exp-CLIP: zero-shot over 8 test sets (RAF-DB, AffectNet-7, AffectNet-8, FERPlus, DFEW, FERV39k, MAFW, AFEW). Average UAR gain over plain CLIP: +2.71 (B/32), +5.47 (B/16), +8.07 (L/14); WAR gains +4.42, +6.75, +8.23. Examples with ViT-L/14 (UAR/WAR): RAF-DB 58.70/65.37 vs CLIP 47.22/41.13; DFEW 40.16/47.09 vs 32.02/38.12; AFEW 38.72/40.42 vs 35.50/38.32 — [arXiv 2405.19100](https://arxiv.org/abs/2405.19100)
- EmoCLIP: trained on MAFW sample-level captions, then evaluated zero-shot (UAR/WAR): DFEW 36.76/46.27 vs CLIP 19.86/10.60; AFEW 36.13/39.90 vs 23.05/11.80; FERV39K 26.73/35.30 vs 20.99/17.10. Supervised EmoCLIP on DFEW reaches 58.04/62.12 — [arXiv 2310.16640](https://arxiv.org/abs/2310.16640)
- FineCLIPER (cross-dataset "zero-shot": train on one DFER dataset, test on the others). Trained on MAFW with fine-grained captions: DFEW 52.26/62.03, FERV39k 39.72/46.01. Trained on DFEW: FERV39k 40.10/46.91. Trained on FERV39k: DFEW 55.13/63.89. EmoCLIP baseline (MAFW video captions): DFEW 36.76/46.27 — [arXiv 2407.02157](https://arxiv.org/abs/2407.02157)
- EmoCapCLIP (zero-shot WAR, ViT-B/32): AffectNet-7 51.91 and SFEW2.0 53.06. These are higher than every supervised cross-dataset model in its Table 3, whose best values are 51.86 on AffectNet-7 and 52.86 on SFEW2.0 (CAFE and others trained on RAF-DB, SFEW2.0 or AffectNet-7). On RAF-DB, 67.03 is lower than 72.69 for the best model trained on AffectNet-7 (CAFE) — [arXiv 2507.21015](https://arxiv.org/abs/2507.21015)
- GPT-4V zero-shot accuracy: CK+ 69.72, SFEW2.0 57.24, RAF-DB 75.81, FERPlus 64.25, AffectNet 42.77. WAR with 3 sampled frames: DFEW (fd1) 54.80, FERV39k 38.72 — [arXiv 2312.04293](https://arxiv.org/abs/2312.04293)

**Prompt learning for domain generalization / open-set (CoOp-style)**
- PromptEmo (AAAI 2026) [abstract only]: CLIP with trainable positive prompts built from LLM-generated emotion descriptions and template-derived negative prompts, for multi-domain open-set FER. It reports +7.7% AUC on unseen classes "across four FER datasets" — [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/38224) (the full text was blocked by a bot check; the details come from the abstract shown in search results)
- NR-GFER (Information Sciences 2025) [abstract only]: a vision-language model with a frequency-domain "Fourier Prompt" and a residual adapter for generalizable FER on unseen targets — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025525005894)
- CoOp / CoCoOp are used as baselines rather than as cross-domain methods. On DFEW, CoOp 44.98/56.68 and CoCoOp 46.80/57.52, below DFER-CLIP 59.61/71.25 ([arXiv 2308.13382](https://arxiv.org/abs/2308.13382)). The MPA-FER CoOp baseline gives RAF-DB 86.15 ([arXiv 2506.21017](https://arxiv.org/abs/2506.21017)).

**Unsupervised DA (labeled source + unlabeled target), non-foundation backbones**
- AGLRLS (IEEE TMM 2024): global–local adversarial alignment, pseudo-labels with a learned dynamic threshold, and a global–local prediction-consistency step. ResNet-50 or MobileNet-v2 initialised from MS-Celeb-1M; whole network trained in two stages (source CE first). Source RAF-DB, ResNet-50, targets CK+ / JAFFE / SFEW2.0 / FER2013 / ExpW: 87.60 / 61.97 / 58.26 / 60.68 / 73.00 (mean 68.30), vs AGRA 66.13, CGLRL 65.75, DT 58.79 (accuracy) — [arXiv 2401.11085](https://arxiv.org/abs/2401.11085)
- LA-CMFER (arXiv 2024): multi-source UDA. Each of 6 datasets is the target in turn and the other 5 are labeled sources. Encoder: ResNet-18 (ImageNet) conv1–conv3; pseudo-label threshold τ = 0.9; 3 runs. Average accuracy: Source-Only 55.67 (both "single best" and "source combine" rows); DANN combine 60.95; MDAN 60.73; M3SDA 56.40; DUML 67.73; LA-CMFER 69.32. Example →RAF-DB: Source-Only (combine) 60.10 vs 77.86 — [arXiv 2407.05688](https://arxiv.org/abs/2407.05688)
- AGRA (older, 2020/2021) set up the unified CD-FER benchmark (source RAF-DB → CK+, JAFFE, SFEW2.0, FER2013, ExpW) — [arXiv 2008.00923](https://arxiv.org/abs/2008.00923); its numbers were read from the AGLRLS table, not from the AGRA paper.

**Domain generalization with multiple sources (no target data)**
- Liu et al. (IJCAI 2023): leave-one-domain-out over Oulu-CASIA, RAF-DB, SFEW 2.0 and JAFFE, with RAF-DB 2.0 as additional source data. Separate domain / expression / identity subspaces, Fourier-phase reconstruction, identity discriminator. ResNet-50 accuracy →JAFFE / →Oulu / →RAF / →SFEW: baseline 49.87 / 52.59 / 58.01 / 41.96, method 71.72 / 67.83 / 74.94 / 60.12. CIRL (a generic DG method) 60.03 / 60.75 / 62.96 / 47.32 — [IJCAI 2023 PDF](https://www.ijcai.org/proceedings/2023/0135.pdf)
- Older context: Suresh, Yeo & Ong (IEEE TAC; arXiv 2021, v2 2023) ran round-robin experiments over 12 datasets with ResNet-50 pre-trained on VGGFace2. The best multi-source DG average is 65.6% (range 34.6–88.6, chance 14.3), 10.8 points below within-corpus accuracy (76.4) — [arXiv 2106.15453](https://arxiv.org/abs/2106.15453)

**Source-free DA**
- CluP (BMVC 2022): source-free UDA starting from a ResNet-18 FER model; AFE / RAF-DB → ExpW / FER2013; top-1 accuracy. CluP (SwAV) 65.00 / 52.51 / 66.60 / 53.71 vs SHOT 54.12 / 49.44 / 53.51 / 49.36. The source-accessible UDA method AGRA gets 65.03 / 51.95 / 69.70 / 54.94 — [arXiv 2210.05246](https://arxiv.org/abs/2210.05246)
- Subject-level SFDA from the ETS Montréal group (pain and stress, not basic emotions):
  - DSFDA (FG 2025): the target domain is one subject with neutral-only video. A generator disentangles identity and expression to synthesise the missing classes. The source model is an Inception-V3 CNN. BioVid average accuracy: source-only 68.62; two rows labeled "DSFDA (ours)" report 80.24 and 81.88 (the difference between the two rows was not resolved in my extraction) — [arXiv 2503.20771](https://arxiv.org/abs/2503.20771)
  - SFDA-PFT (ICLR 2026): ResNet-18; the feature extractor and classifier stay frozen; a feature-space translator pre-trained by "subject-swapping" on source subjects is adapted with neutral target images; BioVid, StressID, BAH, Aff-Wild2 — [arXiv 2508.09202](https://arxiv.org/abs/2508.09202)
  - Subject-based MSDA (FG 2024): every source subject is a separate domain and adaptation targets one unlabeled subject (BioVid 87 subjects, UNBC-McMaster) — [arXiv 2312.05632](https://arxiv.org/abs/2312.05632)

**Test-time adaptation**
- Turnbull et al. (arXiv 2026) present "the first evaluation of TTA methods for FER under natural domain shifts". ViT-L/16 is fine-tuned from the FMAE-IAT checkpoint on AffectNet (about 37k, contempt removed), RAF-DB, FERPlus, and 2-source combinations; 7 classes; 8 TTA methods; streaming batch 16, one step, lr 1e-3, LayerNorm updates; 3 seeds; top-1 accuracy. Results:
  - SHOT on FERPlus→AffectNet: 41.83 → 53.17 (+11.34, the largest gain).
  - T3A on AffectNet→FERPlus: 58.79 → 66.33.
  - TENT on AffectNet→RAF-DB: 72.16 → 74.83.
  - Some methods lower accuracy: SHOT on RAF-DB→FERPlus 74.54 → 60.29; SHOT on RAF-DB→RAF-DB 92.87 → 69.72.
  - Two sources: FERPlus+AffectNet → RAF-DB baseline 76.96 (TENT 80.44), between the single-source baselines 72.16 (AffectNet) and 80.96 (FERPlus).
  - [arXiv 2603.19994](https://arxiv.org/abs/2603.19994)
- TTA-CaP (arXiv 2026): CLIP ViT-B/32 with frozen encoders, fine-tuned on source subjects, then gradient-free cache-based personalization per target subject (10 target subjects per dataset; three caches and a tri-gate). WAR/F1 on BioVid: zero-shot CLIP 50.0/33.3; source-fine-tuned CLIP 69.7/66.6; Exp-CLIP 70.2/66.7; TPT 71.1/67.5; TDA 71.4/68.2; CRG 76.4/73.8; TTA-CaP 81.0/78.6. On StressID 81.5/67.9 and on BAH 69.2/41.1 — [arXiv 2603.21309](https://arxiv.org/abs/2603.21309)
- CLIP-AUTT (arXiv 2026): frozen CLIP with Action Unit text prompts and a temporal module, then test-time prompt tuning per unseen subject using entropy-guided temporal-window selection. BioVid with 77 source and 10 target subjects; also StressID and BAH. Its baselines use CLIP ViT-B/32 — [arXiv 2603.27999](https://arxiv.org/abs/2603.27999)

**Few-shot adaptation**
- EmoCapCLIP runs few-shot linear probing and prompt tuning on RAF-DB "following CoOp", with 5 seeds. It reports that EmoCapCLIP beats Exp-CLIP and FLIP across backbones and shot numbers. The numbers appear only in a figure and could not be extracted — [arXiv 2507.21015](https://arxiv.org/abs/2507.21015)
- MARLIN: label-fraction experiments rather than K-shot. Linear-probe MOSEI emotion accuracy is 80.60 with 100% of labels, 80.59 with 50%, 79.89 with 10% and 78.61 with 1% — [arXiv 2211.06627](https://arxiv.org/abs/2211.06627)
- No FER paper was found that applies Tip-Adapter-style caches or prototype classifiers with few labeled target samples across datasets. The closest cache method is TTA-CaP, which is unsupervised and per subject — [arXiv 2603.21309](https://arxiv.org/abs/2603.21309)

### Inferences
- With foundation models, FER researchers mostly avoid target data (zero-shot or DG) or adapt per subject at test time. UDA with unlabeled target data is done almost only with CNNs trained end to end. The combination "frozen FM features + labeled sources + K target labels + unlabeled target" is absent from the papers examined.
- Two-source results in the TTA paper show that adding a source can lower the source-only baseline relative to the better single source (76.96 vs 80.96 for →RAF-DB). This resembles negative transfer from adding a corpus, but there it comes from fine-tuned ViT-L training rather than linear heads.

### Gaps
- No full text for PromptEmo, NR-GFER or the IEEE Access 2024 CLIP identity paper.
- No source found for UDA on DFER datasets (for example DFEW→FERV39k with unlabeled target) using a foundation model.
- No FER paper reports semi-supervised DA (a few target labels plus unlabeled target) with CLIP or face-MAE features.

## Q3. Standard protocols: datasets, class sets, metrics, domain pairs

### Takeaway
- Static FER uses 7 classes (6 basic + neutral); AffectNet-8 and FERPlus add contempt.
- Static cross-domain papers report overall accuracy; the CD-FER benchmark is "RAF-DB → CK+, JAFFE, SFEW2.0, FER2013, ExpW".
- Dynamic FER uses DFEW (7 classes, 5 folds), FERV39k (7 classes, 80/20 split) and MAFW (11 classes, 5 folds), and always reports both UAR and WAR.
- None of the papers uses a 4-class set like ours, and UAR is not the main metric in static cross-domain FER.

### Cited Findings
- Static datasets as used by Exp-CLIP:
  - RAF-DB: 7 classes, 3,068 test images.
  - AffectNet-7: 3,500 test images; AffectNet-8 (contempt added): 4,000 test images.
  - FERPlus: 8 classes, 3,589 test images.
  - Metrics: WAR (= accuracy) and UAR, the latter defined as "the accuracy per class divided by the number of classes".
  - [arXiv 2405.19100](https://arxiv.org/abs/2405.19100)
- Dynamic datasets as used by Exp-CLIP:
  - DFEW: 11,697 clips, 7 classes, 5-fold CV.
  - FERV39k: 38,935 clips, 7 classes, 80/20 split.
  - MAFW: 10,045 clips.
  - AFEW: 1,809 clips split 773 / 383 / 653. The test split is not public, so the validation split is reported.
  - [arXiv 2405.19100](https://arxiv.org/abs/2405.19100)
- EmoCLIP evaluates DFEW, AFEW and FERV39K with 7 classes and MAFW with 11 classes (also 43 compound classes) — [arXiv 2310.16640](https://arxiv.org/abs/2310.16640)
- CLIPER uses WAR as the validation metric for static FER and both UAR and WAR for DFER. DFEW uses 12,059 clips with the 5 folds provided with the dataset — [arXiv 2303.00193](https://arxiv.org/abs/2303.00193)
- CAFE uses 7 basic classes for RAF-DB, FERPlus, AffectNet, SFEW2.0 and MMA and reports test accuracy; per-class mean accuracy is in its supplement — [arXiv 2408.10614](https://arxiv.org/abs/2408.10614)
- AGLRLS: source RAF-DB or FER2013; targets CK+, JAFFE, SFEW2.0, FER2013/RAF-DB, ExpW; accuracy plus mean recall, precision and F1 — [arXiv 2401.11085](https://arxiv.org/abs/2401.11085)
- LA-CMFER: leave-one-dataset-out over JAFFE (213 images, 10 Japanese women), RAF-DB (12,271 / 3,068), CK+ (last three frames, 1,236 images), Oulu-CASIA, AffectNet (up to 5,000 per class, 33,793 train / 3,500 test) and FER-2013 (28,709 / 3,589); 7 classes; for the lab datasets the training and test sets are the same — [arXiv 2407.05688](https://arxiv.org/abs/2407.05688)
- The TTA evaluation uses pairwise and 2-source transfers among AffectNet, RAF-DB and FERPlus with 7 classes and top-1 accuracy — [arXiv 2603.19994](https://arxiv.org/abs/2603.19994)
- SVFAP uses subject-independent cross-validation for the lab audio-visual datasets: 5-fold for CREMA-D (91 actors), 6-fold for RAVDESS speech (24 actors), 5-fold for eNTERFACE05 — [arXiv 2401.00416](https://arxiv.org/abs/2401.00416)
- The personalization line (TTA-CaP, CLIP-AUTT) uses pain and stress datasets (BioVid, StressID, BAH) with WAR and macro-F1, averaged over 10 target subjects — [arXiv 2603.21309](https://arxiv.org/abs/2603.21309)

### Inferences
- Typical domain pairs:
  - Static UDA: RAF-DB as the large in-the-wild source, lab or small targets (CK+, JAFFE, SFEW2.0) and large web targets (FER2013, ExpW).
  - Static DG: each of 4–6 datasets held out in turn.
  - Video zero-shot: MAFW → DFEW / FERV39k / AFEW; also DFEW ↔ FERV39k.
- Implication for comparison: static cross-domain numbers are accuracy (WAR) under 7 classes with class imbalance, so they are not directly comparable with our 4-class UAR. DFER papers report UAR as well, which makes them the nearer reference.

### Gaps
- No standard cross-dataset protocol for DFER with unlabeled target data was found.
- None of the FER papers maps classes to a 4-class set (Neutral / Happy / Sad / Angry).

## Q4. How do these works handle identity (subject) information in face representations?

### Takeaway
Most CLIP-based FER papers do not discuss identity at all. Identity is handled explicitly in three other strands of work:
- disentanglement or adversarial removal (IJCAI 2023 CD-FER, DSFDA, FMAE-IAT for AUs);
- treating each subject as its own domain for personalization (ETS Montréal group, five papers);
- subject-independent splits in lab datasets.

The strongest quantitative evidence comes from action-unit (AU) data, not basic-emotion FER: a linear probe on a frozen AU model identifies 41 BP4D subjects with 83% accuracy.

### Cited Findings
- Full-text search counts for "identit*": 0 in CAFE, EmoCLIP, CLIPER, DFER-CLIP, FineCLIPER, A3lign-DFER, PE-CLIP and S2D; 1 in Exp-CLIP and in MPA-FER, both only in the reference lists — [CAFE](https://arxiv.org/abs/2408.10614), [EmoCLIP](https://arxiv.org/abs/2310.16640), [CLIPER](https://arxiv.org/abs/2303.00193), [DFER-CLIP](https://arxiv.org/abs/2308.13382), [FineCLIPER](https://arxiv.org/abs/2407.02157), [A3lign-DFER](https://arxiv.org/abs/2403.04294), [PE-CLIP](https://arxiv.org/abs/2503.16945), [S2D](https://arxiv.org/abs/2312.05447), [Exp-CLIP](https://arxiv.org/abs/2405.19100), [MPA-FER](https://arxiv.org/abs/2506.21017)
- FMAE-IAT (AU detection):
  - Linear probing on a frozen, trained AU model recognised BP4D subjects (41 identities; 70 training and 30 test samples per subject) with accuracy up to 83%. The authors conclude the models "effectively 'memorize' subject identities".
  - Identity Adversarial Training (a linear identity head with gradient reversal, λ = 2.0 / 1.0 / 0.5 for BP4D / BP4D+ / DISFA) lowers identity accuracy to 4.6% after 1 epoch and 27.9% at epoch 19.
  - The authors did not apply IAT to FER "because these datasets do not include the identity labels and do not suffer from identity shortcut learning due to the large number of subjects".
  - [arXiv 2407.11243](https://arxiv.org/abs/2407.11243)
- Liu et al. (IJCAI 2023) argue that cross-domain FER methods "suffer from the interference of identity features". They separate domain, expression and identity subspaces and add an identity discriminator. Leave-one-domain-out accuracy gains over the ResNet-50 baseline are 21.85 (→JAFFE), 15.24 (→Oulu), 16.93 (→RAF-DB) and 18.16 (→SFEW) points — [IJCAI 2023 PDF](https://www.ijcai.org/proceedings/2023/0135.pdf)
- DSFDA disentangles identity and expression to generate non-neutral expressions with the target subject's identity; SFDA-PFT's subject-swapping translator learns "the structural relationship between expression and identity-specific features" — [arXiv 2503.20771](https://arxiv.org/abs/2503.20771), [arXiv 2508.09202](https://arxiv.org/abs/2508.09202)
- Subject-as-domain: Subject-based MSDA (FG 2024) "considers each subject as a separate domain" — [arXiv 2312.05632](https://arxiv.org/abs/2312.05632). TTA-CaP builds a static cache by matching target-subject feature statistics to the top-3 closest source subjects — [arXiv 2603.21309](https://arxiv.org/abs/2603.21309). CLIP-AUTT adapts AU prompts per unseen subject — [arXiv 2603.27999](https://arxiv.org/abs/2603.27999)
- Kim et al. (IEEE Access 2024) [abstract / snippet only] use CLIP to retrieve attribute information (gender, age, skin) and train expression and identity features separately, with an objective based on mutual information, to obtain "ID-invariant" expression features — [DOAJ record](https://doaj.org/article/74beb1a28a3246c681992a254290e720)
- In the GPT-4V benchmark, the prompts tell the model to "ignore the speaker's identity and focus on the facial expression" — [arXiv 2312.04293](https://arxiv.org/abs/2312.04293)
- The SSL video models use 5- or 6-fold subject-independent cross-validation for CREMA-D, RAVDESS and eNTERFACE05 — [arXiv 2401.00416](https://arxiv.org/abs/2401.00416)
- MARLIN's discussion notes that its features focus on facial attributes such as hair and on identities, which can introduce bias (the passage was only partly legible in the extracted text) — [arXiv 2211.06627](https://arxiv.org/abs/2211.06627)

### Inferences
- In-the-wild FER benchmarks have thousands of subjects with few images each, so identity shortcuts are seldom tested there; the FMAE-IAT authors give exactly this reason. Corpora with few speakers and many utterances per speaker (IEMOCAP: 10 speakers; also lab corpora such as RAVDESS and CREMA-D) are closer to the AU-dataset case, where identity was shown to be linearly decodable at 83%.
- Measuring how well speaker identity can be decoded from frozen face features, as FMAE-IAT did with a linear probe, is an established diagnostic. It was applied to AU models, not to emotion-category features.

### Gaps
- No paper was found that measures identity decodability from frozen CLIP, FaRL or DINOv2 features and relates it to cross-dataset FER accuracy.
- No FER paper checks speaker or actor overlap between the DFEW / FERV39k / MAFW folds.

## Q5. What counts as the "standard method", and how common is each choice? (counts, per-paper record, differences from our setting)

### Takeaway
Among the 12 CLIP-based FER papers read:
- 8 keep the CLIP image encoder frozen and 3 fine-tune it (EmoCapCLIP not counted, its training scope was not checked);
- ViT-B/32 is used in 7, ViT-B/16 in 6 and ViT-L/14 in 4 (papers can use several);
- 6 evaluate only within one dataset, 5 do zero-shot or cross-dataset transfer without target data, 2 do test-time adaptation per subject, 1 does few-shot (within one dataset), and none does UDA with unlabeled target data.

Among the 9 cross-domain FER papers read, 7 use CNN backbones trained end to end, 1 uses a ViT-L/16 face MAE fine-tuned on the source (TTA evaluation), and 1 was not re-checked. None of the 9 uses expression labels from the target; DSFDA and PFT assume the target data is neutral-only. Accuracy is the metric in the static papers. The de-facto standard is therefore:
- for foundation-model FER: frozen CLIP plus prompts, adapters or a head, evaluated within the dataset or zero-shot;
- for cross-domain FER: an end-to-end CNN with adversarial alignment and pseudo-labels.

### Cited Findings
Counts below are over the papers whose full text was read; the evidence for each row is in Q1–Q4.

**CLIP-based FER papers (n = 12): CAFE, Exp-CLIP, EmoCLIP, CLIPER, DFER-CLIP, FineCLIPER, A3lign-DFER, PE-CLIP, MPA-FER, EmoCapCLIP, TTA-CaP, CLIP-AUTT**
- CLIP image encoder frozen in the main method: 8. CAFE (fixed CLIP features plus a trainable ResNet-18 mask network), Exp-CLIP (linear head), A3lign-DFER, FineCLIPER (adapters), PE-CLIP (adapters and prompts), MPA-FER (prompts), TTA-CaP (cache), CLIP-AUTT (prompts). Image encoder fine-tuned: 3 (CLIPER, DFER-CLIP, EmoCLIP). EmoCapCLIP was not counted — sources in Q1.
- Backbone (a paper can use several): ViT-B/32 in 7 (CAFE, Exp-CLIP, EmoCLIP, CLIPER-DFER, DFER-CLIP, TTA-CaP, EmoCapCLIP); ViT-B/16 in 6 (Exp-CLIP, CLIPER-SFER, FineCLIPER, PE-CLIP, MPA-FER, EmoCapCLIP); ViT-L/14 in 4 (Exp-CLIP, A3lign-DFER, MPA-FER, EmoCapCLIP) — sources in Q1.
- Evaluation type: within one dataset only in 6 (CLIPER, DFER-CLIP, A3lign-DFER, PE-CLIP, MPA-FER, and FineCLIPER's main table); zero-shot or cross-dataset without target data in 5 (CAFE, Exp-CLIP, EmoCLIP, FineCLIPER's zero-shot table, EmoCapCLIP); per-subject TTA with unlabeled target data in 2 (TTA-CaP, CLIP-AUTT); UDA with unlabeled target data in 0; few labeled target samples in 1 (EmoCapCLIP, within RAF-DB) — sources in Q2.
- Linear probe on frozen CLIP appears as a baseline in 4 (CLIPER, DFER-CLIP, EmoCLIP's "EmoCLIP (LP)", EmoCapCLIP's few-shot study), and as the method itself in 0 — [CLIPER](https://arxiv.org/abs/2303.00193), [DFER-CLIP](https://arxiv.org/abs/2308.13382), [EmoCLIP](https://arxiv.org/abs/2310.16640), [EmoCapCLIP](https://arxiv.org/abs/2507.21015)
- Metric: UAR and WAR in all DFER papers (DFER-CLIP, CLIPER-DFER, EmoCLIP, FineCLIPER, A3lign-DFER, PE-CLIP, Exp-CLIP, EmoCapCLIP); accuracy only in the static-only papers CAFE and MPA-FER; WAR and macro-F1 in TTA-CaP.
- An LLM is used to write class or sample descriptions in 5 (Exp-CLIP: Flan-T5-XXL; EmoCLIP: ChatGPT for class descriptions; MPA-FER: ChatGPT; PromptEmo [abstract only]; EmoCapCLIP: MLLM-generated captions) — [Exp-CLIP](https://arxiv.org/abs/2405.19100), [EmoCLIP](https://arxiv.org/abs/2310.16640), [MPA-FER](https://arxiv.org/abs/2506.21017), [EmoCapCLIP](https://arxiv.org/abs/2507.21015)

**Cross-domain FER papers (n = 9): AGLRLS, LA-CMFER, IJCAI-2023 identity DG, CluP, Subject-based MSDA, DSFDA, SFDA-PFT, TTA evaluation, Suresh et al. [older]**
- Backbone:
  - CNN confirmed in 7: ResNet-50/MobileNet-v2 (MS-Celeb init) in AGLRLS; ResNet-18 (ImageNet) in LA-CMFER; ResNet-18/50 in IJCAI 2023; ResNet-18 in CluP and PFT; Inception-V3 in DSFDA; ResNet-50 (VGGFace2) in Suresh et al.
  - Face-pretrained ViT in 1: ViT-L/16 FMAE-IAT, fine-tuned, in the TTA evaluation.
  - Not re-checked: Subject-based MSDA.
  - Sources in Q2.
- Target labels: no expression labels from the target in all 9. In DSFDA and PFT the target data is neutral-only by design, so its class is known. Unlabeled target data is used in 7 (all except IJCAI-2023 DG and Suresh DG). Multi-source in 4 (LA-CMFER, IJCAI 2023, the TTA evaluation's 2-source combinations, Suresh et al.) — sources in Q2.
- Main metric: accuracy in all the static basic-emotion papers (AGLRLS, LA-CMFER, IJCAI 2023, CluP, TTA evaluation, Suresh et al.).

**Per-paper record and difference from our setting**

| Paper | FM / backbone, frozen? | What is trained | Target data used | Data / classes / metric | Difference from our setting | Source |
|---|---|---|---|---|---|---|
| CAFE (ECCV'24) | CLIP ViT-B/32 frozen | ResNet-18 mask network, channel-split logits | none | 5 static datasets, 7 classes, accuracy | images, single source, no target labels, CNN on pixels is trained | [2408.10614](https://arxiv.org/abs/2408.10614) |
| Exp-CLIP (WACV'25) | CLIP B/32, B/16, L/14 frozen | linear projection head (LLM-aligned, unlabeled CAER-S) | none (zero-shot) | 4 static + 4 video, 7/8 classes, UAR/WAR | zero labels anywhere; text-based classifier | [2405.19100](https://arxiv.org/abs/2405.19100) |
| EmoCLIP (FG'24) | CLIP B/32 fine-tuned | both encoders + temporal layer, MAFW captions | none at test | MAFW→DFEW/AFEW/FERV39k, UAR/WAR | caption supervision; video end to end; no labeled target | [2310.16640](https://arxiv.org/abs/2310.16640) |
| CLIPER | CLIP B/16 / B/32 fine-tuned | text descriptors, then image encoder | none (within dataset) | RAF-DB, AffectNet, FERPlus, DFEW, FERV39k | in-domain only | [2303.00193](https://arxiv.org/abs/2303.00193) |
| DFER-CLIP (BMVC'23) | CLIP B/32 image fine-tuned, text frozen | image encoder, temporal model, prompts | none (within dataset) | DFEW, FERV39k, MAFW, UAR/WAR | in-domain only | [2308.13382](https://arxiv.org/abs/2308.13382) |
| FineCLIPER (MM'24) | CLIP B/16 frozen + adapters | adapters (13M) | none; cross-dataset zero-shot table | DFEW, FERV39k, MAFW, UAR/WAR | adds face parsing, landmarks, captions | [2407.02157](https://arxiv.org/abs/2407.02157) |
| A3lign-DFER | CLIP L/14 frozen | label text embeddings, dynamic encoders | none | DFEW, FERV39k, MAFW | in-domain only | [2403.04294](https://arxiv.org/abs/2403.04294) |
| PE-CLIP | CLIP B/16 frozen | adapters + prompts (9M) | none | DFEW, FERV39K | in-domain only | [2503.16945](https://arxiv.org/abs/2503.16945) |
| MPA-FER (ICCV'25) | CLIP B/16, L/14 frozen | visual + text prompts | none | RAF-DB, AffectNet, FERPlus, accuracy | in-domain only | [2506.21017](https://arxiv.org/abs/2506.21017) |
| EmoCapCLIP | CLIP trained on EmoCap100K | contrastive caption training; few-shot LP / CoOp | few-shot labels within RAF-DB | 9 static + 8 video benchmarks, UAR/WAR | few-shot is within one dataset, not cross-corpus | [2507.21015](https://arxiv.org/abs/2507.21015) |
| TTA-CaP | CLIP B/32 frozen at test time | source fine-tuning, then caches (no gradients) | unlabeled frames of one target subject | BioVid, StressID, BAH; WAR/F1 | pain/stress classes; per-subject target | [2603.21309](https://arxiv.org/abs/2603.21309) |
| CLIP-AUTT | frozen CLIP | AU prompts, temporal module; test-time prompt tuning | unlabeled target subject video | BioVid, StressID, BAH | same as above | [2603.27999](https://arxiv.org/abs/2603.27999) |
| AGLRLS (TMM'24) | ResNet-50 / MobileNet-v2 (MS-Celeb), fully trained | whole net, adversarial + pseudo-labels | unlabeled target | RAF-DB→5 targets, 7 classes, accuracy | no FM; no target labels; single source | [2401.11085](https://arxiv.org/abs/2401.11085) |
| LA-CMFER | ResNet-18 (ImageNet), fully trained | whole net, multi-source alignment + pseudo-labels | unlabeled target | leave-one-dataset-out over 6 datasets, 7 classes, accuracy | closest to our multi-source structure, but 5 sources, no target labels, CNN | [2407.05688](https://arxiv.org/abs/2407.05688) |
| IJCAI'23 identity DG | ResNet-18/50, fully trained | domain / expression / identity disentangling | none | leave-one-domain-out over 4 datasets, accuracy | DG; uses identity disentanglement | [IJCAI PDF](https://www.ijcai.org/proceedings/2023/0135.pdf) |
| CluP (BMVC'22) | ResNet-18 FER model | target model (source-free) | unlabeled target | AFE/RAF-DB → ExpW/FER2013, accuracy | source-free; no target labels | [2210.05246](https://arxiv.org/abs/2210.05246) |
| TTA evaluation (2026) | ViT-L/16 FMAE-IAT, fine-tuned on source | LayerNorm / BN parameters at test time | unlabeled target stream | AffectNet / RAF-DB / FERPlus, incl. 2-source, accuracy | no target labels; fine-tuned ViT | [2603.19994](https://arxiv.org/abs/2603.19994) |
| MARLIN | face video MAE (YouTubeFace); LP and FT | linear head (LP) | label fractions within MOSEI | MOSEI emotion, accuracy | closest to frozen features + linear head on MOSEI, but within one corpus | [2211.06627](https://arxiv.org/abs/2211.06627) |
| MMA-DFER | frozen MAE-Face + AudioMAE | fusion bottleneck, prompts | none | DFEW, MAFW, UAR/WAR | audio-visual, in-domain | [2404.09010](https://arxiv.org/abs/2404.09010) |

### Inferences
- Across the papers examined, the "standard method" depends on the goal:
  - in-domain FER/DFER accuracy: frozen CLIP with prompts or adapters, or a fully fine-tuned face video MAE;
  - generalization without target data: frozen CLIP with LLM- or caption-derived text supervision, judged by zero-shot UAR/WAR;
  - adaptation to a new dataset: a CNN with adversarial alignment and confidence-thresholded pseudo-labels (τ = 0.9 in LA-CMFER), measured by accuracy, with RAF-DB as the usual source.
- Our setting differs from all of them in four ways at once: (1) the inputs are fixed utterance-level vectors (one frozen ResNet-50 FER feature per utterance, plus audio and text), not pixels or frame sequences; (2) K labeled target examples per class plus unlabeled target data, whereas these papers use 0 target labels or labels within the same dataset; (3) 4 classes and UAR as the metric, whereas FER uses 7–11 classes and accuracy for static cross-domain work; (4) the sources are multimodal conversational or acted corpora, not FER image datasets.
- The closest structural analogues are LA-CMFER (multi-source, leave-one-dataset-out, unlabeled target, pseudo-labels), the TTA evaluation (2-source combinations, unlabeled target) and MARLIN (frozen face features + linear head on MOSEI). None of them uses target labels.

### Gaps
- The counts cover only the 31 papers read in full. They are not a systematic survey, and they over-represent groups with several papers (Patras/QMUL: DFER-CLIP, EmoCLIP, Exp-CLIP; ETS Montréal: four subject-level papers).
- Few-shot numbers for EmoCapCLIP are only in a figure. The AGRA benchmark numbers were taken from the AGLRLS table, not from AGRA itself.
- The DSFDA row labels (two "ours" rows) and the CLIP-AUTT backbone and numbers were not fully resolved.
