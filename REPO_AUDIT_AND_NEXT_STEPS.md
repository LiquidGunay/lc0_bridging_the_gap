Deep Response

## High-level verdict

This repo is a **good LC0/JAX substrate** and a **partial concept-discovery scaffold**, but I would not yet call it a faithful Schut et al. reproduction. The missing core is that Schut et al. are not simply separating “LC0 positions” from “human positions.” Their central method discovers **sparse latent concept vectors from contrastive MCTS rollouts**: optimal lines chosen by the engine versus subpar alternatives, followed by **novelty filtering** against human-game representation spaces and **teachability filtering** through prototype-based transfer. The repo currently has activation dumping, a sparse CVXPY-style separator, puzzle-tag matching, filtering utilities, and causal patching hooks, but the Schut-style dynamic rollout machinery and the novelty/teachability filters are not yet implemented at research-baseline quality. The repo README says it targets LC0 BT4, JAX/Flax inference, activation dumps, and Schut-style concept discovery, which is the right substrate; the paper’s method, however, explicitly emphasizes policy-value network plus MCTS and filtering for novelty and teachability. ([GitHub][1])

My recommendation: turn this into two things at once:

1. A **research-grade chess concept discovery baseline** centered on LC0 dynamic concepts.
2. A **“Learn Interpretability with Chess” notebook series** where each notebook teaches one interpretability idea through a concrete chess experiment.

---

## What Schut et al. actually require

Schut et al. define concepts as sparse vectors in a neural network’s latent space. They justify a linear concept-vector assumption, then formulate concept discovery as a convex optimization problem with sparsity induced by an (L_1) objective. For dynamic chess concepts, the crucial ingredient is not a dataset label like “fork” or “pin”; it is a contrast between an MCTS-selected optimal rollout and one or more subpar rollouts. ([ar5iv][2])

Let

[
\phi_l(s) \in \mathbb{R}^{d_l}
]

be the activation representation of position (s) at layer (l). For LC0 BT4, I would usually treat the raw activation as

[
Z_l(s) \in \mathbb{R}^{64 \times d}
]

and only later decide whether to flatten, pool, or learn square-local concepts. The current repo’s `_pool_tokens` averages across the 64 board tokens, which is convenient but removes much of the spatial information needed for chess concepts like outposts, files, passed pawns, weak squares, pawn storms, and queen-side regrouping. ([GitHub][3])

For a **static supervised concept**, the mathematically faithful baseline should be:

[
\min_{v,\xi \ge 0} |v|_1 + C \sum_i \xi_i
]

subject to

[
v^\top\left(\phi_l(s_i^+) - \phi_l(s_i^-)\right) \ge \gamma - \xi_i.
]

Here (s_i^+) contains the concept, (s_i^-) does not, (\gamma) is a positive margin, and (v) is the sparse concept vector. The repo’s `svm_cvxpy` branch is close to this static paired-constraint form, but the rest of the pipeline still treats this as an A-vs-B embedding separator rather than as a full concept-discovery and validation framework. ([GitHub][4])

For a **dynamic concept**, define an LC0 MCTS optimal rollout

[
\tau_i^+ = (s_{i,0}^+, s_{i,1}^+, \ldots, s_{i,D}^+)
]

and one or more subpar rollouts

[
\tau_{i,m}^- = (s_{i,0}^-, s_{i,1}^-, \ldots, s_{i,D}^-).
]

Then define an aggregated trajectory representation:

[
\psi_l(\tau) = \frac{1}{|T|} \sum_{t \in T} \operatorname{vec}(Z_l(s_t)).
]

Schut et al. distinguish “single-player” rollout indexing from “both-player” rollout indexing because latent representations alternate perspective as players move; the paper uses every other representation for single-player concepts and every representation for both-player concepts. ([ar5iv][2])

The dynamic sparse concept objective becomes:

[
\min_{v,\xi \ge 0} |v|*1 + C \sum*{i,m} \xi_{i,m}
]

subject to

[
v^\top\left(\psi_l(\tau_i^+) - \psi_l(\tau_{i,m}^-)\right) \ge \gamma - \xi_{i,m}.
]

For prophylactic concepts, where the optimal line avoids increasing some bad latent property, also solve the reversed-sign version:

[
v^\top\left(\psi_l(\tau_{i,m}^-) - \psi_l(\tau_i^+)\right) \ge \gamma - \xi_{i,m}.
]

This active/prophylactic split matters. The paper explicitly describes active planning as increasing concept presence and prophylactic planning as avoiding concept increase in the subpar line. ([ar5iv][2])

After discovering candidate concepts, Schut et al. do not stop at prototype examples. They filter for **teachability** and **novelty**. Teachability is measured by whether a student network improves on concept-related test positions after training on concept prototypes. Novelty is measured by whether the concept vector is better reconstructed from the machine-game latent basis than from the human-game latent basis. ([ar5iv][2])

For novelty, let (X_M) be machine-game activations and (X_H) be human-game activations. Compute SVD bases (V_{M,k}) and (V_{H,k}). For concept vector (v),

[
L_M(k) = \frac{|v - V_{M,k}V_{M,k}^\top v|_2^2}{|v|_2^2},
]

[
L_H(k) = \frac{|v - V_{H,k}V_{H,k}^\top v|_2^2}{|v|_2^2},
]

[
\nu(k) = L_H(k) - L_M(k).
]

A concept is more machine-novel when (\nu(k) > 0) over a grid of (k) values. The paper’s novelty section compares reconstruction losses from AlphaZero-game bases and human-game bases and accepts concepts whose reconstruction is consistently better using the machine-game basis. ([ar5iv][2])

---

## Repo audit against that standard

| Area                      |                                                                                                                    Current repo state | Assessment                                                                                                                        | Required change                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| LC0/JAX substrate         | Strong. The README describes LC0 BT4 inference, ONNX oracle checks, activation dumping, concept tools, UCI support, and export hooks. | Good foundation.                                                                                                                  | Keep this as the engine layer. Do not mix concept logic into model code. ([GitHub][1])                                              |
| Activation representation |                                                                `dump_activations` captures a layer and then mean-pools the 64 tokens. | Major Schut mismatch. Chess concepts are spatial; pooling destroys square/file/rank structure.                                    | Store unpooled (64 \times d) activations; expose flatten, square-local, and pooled modes separately. ([GitHub][3])                  |
| History handling          |        Activation dumping calls `encode_board(board, [], ...)`, so PGN move history is discarded. LC0 input planes are history-aware. | Important modeling gap. Some strategic concepts depend on prior move, repetition, castling, en passant, and side-to-move context. | Store rolling 8-board history from PGNs/chunks, not just FEN. ([GitHub][3])                                                         |
| Sparse concept solver     |                               `concepts.py` includes `mean_diff`, `whitened_mean_diff`, `cov_shift`, `cluster_diff`, and `svm_cvxpy`. | `svm_cvxpy` is the closest to the paper, but the surrounding data construction is not Schut-style dynamic concept discovery.      | Split into `static_sparse_concept` and `dynamic_sparse_concept`, each with explicit constraint datasets. ([GitHub][4])              |
| MCTS rollout pairs        |          Existing filtering uses LC0 evaluation and disagreement tools, but there is no stored optimal-versus-subpar rollout dataset. | This is the main missing piece.                                                                                                   | Add an MCTS pair builder that stores root, optimal PV, subpar PVs, visits, Q/WDL, policy prior, depth, and node budget.             |
| Novelty filtering         |                                                                                              Not implemented as the paper defines it. | Major gap.                                                                                                                        | Add SVD-basis reconstruction comparison: machine basis vs human basis.                                                              |
| Teachability filtering    |                                                               Current causal validation measures patch effects, not student learning. | Useful but not equivalent.                                                                                                        | Add student distillation from concept prototypes and compare against random-prototype curricula.                                    |
| Puzzle tag matching       |                         `match_concepts.py` projects puzzle activations onto concept directions and averages scores by Lichess theme. | Good pedagogical validation, not concept discovery. Also needs normalization, statistical tests, and confound controls.           | Treat tags as external interpretation/evaluation only, never as the unsupervised discovery signal. ([GitHub][5])                    |
| Data filtering            |    `filter_fens_eval.py` supports LC0 engine evaluation, centipawn limits, WDL expectation, phase, ply, pieces, and non-pawn filters. | Good start. The variable named `win_prob` is actually using WDL expectation, so name/calibration should be clarified.             | Rename to `expected_score`; add stability across node budgets, tablebase exclusion, mate filters, and stratification. ([GitHub][6]) |
| Documentation             |                       `docs/schut_audit_report.md` correctly notes that mean difference, whitening, and PCA are not the paper method. | Good start, but it over-compresses the fix into “add sparse SVM.”                                                                 | Expand the audit around dynamic rollout constraints, novelty, teachability, and activation geometry. ([GitHub][7])                  |
| Learning-readiness        |                                                               Many raw source files appear as single-line modules through GitHub raw. | Bad for a notebook-learning project.                                                                                              | Reformat with Black/Ruff and add educational docstrings.                                                                            |

The most serious issue is this: **the current repo can discover a sparse direction separating two activation clouds, but Schut-style concept discovery needs a sparse direction explaining why a searched plan beats an alternative searched plan.** That distinction is the heart of the paper.

---

## The concept-discovery baseline I would build

### Baseline name

**LC0 Dynamic Sparse Concepts**, or **LDSC**.

This should be the main baseline. Other methods such as mean difference, LDA/whitened mean difference, PCA covariance shift, k-means, and sparse autoencoders can be comparison baselines, but the Schut-aligned method should be LDSC.

---

## Data design: the right open data to filter

Use three data tiers.

### Tier A: Pedagogical data

Use this for teaching notebooks, sanity checks, and tag matching.

Sources:

* Lichess puzzle database, because it contains `PuzzleId`, `FEN`, `Moves`, `Rating`, `Themes`, and `OpeningTags`. ([database.lichess.org][8])
* Lichess puzzle theme pages, especially origin categories like Master, Master vs Master, and Super GM, because they support more chess-serious subsets. ([lichess.org][9])

Filters:

[
\text{Rating} \in [1200, 2600]
]

for beginner-to-advanced notebooks, and

[
\text{Rating} \ge 2200
]

for research validation.

Also require:

[
\text{RatingDeviation} \le 100,\quad \text{Popularity} > 0,\quad \text{NbPlays} \ge 50.
]

Remove or separately bucket tactical tags like `mate`, `mateIn1`, `mateIn2`, and `oneMove`, because they can dominate concept vectors with shallow tactical motifs.

### Tier B: Human comparison data

Use this for novelty filtering.

Preferred sources:

* Lichess broadcasts for OTB-style strong human games.
* Lichess monthly standard rated PGNs for larger online samples. As of the current Lichess standard list I checked, the newest standard monthly file listed is `lichess_db_standard_rated_2026-03.pgn.zst`. ([database.lichess.org][10])

Filters:

[
\text{variant} = \text{standard}
]

[
\min(Elo_\text{white}, Elo_\text{black}) \ge 2400
]

for research novelty, with a fallback to 2200 or 2000 only for notebooks and small runs.

Use rapid/classical for online games; for broadcasts, preserve OTB-style events even if the PGN lacks normal Lichess time-control metadata.

Position filters:

[
\text{ply} \ge 18
]

[
0.25 \le \text{phase} \le 0.85
]

[
#\text{pieces} \ge 10
]

[
#\text{non-pawns} \ge 4.
]

Remove:

* illegal FENs,
* variants,
* duplicate normalized board states,
* positions with (\le 7) pieces if tablebase contamination is a concern,
* mate-in-1 / trivial forced mate positions,
* book-only opening positions.

Split by **game**, not by position, to avoid near-duplicate train/test leakage.

### Tier C: Machine concept-discovery data

Use LC0 self-play training chunks and generated LC0 games.

The LCZero training repo points to official training data storage, and notes that training data is packed into tar files containing chunks/games. ([GitHub][11])

For Schut-style novelty, machine positions should not be arbitrary LC0 positions. Use positions where:

1. A strong LC0 network and a weaker checkpoint disagree on top move.
2. The root is not a trivial tactic or tablebase endgame.
3. LC0’s MCTS evaluation is stable across node budgets.
4. The MCTS tree contains a meaningful subpar rollout.

For each root (s), run LC0 MCTS and store:

```text
root_fen
root_history
network_id
node_budget
temperature
best_move
best_pv
best_pv_states
best_visits
best_q_or_wdl
candidate_subpar_moves
subpar_pvs
subpar_visits
subpar_q_or_wdl
policy_priors
search_entropy
phase
ply
material_signature
opening_family
```

The subpar rollout should not be garbage. Select the most-visited alternative satisfying:

[
N(a^-) \ge N_{\min}
]

[
Q(a^+) - Q(a^-) \ge \Delta_Q
]

or

[
N(a^+) - N(a^-) \ge \Delta_N.
]

For notebook-scale runs, use (N_{\text{nodes}} = 800) or (1600). For research validation, use (10{,}000+) nodes, and re-check final prototypes at a higher budget.

---

## Exact baseline pipeline

### Step 1: Activation store

Replace the current pooled activation dump with an explicit activation schema.

```python
ActivationRecord:
    game_id: str
    ply: int
    fen: str
    history_fens: list[str]
    source: Literal["human", "lc0_selfplay", "puzzle", "broadcast"]
    layer: str
    activation: np.ndarray  # [64, d] or [d] depending on layer
    policy_logits: np.ndarray
    wdl: np.ndarray
    metadata: dict
```

The important change is that activations should be stored as (64 \times d), not immediately pooled.

Then expose three scoring modes:

[
\phi_l^{\text{flat}}(s) = \operatorname{vec}(Z_l(s)) \in \mathbb{R}^{64d}
]

[
\phi_l^{\text{mean}}(s) = \frac{1}{64}\sum_{q=1}^{64} Z_l(s)_q \in \mathbb{R}^{d}
]

[
\phi_l^{\text{local}}(s; v) = \max_q v^\top Z_l(s)_q.
]

For Schut-faithful experiments, start with `flat`. For notebooks, use `mean` first because it is easier to explain, then show why it fails for spatial chess concepts.

### Step 2: Static sanity concepts

Before dynamic discovery, prove the machinery works.

Use easy concepts:

* piece present: queen, rook, bishop, knight, pawn,
* side-to-move material imbalance,
* passed pawn,
* open file,
* king safety proxy,
* puzzle tags such as `fork`, `pin`, `skewer`, `deflection`.

For each concept, construct matched positive/negative pairs.

Solve:

[
\min_{v,\xi \ge 0} |v|_1 + C \sum_i \xi_i
]

[
v^\top(\phi_l(s_i^+) - \phi_l(s_i^-)) \ge 1 - \xi_i.
]

Report:

[
\text{constraint satisfaction} =
\frac{1}{n}\sum_i \mathbf{1}\left[v^\top(\phi_l(s_i^+) - \phi_l(s_i^-)) > 0\right].
]

Also report AUC, top prototypes, bottom prototypes, and causal-patching effect.

This becomes the first reproducible baseline.

### Step 3: Dynamic rollout-pair builder

For each selected root (s_i), get:

[
\tau_i^+
]

from the most-visited or best-evaluated principal variation, and subpar rollouts:

[
\tau_{i,1}^-, \ldots, \tau_{i,M}^-.
]

Define:

[
\psi_l(\tau) = \frac{1}{|T|}\sum_{t \in T}\operatorname{vec}(Z_l(s_t)).
]

Use:

[
T_\text{both} = {0,1,2,\ldots,D}
]

and

[
T_\text{single} = {0,2,4,\ldots,D}
]

or the analogous odd-index set for opponent-perspective concepts.

Then build pair differences:

[
\delta_{i,m} = \psi_l(\tau_i^+) - \psi_l(\tau_{i,m}^-).
]

### Step 4: Candidate concept generation

Schut et al. need a set (P_c) of positions/rollouts for a concept. In an open LC0 baseline, we need a reproducible way to form these sets without human labels.

Use clustering over rollout-difference vectors:

[
\delta_{i,m}.
]

For each cluster (C_j), solve:

[
\min_{v_j,\xi \ge 0} |v_j|*1 + C\sum*{(i,m)\in C_j}\xi_{i,m}
]

[
v_j^\top \delta_{i,m} \ge 1 - \xi_{i,m}.
]

Also solve the reversed version for prophylactic concepts:

[
-v_j^\top \delta_{i,m} \ge 1 - \xi_{i,m}.
]

Use multiple random seeds, cluster counts, and node budgets. Keep concepts that are stable across bootstrap resamples:

[
\cos(v_j^{(a)}, v_j^{(b)}) \ge 0.7
]

after sign alignment.

### Step 5: Feature scaling with mathematical care

An (L_1) objective is not scale-invariant. If one activation dimension has a larger variance, the solver may prefer or avoid it for purely numerical reasons.

Use background standardization:

[
\tilde{x}_k = \frac{x_k - \mu_k}{\sigma_k + \epsilon}.
]

Solve in standardized coordinates:

[
\tilde{v}.
]

Map back to raw activation space:

[
v_k = \frac{\tilde{v}_k}{\sigma_k + \epsilon}.
]

Store both:

```text
direction_standardized
direction_raw
feature_mean
feature_std
layer
activation_mode
```

This is necessary for reproducibility and for honest interpretation of sparsity.

### Step 6: Novelty filtering

Build two activation matrices:

[
X_M \in \mathbb{R}^{n_M \times d}
]

from LC0 self-play or LC0-generated positions, and

[
X_H \in \mathbb{R}^{n_H \times d}
]

from high-level human games.

Compute right singular vector bases:

[
X_M = U_M\Sigma_M V_M^\top,
]

[
X_H = U_H\Sigma_H V_H^\top.
]

For a concept (v), define:

[
L_M(k) = \frac{|v - V_{M,k}V_{M,k}^\top v|_2^2}{|v|_2^2},
]

[
L_H(k) = \frac{|v - V_{H,k}V_{H,k}^\top v|_2^2}{|v|_2^2}.
]

Accept if:

[
L_M(k) < L_H(k)
]

for most or all (k) in a grid such as:

[
k \in {32, 64, 128, 256, 512, 1024}.
]

Report:

[
\nu(k)=L_H(k)-L_M(k).
]

This becomes the “machine-novelty” curve.

### Step 7: Prototype selection

For static concepts:

[
\operatorname{score}_c(s)=v_c^\top\phi_l(s).
]

For dynamic concepts:

[
\operatorname{score}_c(s)=
v_c^\top\psi_l(\tau^+(s))
-------------------------

\frac{1}{M}\sum_m v_c^\top\psi_l(\tau_m^-(s)).
]

Select top prototypes subject to diversity constraints:

* no same game within top (K),
* no near-duplicate FEN,
* stratify by opening family,
* require evaluation stability,
* require nontrivial continuation.

Each prototype card should include:

```text
FEN
board diagram
side to move
LC0 best move
human/common move if available
PV / subpar PV
concept score
novelty score
tag enrichments
causal patch effect
student teachability lift
```

### Step 8: Causal validation

The repo’s causal validation already measures value/logit shifts under patching, but it should become more concept-specific. Current patching adds a vector to a named layer and reports value/top-logit changes; useful, but not enough to validate that the concept changes the model toward the discovered plan. ([GitHub][12])

For each concept (v), patch:

[
Z_l'(s) = Z_l(s) + \beta \operatorname{reshape}(v).
]

Measure:

[
\Delta \log \pi(a^+) =
\log \pi'(a^+ \mid s) - \log \pi(a^+ \mid s),
]

[
\Delta \log \pi(a^-) =
\log \pi'(a^- \mid s) - \log \pi(a^- \mid s),
]

[
\Delta_\text{margin}
====================

## [\log \pi'(a^+) - \log \pi'(a^-)]

[\log \pi(a^+) - \log \pi(a^-)].
]

Accept a concept only if:

[
\mathbb{E}[\Delta_\text{margin}] > 0
]

with bootstrap confidence intervals, and random sparse vectors do not produce comparable effects.

### Step 9: Teachability validation

Use a weaker LC0 checkpoint or a small student model.

Teacher:

[
\pi_T(a\mid s)
]

Student:

[
\pi_S(a\mid s).
]

Train on concept prototypes using:

[
\mathcal{L}_\text{KL}
=====================

\frac{1}{n}
\sum_i
\operatorname{KL}\left(
\pi_T(\cdot\mid s_i)
;|;
\pi_S(\cdot\mid s_i)
\right).
]

Evaluate top-1 overlap:

[
\operatorname{Overlap}
======================

\frac{1}{n}
\sum_i
\mathbf{1}
\left[
\arg\max_a \pi_T(a\mid s_i)
===========================

\arg\max_a \pi_S(a\mid s_i)
\right].
]

Then compute lift over random-prototype training:

[
\operatorname{TeachLift}
========================

## \operatorname{Overlap}_\text{concept}

\operatorname{Overlap}_\text{random}.
]

Schut et al. use this kind of prototype curriculum idea to decide whether concepts are useful/teachable, not merely separable in activation space. ([ar5iv][2])

### Step 10: Human-label interpretation, not discovery

Use Lichess puzzle themes only after unsupervised discovery.

For a concept score vector (a \in \mathbb{R}^N) and tag matrix (Y \in {0,1}^{N\times T}), compute standardized effect size:

[
\Delta_t =
\frac{
\mathbb{E}[a_i \mid Y_{it}=1]
-----------------------------

\mathbb{E}[a_i \mid Y_{it}=0]
}{
s_\text{pooled}
}.
]

Also fit:

[
Y_{it} \sim \alpha_t + \beta_t a_i

* \gamma_1 \text{phase}_i
* \gamma_2 \text{eval}_i
* \gamma_3 \text{ply}_i
* \gamma_4 \text{material}_i.
  ]

Use permutation tests grouped by game or puzzle source and apply FDR correction. The current `match_concepts.py` averages raw scores per tag, which is good for a demo but too weak for a baseline. ([GitHub][5])

---

## Proposed repo architecture

I would restructure the interpretability layer like this:

```text
lc0jax/
  interpretability/
    activation_store.py       # unpooled activation records, metadata, loaders
    mcts_rollouts.py          # LC0 UCI/MCTS rollout extraction
    pair_builders.py          # static and dynamic constraint datasets
    sparse_concepts.py        # CVXPY sparse concept solvers
    scoring.py                # static/dynamic concept scores
    novelty.py                # SVD basis reconstruction novelty
    patching.py               # activation patching with shape-safe vectors
    teachability.py           # student distillation experiments
    tag_matching.py           # puzzle-tag enrichment and stats
    reports.py                # concept cards and dashboards
```

CLI tools:

```text
tools/
  build_activation_store.py
  build_mcts_pairs.py
  solve_static_concepts.py
  solve_dynamic_concepts.py
  filter_novel_concepts.py
  build_prototypes.py
  causal_validate_concepts.py
  teachability_eval.py
  match_concepts_to_tags.py
  build_concept_cards.py
```

The current `discover_concepts.py` can remain as a quick demo, but the research baseline should move to explicit pair builders and solver objects.

---

## “Learn Interpretability with Chess” notebook series

The notebooks should be a curriculum, not just experiment logs.

| Notebook | Title                               | Core idea                                                | Output                           |
| -------- | ----------------------------------- | -------------------------------------------------------- | -------------------------------- |
| 00       | Setup: LC0, JAX, and chess boards   | Environment, model loading, FEN display                  | First LC0 policy/value inference |
| 01       | LC0 input planes                    | Board encoding, side-to-move orientation, history planes | Visualized 112-plane input       |
| 02       | Policy and value heads              | Legal move masking, WDL, move logits                     | Top moves and value estimates    |
| 03       | Activations as chess geometry       | Layer activations, token/square structure                | PCA/UMAP by phase/material       |
| 04       | Linear probes for easy concepts     | Piece probes, material, open files                       | First concept vectors            |
| 05       | Sparse concept vectors              | (L_1) optimization, margins, constraint satisfaction     | CVXPY static concept solver      |
| 06       | Prototypes and counterexamples      | Top/bottom scoring positions                             | Concept prototype viewer         |
| 07       | Causal patching                     | Add concept vectors to activations                       | Policy/value shift plots         |
| 08       | MCTS as a source of explanations    | PVs, visits, Q/WDL, subpar lines                         | Search-tree visualizer           |
| 09       | Dynamic concepts                    | Optimal-vs-subpar rollout constraints                    | First Schut-style LC0 concepts   |
| 10       | Novelty: machine vs human subspaces | SVD bases and reconstruction losses                      | Novelty curves                   |
| 11       | Puzzle tags as interpretation       | Lichess themes, tag enrichment, confounds                | Tag-concept report               |
| 12       | Teachability                        | Student network, KL distillation, top-1 overlap          | Teachability lift                |
| 13       | Concept cards                       | Combining prototypes, MCTS, novelty, tags, patching      | Publishable concept cards        |
| 14       | Capstone                            | Discover and audit one new concept end-to-end            | Reproducible concept report      |

This series should have two tracks:

* **Learning track:** small data, CPU/GPU-light, puzzle tags, simple probes.
* **Research track:** LC0 MCTS rollouts, self-play chunks, novelty, teachability, reproducibility.

---

## Priority plan

### P0: Make the baseline mathematically faithful

Implement these first:

1. **Unpooled activation storage**
   Preserve (64 \times d) activations and history-aware inputs.

2. **MCTS rollout extraction**
   Store optimal and subpar trajectories with visits, Q/WDL, priors, and PV states.

3. **Dynamic sparse solver**
   Implement the rollout-pair objective:

   [
   \min |v|_1 + C\sum\xi
   ]

   [
   v^\top(\psi(\tau^+) - \psi(\tau^-)) \ge 1-\xi.
   ]

4. **Constraint-satisfaction validation**
   Train/test split by root position or game, not by sampled pair.

5. **Novelty metric**
   Add SVD reconstruction comparison against human activations.

6. **Concept reports**
   Every concept should produce prototypes, MCTS lines, novelty curves, tag enrichments, and causal effects.

### P1: Make it robust

Add:

* feature standardization,
* sparse-vector stability under bootstrap,
* random-vector and shuffled-pair baselines,
* node-budget stability,
* tablebase/trivial-tactic exclusion,
* stratified sampling by phase/opening/eval/material,
* multiple LC0 checkpoints for final-vs-weaker disagreement.

### P2: Make it educational

Add:

* notebooks,
* board visualizations,
* sparse-vector heatmaps,
* move-logit patching plots,
* “concept card” HTML pages,
* exercises with expected outputs.

---

## Acceptance criteria for a serious baseline

A discovered concept should be accepted only if it passes:

[
\text{HeldoutConstraintSatisfaction} > \text{random baseline} + \delta
]

[
\text{NoveltyArea} = \frac{1}{|K|}\sum_{k\in K} [L_H(k)-L_M(k)] > 0
]

[
\mathbb{E}[\Delta_\text{policy margin under patch}] > 0
]

[
\text{TeachLift} > 0
]

and its top prototypes are nontrivial, diverse, and stable under higher-node LC0 analysis.

---

## Bottom line

The repo’s current value is that it already has much of the LC0/JAX plumbing: model loading, activation dumping, concept-vector experiments, filtering, puzzle-tag extraction, and causal patching hooks. The main upgrade is to stop treating concept discovery as “separate two embedding datasets” and instead implement the Schut et al. object directly:

[
\textbf{sparse concept vector explaining optimal MCTS rollouts over subpar alternatives.}
]

Once that is in place, the project can become both a credible open chess concept-discovery baseline and a very strong interpretability education series.

[1]: https://github.com/LiquidGunay/lc0_bridging_the_gap/tree/main "GitHub - LiquidGunay/lc0_bridging_the_gap: A reproduction of Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero but using LC0 · GitHub"
[2]: https://ar5iv.org/pdf/2310.16410 "[2310.16410] Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero"
[3]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/lc0jax/interpretability/activations.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/lc0jax/interpretability/concepts.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/tools/match_concepts.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/tools/filter_fens_eval.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/docs/schut_audit_report.md "raw.githubusercontent.com"
[8]: https://database.lichess.org/ "lichess.org open database"
[9]: https://lichess.org/training/themes "Puzzle Themes • lichess.org"
[10]: https://database.lichess.org/standard/list.txt "database.lichess.org"
[11]: https://github.com/LeelaChessZero/lczero-training "GitHub - LeelaChessZero/lczero-training: For code etc relating to the network training process. · GitHub"
[12]: https://raw.githubusercontent.com/LiquidGunay/lc0_bridging_the_gap/main/tools/causal_validate.py "raw.githubusercontent.com"
