# Concept Discovery Audit against Schut et al. (arXiv:2310.16410)

## Objective
Audit the current implementation in `lc0jax/interpretability/concepts.py` and `tools/discover_concepts.py` against the mathematical formulations detailed in the Schut et al. paper, "Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero".

## Current State
The existing implementation in `lc0jax/interpretability/concepts.py` offers the following methods for extracting concept directions between two sets of activation embeddings ($A$ and $B$):
- **`mean_diff`**: Subtracts the mean vectors of the two sets: $v = \mu_A - \mu_B$.
- **`whitened_mean_diff`**: Scales the mean difference by the inverse of the pooled covariance matrix: $v = \Sigma^{-1} (\mu_A - \mu_B)$. This is mathematically equivalent to Linear Discriminant Analysis (LDA).
- **`cov_shift`**: Uses PCA (eigendecomposition) on the difference between the covariance matrices of $A$ and $B$.
- **`cluster_diff`**: Uses k-means clustering to find representative concepts.

## Discrepancy Analysis
A detailed review of the `final_draft.tex` source of arXiv:2310.16410 reveals that the paper **does not use whitening, PCA, or mean difference directly for its concept vectors.** Instead, it treats concept discovery strictly as a **Convex Optimization** problem to enforce sparsity and a specific constraint.

According to Equation 1 (for static concepts) and Equation 2/3 (for dynamic concepts), the paper frames the concept extraction as:
$$ \min \| v_{c,l} \|_1 $$
$$ \text{such that} \quad v_{c,l} \cdot z^+_{l} \ge v_{c,l} \cdot z^-_{l} \quad \forall z_l^+ \in Z^+, z_l^- \in Z^- $$
*(Note: To prevent the trivial solution $v_{c,l} = 0$, a positive margin constraint must typically be applied in practice, e.g., $\ge 1$)*

The paper explicitly states:
> "We solve the convex optimisation problem using a standard solver in the package \texttt{cvxpy} \citep{cvxpy1, cvxpy2}."

### Conclusion
Our current implementation deviates significantly from the exact paper methodology. While methods like `whitened_mean_diff` (LDA) and `mean_diff` are fast and widely used in mechanistic interpretability, they are L2-based and do not naturally yield the highly sparse, L1-penalized concept vectors the paper claims to discover.

Update 2026-04-27: `svm_cvxpy` is now present, but this only covers a static paired-difference baseline. The deeper gap is dynamic concept discovery from LC0 MCTS optimal rollouts contrasted with subpar rollouts, followed by novelty and teachability filters. See `REPO_AUDIT_AND_NEXT_STEPS.md` and `IMPLEMENTATION_STATUS_AND_NEXT_WORK.md` for the current status.

## Required Actions
1. **Add `svm_cvxpy` (or L1 Linear SVM) method** to `lc0jax/interpretability/concepts.py`.
   - The method should use `cvxpy` or an L1-regularized `LinearSVC` from `scikit-learn` to solve the exact hard-margin (or soft-margin) constraint formulated in the paper.
   - For performance, random pairing or sub-sampling of $Z^+$ and $Z^-$ points should be used to avoid $O(|Z^+| \times |Z^-|)$ constraints, as done in the paper.
2. Update the default method in `tools/discover_concepts.py` to allow execution of this new CVXPY-based approach, ensuring alignment with the reproduction goals.
