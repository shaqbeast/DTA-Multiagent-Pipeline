# Robust Drug-Target Affinity for Precision Drug Discovery

**Subtitle:** Uncertainty-Aware, Budget-Constrained Compound Ranking for Wet-Lab Decision Making

---

## Overview

Drug-target affinity (DTA) models perform well on curated benchmarks but fail under real-world noise — assay variability, drug form differences, and novel scaffolds degrade predictions silently. Since screening budgets are finite, false positives waste expensive experiments. This project builds a multi-agent pipeline that quantifies prediction uncertainty, detects distribution shift, and ranks compounds by expected utility under a fixed wet-lab budget.

---

## Phase 1 — Data Ingestion & Harmonization

**Goal:** Merge heterogeneous DTA sources into a clean, unified dataset with scaffold-aware splits.

1. Download **DrugForm-DTA** from Zenodo and **Therapeutics Data Commons (TDC)** via its Python API (`pip install PyTDC`).
2. Parse both into a unified schema with the following fields:
   - Compound SMILES
   - Target protein sequence
   - Affinity label (pKd / pKi / IC50)
   - Assay type and conditions
   - Drug form metadata
3. Standardize affinity units — convert all values to a negative-log scale (e.g., pKd).
4. Deduplicate compound–target pairs and flag conflicting labels arising from different assay conditions.
5. Generate **scaffold-aware** train/validation/test splits using RDKit's `MurckoScaffold`, ensuring test-set scaffolds are completely unseen during training.

**Key libraries:** `PyTDC`, `RDKit`, `pandas`

---

## Phase 2 — Molecular Featurization

**Goal:** Convert raw SMILES and protein sequences into model-ready representations.

6. Parse and canonicalize SMILES strings with RDKit; discard invalid molecules.
7. Build molecular graphs:
   - **Nodes (atoms):** element type, degree, formal charge, aromaticity, hybridization
   - **Edges (bonds):** bond type, conjugation, ring membership
8. Compute **Morgan fingerprints** (radius 2, 2048-bit) as auxiliary features for baseline models and OOD detection.
9. Encode target protein sequences using one of:
   - Pretrained **ESM-2** embeddings (recommended for performance)
   - Learned 1D-CNN encoder (lighter weight)
10. Package all features into PyTorch Geometric `Data` objects and build `DataLoader` pipelines with scaffold-aware batch sampling.

**Key libraries:** `RDKit`, `torch_geometric`, `fair-esm`

---

## Phase 3 — Affinity Predictor with Uncertainty Estimation

**Goal:** Train a model that predicts binding affinity *and* quantifies how much it trusts each prediction.

11. Build a **GNN encoder** (GIN or AttentiveFP) for compound graph representations.
12. Add a **protein encoder** — either a 1D CNN or frozen ESM-2 embeddings with a projection head.
13. Concatenate compound + target representations and pass through a **multi-head output layer:**
    - **Head 1:** Affinity regression (predicted mean)
    - **Head 2:** Aleatoric uncertainty (predicted log-variance)
14. Train with **heteroscedastic Gaussian NLL loss:**
    ```
    L = 0.5 * [log(σ²) + (y - μ)² / σ²]
    ```
15. Wrap the model with **MC Dropout** — at inference, perform ~30 stochastic forward passes. The variance across passes estimates epistemic uncertainty.
16. Optionally train a **deep ensemble** of 3–5 independently initialized models for improved calibration and robustness.

**Key libraries:** `torch`, `torch_geometric`, `fair-esm`

---

## Phase 4 — Distribution Shift Detection & OOD Flagging

**Goal:** Identify test compounds that fall outside the training distribution so their predictions can be treated with appropriate caution.

17. On the training set, compute reference distributions:
    - GNN embedding centroids per scaffold cluster
    - Baseline prediction variance statistics
20. Log which test compounds are flagged and categorize the reason:
    - Novel scaffold (structural novelty)
    - Unusual assay condition (metadata shift)
    - Both

**Key libraries:** `scikit-learn`, `numpy`, `torch`

---

## Phase 5 — Budget-Constrained Ranking & Selection

**Goal:** Rank candidate compounds by expected utility and select the best set to test within a fixed experimental budget.

21. Define the **utility function:**
    ```
    U(x) = predicted_affinity(x) − λ · total_uncertainty(x)
    ```
    where λ is a tunable risk-aversion parameter.
22. Sort all candidates by U; select the **top-N** that fit within the wet-lab budget.
23. Implement an **abstention policy** — exclude any compound whose total uncertainty exceeds a threshold, regardless of predicted affinity. This avoids wasting budget on high-risk predictions.
24. Evaluate using **Hit Rate@Budget**: the fraction of selected compounds that are true actives in the ground-truth data.

---

## Phase 6 — Evaluation & Transparent Reporting

**Goal:** Rigorously measure pipeline performance under realistic conditions and report degradation honestly.

25. Compute **Kendall's τ** between predicted and true rankings across three regimes:
    - IID split (baseline)
    - Scaffold split (structural generalization)
    - Assay-shift split (condition generalization)
26. Plot **calibration curves** (predicted confidence vs. observed hit rate) and compute **Expected Calibration Error (ECE)**.
27. Report **performance degradation transparently** — quantify exactly how much ranking quality drops from IID → scaffold split → assay shift.
28. Run **ablation studies:**
    - Uncertainty-aware ranking vs. naive "sort by predicted affinity"
    - MC Dropout vs. deep ensemble vs. combined
    - Effect of λ on hit rate vs. coverage tradeoff

---

## Specialized Agents Summary

| Agent | Responsibility |
|---|---|
| **Molecule Agent** | SMILES parsing, graph encoding, fingerprint generation |
| **Uncertainty Agent** | MC Dropout inference, calibration scoring, OOD detection |
| **Selection Agent** | Budget-aware ranking, abstention policy, utility optimization |

---

## Environment & Dependencies

```
python >= 3.10
torch >= 2.0
torch_geometric
rdkit
PyTDC
fair-esm
scikit-learn
pandas
numpy
matplotlib
```

---

## Success Criteria

- **Ranking Stability:** Kendall's τ ≥ 0.6 under scaffold shift
- **Hit Rate@Budget:** ≥ 30% improvement over naive affinity ranking
- **Calibration:** ECE < 0.10
- **Abstention Quality:** ≥ 80% precision on "don't test" decisions