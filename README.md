# BAMD-v3

BAMD-v3 is a teacher-guided data selection pipeline for imbalanced tabular classification.

The method combines:

- an MLP teacher,
- permutation feature importance,
- class-specific weighted representations,
- class-wise KMeans candidate generation,
- boundary-aware and diversity-aware final sample selection.

The final selected subset contains:

- 389 samples from Class 0
- 244 samples from Class 1
- 633 samples in total

---

## 1. Pipeline

```text
Raw train / validation data
        ↓
Step 1 - Train MLP Teacher
        ↓
Teacher checkpoint + probabilities
        ↓
Step 2 - Permutation Feature Importance
        ↓
Global and class-specific feature importance
        ↓
Step 3 - Build Candidate Pool
        ↓
Class-specific weighted representation
        ↓
Class-wise MiniBatchKMeans
        ↓
2,532 candidate samples
        ↓
BAMD-v3 Selection
        ↓
R + B + D_mix
        ↓
633 selected samples
        ↓
Downstream MLP Evaluation
```

---

## 2. Repository Structure

```text
BAMD-v3/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── configs/
│
├── scripts/
│   └── run_bamd_v3.sh
│
├── src/
│   ├── step1_train_teacher.py
│   ├── step2_permutation_importance.py
│   ├── step3_build_candidate_pool.py
│   ├── bamd_v3.py
│   └── evaluate_subset.py
│
└── outputs/
```

---

## 3. Installation

Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
scikit-learn
torch
joblib
```

---

## 4. Dataset

Place the dataset files inside:

```text
data/
├── train.csv
├── val.csv
└── test.csv
```

The current implementation expects the label column:

```text
label
```

Categorical features:

```text
node_id
parent_id
rpl_ver
```

The remaining input features are treated as numerical features.

---

# 5. Step 1 - Train Teacher

The teacher is an MLP designed for mixed numerical and categorical tabular data.

Categorical features are converted to integer IDs and passed through learnable embedding layers.

For categorical feature $j$, the embedding dimension is:

$
d_j =
\min
\left(
16,
\max
\left(
4,
\left\lfloor
\frac{C_j}{2}
\right\rfloor
\right)
\right)
$

where $C_j$ is the categorical cardinality.

The teacher backbone is:

```text
Numerical features
        +
Categorical embeddings
        ↓
Linear → 256
        ↓
ReLU
        ↓
Dropout
        ↓
Linear → 128
        ↓
ReLU
        ↓
Dropout
        ↓
Linear → 64
        ↓
ReLU
        ↓
Classifier → 2 classes
```

Run:

```bash
python src/step1_train_teacher.py \
    --train data/train.csv \
    --val data/val.csv \
    --output-dir outputs/teacher
```

Main outputs:

```text
outputs/teacher/
├── teacher_best.pt
├── preprocessor.json
├── scaler.pkl
├── train_hidden.npy
├── train_logits.npy
├── train_probs.npy
├── train_loss.npy
├── train_labels.npy
└── train_indices.npy
```

The teacher checkpoint is selected using validation loss.

---

# 6. Step 2 - Permutation Feature Importance

Permutation Feature Importance measures how much the teacher performance degrades when one feature is randomly permuted.

For feature $j$:

$
I_j =
L_{\text{permuted},j}
-
L_{\text{baseline}}
$

A larger value indicates that the teacher relies more strongly on that feature.

The implementation computes:

- global importance,
- Class 0 importance,
- Class 1 importance.

Run:

```bash
python src/step2_permutation_importance.py \
    --train data/train.csv \
    --teacher-dir outputs/teacher \
    --result-dir outputs/pfi
```

Main output:

```text
outputs/pfi/feature_importance.csv
```

The file contains global and class-specific feature importance values and normalized importance weights.

---

# 7. Step 3 - Candidate Pool Construction

Step 3 reduces the full training set to a smaller candidate pool before final BAMD-v3 selection.

## 7.1 Teacher input representation

For each sample:

$
Z =
[
X_{\text{num}},
E_{\text{node}},
E_{\text{parent}},
E_{\text{rpl}}
]
$

where:

- $X_{\text{num}}$ contains standardized numerical features,
- $E$ denotes learned categorical embeddings from the teacher.

Categorical embedding blocks are normalized before further processing.

---

## 7.2 Class-specific feature weighting

Feature importance is used to construct separate weights for Class 0 and Class 1.

For Class 0:

$
W_0 =
\alpha_g I_{\text{global}}
+
\alpha_c I_{C0}
$

For Class 1:

$
W_1 =
\alpha_g I_{\text{global}}
+
\alpha_c I_{C1}
$

Default values:

$
\alpha_g = 0.7
$

$
\alpha_c = 0.3
$

Each feature block is scaled by:

$
\sqrt{w_j}
$

This produces:

```text
weighted_representation_class0.npy
weighted_representation_class1.npy
```

The square-root scaling ensures that squared Euclidean distance becomes a feature-weighted distance:

$
d^2(x,y)
=
\sum_j
w_j
\|x_j-y_j\|^2
$

---

## 7.3 Candidate generation

Candidate selection is performed separately for Class 0 and Class 1 using MiniBatchKMeans.

The default candidate pool size is:

$
4 \times 633 = 2532
$

with approximately:

```text
Class 0: 1772
Class 1: 760
```

For each cluster, the samples nearest to the centroid are retained as candidate samples.

Run:

```bash
python src/step3_build_candidate_pool.py \
    --train data/train.csv \
    --teacher-dir outputs/teacher \
    --importance-csv outputs/pfi/feature_importance.csv \
    --result-dir outputs/candidates \
    --final-ratio 0.01 \
    --candidate-multiplier 4.0 \
    --candidate-class1-ratio 0.30 \
    --candidates-per-cluster 2 \
    --global-weight 0.7 \
    --class-weight 0.3
```

Main outputs:

```text
outputs/candidates/
├── teacher_input_representation.npy
├── weighted_representation_class0.npy
├── weighted_representation_class1.npy
├── class_specific_feature_weights.csv
├── candidate_pool.csv
├── candidate_indices.npy
└── step3_summary.json
```

---

# 8. BAMD-v3 Selection

BAMD-v3 performs the final selection from the candidate pool.

The final score is:

$
S =
0.55R
+
0.30B
+
0.15D_{\text{mix}}
$

where:

- $R$: representativeness,
- $B$: bounded boundary support,
- $D_{\text{mix}}$: mixed diversity.

---

## 8.1 Representativeness

For each class, KMeans is run with the number of clusters equal to the class budget.

```text
Class 0:
1772 candidates → 389 clusters

Class 1:
760 candidates → 244 clusters
```

Samples closer to their cluster centroid receive higher representativeness rank.

Only one sample is finally selected from each cluster.

---

## 8.2 Boundary Score

Teacher uncertainty is defined as:

$
U_i =
\min(p_{i0}, p_{i1})
$

For each candidate, the method computes:

- mean distance to the nearest same-class samples,
- mean distance to the nearest opposite-class samples.

Boundary support is:

$
L_i =
\frac{d_{\text{same}}}
{d_{\text{same}} + d_{\text{opp}} + \epsilon}
$

The final boundary value is:

$ 
B_i =
U_i L_i 
$

Default nearest-neighbor settings:

```text
k_same = 10
k_opp  = 10
```

---

## 8.3 Mixed Diversity

Mixed diversity combines latent-space and categorical diversity:

$
D_{\text{mix}}
=
\alpha D_{\text{latent}}
+
(1-\alpha)D_{\text{cat}}
$

Default:

$
\alpha = 0.8
$

Therefore:

$
D_{\text{mix}}
=
0.8D_{\text{latent}}
+
0.2D_{\text{cat}}
$

Latent diversity uses cosine distance.

Categorical diversity uses PFI-weighted categorical mismatch over:

```text
node_id
parent_id
rpl_ver
```

with separate categorical PFI weights for Class 0 and Class 1.

---

## 8.4 Greedy Selection

Selection is iterative.

At each iteration:

1. remove candidates belonging to already selected clusters,
2. compute representativeness,
3. compute boundary score,
4. compute diversity relative to the already selected set,
5. compute the final score,
6. select the highest-scoring candidate,
7. mark its cluster as filled,
8. update diversity distances.

This continues until all class-specific budgets are filled.

Final subset:

```text
Class 0: 389
Class 1: 244
Total:   633
```

---

# 9. Run BAMD-v3 and Downstream Evaluation

The repository includes:

```text
scripts/run_bamd_v3.sh
```

This script performs:

```text
BAMD-v3 selection
        ↓
Final subset validation
        ↓
Downstream MLP evaluation
```

Run:

```bash
chmod +x scripts/run_bamd_v3.sh
./scripts/run_bamd_v3.sh
```

The script expects the outputs from Steps 1-3 to already exist.

---

# 10. Downstream Evaluation

The selected subset is evaluated using an MLP downstream classifier.

Default evaluation seeds:

```text
0
1
2
3
4
```

Typical training settings:

```text
epochs        = 100
batch size    = 256
learning rate = 1e-3
weight decay  = 1e-4
patience      = 20
```

Evaluation output is stored under:

```text
outputs/bamd_v3/evaluation/
```

including:

```text
comparison.csv
```

---

# 11. BAMD-v3 Outputs

After running the final selection stage:

```text
outputs/bamd_v3/
├── model_subset_bamd_v3.csv
├── bamd_v3_manifest.csv
├── bamd_v3_c0_selected_389.csv
├── bamd_v3_c1_selected_244.csv
├── config.json
└── evaluation/
```

`model_subset_bamd_v3.csv` is the final condensed real-data subset.

---

# 12. Reproducibility

Default seeds:

```text
Teacher seed:      42
PFI seed:          42
Candidate seed:    42
BAMD seed:         42

Evaluation seeds:
0 1 2 3 4
```

Important BAMD-v3 parameters:

```text
Candidate pool:
    2532 samples

Final subset:
    Class 0 = 389
    Class 1 = 244
    Total   = 633

Score:
    Representativeness = 0.55
    Boundary           = 0.30
    Diversity          = 0.15

Mixed diversity:
    Latent      = 0.80
    Categorical = 0.20

Boundary:
    k_same = 10
    k_opp  = 10
```