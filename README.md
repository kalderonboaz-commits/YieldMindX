# Yield Correlation / Yield Intelligence Project

## Overview

This project is intended to build an explainable Machine Learning system for semiconductor wafer Yield Intelligence.

The main objective is not only to predict `SortingYield`, but also to discover and explain which process, electrical, metrology, and derived parameters are associated with high or low yield.

The system is designed to identify:

- Direct relationships between parameters and yield
- Nonlinear relationships
- Threshold effects
- Process-window behavior
- Technology-specific behavior
- Two-way and higher-order interactions
- Derived changes between process stages
- Engineering-oriented root-cause hypotheses

The current dataset is synthetic and was created with intentionally planted relationships. This makes it possible to validate whether the analysis and Machine Learning pipeline is able to rediscover known patterns without being explicitly told the answers.

---

## Project Root

Current local project path:

```text
D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table
```

Expected structure:

```text
correlation_data_table/
│
├── data/
│   └── correlation_data_table.csv
│
├── output/
│   ├── data_audit_columns.csv
│   └── eda/
│
├── correlation_data_table.docx
├── data_audit.py
├── eda.py
└── README.md
```

---

## Dataset

The current synthetic dataset contains approximately:

```text
Rows:    1500 wafers
Columns: 1430
Target:  SortingYield
```

The dataset contains a large number of process, electrical, metrology, metadata, identification, and yield-related columns.

Because the number of columns is very large relative to the number of wafers, predictors must not automatically be passed into a Machine Learning model without screening.

---

## Target Variable

The main prediction and explanation target is:

```text
SortingYield
```

`SortingYield` is the dependent variable, or `Y`.

The remaining eligible engineering parameters are candidate predictors, or `X` variables.

The goal is to understand which predictors, ranges, deviations, combinations, and process-stage changes are associated with variation in `SortingYield`.

---

## Important Methodological Principle

This project is not a simple correlation-ranking project.

A predictor may have a weak Pearson correlation with `SortingYield` and still be highly important.

For example, a process-window relationship can look like:

```text
Low Yield  ←  Optimal Region  →  Low Yield
```

In this case, a linear correlation can be close to zero even though the parameter strongly affects yield.

Therefore, the project must support more than simple Pearson correlation.

Required discovery types include:

- Linear relationships
- Monotonic nonlinear relationships
- U-shaped or process-window relationships
- Threshold effects
- Absolute-deviation effects
- Technology-specific behavior
- Interactions
- Derived engineering features

---

## Ground-Truth / Reference Document

The project contains:

```text
correlation_data_table.docx
```

This document is the primary specification and ground-truth reference for the synthetic dataset.

It describes the intentionally planted relationships and the expected methodological behavior of the analysis pipeline.

The document should be treated as a validation reference, not as information that should be directly hard-coded into the Machine Learning model.

The model should eventually be evaluated on whether it can rediscover the planted relationships from data.

---

## Known Planted Relationships

### 1. Gate CD Process Window

Parameter:

```text
GATE2_CD_Gate_Leg_Length
```

The relationship with yield is nonlinear.

Each technology has a different nominal Gate CD target:

```text
PHEMT_0.15um → approximately 0.15
PHEMT_0.25um → approximately 0.25
PHEMT_0.35um → approximately 0.35
```

Yield is expected to be better near the technology-specific target and lower as Gate CD deviates from that target.

This means Gate CD must not be interpreted only through global linear correlation.

### 2. PAE

Parameter:

```text
TOPSIN_ET_Load_Pull_P.A.E_Last
```

PAE stands for Power Added Efficiency.

The synthetic data contains a positive association:

```text
Higher PAE → Higher SortingYield
```

### 3. Drain Lag

Parameter:

```text
TOPSIN_ET_DC_PIV_Drain_Lag_1
```

Drain Lag contains a threshold-like negative relationship. At lower values, the effect on yield is relatively small. At higher values, yield begins to decrease more strongly.

The exploratory analysis showed a visible change around the neighborhood of approximately 7, but exact thresholds must be learned and validated statistically rather than hard-coded from visual inspection.

### 4. Gate CD × SiN Thickness Interaction

Parameters:

```text
GATE2_CD_Gate_Leg_Length
SIN_Thickness_SiN_Thickness
```

The relevant engineering representation is based on deviation from nominal conditions:

```text
Gate_CD_Deviation = |Gate CD - technology-specific Gate target|
SiN_Deviation     = |SiN Thickness - 0.150|
```

The effect should be evaluated separately by technology because the Gate CD target changes between technologies.

### 5. Rc × Rsh Interaction

Parameters:

```text
OHMIC_ET_Rc_an_ct2
OHMIC_ET_Rsh_an_ct2
```

The important relationship is not necessarily the effect of each variable in isolation. The synthetic data contains an interaction where unfavorable values of both parameters together are associated with lower yield.

### 6. Alignment

Parameter:

```text
GATE2_Ali_Litho_Gate_to_recess_alignment
```

The relevant effect is based on absolute deviation from zero, rather than the sign.

Derived feature:

```text
Abs_Alignment = |Alignment|
```

A large alignment deviation is expected to be associated with reduced yield.

### 7. Delta Gm

Parameters:

```text
FIC_ET_Gm_max
TOPSIN_ET_Gm_max
```

The important feature is the change between stages:

```text
Delta_Gm = TOPSIN_ET_Gm_max - FIC_ET_Gm_max
```

A large negative Delta Gm indicates degradation between stages and is associated with lower yield in the synthetic data.

---

## Feature Eligibility and Target Leakage

A major principle of the project is that not every column in the raw table is automatically a valid predictor.

Before full-model training, predictors must be screened for methodological eligibility.

### Target Leakage Columns

The following yield/output columns must not be used as predictors for `SortingYield`:

```text
Actual Sorting Yield (%)
LineYield
BackEndYield
VIYield
LTYield
TotalYield
```

These variables are closely related to the target and can provide the model with direct or indirect information about the answer.

Including them can create artificially strong metrics while producing a model that does not truly learn process relationships.

### Constant / Near-Constant Columns

The data audit identified examples such as:

```text
Process
FinishLineYear
Exception
```

Columns with no meaningful variation provide little or no predictive information and should not be treated as ordinary model predictors.

### IDs and Administrative Metadata

Examples may include:

```text
LotName
WaferNum
Sorting Date
FinishLineDate
Version
Year_WW
```

Such columns can cause the model to memorize lots, wafers, dates, or operational periods instead of learning engineering relationships.

They should be excluded or handled deliberately unless there is a justified modeling reason to keep them.

### Downstream Information

Any variable measured after the prediction point, or any variable that would not be available when the model is expected to make a real prediction, must be reviewed for leakage.

Leakage is not determined only by column name. Process timing and data availability matter.

### Recommended Feature-Screening Output

Before full Machine Learning discovery, the project should eventually generate:

```text
included_features.csv
excluded_features.csv
```

Each excluded feature should have a reason, for example:

```text
Actual Sorting Yield (%) → target_leakage
Process                  → constant
WaferNum                 → identifier
Sorting Date             → metadata
```

The objective is not to aggressively remove engineering predictors. The objective is to prevent leakage, memorization, and non-informative inputs while allowing the model to discover which legitimate engineering variables matter.

---

## Work Completed So Far

### 1. Data Audit

Script:

```text
data_audit.py
```

The data audit checks:

- Dataset dimensions
- Target existence
- Target statistics
- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- Numeric and non-numeric column counts
- Target-leakage columns
- Ground-truth parameter availability
- ID / metadata candidates

Observed dataset state:

```text
Rows: 1500
Columns: 1430
Missing SortingYield values: 0
Duplicate rows: 0
```

The audit output includes:

```text
output/data_audit_columns.csv
```

### 2. Exploratory Data Analysis

Script:

```text
eda.py
```

The EDA currently creates:

- `SortingYield` histogram
- `SortingYield` boxplot
- Feature distributions
- Feature-vs-yield scatter plots
- Binned relationship plots
- Pearson correlation summary
- Spearman correlation summary
- Delta Gm analysis
- Absolute alignment analysis
- Technology-specific Gate CD analysis
- Rc × Rsh interaction heatmap
- Technology-specific Gate CD deviation × SiN deviation heatmaps
- Heatmap cell-count CSV files
- Technology yield summary

Outputs are stored under:

```text
output/eda/
```

---

## Why Binned Plots Are Used

Raw scatter plots can be difficult to interpret because the dataset contains many wafers and multiple overlapping effects.

Binned plots divide a predictor into groups and calculate the mean `SortingYield` for each group.

This helps reveal thresholds, process windows, curvature, nonlinear trends, and local changes in behavior.

Binned plots are exploratory tools and should not be treated as proof of causality.

---

## Why Heatmaps Are Used

Heatmaps are used to investigate interactions.

Each heatmap cell represents a group of wafers that fall within one range of predictor X and one range of predictor Y. The color represents the average `SortingYield` for that group.

A separate counts table is generated to show how many wafers contribute to each heatmap cell. This prevents over-interpreting an extreme-looking cell that is based on only a few wafers.

---

## Main EDA Findings So Far

### PAE

A positive relationship was observed:

```text
Higher PAE → Higher SortingYield
```

### Drain Lag

A negative threshold-like relationship was observed. Yield is relatively stable at lower values and decreases more strongly at elevated Drain Lag values.

### Gate CD

Global Pearson correlation was close to zero. However, technology-specific analysis revealed meaningful process-window behavior.

This demonstrates why simple linear correlation is insufficient for this project.

### Delta Gm

The derived `Delta_Gm` feature showed a meaningful relationship with yield. Large negative changes between FIC and TOPSIN Gm were associated with lower yield.

### Absolute Alignment

The raw signed alignment showed little linear correlation. Using absolute deviation produced a more meaningful signal, although the relationship remains noisier than PAE or Drain Lag.

### Rc × Rsh

The interaction heatmap showed evidence that unfavorable values of Rc and Rsh together can be associated with lower yield.

### Gate CD × SiN

The first global heatmap was methodologically problematic because it mixed technologies with different Gate CD targets.

The analysis was corrected by:

1. Computing Gate CD deviation from the technology-specific target
2. Computing SiN deviation from 0.150
3. Building one interaction heatmap per technology

The corrected analysis showed clearer interaction structure, especially in the 0.15 µm and 0.25 µm technologies.

---

## Important Statistical Interpretation

This project distinguishes between:

```text
Correlation
Association
Prediction
Interaction
Causality
```

A strong statistical relationship does not automatically prove physical causality.

The Machine Learning system should generate evidence and engineering hypotheses. Final causal conclusions require engineering validation, domain expertise, controlled experiments, or additional process evidence.

---

## Current Project Status

```text
Data Audit                     DONE
Basic EDA                      DONE
Pearson / Spearman Analysis    DONE
Scatter Plots                  DONE
Binned Relationships           DONE
Technology Analysis            DONE
Derived Features               DONE
Interaction Heatmaps           DONE
Heatmap Count Validation       DONE
Machine Learning Model         NOT YET IMPLEMENTED
Full Feature Screening         NOT YET IMPLEMENTED
```

---

## Recommended Next Step

The next phase should begin with reproducible feature eligibility screening.

The approximately 1,430 raw columns should be classified into:

```text
Eligible engineering predictors
Excluded target-leakage columns
Excluded identifiers
Excluded administrative metadata
Excluded constant / near-constant columns
Excluded downstream information
Categorical variables requiring deliberate handling
```

After that, the first interpretable Machine Learning model can be trained.

A strong candidate is an Explainable Boosting Machine (EBM), because the project requires both prediction and interpretable nonlinear response shapes.

The model should initially be tested in a controlled setting before being expanded to the full screened candidate feature set.

---

## Planned Machine Learning Roadmap

```text
Raw Dataset
    ↓
Data Audit
    ↓
EDA
    ↓
Feature Eligibility Screening
    ↓
Derived Feature Generation
    ↓
Controlled Interpretable Model
    ↓
Evaluation on Unseen Data
    ↓
Interaction Discovery
    ↓
XGBoost / Strong Nonlinear Model
    ↓
SHAP / Explainability
    ↓
Full Screened Feature Discovery
    ↓
Engineering-Oriented Yield Intelligence
```

---

## Model Evaluation

Models should be evaluated on unseen data rather than only on training performance.

Potential metrics include:

```text
MAE
RMSE
R²
```

Model quality should not be judged only by predictive accuracy.

The model should also be evaluated for:

- Stability
- Generalization
- Leakage risk
- Interpretability
- Engineering usefulness
- Ability to rediscover known synthetic ground-truth relationships

---

## Final System Goal

The final system should be able to produce engineering-oriented insights such as:

```text
Drain Lag shows a threshold-like relationship with SortingYield.

PAE is positively associated with SortingYield.

Gate CD has a technology-dependent process window.

Large Gate CD and SiN deviations together are associated with additional yield loss.

High Rc combined with high Rsh is associated with reduced yield.

Large negative Delta Gm indicates degradation between process stages and is associated with lower yield.
```

The goal is not merely to rank features.

The goal is to create a reusable Yield Intelligence pipeline that can help engineers understand:

```text
Which parameter?
Which range?
Which direction?
Which threshold?
Which combination?
Which process stage?
What is associated with yield loss?
```

while avoiding leakage and preserving explainability.

---

## Current Principle for Predictor Selection

The project does **not** aim to remove legitimate engineering predictors based only on intuition.

Instead:

```text
Keep legitimate engineering candidates
Remove or isolate leakage
Remove non-informative columns
Remove identifiers that encourage memorization
Handle metadata deliberately
Handle categorical variables deliberately
Let the ML model discover which valid engineering predictors matter
```

This distinction is critical to the validity of the final model.

---

## Notes

- The current dataset is synthetic.
- Numeric thresholds in the dataset are training/reference values and should not automatically be interpreted as real production specifications.
- Visual EDA provides evidence, not proof.
- Correlation does not imply causation.
- The ground-truth document is used to validate discovery quality, not to hard-code model answers.
- Feature leakage must be controlled before final model training.
