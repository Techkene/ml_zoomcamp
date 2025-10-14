# %%
# Course Lead Scoring - Jupyter Notebook
# This notebook performs data preparation, exploratory analysis, model training,
# and answers the multiple-choice questions described by the user.

# %%
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score

# %%
# Load dataset
path = '/mnt/data/course_lead_scoring.csv'
df = pd.read_csv(path)
print('Dataset loaded. Shape:', df.shape)

# %%
# Data preparation: check missing values and impute
missing_summary = df.isnull().sum()
print('\nMissing values per column:\n', missing_summary)

# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Impute: categorical -> 'NA', numerical -> 0.0
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna('NA')
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(0.0)

print('\nAfter imputation, missing values per column:\n', df.isnull().sum())

# %%
# Q1: What is the mode for the column 'industry'?
# Print the most frequent observation (mode)
if 'industry' in df.columns:
    mode_industry = df['industry'].mode()
    if len(mode_industry) > 0:
        mode_industry = mode_industry.iloc[0]
    else:
        mode_industry = None
    print('\nQ1 - Mode of industry column:', mode_industry)
else:
    print('\nQ1 - Column "industry" not found in dataset')

# %%
# Q2: Correlation matrix for numerical features and find the pair with biggest correlation
num_df = df[num_cols].copy()
# Drop any constant columns to avoid NaNs in correlation
num_df = num_df.loc[:, num_df.nunique() > 1]
corr = num_df.corr()
print('\nNumeric columns used for correlation:\n', num_df.columns.tolist())
print('\nCorrelation matrix:\n', corr)

# Compute absolute correlations and inspect candidate pairs
candidates = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]
candidate_corrs = {}
for a, b in candidates:
    if a in corr.index and b in corr.columns:
        candidate_corrs[(a, b)] = corr.loc[a, b]
    else:
        candidate_corrs[(a, b)] = np.nan

print('\nQ2 - Candidate pair correlations:')
for k, v in candidate_corrs.items():
    print(f'  {k}: {v}')

# Find the pair (from the provided list) with the largest absolute correlation
best_pair = max(candidate_corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -1)[0]
print('\nQ2 - Pair with largest correlation among candidates:', best_pair)

# %%
# Prepare target and split the data into train/val/test with 60/20/20
# Ensure target column name - problem statement uses 'converted'
if 'converted' not in df.columns:
    raise ValueError('Target column "converted" not found in the dataset')

# We'll stratify splits by the target to maintain class balance
X_full = df.drop(columns=['converted']).copy()
y_full = df['converted'].copy()

# First split: train (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.4, random_state=42, stratify=y_full
)
# Second split: from temp, create val (50% of temp => 20% overall) and test (50% of temp => 20% overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print('\nSplit sizes:')
print('  train:', X_train.shape, y_train.shape)
print('  val:  ', X_val.shape, y_val.shape)
print('  test: ', X_test.shape, y_test.shape)

# %%
# Q3: Calculate mutual information score between converted and categorical variables using training set only.
# We'll compute mutual_info_score for each categorical variable
# Identify categorical columns in original X (based on dtype object/category)
train_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print('\nCategorical columns considered for mutual information on training set:\n', train_cat_cols)

mi_scores = {}
for c in train_cat_cols:
    # mutual_info_score expects discrete labels; convert to strings if needed
    mi = mutual_info_score(y_train, X_train[c].astype(str))
    mi_scores[c] = round(mi, 2)

print('\nQ3 - Mutual information scores (rounded to 2 decimals):')
for k, v in mi_scores.items():
    print(f'  {k}: {v}')

# Determine which of the provided options has the biggest MI: industry, location, lead_source, employment_status
options = ['industry', 'location', 'lead_source', 'employment_status']
present_options = {opt: mi_scores.get(opt, None) for opt in options}
print('\nQ3 - Scores for provided options:')
for k, v in present_options.items():
    print(f'  {k}: {v}')

best_mi_feature = max((opt for opt in options if present_options.get(opt) is not None),
                      key=lambda x: present_options[x]) if any(present_options.values()) else None
print('\nQ3 - Feature with biggest mutual information among the options:', best_mi_feature)

# %%
# Q4: Train logistic regression with one-hot encoding for categorical variables.
# We'll use pd.get_dummies on training set, and align val/test to train columns

def prepare_onehot(X_train, X_val, X_test):
    X_tr = pd.get_dummies(X_train, drop_first=False)
    X_v = pd.get_dummies(X_val, drop_first=False)
    X_te = pd.get_dummies(X_test, drop_first=False)
    # Align columns
    X_v = X_v.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr, X_v, X_te

X_train_enc, X_val_enc, X_test_enc = prepare_onehot(X_train, X_val, X_test)

print('\nAfter one-hot, number of features:', X_train_enc.shape[1])

# Fit logistic regression as specified
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_enc, y_train)

# Accuracy on validation set
y_val_pred = model.predict(X_val_enc)
val_accuracy = accuracy_score(y_val, y_val_pred)
print('\nQ4 - Validation accuracy (unrounded):', val_accuracy)
print('Q4 - Validation accuracy (rounded to 2 decimals):', round(val_accuracy, 2))

# %%
# Q5: Feature elimination (exclude each feature individually and compute accuracy drop)
# We're asked to test excluding 'industry', 'employment_status', 'lead_score'
base_accuracy = val_accuracy
print('\nBase validation accuracy used for feature elimination:', base_accuracy)

features_to_test = ['industry', 'employment_status', 'lead_score']
accuracy_diffs = {}

for feat in features_to_test:
    X_tr_mod = X_train.copy()
    X_v_mod = X_val.copy()
    # If numeric feature, drop directly; if categorical, drop all one-hot columns starting with feat or equal to feat
    if feat in X_tr_mod.columns and pd.api.types.is_numeric_dtype(X_tr_mod[feat].dtype):
        X_tr_mod = X_tr_mod.drop(columns=[feat])
        X_v_mod = X_v_mod.drop(columns=[feat])
    else:
        # drop categorical columns from the one-hot encoding
        # Create one-hot for full set then drop columns that reference the feature
        X_tr_ohe = pd.get_dummies(X_tr_mod, drop_first=False)
        cols_to_drop = [c for c in X_tr_ohe.columns if c.startswith(feat + '_') or c == feat]
        # Remove these columns from original X by dropping the corresponding categorical column before encoding
        if feat in X_tr_mod.columns:
            X_tr_mod = X_tr_mod.drop(columns=[feat])
        if feat in X_v_mod.columns:
            X_v_mod = X_v_mod.drop(columns=[feat])
    # Re-encode after dropping
    X_tr_enc_mod, X_v_enc_mod, _ = prepare_onehot(X_tr_mod, X_v_mod, X_test.head(0))
    # Fit model
    m = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    m.fit(X_tr_enc_mod, y_train)
    y_v_pred_mod = m.predict(X_v_enc_mod)
    acc_mod = accuracy_score(y_val, y_v_pred_mod)
    diff = base_accuracy - acc_mod
    accuracy_diffs[feat] = diff
    print(f'Feature removed: {feat} -> val acc: {acc_mod:.6f}, diff: {diff:.6f}')

# Which feature has the smallest difference?
smallest_diff_feat = min(accuracy_diffs.items(), key=lambda x: x[1])[0]
print('\nQ5 - Feature with the smallest difference:', smallest_diff_feat)

# %%
# Q6: Regularized logistic regression. Try Cs = [0.01, 0.1, 1, 10, 100]
Cs = [0.01, 0.1, 1, 10, 100]
results_C = {}
for C in Cs:
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    m.fit(X_train_enc, y_train)
    y_v_pred = m.predict(X_val_enc)
    acc = accuracy_score(y_val, y_v_pred)
    results_C[C] = round(acc, 3)
    print(f'C={C} -> val accuracy: {results_C[C]:.3f}')

# Find the best accuracy; if multiple, choose smallest C
best_acc = max(results_C.values())
best_Cs = sorted([C for C, acc in results_C.items() if acc == best_acc])
best_C = best_Cs[0]
print('\nQ6 - Best C (ties broken by smallest C):', best_C, 'with accuracy', best_acc)

# %%
# Final summary prints for quick reference
print('\n--- SUMMARY OF ANSWERS ---')
print('Q1 (mode of industry):', mode_industry)
print('Q2 (pair with biggest correlation among candidates):', best_pair)
print('Q3 (feature with biggest mutual information among options):', best_mi_feature)
print('Q4 (validation accuracy rounded to 2 decimals):', round(val_accuracy, 2))
print('Q5 (feature with smallest accuracy difference when removed):', smallest_diff_feat)
print('Q6 (best C):', best_C)
print('\nNotebook run complete.')
