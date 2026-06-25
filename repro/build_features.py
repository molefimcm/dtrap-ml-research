"""
Phase 2/3 reproduction - Step 1: feature engineering.

Faithfully reproduces the data-cleaning steps of DataPreprocessor.ipynb
(filtering, null-column drop, single-value-column drop, array->string
flattening, label construction), but FIXES the issues identified during
code archaeology:

  1. The original notebooks build `multi_data` by one-hot-encoding the
     label column (tags_str) with pd.get_dummies() and then slicing
     `multi_data.iloc[:, 0:N]` as the feature matrix X *without*
     excluding those new dummy columns. Because pd.get_dummies() inserts
     the dummy columns in place of the original label column (which sits
     well inside the first N columns), X ends up containing a one-hot
     encoding of the label itself. This is the source of the ~99-100%
     SVM/RF/GB numbers in the submitted manuscript. FIX: the label is
     kept completely separate from X; no dummy/derived label columns are
     ever joined into the feature matrix.

  2. StandardScaler and CatBoostEncoder were both fit on the *entire*
     dataset (including what later became the test set), and CatBoost
     encoding additionally uses the target column, so any fit on the
     full data leaks test-set label information into the encoding of
     train rows that share a category with test rows. FIX: every fitted
     transformer (scaler, encoder, correlation-drop list) is fit on the
     TRAIN split only and applied to the TEST split.

  3. The numeric "X = iloc[:, 0:60]" style slicing is replaced by an
     explicit, named feature list so it is auditable.

Output: writes train/test feature matrices + labels + a feature/label
metadata file to repro/artifacts/.
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce

pd.set_option('display.max_columns', None)

DATA_PATH = '../data/auditbeat_dataset.parquet'
ART_DIR = 'artifacts'
os.makedirs(ART_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load + filter (identical to original notebooks)
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
print('raw shape', df.shape)

df = df.query(
    'event_action != "network_flow" and process_name != "firefox" and '
    'process_executable != "/usr/bin/sleep" and '
    'user_selinux_user != "snap.notepad-plus-plus.notepad-plus-plus"',
    engine='python',
)
print('filtered shape', df.shape)

# keep timestamp for the time-based split, then drop it from features
timestamps = pd.to_datetime(df['timestamp']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Drop columns with >95% nulls, drop timestamp
# ---------------------------------------------------------------------------
df1 = df.dropna(thresh=int(df.shape[0] * 0.05), axis=1)
df1 = df1.drop(['timestamp'], axis=1).reset_index(drop=True)

orginal_cols = df1.columns

# arr_cols exactly as in DataPreprocessor.ipynb cell 9 (the live, uncommented one)
arr_cols = ['user_effective_name', 'user_effective_id', 'user_effective_group_id',
            'system_audit_package_version', 'system_audit_package_url',
            'system_audit_package_summary', 'system_audit_package_name',
            'system_audit_package_entity_id', 'system_audit_package_arch',
            'process_start', 'process_hash_sha1', 'process_entity_id',
            'package_version', 'package_type', 'package_reference', 'package_name',
            'package_description', 'package_architecture', 'host_os_version',
            'host_os_type', 'host_os_platform', 'host_os_name', 'host_os_kernel',
            'host_os_family', 'host_os_codename', 'host_name', 'host_id',
            'host_hostname', 'host_architecture', 'message', 'ecs_version',
            'event_id', 'auditd_data_terminal', 'auditd_data_op',
            'auditd_data_grantors', 'auditd_data_argc', 'agent_version',
            'agent_type', 'agent_name', 'agent_id', 'agent_hostname',
            'auditd_data_acct', 'auditd_paths', 'event_type', 'host_ip', 'host_mac',
            'process_args', 'related_user', 'tags']

arr_cols = [c for c in arr_cols if c in df1.columns]
str_cols = list((Counter(orginal_cols) - Counter(arr_cols)).elements())

# ---------------------------------------------------------------------------
# 3. Drop single-unique-value columns (among str_cols, as in original)
# ---------------------------------------------------------------------------
for col in str_cols:
    if df1[col].nunique(dropna=False) == 1:
        df1 = df1.drop(columns=[col])

# ---------------------------------------------------------------------------
# 4. Flatten array columns to '~'-delimited strings, build tags_str label
# ---------------------------------------------------------------------------
def join_arr(arr_ele):
    try:
        return '~'.join(map(str, arr_ele))
    except TypeError:
        return ''

arr_cols_present = [c for c in arr_cols if c in df1.columns]
for c in arr_cols_present:
    df1[c + '_str'] = [join_arr(a) for a in df1[c]]

LABEL_MAP = {
    '': 'Others',
    'exec': 'exec',
    'access': 'access',
    'T1166_Seuid_and_Setgid': 'T1166_Seuid_and_Setgid',
    'T1087_Account_Discovery': 'T1087_Account_Discovery',
    'T1169_Sudo': 'T1169_Sudo',
    'T1082_System_Information_Discovery': 'T1082_System_Information_Discovery',
    'T1016_System_Network_Configuration_Discovery': 'T1016_System_Network_Configuration_Discovery',
}
df1['tags_label'] = df1['tags_str'].map(LABEL_MAP)
# anything not in the 8 known buckets (e.g. residual 32bit-abi combos) -> Others
df1['tags_label'] = df1['tags_label'].fillna('Others')

print('\nLabel distribution:')
print(df1['tags_label'].value_counts())

# drop the original array columns now that *_str versions exist
df1 = df1.drop(columns=arr_cols_present)

# ---------------------------------------------------------------------------
# 5. Identify numeric vs categorical columns
# ---------------------------------------------------------------------------
exclude = {'tags_str', 'tags_label'}
numeric_cols = [c for c in df1.select_dtypes(include='number').columns if c not in exclude]
categorical_cols = [c for c in df1.columns
                     if df1[c].dtype == 'object' and c not in exclude]

print(f'\n#numeric cols: {len(numeric_cols)}')
print(f'#categorical cols: {len(categorical_cols)}')

le = LabelEncoder()
y_all = le.fit_transform(df1['tags_label'])
np.save(os.path.join(ART_DIR, 'label_classes.npy'), le.classes_, allow_pickle=True)
print('\nClasses:', list(le.classes_))

X_raw = df1[numeric_cols + categorical_cols].copy()
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

# ---------------------------------------------------------------------------
# 6. Time-based split (70% earliest events = train, 30% latest = test)
# ---------------------------------------------------------------------------
order = np.argsort(timestamps.values, kind='stable')
n = len(order)
cut = int(n * 0.70)
train_idx = order[:cut]
test_idx = order[cut:]

print(f'\nTime-based split: train n={len(train_idx)}, test n={len(test_idx)}')
print('train time range:', timestamps.iloc[train_idx].min(), '->', timestamps.iloc[train_idx].max())
print('test  time range:', timestamps.iloc[test_idx].min(), '->', timestamps.iloc[test_idx].max())

# ---------------------------------------------------------------------------
# 7. Fit StandardScaler + CatBoostEncoder on TRAIN ONLY, apply to both
# ---------------------------------------------------------------------------
X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
X_test_raw = X_raw.iloc[test_idx].reset_index(drop=True)
y_train = y_all[train_idx]
y_test = y_all[test_idx]

scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train_raw[numeric_cols]) if numeric_cols else np.zeros((len(X_train_raw), 0)),
    columns=numeric_cols)
X_test_num = pd.DataFrame(
    scaler.transform(X_test_raw[numeric_cols]) if numeric_cols else np.zeros((len(X_test_raw), 0)),
    columns=numeric_cols)

cbe = ce.cat_boost.CatBoostEncoder()
X_train_cat = cbe.fit_transform(X_train_raw[categorical_cols], y_train)
X_test_cat = cbe.transform(X_test_raw[categorical_cols])

X_train = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

print('\nFeature matrix shape before correlation filter:', X_train.shape)

# ---------------------------------------------------------------------------
# 8. Drop highly-correlated features (>0.99), computed on TRAIN ONLY
# ---------------------------------------------------------------------------
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.99)]
print(f'\nDropping {len(to_drop)} highly-correlated columns:', to_drop)

X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

print('\nFinal feature matrix shape:', X_train.shape, X_test.shape)

# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------
X_train.to_parquet(os.path.join(ART_DIR, 'X_train.parquet'))
X_test.to_parquet(os.path.join(ART_DIR, 'X_test.parquet'))
np.save(os.path.join(ART_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(ART_DIR, 'y_test.npy'), y_test)

# also save the full (unsplit) feature matrix + labels for the CV protocol
X_all_num = pd.DataFrame(StandardScaler().fit_transform(X_raw[numeric_cols]) if numeric_cols else np.zeros((len(X_raw), 0)),
                          columns=numeric_cols)
cbe_full = ce.cat_boost.CatBoostEncoder()
X_all_cat = cbe_full.fit_transform(X_raw[categorical_cols], y_all)
X_all = pd.concat([X_all_num.reset_index(drop=True), X_all_cat.reset_index(drop=True)], axis=1)
X_all = X_all.drop(columns=[c for c in to_drop if c in X_all.columns])
X_all.to_parquet(os.path.join(ART_DIR, 'X_all.parquet'))
np.save(os.path.join(ART_DIR, 'y_all.npy'), y_all)

meta = {
    'n_total': int(n),
    'n_train': int(len(train_idx)),
    'n_test': int(len(test_idx)),
    'classes': list(le.classes_),
    'feature_columns': list(X_train.columns),
    'dropped_corr_columns': to_drop,
    'label_distribution_total': df1['tags_label'].value_counts().to_dict(),
    'train_time_range': [str(timestamps.iloc[train_idx].min()), str(timestamps.iloc[train_idx].max())],
    'test_time_range': [str(timestamps.iloc[test_idx].min()), str(timestamps.iloc[test_idx].max())],
}
with open(os.path.join(ART_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2, default=str)

print('\nDone. Artifacts written to', ART_DIR)
