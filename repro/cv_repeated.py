"""
Phase 3 - Stratified k-fold CV (mean +/- std).

Protocols A/B use a single time-based train/test split, which is the right
choice for the *headline* result (it respects chronology and avoids the
session/run leakage discussed in REPRO_NOTES.md section 6). But a single
split gives no sense of variance. This script reports stratified 5-fold CV
on Random Forest and Gradient Boosting only (the two models cheap enough to
refit several times; SVM and DNN are not repeated here for runtime reasons
and that is stated explicitly).

To avoid the leakage in section 3.2 of REPRO_NOTES.md, StandardScaler and
CatBoostEncoder are placed INSIDE an sklearn Pipeline together with the
classifier, so they are re-fit on each fold's training data only.

Note: this CV is over RANDOM stratified folds, not session/time-aware folds
(no valid session id exists - see REPRO_NOTES.md section 2.2, and CV by
construction cannot be time-ordered). It is reported as a variance estimate
for the model/feature pipeline, NOT as a leakage-free generalisation
estimate - that role is filled by the time-based Protocols A/B. This
distinction must be stated in the manuscript.

Originally written as 5-fold x 2-repeat (20 fits across two separate
cross_val_score calls); that did not complete within the time available for
this revision. This version uses a single cross_validate call (one fit per
fold, both metrics scored from the same fitted pipeline) with 5-fold x
1-repeat to make the run tractable, and is reported as such (n=5 per model).
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
import category_encoders as ce

DATA_PATH = '../data/auditbeat_dataset.parquet'
ART_DIR = 'artifacts'
RESULTS_DIR = 'results_cv'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Rebuild X_raw / y_all exactly as build_features.py does, but stop before
# scaling/encoding (those go inside the CV pipeline instead).
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
df = df.query(
    'event_action != "network_flow" and process_name != "firefox" and '
    'process_executable != "/usr/bin/sleep" and '
    'user_selinux_user != "snap.notepad-plus-plus.notepad-plus-plus"',
    engine='python',
)
df1 = df.dropna(thresh=int(df.shape[0] * 0.05), axis=1)
df1 = df1.drop(['timestamp'], axis=1).reset_index(drop=True)

orginal_cols = df1.columns
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

for col in str_cols:
    if df1[col].nunique(dropna=False) == 1:
        df1 = df1.drop(columns=[col])

def join_arr(arr_ele):
    try:
        return '~'.join(map(str, arr_ele))
    except TypeError:
        return ''

arr_cols_present = [c for c in arr_cols if c in df1.columns]
for c in arr_cols_present:
    df1[c + '_str'] = [join_arr(a) for a in df1[c]]

LABEL_MAP = {
    '': 'Others', 'exec': 'exec', 'access': 'access',
    'T1166_Seuid_and_Setgid': 'T1166_Seuid_and_Setgid',
    'T1087_Account_Discovery': 'T1087_Account_Discovery',
    'T1169_Sudo': 'T1169_Sudo',
    'T1082_System_Information_Discovery': 'T1082_System_Information_Discovery',
    'T1016_System_Network_Configuration_Discovery': 'T1016_System_Network_Configuration_Discovery',
}
df1['tags_label'] = df1['tags_str'].map(LABEL_MAP).fillna('Others')
df1 = df1.drop(columns=arr_cols_present)

exclude = {'tags_str', 'tags_label'}
numeric_cols = [c for c in df1.select_dtypes(include='number').columns if c not in exclude]
categorical_cols = [c for c in df1.columns if df1[c].dtype == 'object' and c not in exclude]

# Drop the same 5 identifier/run-proxy columns as Protocol B (section 6)
DROP_COLS = ['auditd_session', 'agent_ephemeral_id', 'auditd_sequence',
              'process_entity_id_str', 'process_start_str']
numeric_cols = [c for c in numeric_cols if c not in DROP_COLS]
categorical_cols = [c for c in categorical_cols if c not in DROP_COLS]

le = LabelEncoder()
y_all = le.fit_transform(df1['tags_label'])
classes = le.classes_
n_classes = len(classes)

X_raw = df1[numeric_cols + categorical_cols].copy()
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

print('X_raw shape', X_raw.shape, 'n_classes', n_classes)
print('label distribution:', dict(zip(*np.unique(y_all, return_counts=True))))

# ---------------------------------------------------------------------------
# Pipeline: ColumnTransformer(StandardScaler, CatBoostEncoder) -> classifier
# ---------------------------------------------------------------------------
pre = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', ce.cat_boost.CatBoostEncoder(), categorical_cols),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, clf in [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]:
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    print(f'\nRunning 5-fold CV for {name} (5 fits)...')
    cv_res = cross_validate(pipe, X_raw, y_all, cv=skf,
                             scoring=['f1_macro', 'accuracy'], n_jobs=1)
    scores_f1 = cv_res['test_f1_macro']
    scores_acc = cv_res['test_accuracy']
    print(f'{name}: macro-F1 = {scores_f1.mean()*100:.2f} +/- {scores_f1.std()*100:.2f}')
    print(f'{name}: accuracy  = {scores_acc.mean()*100:.2f} +/- {scores_acc.std()*100:.2f}')
    results[name] = {
        'macro_f1_mean': scores_f1.mean() * 100,
        'macro_f1_std': scores_f1.std() * 100,
        'accuracy_mean': scores_acc.mean() * 100,
        'accuracy_std': scores_acc.std() * 100,
        'macro_f1_folds': (scores_f1 * 100).tolist(),
        'accuracy_folds': (scores_acc * 100).tolist(),
    }
    with open(os.path.join(RESULTS_DIR, 'cv_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

print('\nDone.')
