"""
Phase 3 - Identifier/run-proxy leakage ablation.

train_models.py (Protocol A) trains on all 57 features surviving the
correlation filter and gets RF accuracy=99.98% / macro-F1=99.99%, with
PERFECT per-class scores even for T1166 (43 test rows). That is too good
given GB and SVM on the SAME matrix score far lower (macro-F1 72.6% /
74.6%), which suggests RF is exploiting a handful of columns that act as
run/session identifiers rather than behavioural signal:

  - auditd_session        (29 distinct values total)
  - agent_ephemeral_id     (6 distinct values = "which of the 6 collection runs")
  - auditd_sequence        (monotonically increasing per-host event counter,
                             i.e. a proxy for time/order)
  - process_entity_id_str  (per-process-instance identifier)
  - process_start_str      (process start time, encoded as a string)

Because attack-technique labels cluster heavily by session/run/time window,
these columns let a model memorise "which run is this row from" and back out
the label without learning anything about the technique's audit-log
behaviour. This script (Protocol B) drops these five columns and re-trains
the same four models on the remaining 52 features, to report a more
conservative, behaviourally-grounded result set alongside Protocol A.
"""
import json
import os
import random

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

SEED = 42
random.seed(SEED)

import numpy as np
import pandas as pd

np.random.seed(SEED)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)

ART_DIR = 'artifacts'
RESULTS_DIR = 'results_ablation'
os.makedirs(RESULTS_DIR, exist_ok=True)

DROP_COLS = ['auditd_session', 'agent_ephemeral_id', 'auditd_sequence',
              'process_entity_id_str', 'process_start_str']

X_train = pd.read_parquet(os.path.join(ART_DIR, 'X_train.parquet'))
X_test = pd.read_parquet(os.path.join(ART_DIR, 'X_test.parquet'))
present = [c for c in DROP_COLS if c in X_train.columns]
print('Dropping identifier/run-proxy columns:', present)
X_train = X_train.drop(columns=present).to_numpy()
X_test = X_test.drop(columns=present).to_numpy()

y_train = np.load(os.path.join(ART_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(ART_DIR, 'y_test.npy'))
classes = np.load(os.path.join(ART_DIR, 'label_classes.npy'), allow_pickle=True)
n_classes = len(classes)

print('X_train', X_train.shape, 'X_test', X_test.shape, 'n_classes', n_classes)

all_results = {}


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    report = classification_report(y_true, y_pred, target_names=classes,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    print(f'\n=== {name} ===')
    print(f'accuracy={acc:.2f}  macro_f1={macro_f1:.2f}')
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    all_results[name] = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }
    np.save(os.path.join(RESULTS_DIR, f'{name}_confmat.npy'), cm)
    np.save(os.path.join(RESULTS_DIR, f'{name}_y_pred.npy'), y_pred)


print('\nTraining Linear SVM...')
svm = SVC(kernel='linear', gamma='auto')
svm.fit(X_train, y_train)
evaluate('SVM', y_test, svm.predict(X_test))

print('\nTraining Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate('RandomForest', y_test, rf.predict(X_test))

print('\nTraining Gradient Boosting...')
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
evaluate('GradientBoosting', y_test, gb.predict(X_test))

print('\nTraining DNN...')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(SEED)

dnn = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
dnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = dnn.fit(X_train, y_train, epochs=20, batch_size=64, verbose=2,
                   validation_split=0.1)

y_pred_dnn = np.argmax(dnn.predict(X_test), axis=1)
evaluate('DNN', y_test, y_pred_dnn)

with open(os.path.join(RESULTS_DIR, 'dnn_history.json'), 'w') as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print('\nSummary (test set, time-based split, identifier columns dropped):')
for name, r in all_results.items():
    print(f"  {name:18s}  accuracy={r['accuracy']:.2f}%  macro-F1={r['macro_f1']:.2f}%")
