"""
Phase 2/3 - Step 2: train DNN, SVM, RF, GB under ONE consistent protocol
(time-based 70/30 split produced by build_features.py), and report
accuracy, macro-F1, per-class precision/recall, and confusion matrices.

Fixes vs the original notebooks:
  - DNN output layer is Dense(n_classes, softmax) trained with
    sparse_categorical_crossentropy (the original used Dense(1, sigmoid/
    softmax) + categorical_crossentropy against integer labels, which is
    mathematically meaningless - loss was always 0).
  - SVM uses a linear kernel on the (now leakage-free) feature matrix.
  - All models trained/evaluated on the same X_train/X_test/y_train/y_test.
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
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = pd.read_parquet(os.path.join(ART_DIR, 'X_train.parquet')).to_numpy()
X_test = pd.read_parquet(os.path.join(ART_DIR, 'X_test.parquet')).to_numpy()
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


# ---------------------------------------------------------------------------
# Linear SVM
# ---------------------------------------------------------------------------
print('\nTraining Linear SVM...')
svm = SVC(kernel='linear', gamma='auto')
svm.fit(X_train, y_train)
evaluate('SVM', y_test, svm.predict(X_test))

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
print('\nTraining Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate('RandomForest', y_test, rf.predict(X_test))

# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------
print('\nTraining Gradient Boosting...')
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
evaluate('GradientBoosting', y_test, gb.predict(X_test))

# ---------------------------------------------------------------------------
# DNN (Keras) - FIXED architecture: softmax over n_classes, sparse CE
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Save all results
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print('\nSummary (test set, time-based split):')
for name, r in all_results.items():
    print(f"  {name:18s}  accuracy={r['accuracy']:.2f}%  macro-F1={r['macro_f1']:.2f}%")
