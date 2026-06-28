"""
DNN seed-variance check for the Protocol A vs Protocol B comparison.

train_models.py / train_models_ablation.py only call tf.random.set_seed(42)
before building the DNN. NumPy/Python RNG and TF op-level nondeterminism are
not pinned, and CatBoostEncoder/Dropout/validation_split shuffling are not
seeded either. A manual check (4 single runs of train_models.py /
train_models_ablation.py) showed DNN macro-F1 ranging ~61-85% under Protocol
A alone and ~57-80% under Protocol B alone - a spread as large as the
"+19.07 point" Protocol A -> B effect the manuscript reports from a single
run pair (Table 5, Sections 4.1/4.3).

This script re-derives that comparison properly: full determinism pinning,
N independent seeds per protocol, and the resulting mean +/- std macro-F1
and accuracy for each protocol, so the A vs B comparison can be judged
against the model's own run-to-run noise instead of a single anecdote.
"""
import json
import os
import random

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ART_DIR = 'artifacts'
RESULTS_DIR = 'results_dnn_variance'
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))

DROP_COLS = ['auditd_session', 'agent_ephemeral_id', 'auditd_sequence',
             'process_entity_id_str', 'process_start_str']

X_train_full = pd.read_parquet(os.path.join(ART_DIR, 'X_train.parquet'))
X_test_full = pd.read_parquet(os.path.join(ART_DIR, 'X_test.parquet'))
y_train = np.load(os.path.join(ART_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(ART_DIR, 'y_test.npy'))
classes = np.load(os.path.join(ART_DIR, 'label_classes.npy'), allow_pickle=True)
n_classes = len(classes)

present = [c for c in DROP_COLS if c in X_train_full.columns]
PROTOCOLS = {
    'A': (X_train_full.to_numpy(), X_test_full.to_numpy()),
    'B': (X_train_full.drop(columns=present).to_numpy(),
          X_test_full.drop(columns=present).to_numpy()),
}


def train_one(seed, X_train, X_test):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    from tensorflow.keras import layers, Sequential

    dnn = Sequential([
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
    dnn.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0, validation_split=0.1)

    y_pred = np.argmax(dnn.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
    return acc, macro_f1


results = {}
for protocol, (X_train, X_test) in PROTOCOLS.items():
    print(f'\n=== Protocol {protocol}: X_train {X_train.shape}, X_test {X_test.shape} ===')
    runs = []
    for seed in SEEDS:
        acc, macro_f1 = train_one(seed, X_train, X_test)
        print(f'  seed={seed}  accuracy={acc:.2f}%  macro_f1={macro_f1:.2f}%')
        runs.append({'seed': seed, 'accuracy': acc, 'macro_f1': macro_f1})

    accs = np.array([r['accuracy'] for r in runs])
    f1s = np.array([r['macro_f1'] for r in runs])
    results[protocol] = {
        'runs': runs,
        'accuracy_mean': float(accs.mean()), 'accuracy_std': float(accs.std()),
        'macro_f1_mean': float(f1s.mean()), 'macro_f1_std': float(f1s.std()),
    }
    print(f'  Protocol {protocol} summary: accuracy={accs.mean():.2f}+/-{accs.std():.2f}%  '
          f'macro_f1={f1s.mean():.2f}+/-{f1s.std():.2f}%')

delta_mean = results['B']['macro_f1_mean'] - results['A']['macro_f1_mean']
pooled_std = float(np.sqrt(
    (np.array([r['macro_f1'] for r in results['A']['runs']]).std() ** 2 +
     np.array([r['macro_f1'] for r in results['B']['runs']]).std() ** 2) / 2
))
print(f'\nProtocol B - Protocol A macro-F1 (mean over {N_SEEDS} seeds each): '
      f'{delta_mean:+.2f} points (per-protocol std ~{pooled_std:.2f} points)')

with open(os.path.join(RESULTS_DIR, 'dnn_seed_variance.json'), 'w') as f:
    json.dump({'n_seeds': N_SEEDS, 'protocols': results,
                'delta_macro_f1_B_minus_A_mean': delta_mean,
                'pooled_std_macro_f1': pooled_std}, f, indent=2)

print(f'\nResults written to {RESULTS_DIR}/dnn_seed_variance.json')
