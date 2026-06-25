"""
Phase 3 - Rule-based ATT&CK baseline.

This implements a simple, expert-curated, rule-based classifier that mirrors
how a SIEM (Wazuh/Elastic/Splunk/Sentinel) would map Auditbeat fields to
MITRE technique labels using the SAME audit-rule keys defined in
linux_audit.rules (file-path watches and syscall filters), but WITHOUT using
the `tags`/`tags_str` field itself (which IS the ground-truth label and would
make the comparison circular).

Rules are applied in priority order to each event's raw `file_path`,
`process_executable`, and `auditd_data_syscall` fields:

  1. file_path in {/etc/resolv.conf}                         -> T1016
  2. file_path in {/etc/hostname, /etc/login.defs}            -> T1082
  3. file_path in {/etc/passwd, /etc/group, /etc/shadow,
                    /etc/gshadow}                              -> T1087
  4. auditd_data_syscall in {setuid, setgid, seteuid,
                              setegid}                         -> T1166
  5. file_path == /etc/sudoers or process_executable in
     {/usr/bin/sudo, /bin/su}                                  -> T1169
  6. auditd_data_syscall == execve                             -> exec
  7. auditd_data_syscall in {open, openat, read, ...}          -> access
  8. everything else                                           -> Others

This baseline is evaluated on the SAME time-based test split (26,007 rows)
used for Protocol A/B, using the SAME ground-truth labels, for a fair
comparison with the ML models.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = '../data/auditbeat_dataset.parquet'
ART_DIR = 'artifacts'
RESULTS_DIR = 'results_rule_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH, engine='pyarrow')
df = df.query(
    'event_action != "network_flow" and process_name != "firefox" and '
    'process_executable != "/usr/bin/sleep" and '
    'user_selinux_user != "snap.notepad-plus-plus.notepad-plus-plus"',
    engine='python',
)
timestamps = pd.to_datetime(df['timestamp']).reset_index(drop=True)
df = df.reset_index(drop=True)

LABEL_MAP = {
    '': 'Others', 'exec': 'exec', 'access': 'access',
    'T1166_Seuid_and_Setgid': 'T1166_Seuid_and_Setgid',
    'T1087_Account_Discovery': 'T1087_Account_Discovery',
    'T1169_Sudo': 'T1169_Sudo',
    'T1082_System_Information_Discovery': 'T1082_System_Information_Discovery',
    'T1016_System_Network_Configuration_Discovery': 'T1016_System_Network_Configuration_Discovery',
}

def join_arr(arr_ele):
    try:
        return '~'.join(map(str, arr_ele))
    except TypeError:
        return ''

tags_str = df['tags'].apply(join_arr)
y_true_label = tags_str.map(LABEL_MAP).fillna('Others')

classes = np.load(os.path.join(ART_DIR, 'label_classes.npy'), allow_pickle=True)
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
y_true = y_true_label.map(class_to_idx).to_numpy()

# ---------------------------------------------------------------------------
# Apply the rule cascade
# ---------------------------------------------------------------------------
file_path = df['file_path'].astype(str)
proc_exec = df['process_executable'].astype(str)
syscall = df['auditd_data_syscall'].astype(str)

pred_label = pd.Series('Others', index=df.index)

mask = syscall.isin(['open', 'openat', 'read', 'readlink', 'chmod', 'chown', 'write'])
pred_label[mask] = 'access'

mask = syscall == 'execve'
pred_label[mask] = 'exec'

mask = (file_path == '/etc/sudoers') | proc_exec.isin(['/usr/bin/sudo', '/bin/su'])
pred_label[mask] = 'T1169_Sudo'

mask = syscall.isin(['setuid', 'setgid', 'seteuid', 'setegid'])
pred_label[mask] = 'T1166_Seuid_and_Setgid'

mask = file_path.isin(['/etc/passwd', '/etc/group', '/etc/shadow', '/etc/gshadow'])
pred_label[mask] = 'T1087_Account_Discovery'

mask = file_path.isin(['/etc/hostname', '/etc/login.defs'])
pred_label[mask] = 'T1082_System_Information_Discovery'

mask = file_path == '/etc/resolv.conf'
pred_label[mask] = 'T1016_System_Network_Configuration_Discovery'

y_pred = pred_label.map(class_to_idx).fillna(class_to_idx['Others']).astype(int).to_numpy()

# ---------------------------------------------------------------------------
# Evaluate on the SAME time-based 70/30 split as Protocols A/B
# ---------------------------------------------------------------------------
order = np.argsort(timestamps.values, kind='stable')
n = len(order)
cut = int(n * 0.70)
test_idx = order[cut:]

y_true_test = y_true[test_idx]
y_pred_test = y_pred[test_idx]

acc = accuracy_score(y_true_test, y_pred_test) * 100
macro_f1 = f1_score(y_true_test, y_pred_test, average='macro') * 100
report = classification_report(y_true_test, y_pred_test, target_names=classes,
                                output_dict=True, zero_division=0)
cm = confusion_matrix(y_true_test, y_pred_test, labels=range(n_classes))

print(f'Rule-based baseline on test split (n={len(test_idx)})')
print(f'accuracy={acc:.2f}  macro_f1={macro_f1:.2f}')
print(classification_report(y_true_test, y_pred_test, target_names=classes, zero_division=0))

with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        'accuracy': acc,
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }, f, indent=2)
np.save(os.path.join(RESULTS_DIR, 'confmat.npy'), cm)
