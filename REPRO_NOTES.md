# Reproduction Notes — DTRAP-2025-0106

This document records what was actually re-derived from the code and data in
this repository, as the single source of truth for the revised manuscript.
Per the project's prime directive: **the code and data win over any number
previously written in any draft.** Where this document's numbers disagree
with `DTRAP_2025_0106_Revised_Manuscript_v7.docx`, the manuscript is wrong
and must be corrected to match this document.

All steps are reproducible from a clean checkout via:

```
repro/build_features.py      # feature engineering, time-based split, artifacts/
repro/train_models.py        # Protocol A: all 57 retained features
repro/train_models_ablation.py  # Protocol B: drop 5 run/session-identifier columns
```

Environment: Python 3.11 venv (`.venv311`) with pandas 2.3.3, pyarrow,
scikit-learn 1.9.0, category_encoders, tensorflow 2.21.0.

---

## 1. Pipeline architecture (as built, not as described in early drafts)

1. **Attack simulation** — `simulator.sh` + `commands.txt`: a custom set of
   Unix shell scripts, explicitly commented with `#Atomic Test <N>` and MITRE
   technique IDs (e.g. T1003.008, T1222.002, T1087.001), referencing
   `/tmp/AtomicRedTeam/atomics/...` paths. This is an **Atomic-Red-Team-style
   custom simulator**, not Atomic Red Team itself, and not Auditbeat.
2. **Telemetry collection** — Elastic **Auditbeat v7.16.3**, configured via
   `linux_audit.rules`, which maps Linux audit-framework watches/syscall
   filters to MITRE technique keys via `-k <TECHNIQUE_ID>` (e.g.
   `-w /etc/passwd -p wa -k TT1087_Account_Discovery`,
   `-w /bin/su -p x -k T1169_Sudo`). Auditbeat forwards events to
   Elasticsearch. **Auditbeat is the collector, not the attack generator —
   these are architecturally separate components**, which is the correction
   needed for Table 1 (reviewer comment #2).
3. **Storage / extraction** — Elasticsearch `auditbeat-*` index →
   `ESSparkReader.scala` / `DataExtractorfromES.scala` (Apache Spark) flatten
   the nested JSON, rename fields (`.` → `_`, lowercase), and write Parquet.
4. **Dataset** — single Parquet file
   `part-00000-69e58b79-b94a-4504-9442-48d106d2f888-c000.snappy.parquet`,
   **226,760 rows × 251 columns** raw.
5. **Feature engineering** — `DataPreprocessor.ipynb` (+ two near-duplicate
   copies). Re-implemented faithfully but with two leakage bugs fixed (see
   §3) in `repro/build_features.py`.
6. **Models** — Keras DNN, linear SVM, Random Forest, Gradient Boosting,
   re-implemented in `repro/train_models.py` / `train_models_ablation.py`.
   MLflow is referenced in the original notebooks but no `mlruns/` directory
   exists in this repo — MLflow tracking was not actually used to produce any
   numbers that survive into this document.

---

## 2. Dataset, after the same filter the original notebooks apply

Filter: `event_action != "network_flow" and process_name != "firefox" and
process_executable != "/usr/bin/sleep" and user_selinux_user !=
"snap.notepad-plus-plus.notepad-plus-plus"`

- Raw: 226,760 rows × 251 columns
- Filtered: **86,689 rows × 251 columns**

### 2.1 Label scheme (8 classes, not 9 and not 12)

The label is derived from the first element of the `tags` array
(`tags_str`), mapped as follows. Any `tags_str` value not in this map
collapses to `Others`:

| Raw `tags_str` | Manuscript class |
|---|---|
| `` (empty) | Others |
| `exec` | exec |
| `access` | access |
| `T1166_Seuid_and_Setgid` | T1166_Seuid_and_Setgid |
| `T1087_Account_Discovery` | T1087_Account_Discovery |
| `T1169_Sudo` | T1169_Sudo |
| `T1082_System_Information_Discovery` | T1082_System_Information_Discovery |
| `T1016_System_Network_Configuration_Discovery` | T1016_System_Network_Configuration_Discovery |

**Total label distribution (n = 86,689):**

| Class | Count | % |
|---|---|---|
| access | 31,630 | 36.5% |
| Others | 25,979 | 30.0% |
| exec | 20,062 | 23.1% |
| T1166_Seuid_and_Setgid | 5,119 | 5.9% |
| T1087_Account_Discovery | 1,859 | 2.1% |
| T1169_Sudo | 1,775 | 2.0% |
| T1082_System_Information_Discovery | 259 | 0.3% |
| T1016_System_Network_Configuration_Discovery | 6 | 0.007% |

This is **5 MITRE-technique classes + 3 non-attack classes (exec, access,
Others) = 8 classes total.** It is NOT the 12-class / "9 attack technique +
3 benign" scheme described in v7's abstract and §3.3/§3.4 — that scheme does
not correspond to anything present in the actual dataset. `linux_audit.rules`
does define additional rules for T1003, T1033, T1049, T1057, T1078, T1081,
T1219, T1072, T1105, but **none of these technique IDs appear as a `tags`
value in the dataset that survives the filter** — only T1166, T1087, T1169,
T1082, T1016 do.

### 2.2 Session/run granularity

v7 describes a fine-grained `session_id` column with 985 sessions used for
`GroupShuffleSplit`. **No such column exists.** The closest candidates:

- `auditd_session`: 29 distinct values, only 58,213/86,689 rows non-null.
- `agent_ephemeral_id`: 6 distinct values, all rows non-null — corresponds to
  6 separate Auditbeat collection runs/hosts.
- `process_entity_id`: 2,840 distinct values, only 6,210/86,689 rows non-null.

None of these support an 985-session, three-way (70/10/20) GroupShuffleSplit
as described in v7. Per-`agent_ephemeral_id` inspection shows 6 runs spanning
**2022-01-26 to 2022-02-12**, with the large majority of MITRE-labelled
(non-exec/access/Others) events concentrated on **2022-01-27 and
2022-01-28**. v7's "246-hour equivalent period" framing is also not
supported by this timestamp range (~17 days wall-clock, but highly
non-uniform).

**Decision (documented deviation from v7):** because no valid session
identifier exists at the granularity v7 describes, this reproduction uses a
**time-based (chronological) 70/30 split** as the primary, leakage-resistant
protocol. This is weaker than a true session-based split (it does not
guarantee independence of overlapping processes that straddle the split
point) but is strictly stronger than the random/event-level split that
produces the leakage the reviewer flagged, and it is the strongest split
defensible from the data actually present. This must be stated explicitly and
honestly in the revised manuscript's methodology and limitations sections.

- Train: 60,682 rows, 2022-01-26 18:22:57.677 → 2022-01-28 12:07:28.865
- Test: 26,007 rows, 2022-01-28 12:07:28.873 → 2022-02-12 10:15:20.601

**Per-class counts by split** (note severe class/split imbalance for two
rare classes — flagged as a limitation):

| Class | Train | Test |
|---|---|---|
| Others | 20,106 | 5,873 |
| T1016_System_Network_Configuration_Discovery | 4 | 2 |
| T1082_System_Information_Discovery | 158 | 101 |
| T1087_Account_Discovery | 1,135 | 724 |
| T1166_Seuid_and_Setgid | 5,076 | 43 |
| T1169_Sudo | 1,075 | 700 |
| access | 19,395 | 12,235 |
| exec | 13,733 | 6,329 |

T1016 has only 6 total examples (4 train / 2 test) — any per-class metric for
this class is **not statistically meaningful** and must be reported as such.
T1166 is heavily front-loaded into train (5,076 vs 43 in test) because nearly
all T1166-labelled activity occurred on 2022-01-27, before the split point.

---

## 3. Bugs found in the original notebooks, and how they were fixed

### 3.1 Label leakage via one-hot-encoded target in the feature matrix

The original notebooks build `multi_data` by calling `pd.get_dummies()` on
the label column (`tags_str`) and then slicing `multi_data.iloc[:, 0:N]` as
the feature matrix `X`, **without excluding the new dummy columns**.
`pd.get_dummies()` inserts the one-hot columns in place of the original label
column, which sits well inside the first N columns, so `X` ends up
containing a one-hot encoding of the label itself. This is the root cause of
the ~99.98–99.99% SVM/RF/GB numbers reported in earlier drafts.

**Fix** (`repro/build_features.py`): the label (`y`) is kept completely
separate from the feature matrix (`X`) at all times; no dummy/derived label
columns are ever joined into `X`.

### 3.2 StandardScaler / CatBoostEncoder fit on the full dataset (incl. test)

The original notebooks fit `StandardScaler` and `CatBoostEncoder` on the
entire dataset before splitting. `CatBoostEncoder` additionally uses the
target column, so fitting on the full data leaks test-set label information
into the encoding of train rows that share a categorical value with test
rows.

**Fix**: scaler, encoder, and the correlation-based column-drop list are all
fit on the **train split only** and applied (not re-fit) to the test split.

### 3.3 Broken DNN architecture

The original DNN used `Dense(1, activation='sigmoid'/'softmax')` as the
output layer with `categorical_crossentropy` loss against integer-encoded
labels. A single-unit output cannot represent an 8-class softmax
distribution, and `categorical_crossentropy` against integer (non-one-hot)
labels is not a valid loss pairing — saved notebook outputs show training
loss permanently at `0.0000e+00` and accuracy of only 36.7–59.7%, i.e. the
network never learned anything.

**Fix** (`repro/train_models.py`): `Dense(n_classes, activation='softmax')`
output layer, `sparse_categorical_crossentropy` loss, which is the correct
pairing for integer class labels.

---

## 4. Feature matrix (after both fixes)

- 5 numeric columns, 99 categorical columns identified from the 86,689 × 251
  filtered/cleaned dataframe.
- Before correlation filtering: **(60,682 train rows) × 104 columns**.
- 47 columns dropped at the |r| > 0.99 correlation threshold (computed on
  train only) — includes near-duplicate ID/metadata columns
  (`user_audit_id`/`user_audit_name`, `user_saved_*`, `user_filesystem_*`,
  many `*_str` host/package metadata columns that are constant or
  near-constant across this single-host dataset).
- **Final feature matrix: 60,682 × 57 (train), 26,007 × 57 (test).**

This is the figure that replaces v7's "32 features across 58,998 events"
(Table 4) — the real numbers are **57 features across 86,689 events**
(60,682 train / 26,007 test under the time-based split).

---

## 5. Protocol A — all 57 features, time-based 70/30 split

Single consistent run, `repro/train_models.py`, results in
`repro/results/metrics.json`.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Linear SVM | 97.13% | 74.62% |
| Random Forest (100 trees) | 99.98% | 99.99% |
| Gradient Boosting (100 trees) | 98.82% | 72.61% |
| DNN (fixed architecture, 20 epochs) | 96.10% | 61.05% |

Per-class precision/recall/F1 (full classification reports and confusion
matrices in `repro/results/`):

- **SVM**: perfect on Others/T1016/T1082/T1087/access/exec, but **0.00
  precision/recall on T1166 and T1169** (both collapse to majority-adjacent
  classes).
- **Random Forest**: perfect (1.00/1.00/1.00) on every class, including
  T1166 (43 test rows) and T1016 (2 test rows) — see §6, this is suspicious.
- **Gradient Boosting**: perfect on Others/T1087/T1166/access/exec, **0.00 on
  T1016 and T1082**, and only 0.71 recall on T1169.
- **DNN**: near-perfect on Others/T1016/access/exec, 0.91 F1 on T1087, but
  **0.00 on T1166 and T1169** and essentially 0 recall (0.01) on T1082.

**Headline finding so far:** under one consistent, leakage-fixed protocol,
results are *not* a tight cluster (contrary to v7's narrative of "four of
five models within a point of each other"). SVM, GB, and DNN all collapse on
at least one of the rare classes (T1166 and/or T1169 and/or T1082/T1016), and
only Random Forest reports near-perfect scores everywhere — including on the
two classes (T1016, T1166) with the most severe train/test imbalance, which
is the opposite of what severe imbalance should produce.

---

## 6. Protocol B — identifier/run-proxy column ablation (in progress)

Random Forest's Protocol-A result (99.98% / 99.99%, perfect even on T1166's
43-row test set) is implausibly strong given GB and SVM on the *same* matrix
score far lower. Five of the 57 retained features are not behavioural
signal about *what the process did* but instead identify *which
session/run/time the event came from*:

- `auditd_session` (29 distinct values total)
- `agent_ephemeral_id` (6 distinct values — "which of the 6 collection runs")
- `auditd_sequence` (monotonically increasing per-host event counter — a
  direct proxy for time/order)
- `process_entity_id_str`, `process_start_str` (per-process-instance ID /
  start timestamp)

Because attack-technique labels cluster heavily by session/run/time window
(§2.2), a tree ensemble can use these columns to infer "which run is this
row from" and back out the label without learning anything about the
technique's audit-log signature — a second, subtler leakage channel beyond
the two fixed in §3, and one that survives a correct time-based split with
train-only-fit transformers.

`repro/train_models_ablation.py` drops these 5 columns (52 features
remaining) and re-trains all four models under the same time-based split.

### 6.1 Protocol B results

| Model | Accuracy | Macro-F1 | Δ macro-F1 vs Protocol A |
|---|---|---|---|
| Linear SVM | 97.13% | 74.62% | +0.00 |
| Random Forest | 99.98% | 99.96% | −0.03 |
| Gradient Boosting | 98.82% | 72.61% | +0.00 |
| DNN | 98.48% | **80.12%** | **+19.07** |

**Interpretation (honest, not the leakage story I expected):** dropping the
5 run/session-identifier columns leaves SVM, Random Forest, and Gradient
Boosting **essentially unchanged** (all within 0.03 points). Random Forest
remains at 99.98% accuracy / 99.96% macro-F1 with near-perfect per-class
scores, including on T1166 (43 test rows) and T1016 (2 test rows). So the
identifier-leakage hypothesis in §6 is **not the primary explanation** for
RF's near-perfect score — that hypothesis is recorded here because it was a
reasonable thing to check and the check is part of the evidence trail, but it
did not pan out as expected, and the result must be reported as such rather
than silently dropped.

The one model that *did* change substantially is the **DNN**, whose macro-F1
rose from 61.05% (Protocol A) to **80.12%** (Protocol B) — driven mainly by
T1082 (F1 0.02 → 0.68) and T1169 (F1 0.00 → 0.82). T1166 remains at 0.00 F1 in
both protocols. This suggests the DNN, specifically, was using the
run/session-identifier columns as a shortcut in Protocol A, and performs
better — though still far behind the ensembles — once forced to rely on
behavioural features.

The most defensible reading of RF's persistently near-perfect score is the
**controlled-simulation effect that v7 itself already names** (§3.3/§5.1):
because each MITRE technique was generated by a small number of repeated,
scripted command invocations, the audit-rule-triggering fields
(`file_path`, `process_executable`, `auditd_data_syscall`, etc.) take a
narrow, near-deterministic set of values per technique. A tree ensemble with
CatBoost-encoded categorical features can therefore separate these classes
almost perfectly **in this lab dataset**, even for rare classes, because
"rare" here means "few examples" but not "ambiguous examples." This is a
genuine result, not an artifact of the two leakage bugs fixed in §3 — but it
is also a strong caveat on external validity, and must be stated as such in
Limitations: near-perfect RF performance reflects the narrow, scripted
nature of this lab simulation and should not be read as evidence of
real-world detection performance.

---

## 6.2 Rule-based ATT&CK baseline (`repro/rule_baseline.py`)

Addresses reviewer comment on whether ML provides measurable benefit over
rule-based SIEM detection (Wazuh/Elastic/Splunk/Sentinel-style). Implements a
simple priority-ordered rule cascade using the SAME audit-rule logic as
`linux_audit.rules` (file-path watches for `/etc/passwd`, `/etc/shadow`,
`/etc/sudoers`, `/etc/resolv.conf`, etc., and syscall filters for
`setuid`/`setgid`/`execve`/etc.), applied to the raw `file_path`,
`process_executable`, and `auditd_data_syscall` fields — **without using the
`tags` field itself** (which is the ground-truth label and would make the
comparison circular). Evaluated on the same 26,007-row time-based test split.

**Result: accuracy = 91.25%, macro-F1 = 77.13%.**

Per-class: perfect on T1016, T1082, T1087, access; strong on Others (0.83 F1)
and exec (0.96 F1); **T1169_Sudo collapses to 0.24 precision / 1.00 recall**
(the rule for `/usr/bin/sudo`/`/bin/su`/`/etc/sudoers` over-fires on benign
sudo usage that the ML labels classify as `access`/`exec`/`Others`); **T1166
scores 0.00** (no event in this dataset has `auditd_data_syscall` literally
equal to `setuid`/`setgid`/`seteuid`/`setegid` — T1166-labelled events are
captured under a different syscall name in this data, so the naive rule never
fires).

This is a genuine, non-circular baseline: macro-F1 77.13% is **higher than
GB (72.61%) and DNN (61.05%) in Protocol A**, and within range of SVM
(74.62%). This directly supports an honest "ML as complement, not wholesale
replacement" framing for §4.4 — the simple rule cascade is already
competitive with three of the four ML models on this dataset, and where it
fails (T1169 precision, T1166 entirely) those are concrete, explainable
failure modes that a combined rule+ML approach could address.

---

## 7. Discrepancies between v7 and the reproduced pipeline (for Phase 5)

| v7 claim | Reproduced reality |
|---|---|
| 58,998 events, 985 sessions, 12 classes (9 attack + 3 benign) | 86,689 events, 8 classes (5 MITRE techniques + exec/access/Others); no 985-session structure exists in the data |
| Session-based GroupShuffleSplit, 70/10/20 (688/97/200 sessions) on `session_id` | No `session_id` column at that granularity exists; replaced with time-based 70/30 chronological split (60,682/26,007 events) |
| Final feature matrix: 32 features across 58,998 events | 57 features across 86,689 events (60,682 train / 26,007 test) |
| "246-hour equivalent" collection window | ~17 days (2022-01-26 to 2022-02-12), highly non-uniform; most MITRE-labelled activity on 2022-01-27/28 |
| Table 6: RF 99.4%/98.0%, GB 99.4%/98.0%, DT 99.3%/97.8%, SVM 94.0%/91.7%, DNN 98.4%/96.0% | Protocol A: SVM 97.1%/74.6%, RF 99.98%/99.99% (likely identifier leakage, see §6), GB 98.8%/72.6%, DNN 96.1%/61.1%. No Decision Tree was run (not in original notebooks' final model set) |
| Table 5 (DNN 9-config hyperparameter search, 88–98.8% val. acc.) | No evidence in the notebooks of a 9-configuration DNN search; the original DNN code was structurally broken (loss=0) and produced 36.7–59.7% accuracy |
| Table 4 (8-step feature pipeline, "4 columns removed: user_group_id, user_fs_id, exit_success, ppid_bracket") | Real correlation filter removes 47 columns (full list in `repro/artifacts/meta.json`), not 4 |
| §4.5 Dataset-size sensitivity analysis (10 sizes, 5k–300k events) | Not run; no code in the repo implements this. To be addressed in Phase 3/4 — either run a reduced version or remove this section and document as not performed |
| §4.7 Cross-dataset coverage analysis vs Karim et al. Linux-APT-Dataset-2024 | Not run; no code or data for this dataset exists in the repo. To be removed or rewritten as future work unless the dataset can be obtained and the comparison actually run |

---

## 8. Outstanding items

- [x] Fill in Protocol B (§6) - `train_models_ablation.py` completed; results
      in §6.1.
- [x] Repeated stratified k-fold CV (mean ± std) for Phase 3 -
      the original `repro/cv_repeated.py` (RF + GB, 5-fold x 2-repeat = 20 fits
      via two separate `cross_val_score` calls) did not complete after being
      left running for the entire prior session. The script was rewritten to a
      single-pass stratified 5-fold `cross_validate` (5 fits per model,
      `f1_macro` + `accuracy` from one fitted pipeline, results written
      incrementally to `results_cv/cv_metrics.json` after each model). Re-run:
      Random Forest's 5-fold CV completed -
      macro_f1_mean=92.4961%, macro_f1_std=6.1269% (folds: 87.49, 87.50, 100.0,
      87.49, 100.0), accuracy_mean=99.9954%, accuracy_std=0.0043% (folds:
      99.9942, 99.9942, 100.0, 99.9885, 100.0). The bimodal macro-F1 pattern
      tracks whether the rarest class (T1016, n=6) was correctly classified in
      that fold. Gradient Boosting's 5-fold CV did not complete (background run
      died silently after ~12h, no output/log, process no longer in `tasklist`)
      and is NOT reported - no fabricated numbers. `repro/make_figures.py` was
      fixed (matplotlib `tick_labels=` instead of deprecated `labels=`) and
      re-run, producing `repro/figures/cv_boxplot.png` showing only the
      RandomForest result with the mean±std annotation "92.50 +/- 6.13". A new
      §5.1 limitations paragraph ("Cross-validation variance estimate")
      reports this finding in text, including the GB-not-completed caveat. The
      old Figure 8 (5-fold CV boxplot for Decision Tree/RF/GB on a
      session-based split) remains removed from the manuscript body; the new
      `cv_boxplot.png` (RF only) is provided in the supplementary repository's
      `repro/figures/` as the artefact backing the §5.1 text, but has not been
      inserted as a numbered inline figure in v8.
- [x] Rule-based ATT&CK baseline - `repro/rule_baseline.py` completed;
      91.25%/77.13%, results in §6.2.
- [x] §4.5 and §4.7 of v7 - reproduced-from-scratch result: no code/data for
      either exists in this repo. Reframed as future work in the v8
      manuscript (option b), figures A/B/C removed.
- [x] Figure regeneration (Phase 4) - class distribution, Protocol A/B
      confusion matrices, per-class P/R/F1 bars, a model-comparison bar chart
      (`model_comparison.png`), and a CV boxplot (`cv_boxplot.png`, RF only -
      see CV item above) all done in `repro/figures/`.
- [x] Manuscript v8.docx inline figures (Phase 4) - all 9 inline
      code-screenshot images reviewed:
        - Figure 1 (attack simulator code snippet) - kept, no numeric claims.
        - Figure 3 (pipeline architecture diagram) - kept; added a note that
          the MLflow tracking box reflects the originally planned pipeline,
          not the submitted code (which writes results/ JSON directly).
        - Figure 4 (technique distribution, old: 58,998 events/12 classes) -
          replaced with `class_distribution.png` (86,689 events/8 classes).
        - Figure 5 (broken DNN architecture diagram, 1000->100->500/sigmoid)
          - removed.
        - Figure 6 (fabricated DNN training/validation curves from the
          9-config search) - removed.
        - Figure 9 (model comparison, old session-based numbers) - replaced
          with new `model_comparison.png` (Protocol A + rule baseline, real
          numbers).
        - Figure 7 (Linear SVM confusion matrix, old n=120/12-class) -
          replaced with `confmat_SVM.png` (Protocol A, real test set).
        - Figure 10 (Random Forest confusion matrix, session-based) -
          replaced with `confmat_RandomForest.png` (Protocol A, real test
          set).
        - Figure 8 (5-fold CV boxplot, Decision Tree/RF/GB) - removed (see CV
          item above).
- [x] §3.7 MLflow claim (was para 86 in earlier numbering) - grep of the
      entire project found no MLflow usage outside `.venv311` library
      dependencies. The claim "MLflow was used to log all hyperparameter
      configurations..." has been replaced with an accurate description of
      what the supplementary repo actually contains (scripts + saved
      results/figures).
- [x] Anonymous supplementary repo packaging (Phase 4) - assembled in
      `supplementary/` and zipped as `DTRAP_2025_0106_Supplementary.zip`
      (code, saved metrics/figures, REPRO_NOTES.md, README.md,
      linux_audit.rules, simulator.sh). `simulator.sh`'s one
      `/home/naveen/...` path was already changed to `/home/user/...`
      (cosmetic, not a feature value).
- [x] Anonymity redaction of the raw dataset parquet - the original
      (`part-00000-69e58b79-b94a-4504-9442-48d106d2f888-c000.snappy.parquet`)
      contained the author's real username/hostname ("naveen"/"lab1") as
      literal values in 39 columns (e.g. `user_name`, `group_name`,
      `host_hostname`, `agent_hostname`, file paths under `auditd_data_name`),
      2,773,383 cells total. `redact_pii.py` string-replaces
      naveen->user and lab1->lab across every string/list-of-string cell and
      writes a redacted copy. The full pipeline (`build_features.py`,
      `train_models.py`, `train_models_ablation.py`, `rule_baseline.py`) was
      re-run against the redacted file in an isolated directory and compared
      to the original results: SVM, Random Forest, Gradient Boosting, and the
      rule-based baseline are bit-identical to 4 decimal places in both
      Protocol A and B; the DNN (which is not seeded against
      CatBoostEncoder/TF internal nondeterminism) shows the same order of
      run-to-run variance it already exhibits between any two unseeded runs
      (Protocol A: 96.10%/61.05% -> 97.29%/60.11%; Protocol B:
      98.48%/80.12% -> 99.05%/85.28%), not a redaction-induced change. The
      redacted parquet is the one published in the public GitHub repository
      (`molefimcm/dtrap-ml-research`) as `data/auditbeat_dataset.parquet`.
- [x] Manuscript v8.docx - Abstract, Table 1 (data-generation/MLflow rows),
      Tables 2-6 (docx Table[0]-[5]), Introduction contributions/RQs/keywords,
      §3.4/3.6/3.7, §4.1-4.4, §4.5/§4.7 reframe, §5.1/§5.2, §6/§6.1, and all 9
      inline figures updated with reproduced numbers/plots. A stale duplicate
      "Revised: May 2026" line was also removed. Phase 5 text revisions are
      complete.
- [x] Phase 6: point-by-point cover letter - `DTRAP_2025_0106_Cover_Letter.docx`
      written, responding to all 8 reviewer comments with specific references
      to the sections/tables/figures that address each one.

## 9. Final deliverables for this revision

- `DTRAP_2025_0106_Revised_Manuscript_v8.docx` - revised manuscript.
- `DTRAP_2025_0106_Cover_Letter.docx` - point-by-point response to reviewer.
- `DTRAP_2025_0106_Supplementary.zip` (built from `supplementary/`) - code,
  saved results/metrics, figures, REPRO_NOTES.md, README.md.
- This `REPRO_NOTES.md` - full provenance record for every number in v8.
- Public code/data repository: https://github.com/molefimcm/dtrap-ml-research
  (redacted dataset, all repro scripts, saved results and figures).

## 10. Known remaining gaps (disclosed honestly in the manuscript itself)

- Stratified 5-fold CV completed for Random Forest (macro-F1 92.50% ± 6.13%,
  accuracy 99.996% ± 0.004%) and is reported in §5.1; Gradient Boosting's CV did
  not complete in time and is not reported - no fabricated numbers,
  `cv_repeated.py` provided for completion in a future revision. SVM and DNN
  were not repeated for runtime reasons (stated in §5.1).
