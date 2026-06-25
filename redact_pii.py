"""
Redact author PII ("naveen" username, "lab1" hostname) from the raw Auditbeat
parquet before it goes into the public GitHub repo. Replaces literal
substrings inside every string and list-of-string cell, across all columns,
case-sensitive (no capitalized variants were found in the data). Writes a new
parquet; the original is left untouched.
"""
import pandas as pd

SRC = 'part-00000-69e58b79-b94a-4504-9442-48d106d2f888-c000.snappy.parquet'
DST = 'part-00000-69e58b79-b94a-4504-9442-48d106d2f888-c000.snappy.redacted.parquet'

REPLACEMENTS = [('naveen', 'user'), ('lab1', 'lab')]


def redact_value(v):
    if isinstance(v, str):
        for old, new in REPLACEMENTS:
            v = v.replace(old, new)
        return v
    if isinstance(v, list):
        return [redact_value(x) for x in v]
    if hasattr(v, 'tolist'):
        return redact_value(v.tolist())
    return v


df = pd.read_parquet(SRC, engine='pyarrow')

n_changed_cells = 0
for col in df.columns:
    if df[col].dtype != object:
        continue
    before = df[col]
    after = before.apply(redact_value)
    changed = (before.astype(str) != after.astype(str))
    n = changed.sum()
    if n:
        print(f'{col}: {n} cells changed')
        n_changed_cells += n
        df[col] = after

print(f'Total cells changed: {n_changed_cells}')
df.to_parquet(DST, engine='pyarrow')
print(f'Wrote {DST}')
