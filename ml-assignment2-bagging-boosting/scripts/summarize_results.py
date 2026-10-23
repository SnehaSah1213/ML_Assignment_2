"""Print every result table into logs/final_tables.txt (human-readable copy of the final run)."""
import glob, os, json, pandas as pd
pd.set_option('display.width', 250, 'display.max_columns', 50)
with open('logs/final_tables.txt', 'w') as f:
    f.write('selected configuration:\n' + json.dumps(json.load(open('configs/selected.json')), indent=1) + '\n')
    for p in sorted(glob.glob('results/tables/*.csv')):
        f.write(f'\n== {os.path.basename(p)}\n' + pd.read_csv(p).round(4).to_string() + '\n')
print('wrote logs/final_tables.txt')
