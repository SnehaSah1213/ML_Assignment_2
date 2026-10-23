# ML Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

Roll number: M26CSA008 (seed 26008). Bagging and SAMME AdaBoost are implemented from scratch; only `DecisionTreeClassifier` comes from scikit-learn.

## Layout
| path | content |
|---|---|
| `src/assignment2.py` | complete code for Q1, Q2, Q3 (the run that produced the report) |
| `notebooks/assignment2.ipynb` | same code as Colab cells |
| `split/split.npz` | saved 6/2/2 split (train/val/test indices into the Olivetti array) |
| `configs/settings.json`, `configs/selected.json` | fixed settings and the choices made on validation accuracy |
| `results/tables/*.csv`, `results/figures/*.png` | final results of the run used in the report |
| `logs/final_tables.txt` | all final tables in one text file |
| `scripts/` | `check_split.py`, `make_figures.py`, `summarize_results.py` |
| `requirements.txt` | Python dependencies |

## Reproduce (Google Colab, GPU)
1. Runtime > Change runtime type > T4 GPU.
2. Run in a cell:
```
!git clone https://github.com/Givhel/ml-assignment2-bagging-boosting.git
%cd ml-assignment2-bagging-boosting
!pip install -q -r requirements.txt
!python src/assignment2.py
```
The run takes about 1 hour (Q3 with 30 refits is the slow part, tree fitting is CPU-bound). It writes tables, figures and `split.npz` to `out/` and `assignment2_outputs.zip`. Compare `out/*.csv` with `results/tables/*.csv`.

## Reproduce (local machine)
```
git clone https://github.com/Givhel/ml-assignment2-bagging-boosting.git
cd ml-assignment2-bagging-boosting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/assignment2.py
```
Without a GPU the vote aggregation falls back to CPU automatically.

## Checks and figure scripts (no GPU, no dataset download)
```
python scripts/check_split.py        # rebuilds the split from seed 26008 and compares it with split/split.npz
python scripts/make_figures.py       # redraws table-based figures into results/figures_regen/
python scripts/summarize_results.py  # rewrites logs/final_tables.txt from results/tables
```
`q1d_cases`, `q2c_heavy_face` and `q2d_face` need face images and per-round boosting weights, so they are drawn inside `src/assignment2.py`.

## Notes
- Models are trained on clean images only; the masked test set zeroes rows and columns 24 to 39.
- Settings are hard-coded in `src/assignment2.py`; `configs/settings.json` documents them.
- Q3(b) depths are 2, 8 and None because the selected tree (`max_depth=None`) is already fully grown.
- Library versions of the final run: record them with `pip freeze | grep -iE "numpy|pandas|scikit|matplotlib|joblib|torch"` (see `logs/environment.txt` if present).
