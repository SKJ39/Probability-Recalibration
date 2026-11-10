# Topic 20 — November 9 baseline checkpoint (GridSearchCV)

This is the standalone **Spambase checkpoint** that generates the numbers in `baseline.txt`. It intentionally preserves the earlier professor-aligned experiment: seed 2026, the original four-value C grid and the original saved validation predictions. **It does not change the final paper or the final-test analysis.**

## Run

From this folder in Python >= 3.10:

```bash
python -m pip install -r requirements.txt
python -m src.verify_validation
python -m src.baseline_interval
python -m src.study --dataset spambase --draws 50
```

`src.study.fit_models()` uses an ordinary `LogisticRegression` inside a complete `Pipeline` with `StandardScaler`, with **external `GridSearchCV`** and **`StratifiedGroupKFold(5)`**, so each CV training fold fits its own scaler. The original grid is `{0.01, 0.1, 1, 10}`, selected C is 1.0. We deliberately did not substitute an 11-point grid because it would require a fresh experiment and could change the already documented results.

`python -m src.verify_validation` verifies the saved predictions, group-disjoint partitions, and calibration outputs. `python -m src.baseline_interval` calculates the Brier score, log loss, cost, and 95% group bootstrap interval based on 10,000 group resamples. `python -m src.study ...` regenerates the Spambase validation experiment and figures. The saved outputs provide a comparison point for reproduction.

## Contents

- `baseline.txt` — Canvas plain-text checkpoint, fewer than 500 words in the body, exact headings.
- `baseline.png` — optional Nov 9 figure, cost changes for three professor-requested cost ratios.
- `src/` — runnable reproducibility, baseline interval, and verification scripts.
- `data/raw/spambase.zip` — original UCI Spambase data; see https://archive.ics.uci.edu/dataset/94/spambase (CC BY 4.0).
- `results/validation/spambase/` — saved validation predictions and calculations; **no final-test outcomes** in this package.

## Before submitting

1. Confirm the official Canvas project-group label is **CalibraLab**.
2. Confirm the third member's first-name spelling: the supplied baseline text says **Ashir**, while older project records say **Ashish**. This package preserves the name from the most recently supplied baseline text.
3. Upload the extracted source and dataset to GitHub, confirm commands run, and replace the `## Code` placeholder with the **real, accessible repository URL**. Do not use the earlier teammate Colab URL as the source for these different scores.
4. Submit `baseline.txt` and optional `baseline.png` to Canvas. The word limit excludes headings. Do not submit this ZIP as the Canvas baseline unless your instructor explicitly requests it.
5. Historical note: the larger final project already scored its nominal test set. The Nov 9 checkpoint reports validation only, and cannot reverse that history.
