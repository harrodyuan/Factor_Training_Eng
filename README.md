# Factor Training (English)

English lecture notes derived from the Factor Training series.

## Contents

- `lecture_notes/part22_feature_engineering.ipynb`
- `lecture_notes/part22_feature_engineering.tex`
- `lecture_notes/part23_feature_selection_validation.ipynb`
- `lecture_notes/part23_feature_selection_validation.tex`
- `lecture_notes/part24_signal_construction_long_short.ipynb`
- `lecture_notes/part24_signal_construction_long_short.tex`
- `lecture_notes/part25_factor_exposures_risk_adjustment.ipynb`
- `lecture_notes/part25_factor_exposures_risk_adjustment.tex`
- `lecture_notes/lecture_utils.py`
- `lecture_notes/backtest_utils.py`

The Part 22 notebook includes a LOBSTER microstructure example with feature engineering and prediction.
The Part 23 notebook continues with feature screening, combined ranking, stability checks, and temporal validation.
The Part 24 notebook moves from selected features to cross-sectional signals and an illustrative long-short portfolio.
The Part 25 notebook studies factor exposures, rolling risk adjustment, and residualized signals.

## Notes

LOBSTER sample data is not included in this repo. Provide your own licensed files and update paths if needed.

Lecture-note HTML/PDF exports are generated with `tools/export_lecture_note.py`.
Shared notebook helpers are split between `lecture_utils.py` for data/feature support and `backtest_utils.py` for ranking, portfolio construction, and factor-regression support.
