# AMR Lens

A dashboard of antimicrobial resistance by country, pathogen and antibiotic,
built on the WHO GLASS 2017–2020 data compiled by Leclerc (2022). Reported
values are shown as a hospital-style antibiogram; unreported squares can be
filled with a scikit-learn estimate.

**Model task:** estimate a *missing surveillance cell*, i.e. the % resistant for a
country × pathogen × antibiotic × specimen that was not reported.
It is not a forecaster and not a clinical tool.

## Results (grouped 5-fold CV, whole cells held out)

| Model | MAE (pct points) | MAE, cells ≥100 isolates | R² |
|---|---|---|---|
| Global average (DummyRegressor) | 23.8 | 22.6 | 0.00 |
| Additive baseline (Ridge on one-hot: country + bug–drug) | 10.7 | 9.3 | 0.72 |
| Random forest | 9.2 | 7.6 | 0.76 |

The forest beats the additive baseline, but modestly. Year contributes almost
nothing to predictions (permutation importance +0.6 vs +12.6 for antibiotic).

## Project layout

```
train.py              # local: load data, evaluate, save model + dashboard data
api/index.py          # FastAPI: POST /api/predict  (Vercel Python Function)
api/model.joblib      # trained models (additive + forest)
public/index.html     # dashboard (no build step, no JS framework)
public/data.json      # data exported by train.py
serve_local.py        # run everything locally
vercel.json           # serves public/, routes /api/* to api/index.py
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py          # optional: model + data are already included
python serve_local.py    # open http://127.0.0.1:8000
```

## Deploy to Vercel

1. Push this folder to a GitHub repository.
2. In Vercel: **Add New → Project**, import the repo, keep Framework Preset **Other**.
   `vercel.json` sets the output directory to `public`.
3. Deploy. Check `https://<your-app>.vercel.app/api/health` returns `{"ok": true, ...}`.

Or from the command line: `npm i -g vercel && vercel` in this folder.

**Version pinning matters.** `model.joblib` was saved with scikit-learn 1.8.0 and
numpy 2.4.4 on Python 3.12. `requirements.txt` and `.python-version` pin these. If you
retrain with different versions, update `requirements.txt` to match, or the model
may fail to load on Vercel.

**If `/api/health` returns 404:** open the deployment's Functions tab in Vercel to
confirm `api/index.py` was built. The API answers on both `/api/...` and `/...`
paths, so prefix handling should not be the problem.

## Known limitations

- **Country coverage is biased by year.** 2019 has 71 countries; 2017, 2018 and 2020
  only the 46 that reported every year. Compare years within a country only.
- **2020 coincides with COVID-19** disruption to testing and surveillance.
- **Surveillance isolates skew to hospitalised, sicker patients**, so % resistant is
  not community prevalence.
- **Data ends in 2020.** Newer GLASS releases may exist; check WHO before relying on this.
- **Estimates for unreported cells carry roughly ±9 points of typical error**, more
  where the two models disagree.
- `TotalSpecimenIsolates` was dropped: it is aggregated across pathogens and
  antibiotics, so it is not a sample size for any bug–drug pair.
- `Resistant` and `InterpretableAST` are never model inputs (they define the target).

## Data and licence

Leclerc, Q.J. (2022). *Compiled data from WHO GLASS 2022 report* (v2.0).
doi:10.5281/zenodo.7486150. Original data: WHO GLASS reports 2021 and 2022.
Licensed CC BY-NC-SA 3.0 IGO. Because of the ShareAlike term, the data outputs
of this project (`public/data.json`, the model) are shared under the same licence.
WHO is not responsible for this compilation or app.
