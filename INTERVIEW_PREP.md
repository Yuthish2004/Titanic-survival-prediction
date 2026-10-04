# Interview Prep — Explaining This Project In Depth

Use this to prepare for anyone — recruiter, interviewer, or curious
peer — asking you to walk through this project. Read it, don't
memorize it word-for-word; explain it in your own words.

---

## 1. The 30-Second Pitch (say this first, always)

> "I built an end-to-end machine learning project predicting Titanic
> passenger survival. I did exploratory data analysis to find what
> actually drove survival — mainly sex and passenger class — engineered
> a few new features like family size, trained and compared Logistic
> Regression against Random Forest using cross-validation, and deployed
> the winning model as a live Streamlit web app where anyone can input
> passenger details and get a prediction."

---

## 2. "Walk me through your process."

1. **Loaded the data** — 891 passengers, 15 raw columns, from the
   classic Titanic dataset (via seaborn's built-in loader).
2. **EDA first, before touching the model** — I plotted survival rate by
   class, sex, and age to understand what actually mattered, rather than
   throwing every column into a model blindly.
3. **Cleaned missing data** — `age` was missing for ~20% of passengers.
   Instead of filling with a single global median (which ignores that a
   1st-class woman's typical age differs from a 3rd-class man's), I
   imputed per `(pclass, sex)` group — a more accurate estimate.
4. **Engineered features** — added `family_size`, `is_alone`,
   `fare_per_person`, and `age_band` because raw `sibsp`/`parch` alone
   don't capture the "traveling alone vs. with a small family" effect
   that the EDA hinted at.
5. **Trained two models** — Logistic Regression (simple, interpretable
   baseline) and Random Forest (captures non-linear interactions), with
   5-fold cross-validation and a grid search over Random Forest's
   hyperparameters.
6. **Evaluated properly** — not just accuracy. Precision, recall, F1,
   ROC-AUC, and a confusion matrix, because this is a survival/death
   prediction — the *type* of error matters, not just the error rate.
7. **Picked the winner by F1**, not accuracy, and explain why (next
   section).
8. **Deployed it** — wrapped the trained model in a Streamlit app so it's
   not just code sitting in a repo; anyone can interact with it live.

---

## 3. "Why did you choose F1-score over accuracy to pick your model?"

Accuracy can be misleading when classes aren't balanced — here, ~62%
died and ~38% survived, so a model that *always* predicts "died" would
still get ~62% accuracy while being useless. F1 balances **precision**
(of the people I predicted would survive, how many actually did?) and
**recall** (of the people who actually survived, how many did I catch?)
— so it penalizes a model that's lazily biased toward the majority class.

---

## 4. "Why Logistic Regression over Random Forest, if Random Forest is
usually more powerful?"

Two reasons, and both are legitimate answers:

- **Logistic Regression scored higher** on this dataset specifically —
  F1 of 0.788 vs. 0.750, and ROC-AUC of 0.869 vs. 0.854. With only ~900
  rows and a handful of strongly linear signals (sex, class), a simpler
  model generalizes better here. Random Forest's extra flexibility can
  overfit on small datasets — the cross-validation scores confirmed
  Logistic Regression was also more *stable* across folds.
- **Interpretability matters for this use case.** A survival predictor
  that can say "this prediction is driven mainly by sex and class" is
  more trustworthy and explainable than a black-box ensemble — in a
  real-world risk-prediction setting (insurance, medical triage,
  etc.), being able to explain *why* often matters as much as the score.

*(Be honest if asked "did you try tuning Random Forest harder?" — yes,
via GridSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf` —
and it still didn't beat Logistic Regression on this dataset. That's a
legitimate, common real-world outcome — bigger models don't always win.)*

---

## 5. "What is a confusion matrix, and what does yours show?"

A confusion matrix breaks predictions into four buckets:
- **True Positive**: predicted survived, actually survived
- **True Negative**: predicted died, actually died
- **False Positive**: predicted survived, actually died (a "false
  hope" error)
- **False Negative**: predicted died, actually survived (a "missed
  them" error)

My model's recall on the "survived" class was 0.75 — meaning it
correctly identified 75% of actual survivors, missing about 1 in 4. In
a real risk-prediction setting, which error type is worse depends on
context (e.g., in medical screening, missing a positive case is usually
worse than a false alarm) — worth mentioning you understand this
tradeoff exists, even if this project didn't need to optimize for one
specific error type.

---

## 6. "What's ROC-AUC and why did you plot it?"

ROC-AUC measures how well the model separates the two classes across
*all* possible decision thresholds, not just the default 0.5 cutoff.
An AUC of 0.869 means: if I randomly pick one passenger who survived
and one who died, the model ranks the survivor's predicted probability
higher about 87% of the time. It's a more threshold-independent way to
compare models than accuracy alone.

---

## 7. "Why did you engineer `family_size` and `fare_per_person` instead
of just using the raw columns?"

The EDA showed survival wasn't simply "more family = better" or
"higher fare = better" — it was roughly a **sweet spot**: solo
travelers and very large families both fared worse than small families
(2-4 people). A model trained only on raw `sibsp`/`parch` can't easily
learn that non-linear "sweet spot" pattern unless it's a complex model;
explicitly creating `family_size` and `is_alone` makes that signal easy
for even a simple Logistic Regression to pick up. `fare_per_person`
exists because raw `fare` is partly just a proxy for family size (a
family of 5 paid for 5 tickets) — dividing it out isolates the
"wealth per individual" signal from the "group size" signal, reducing
collinearity between features.

---

## 8. "How would you improve this if you had more time?"

Good, honest answers:
- Try gradient-boosted models (XGBoost/LightGBM) — often stronger than
  Random Forest on tabular data
- Add SHAP values for per-prediction explainability (not just global
  feature importance)
- Engineer a feature from the `name`/`title` field (Mr./Mrs./Miss/Master
  correlates with age and social status — a well-known trick on this
  dataset I deliberately kept simple here)
- More rigorous hyperparameter search (Optuna/Bayesian search instead
  of grid search)
- Proper train/validation/test split (3-way) if this were going into
  production, rather than just train/test with cross-validation

---

## 9. "Is this just a tutorial project? What makes it yours?"

Be honest and confident: Titanic is a well-known **learning dataset** —
that's not a weakness to hide, it's a legitimate entry point used in
official scikit-learn and Kaggle tutorials precisely because the data
is clean enough to focus on *process* rather than data-wrangling pain.
What makes it yours is the actual decisions you made and can defend:
the grouped-median imputation, the specific engineered features, the
F1-over-accuracy reasoning, choosing the simpler model when it
genuinely performed better, and deploying it as a working app instead
of stopping at a notebook. Be ready to say what you'd do differently on
a messier, real-world dataset — that shows you understand this was a
controlled first step, not the ceiling of your ability.

---

## 10. Quick-reference numbers (memorize these 4)

| Metric | Logistic Regression (chosen) |
|---|---|
| Accuracy | 84.4% |
| Precision | 82.5% |
| Recall | 75.4% |
| F1 / ROC-AUC | 0.788 / 0.869 |
