# House Price Prediction with Linear Regression (Ames, Iowa)

A machine learning project that predicts house sale prices from 80 property features, using **Linear, Ridge and Lasso regression** on the Ames Housing dataset. The final model explains about **93% of the variation in prices** on houses it has never seen, with an average error of about **$13,000 (roughly 7% of the average price)**.

This is my first data science project. It covers the full workflow: exploring the data, handling outliers and missing values, encoding categories, comparing models with cross-validation, diagnosing the results, and saving the final model.

---

## Table of contents

1. [Dataset](#dataset)
2. [Project workflow](#project-workflow)
3. [Results](#results)
4. [How to read the results](#how-to-read-the-results)
5. [What drives the price](#what-drives-the-price)
6. [Limitations](#limitations)
7. [Possible improvements](#possible-improvements)
8. [How to run](#how-to-run)
9. [Using the saved model](#using-the-saved-model)
10. [Tech stack](#tech-stack)
11. [Reference](#reference)

---

## Dataset

The **Ames Housing dataset** contains residential home sales in Ames, Iowa between 2006 and 2010.

| | |
|---|---|
| Rows | 2,930 houses |
| Columns | 81 (36 numeric after cleaning, 40 categorical) |
| Target | `SalePrice` (USD) |
| Average price | about $180,000 (median $160,000) |

The features describe the lot (size, shape, frontage), the building (quality, condition, year built, living area, basement, garage, porches, pool) and the sale itself (type, condition, month, year).

---

## Project workflow

### 1. Exploration and outlier detection

- Looked at the correlation of each numeric feature with `SalePrice`. The strongest are `Overall Qual` (0.80), `Gr Liv Area` (0.73), `Total Bsmt SF` (0.66) and `Garage Cars` (0.65).
- Scatter plots of `SalePrice` against `Overall Qual` and `Gr Liv Area` revealed **three houses** (indices 1498, 2180, 2181) that are very large and high quality but sold for far less than expected.
- All three have `Sale Condition = Partial`: new homes sold before construction was fully assessed. They don't reflect normal market behavior and would distort a linear fit, so I removed them (2,930 to 2,927 rows).

### 2. Missing values

The right way to fill a missing value depends on what the missing value means:

| Situation | Treatment |
|---|---|
| `NaN` means "this house doesn't have it" (basement, garage, masonry veneer, fireplace) | Categorical columns filled with `"None"`, numeric columns with `0` |
| Only 1 to 2 rows missing in `Electrical` and `Garage Cars` | Dropped those rows |
| `Lot Frontage` (about 17% missing) | Filled with the **mean of the same neighborhood**, since lot size depends on location |
| `Pool QC`, `Misc Feature`, `Alley`, `Fence` (mostly empty) | Dropped the columns |
| `PID` (parcel ID) | Dropped, since it carries no information about price |

Final dataset after cleaning: **2,925 rows**.

### 3. Encoding categorical data

- `MS SubClass` is stored as numbers (20, 30, 60...) but they are only labels: 60 is not "more" than 20. I converted it to text so it is treated as a category.
- All categorical columns were **one-hot encoded** with `drop_first=True` to avoid redundant columns.
- The result is **273 features**.

### 4. Train / test split

- 80% training (2,340 houses), 20% testing (585 houses), with a fixed `random_state` for reproducibility.
- The test set is used **once**, at the end. Model selection uses cross-validation on the training set only.

### 5. Log-transforming the target

`SalePrice` is right-skewed (skewness **1.74**): a few very expensive houses stretch the tail. Linear regression works best when errors have a similar spread everywhere, so I trained on `log(1 + SalePrice)` (skewness **-0.03**, essentially symmetric). Errors on the log scale are roughly percentage errors, and predictions are converted back to dollars with the inverse function.

### 6. Models and cross-validation

Four models were compared with **5-fold cross-validation**:

| Model | Purpose |
|---|---|
| Baseline (always predicts the mean) | A floor that any real model must beat |
| Linear Regression | Plain least squares, no regularization |
| Ridge (`RidgeCV`) | L2 penalty: shrinks all coefficients, handles correlated features well |
| Lasso (`LassoCV`) | L1 penalty: can set coefficients to exactly 0, so it also selects features |

Each model sits in a **pipeline with `StandardScaler`**. Scaling matters for Ridge and Lasso because their penalty depends on the size of the coefficients, which depends on each feature's units. Putting the scaler inside the pipeline means it is fitted only on the training part of each fold, which prevents data leakage. Ridge and Lasso choose their own regularization strength (`alpha`) by internal cross-validation.

### 7. Final model

The chosen model (Ridge, selected by cross-validation) was refit on **all the data** and saved with `joblib`. It is wrapped in a `TransformedTargetRegressor`, so the saved model takes features and returns prices **in dollars** directly.

---

## Results

### Cross-validation (training set, log scale)

| Model | CV RMSE (log) | Std across folds |
|---|---|---|
| **Ridge** | **0.1204** | 0.0096 |
| Lasso | 0.1205 | 0.0102 |
| Linear | 0.1279 | 0.0145 |
| Baseline (mean) | 0.4123 | 0.0146 |

### Test set (585 unseen houses, in dollars)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | $12,826 | $19,393 | 0.937 |
| Lasso | $13,326 | $19,887 | 0.934 |
| Ridge | $13,264 | $20,146 | 0.932 |
| Baseline (mean) | $54,894 | $79,320 | -0.055 |

Lasso kept **120 of the 273 features** and set the other 153 to zero.

---

## How to read the results

**What the metrics mean**

- **MAE (mean absolute error):** the average size of the miss. An MAE of about $13,000 on houses averaging $180,000 is roughly **7%**.
- **RMSE:** like MAE, but big misses count more. Here it is about $20,000, noticeably higher than MAE, so a few houses are predicted much worse than the rest.
- **R²:** the share of price variation the model explains. 0.93 means 93%.
- **CV RMSE on the log scale:** a value of 0.12 corresponds to a typical error of about 12 to 13%.

**What the numbers tell us**

1. **The models clearly learn.** The error falls from 0.41 (guessing the average) to 0.12, a reduction of about 70%.
2. **Ridge and Lasso are tied.** Their difference (0.0001) is about 100 times smaller than the fold-to-fold variation (about 0.01), so neither is meaningfully better.
3. **Regularization helps only modestly.** Linear is slightly worse in cross-validation and has the most unstable scores across folds, which is expected with 273 features and no penalty.
4. **Linear scored best on the test set, but I did not choose it for that reason.** The gaps are small (about $400 to $500 in MAE) and the test set has only 585 houses, so a few unusual sales can reorder models this close. Choosing the model by its test score would turn the test set into part of the tuning and make the final result optimistic. The model was chosen using cross-validation only.
5. **Baseline R² is negative on purpose.** It always predicts the training mean, which differs a little from the test mean, so it does slightly worse than "perfect knowledge of the average".

**Residual analysis (test set)**

- **Actual vs predicted:** points follow the diagonal across the whole range. The most expensive houses ($500k+) are slightly underpredicted, which is common because there are few of them to learn from.
- **Residuals vs predicted:** a random cloud around zero, with no curve and no funnel. The model is not missing a non-linear pattern, and the log transform kept the error spread even for cheap and expensive houses.
- **Residual distribution:** bell-shaped and centered on zero, with most errors within about ±20%. The left tail is heavier: a small number of houses sold for much less than predicted. These are what push RMSE above MAE.

---

## What drives the price

Lasso coefficients are on standardized features, so they show the change in log price for a one-standard-deviation increase in the feature (roughly a percentage change).

| Feature | Approx. effect of +1 std. dev. |
|---|---|
| `Gr Liv Area` (living area) | about +14% |
| `Overall Qual` (material and finish quality) | about +9% |
| `Year Built` | about +5% to 6% |
| `Overall Cond`, `Total Bsmt SF` | about +4% each |

Next in importance are basement finished area, garage capacity, normal sale condition, new construction, and a few neighborhoods. The ranking matches real-world intuition: size, quality, age, condition and basement lead.

Notes on interpretation:

- Coefficients show **association, not causation**. They do not say that renovating a house raises its price by a fixed percentage.
- When features are strongly correlated (for example `Garage Cars` and `Garage Area`), Lasso tends to keep one and drop the other, so a feature missing from the list is not necessarily unimportant.
- For dummy variables, the coefficient is relative to the category that was dropped by `drop_first=True`.

---

## Limitations

- **Depends on a human rating.** `Overall Qual` is probably assigned by an appraiser and already summarizes many other features. Any new house needs this rating, so the model is not fully automatic.
- **Local and dated.** The data covers one city between 2006 and 2010. The model should not be used to price houses in other markets or time periods.
- **Preprocessing is outside the saved model.** Cleaning and one-hot encoding happen in the notebook, not inside the `.pkl` file, so new data must be prepared the same way.
- **Simple filling choices.** `Garage Yr Blt` is filled with `0` for houses without a garage and unmatched `Lot Frontage` values with `0`. For a linear model these artificial values can distort coefficients.
- **Imputation before the split.** Neighborhood means for `Lot Frontage` were computed on the full dataset, which leaks a very small amount of information from the test set. The effect is likely tiny but not zero.
- **One train/test split.** The test score depends partly on which 585 houses ended up in the test set.
- **Linear model assumptions.** The model assumes mostly additive effects. It cannot capture interactions such as "a large area is worth more in a high-quality house".

---

## Possible improvements

- Put cleaning, imputation and encoding in a scikit-learn `Pipeline` with a `ColumnTransformer`, so one saved file accepts raw input and nothing leaks across the split.
- Fill `Garage Yr Blt` with `Year Built` (or drop it) and use the overall median for leftover `Lot Frontage` values, then compare the CV scores.
- Log-transform skewed features such as `Lot Area`.
- Investigate the houses with the largest residuals (for example by `Sale Condition`) to see whether they are unusual sales.
- Try ElasticNet and non-linear models (Random Forest, Gradient Boosting) and compare them with the linear baseline.

---

## How to run

```bash
git clone <your-repo-url>
cd <your-repo>
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook House_Price_Linear_Regression.ipynb
```

Place the dataset where the notebook expects it (`DATA/RAW DATA/Ames_Housing_Data.csv`) and adjust the path in the first data-loading cell if your folder layout is different.

---

## Using the saved model

The saved model returns prices in dollars. New data must be cleaned the same way as the training data (missing values filled, `MS SubClass` as text, no `SalePrice` column) and then aligned to the training columns:

```python
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")

new_X = pd.get_dummies(new_df_clean)                      # same cleaning as training data
new_X = new_X.reindex(columns=columns, fill_value=0)      # add missing dummies, fix column order
print(model.predict(new_X))
```

Only load `.pkl` files you trust, and use the same scikit-learn version that saved the file (`pip freeze` shows yours).

---

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib.

---

## Reference

De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3).

---

## Author

**[DAHMANI Mohamed Anis]**: [LinkedIn : Mohamed Anis DAHMANI / email : dahmani.med.anis@gmail.com / portfolio link : letter]
