"""
EDA & Preprocessing — Titanic Survival Prediction
---------------------------------------------------
Loads the raw Titanic dataset, explores it, cleans it, engineers
features, and saves a model-ready CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/titanic_raw.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 2. EDA — save a few key plots for the README / portfolio
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

sns.countplot(data=df, x="survived", ax=axes[0, 0])
axes[0, 0].set_title("Survival Count (0 = Died, 1 = Survived)")

sns.barplot(data=df, x="pclass", y="survived", ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Passenger Class")

sns.barplot(data=df, x="sex", y="survived", ax=axes[1, 0])
axes[1, 0].set_title("Survival Rate by Sex")

sns.histplot(data=df, x="age", hue="survived", multiple="stack", bins=30, ax=axes[1, 1])
axes[1, 1].set_title("Age Distribution by Survival")

plt.tight_layout()
plt.savefig("assets/eda_overview.png", dpi=130)
plt.close()
print("Saved assets/eda_overview.png")

# Correlation heatmap (numeric features only)
plt.figure(figsize=(7, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png", dpi=130)
plt.close()
print("Saved assets/correlation_heatmap.png")

# ---------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------
# 'deck' has ~77% missing -> drop. 'embark_town' duplicates 'embarked'.
# 'alive' duplicates 'survived' (string version) -> drop (data leakage risk).
# 'who'/'adult_male' are derived from age+sex -> keep 'who' as a feature, drop 'adult_male'.
df = df.drop(columns=["deck", "embark_town", "alive", "adult_male", "class"])

# Age: impute with median grouped by (pclass, sex) — more accurate than a single global median
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))

# Embarked: only 2 missing — impute with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Fare: 0 missing in this dataset version, but guard anyway
df["fare"] = df["fare"].fillna(df["fare"].median())

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------
# Family size & "is alone" — known to matter for survival (small families fared better)
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

# Fare per person — reduces class/fare collinearity
df["fare_per_person"] = df["fare"] / df["family_size"]

# Age bands — survival isn't linear in age (children prioritized)
df["age_band"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["child", "teen", "young_adult", "adult", "senior"])

# ---------------------------------------------------------------
# 5. Encode categoricals
# ---------------------------------------------------------------
df_model = pd.get_dummies(
    df,
    columns=["sex", "embarked", "who", "age_band"],
    drop_first=True
)

# Drop the raw 'alone' bool column (duplicate of is_alone) if present
if "alone" in df_model.columns:
    df_model = df_model.drop(columns=["alone"])

df_model.to_csv("data/titanic_model_ready.csv", index=False)
print("\nSaved data/titanic_model_ready.csv — shape:", df_model.shape)
print("Final columns:", df_model.columns.tolist())
