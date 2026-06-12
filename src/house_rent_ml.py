"""
Projet 4 — House Rent Prediction
Auteur : Zekerya Ethmane Sow (C34636)
Pipeline ML complet : EDA, Preprocessing, Modeles, Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# 1. CHARGEMENT & EDA
# ══════════════════════════════════════════════════
df = pd.read_csv('House_Rent_Dataset.csv')
print("=" * 60)
print("ANALYSE EXPLORATOIRE")
print("=" * 60)
print(f"Forme initiale : {df.shape}")
print(f"\nTypes :\n{df.dtypes}")
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
print(f"\nStatistiques Rent :\n{df['Rent'].describe()}")
print(f"\nVilles : {df['City'].unique().tolist()}")
print(f"Skewness Rent : {df['Rent'].skew():.2f}")

# Visualisation EDA
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
df['Rent'].hist(ax=axes[0,0], bins=50, color='steelblue')
axes[0,0].set_title('Distribution Loyers')
df.groupby('City')['Rent'].mean().plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title('Loyer moyen par Ville')
df.groupby('Furnishing Status')['Rent'].mean().plot(kind='bar', ax=axes[0,2])
axes[0,2].set_title('Loyer par Meublage')
df.groupby('BHK')['Rent'].mean().plot(kind='bar', ax=axes[1,0], color='mediumseagreen')
axes[1,0].set_title('Loyer par BHK')
df[['BHK','Rent','Size','Bathroom']].corr()['Rent'].drop('Rent').plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Correlations avec Rent')
np.log1p(df['Rent']).hist(ax=axes[1,2], bins=50, color='mediumpurple')
axes[1,2].set_title('Log(Rent) — Apres transformation')
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120, bbox_inches='tight')
print("\nVisualisations EDA sauvegardees : eda_plots.png")

# ══════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Doublons
n_dup = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Doublons supprimes : {n_dup}")

# Valeurs aberrantes
q_low, q_high = df['Rent'].quantile(0.01), df['Rent'].quantile(0.99)
df = df[(df['Rent'] >= q_low) & (df['Rent'] <= q_high)]
df = df[(df['Size'] > 50) & (df['Size'] < 10000)]
print(f"Forme apres nettoyage : {df.shape}")

# Extraction etage
def extract_floor(floor_str):
    if pd.isna(floor_str): return 0
    parts = str(floor_str).lower().split()
    if 'ground' in parts: return 0
    if 'upper' in parts or 'lower' in parts: return 1
    try: return int(parts[0])
    except: return 0

df['Floor_Num'] = df['Floor'].apply(extract_floor)

# Log-transform
df['log_Rent'] = np.log1p(df['Rent'])
df['log_Size'] = np.log1p(df['Size'])

# Encodage
le = {}
cat_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
for col in cat_cols:
    le[col] = LabelEncoder()
    df[col + '_enc'] = le[col].fit_transform(df[col].astype(str))
    print(f"Encode {col} : {le[col].classes_.tolist()}")

# ══════════════════════════════════════════════════
# 3. FEATURES & SPLIT
# ══════════════════════════════════════════════════
feature_cols = ['BHK', 'log_Size', 'Floor_Num', 'Bathroom',
                'Area Type_enc', 'City_enc', 'Furnishing Status_enc',
                'Tenant Preferred_enc', 'Point of Contact_enc']
X = df[feature_cols]
y = df['log_Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain : {X_train.shape}, Test : {X_test.shape}")

# Standardisation (pour LR et SVR)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ══════════════════════════════════════════════════
# 4. MODELISATION
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTATS DES MODELES")
print("=" * 60)
print(f"{'Modele':<25} {'R²':>8} {'MAE (Rs)':>12} {'RMSE (Rs)':>12}")
print("-" * 60)

models = {
    'Linear Regression': (LinearRegression(), True),
    'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
    'XGBoost': (xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0), False),
    'SVR (RBF)': (SVR(kernel='rbf'), True),
}

for name, (model, use_sc) in models.items():
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc if use_sc else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    print(f"{name:<25} {r2:>8.4f} {mae:>12.0f} {rmse:>12.0f}")

# ══════════════════════════════════════════════════
# 5. XGBOOST — TUNING & VALIDATION CROISEE
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("XGBOOST — GridSearchCV + Validation Croisee")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
xgb_gs = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
)
xgb_gs.fit(X_train, y_train)
print(f"Meilleurs hyperparametres : {xgb_gs.best_params_}")

best_model = xgb_gs.best_estimator_
y_pred_best = best_model.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_best))
rmse_best = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_best)))
print(f"XGBoost (tune) : R²={r2_best:.4f}, MAE={mae_best:.0f}, RMSE={rmse_best:.0f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"CV R² (5-fold) : {cv_scores.round(4)}")
print(f"Moyenne : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Feature importance
feat_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nImportance des variables :\n{feat_imp.round(4)}")

# ══════════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════════
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoders.pkl')
print("\nModeles sauvegardes : best_model.pkl, scaler.pkl, label_encoders.pkl")

# ══════════════════════════════════════════════════
# 7. PREDICTION EXEMPLE
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXEMPLE DE PREDICTION")
print("=" * 60)

# Mumbai, 2 BHK, 800 sqft, 2eme etage, 1 sdb, Semi-Furnished, Family, Owner
example = pd.DataFrame([[2, np.log1p(800), 2, 1,
                          le['Area Type'].transform(['Super Area'])[0],
                          le['City'].transform(['Mumbai'])[0],
                          le['Furnishing Status'].transform(['Semi-Furnished'])[0],
                          le['Tenant Preferred'].transform(['Bachelors/Family'])[0],
                          le['Point of Contact'].transform(['Contact Owner'])[0]]],
                        columns=feature_cols)

pred_log = best_model.predict(example)[0]
pred_rent = int(np.expm1(pred_log))
print(f"Mumbai, 2BHK, 800sqft, Semi-Furnished, Owner → Loyer estime : ₹{pred_rent:,}")
print("\nPipeline complete avec succes !")
