import pandas as pd
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# ==================== 1. Load Data ====================
dataset = pd.read_csv(r'D:\Project_Data\king_county_100k_synthetic.csv')
print(f"Dataset loaded: {dataset.shape[0]:,} rows, {dataset.shape[1]} columns")

# ==================== 2. Data Preprocessing ====================
print("\n" + "=" * 50)
print("Data Preprocessing...")

df = dataset.copy()
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# --- Missing Values ---
print("\n Missing Values per column:")
print(df.isnull().sum())
missing_total = df.isnull().sum().sum()
print(f"Total missing: {missing_total}")
df = df.dropna()
print(f"Shape after handling missing values: {df.shape}")

# --- Duplicates ---
dups = df.duplicated().sum()
print(f"\nDuplicated rows: {dups}")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# --- Outliers (IQR on price) ---
print("\nPrice stats BEFORE outlier removal:")
print(df['price'].describe())
Q1  = df['price'].quantile(0.25)
Q3  = df['price'].quantile(0.75)
IQR = Q3 - Q1
before = len(df)
df = df[(df['price'] >= Q1 - 1.5 * IQR) &
        (df['price'] <= Q3 + 1.5 * IQR)]
after = len(df)
print(f"Removed {before - after:,} outliers → shape now: {df.shape}")

# ==================== 3. Feature Engineering ====================
print("\n" + "=" * 50)
print("Feature Engineering...")

df['date']      = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
df['sale_year'] = df['date'].dt.year
df['sale_month']= df['date'].dt.month
df = df.drop(['date'], axis=1)

df['house_age']           = df['sale_year'] - df['yr_built']
df['is_renovated']        = (df['yr_renovated'] > 0).astype(int)
df['sqft_per_bedroom']    = df['sqft_living'] / (df['bedrooms'] + 1e-6)
df['living_to_lot_ratio'] = df['sqft_living'] / (df['sqft_lot']  + 1e-6)
df['quality_score']       = df['grade'] * df['condition']

zip_median = df.groupby('zipcode')['price'].median().to_dict()
df['zipcode_median_price'] = df['zipcode'].map(zip_median)
df = df.drop(['zipcode'], axis=1)

print("New features added: house_age, is_renovated, sqft_per_bedroom,")
print("living_to_lot_ratio, quality_score, zipcode_median_price")

# ==================== 4. Visualization & Analysis ====================
print("\n" + "=" * 50)
print("Generating Analysis Plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('King County Housing - Data Analysis', fontsize=16, fontweight='bold')

axes[0,0].hist(df['price'], bins=60, color='steelblue', edgecolor='black')
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price ($)')
axes[0,0].set_ylabel('Count')

axes[0,1].scatter(df['sqft_living'], df['price'], alpha=0.2, color='coral', s=5)
axes[0,1].set_title('Living Area vs Price')
axes[0,1].set_xlabel('sqft_living')
axes[0,1].set_ylabel('Price ($)')

df.groupby('grade')['price'].mean().plot(kind='bar', ax=axes[0,2], color='teal')
axes[0,2].set_title('Average Price by Grade')
axes[0,2].set_xlabel('Grade')
axes[0,2].set_ylabel('Avg Price ($)')

df[df['bedrooms'] <= 7].boxplot(column='price', by='bedrooms', ax=axes[1,0])
axes[1,0].set_title('Price by Bedrooms')
axes[1,0].set_xlabel('Bedrooms')
axes[1,0].set_ylabel('Price ($)')
plt.sca(axes[1,0])
plt.title('Price by Bedrooms')

axes[1,1].scatter(df['house_age'], df['price'], alpha=0.2, color='purple', s=5)
axes[1,1].set_title('House Age vs Price')
axes[1,1].set_xlabel('House Age (years)')
axes[1,1].set_ylabel('Price ($)')

corr_cols = ['price','sqft_living','grade','house_age',
             'quality_score','bedrooms','bathrooms','condition']

sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=axes[1,2])

axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()

# Save Plot as PNG
plt.savefig(
    r'C:\Users\dell\OneDrive - delta technological university\Desktop\housing_analysis.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Plots saved")

# ==================== 5. Prepare X & y ====================
print("\n" + "=" * 50)
print("Preparing Features & Target...")

X = df.drop("price", axis=1)
y = df["price"]
X = pd.get_dummies(X, drop_first=True)
print(f"Total Features after encoding: {X.shape[1]}")
print(f"Training samples available:   {X.shape[0]:,}")

# ==================== 6. Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ==================== 7. Pipeline & Training ====================
print("\n" + "=" * 50)
print("Building Pipeline & Training Model...")
print("Model: Random Forest Regressor")
print("n_estimators=300 | max_depth=20 | Scaling=StandardScaler")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

tic = time.time()
pipeline.fit(X_train, y_train)
toc = time.time()
print(f"Training completed in: {toc - tic:.2f} seconds")

# ==================== 8. Evaluation ====================
print("\n" + "=" * 50)
print("Model Evaluation:")

y_pred = pipeline.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
acc  = 100 - mape

print(f"   R² Score :  {r2:.4f}")
print(f"   Accuracy :   {acc:.2f}%")
print(f"   MAE     :  ${mae:>12,.2f}")
print(f"   RMSE:                      ${rmse:>12,.2f}")
print(f"   MSE:                       {mse:>18,.2f}")

# ==================== 9. Overfitting / Underfitting Check ====================
print("\n" + "=" * 50)
print(" Overfitting / Underfitting Check:")

train_r2 = pipeline.score(X_train, y_train)
test_r2  = pipeline.score(X_test,  y_test)
diff     = train_r2 - test_r2

print(f"   Train R²: {train_r2:.4f}")
print(f"   Test  R²: {test_r2:.4f}")
print(f"   Gap:      {diff:.4f}")

if diff > 0.15:
    print("Overfitting detected!")
    print("Solution: reduce max_depth, increase min_samples_leaf,")
    print("or use fewer n_estimators")
elif test_r2 < 0.70:
    print("Underfitting detected!")
    print("Solution: add more features, increase n_estimators,")
    print("or remove max_depth restriction")
else:
    print("Model is well-balanced — no serious overfitting/underfitting")

# ==================== 10. Cross Validation ====================
print("\n" + "=" * 50)
print("Cross Validation (5-Fold) on 10,000 sample...")

X_sample, y_sample = resample(X, y, n_samples=10000, random_state=42)
cv_scores = cross_val_score(
    pipeline, X_sample, y_sample, cv=5, scoring='r2', n_jobs=-1
)

print(f"   R² per fold: {np.round(cv_scores, 4)}")
print(f"   Mean R²:     {cv_scores.mean():.4f}")
print(f"   Std  (±):    {cv_scores.std():.4f}")

if cv_scores.std() < 0.02:
    print("Model is stable across folds (low variance)")
else:
    print("High variance between folds — model may be unstable")
    
    
    
    

# ==================== 11. Save & Load Model ====================
print("\n" + "=" * 50)
model_path = r'C:\Users\dell\OneDrive - delta technological university\Desktop\Housing_model.sav'
pickle.dump(pipeline, open(model_path, 'wb'))
print(f"Model saved  → {model_path}")

model = pickle.load(open(model_path, 'rb'))
print("Model loaded successfully")

# ==================== 12. Single Prediction Test ====================
print("\n" + "=" * 50)
print("Single Record Prediction Test:")

sample          = X_test.iloc[[0]]
actual_price    = y_test.iloc[0]
predicted_price = model.predict(sample)[0]
error_pct       = abs(actual_price - predicted_price) / actual_price * 100

print(f"   Predicted Price: ${predicted_price:>12,.2f}")
print(f"   Actual    Price: ${actual_price:>12,.2f}")
print(f"   Difference:      ${abs(actual_price - predicted_price):>12,.2f}")
print(f"   Error %:         {error_pct:.2f}%")