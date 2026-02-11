# ===============================================================
# BODYMETA HUMAN-ONLY MACHINE LEARNING STARTER
# ===============================================================
# This script uses the human-only files you prepared.
# It automatically adapts to the files that are available.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Set working directory to where the CSV files are
os.chdir(r"C:\Users\ongt9\Downloads\BodyMeta_Human_Only_Final")

print("Loading human-only BodyMeta data...")
loaded = {}

# Try to load each file (some may be missing)
try:
    df_16s = pd.read_csv("Human_Only_16S_Data.csv", encoding='utf-8-sig')
    loaded['16s'] = df_16s
    print(f"✓ 16S data: {df_16s.shape}")
except FileNotFoundError:
    print("⚠ 16S data file not found.")

try:
    df_microbes = pd.read_csv("Human_Only_Associated_Microbes.csv", encoding='utf-8-sig')
    loaded['microbes'] = df_microbes
    print(f"✓ Associated microbes: {df_microbes.shape}")
except FileNotFoundError:
    print("⚠ Associated microbes file not found.")

try:
    df_lit = pd.read_csv("Human_Only_Literature_Based.csv", encoding='utf-8-sig')
    loaded['lit'] = df_lit
    print(f"✓ Literature-based: {df_lit.shape}")
except FileNotFoundError:
    print("⚠ Literature-based file not found.")

if not loaded:
    print("❌ No data files found. Exiting.")
    exit()

# ===============================================================
# EXPLORE THE AVAILABLE DATA
# ===============================================================

# Pick the main data source (prefer 16S because it has rich metadata)
if '16s' in loaded:
    df_main = loaded['16s']
    src = "16S"
elif 'microbes' in loaded:
    df_main = loaded['microbes']
    src = "Associated Microbes"
else:
    df_main = loaded['lit']
    src = "Literature"

print(f"\n🔍 Using {src} as main data source.")
print("Columns:", list(df_main.columns)[:10], "…")

# Find condition/disease column
condition_cols = [col for col in df_main.columns 
                 if any(k in col.lower() for k in 
                 ['condition', 'disease', 'diagnosis', 'pathology', 'phenotype'])]

target_col = None
if condition_cols:
    target_col = condition_cols[0]
    print(f"\n🎯 Target column: '{target_col}'")
    print(f"   Unique conditions: {df_main[target_col].nunique()}")
    print("   Sample:", df_main[target_col].dropna().unique()[:5])
else:
    print("\n⚠ No obvious condition/disease column found.")

# Find microbe-related columns (potential features)
microbe_cols = [col for col in df_main.columns 
                if any(k in col.lower() for k in 
                ['bacter', 'microbe', 'taxon', 'genus', 'species', 'family'])]
print(f"\n🧬 Found {len(microbe_cols)} microbe-related columns.")

# ===============================================================
# SIMPLE MACHINE LEARNING EXAMPLE (if we have features and a target)
# ===============================================================

if target_col and len(microbe_cols) > 0:
    # Prepare feature matrix X and labels y
    X = df_main[microbe_cols].fillna(0)
    y = df_main[target_col]
    
    # Remove rows with missing labels
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"\n📊 After cleaning: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Only proceed if we have at least two classes
    if y.nunique() >= 2:
        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        print(f"\n🚀 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        test_acc  = rf.score(X_test, y_test)
        print(f"   Training accuracy: {train_acc:.3f}")
        print(f"   Test accuracy:     {test_acc:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'microbe': microbe_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 important microbes:")
        print(importance.head(10).to_string(index=False))
    else:
        print("\n⚠ Need at least 2 different condition values for classification.")
else:
    if not target_col:
        print("\n⚠ Cannot run ML: no target column identified.")
    elif len(microbe_cols) == 0:
        print("\n⚠ Cannot run ML: no microbe-related columns found.")

print("\n✅ ML starter script finished.")
print("\n📌 Next ideas:")
print("   - Try different models (XGBoost, SVM, neural nets)")
print("   - Add feature selection")
print("   - Use cross-validation")
print("   - Explore other target variables")
