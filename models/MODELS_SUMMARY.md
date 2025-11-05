# 📦 Models Folder Summary

## Current Model Files (5 total)

### ✅ Essential Models (4 files) - Used by Main App

These models are **loaded and used** by `lightfm_study_recommender.py`:

1. **model.pkl** (848 KB)
   - Type: LightGBM Classifier
   - Features: 58 engineered features
   - Purpose: Main prediction model for student success classification
   - Used by: Main Streamlit app
   - Training: notebooks/Final.ipynb (Cell 79)

2. **scaler.pkl** (593 B)
   - Type: StandardScaler (sklearn)
   - Purpose: Normalizes numerical features before prediction
   - Used by: Main Streamlit app
   - Training: notebooks/Final.ipynb

3. **encoder.pkl** (1.5 KB)
   - Type: OneHotEncoder (sklearn)
   - Purpose: Encodes categorical variables
   - Used by: Main Streamlit app
   - Training: notebooks/Final.ipynb

4. **target_encoder.pkl** (412 B)
   - Type: LabelEncoder (sklearn)
   - Purpose: Encodes target labels (Fail, Pass, Distinction, Withdrawn)
   - Used by: Main Streamlit app
   - Training: notebooks/Final.ipynb

### 📊 Analysis Model (1 file) - For Research/Visualization

This model is **saved for analysis** but not loaded by the main app:

5. **umap_reducer.pkl** (~14 MB) ⚠️ **NEEDS REGENERATION**
   - Type: UMAP (Uniform Manifold Approximation and Projection)
   - Purpose: Dimensionality reduction for clustering visualization
   - Use Cases:
     - Student behavior clustering analysis
     - 2D/3D visualization of student patterns
     - Exploratory data analysis
     - Understanding feature relationships
   - Training: notebooks/Final.ipynb (Cell 77)
   - Status: **DELETED - needs to be regenerated**

---

## How to Regenerate umap_reducer.pkl

The UMAP reducer was accidentally deleted during cleanup. To regenerate it:

### Option 1: Run Cell 77 Only (Quick Method)
```python
# In notebooks/Final.ipynb, run Cell 77 (ID: ab416c29)
# This assumes 'best_result' variable exists in your kernel
```

The cell contains:
```python
def save_model_for_streamlit():
    """Save necessary models for Streamlit app"""
    # ... other code ...
    
    if 'best_result' in globals() and 'umap_reducer' in best_result:
        with open('../models/umap_reducer.pkl', 'wb') as f:
            pickle.dump(best_result['umap_reducer'], f)
        print("✅ Saved umap_reducer.pkl")
```

### Option 2: Re-run UMAP Analysis (Complete Method)
1. Open `notebooks/Final.ipynb`
2. Run all cells that perform UMAP analysis (creates `best_result` variable)
3. Run Cell 77 to save the UMAP reducer
4. Verify file exists: `ls -lh models/umap_reducer.pkl`

---

## Removed Models (Redundant/Outdated)

These models were deleted on Nov 5, 2025 to clean up redundancy:

❌ **lgb_58_features.pkl** (848 KB)
   - Reason: Duplicate of `model.pkl`
   - Deleted: Yes

❌ **lgb_best_model.pkl** (1.0 MB)
   - Reason: Old version, replaced by `model.pkl`
   - Deleted: Yes

❌ **model_backup_89features.pkl** (1.0 MB)
   - Reason: Wrong feature count (89 instead of 58)
   - Deleted: Yes

❌ **model_backup_randomforest.pkl** (72 MB)
   - Reason: Old RandomForest model, not used
   - Deleted: Yes

**Total Space Freed:** ~75 MB (excluding UMAP which should be kept)

---

## Model Loading in lightfm_study_recommender.py

The main app loads these models on startup:

```python
# Load the trained model and preprocessors
model = joblib.load('models/model.pkl')              # LightGBM classifier
scaler = joblib.load('models/scaler.pkl')            # StandardScaler
encoder = joblib.load('models/encoder.pkl')          # OneHotEncoder
target_encoder = joblib.load('models/target_encoder.pkl')  # LabelEncoder

# Note: umap_reducer.pkl is NOT loaded by the main app
# It's saved for future analysis/visualization purposes
```

---

## Model Sizes & Total Storage

### Current (4 files):
- model.pkl: 848 KB
- scaler.pkl: 593 B
- encoder.pkl: 1.5 KB
- target_encoder.pkl: 412 B
- **Total:** ~850 KB

### After Regenerating UMAP (5 files):
- model.pkl: 848 KB
- scaler.pkl: 593 B
- encoder.pkl: 1.5 KB
- target_encoder.pkl: 412 B
- umap_reducer.pkl: ~14 MB
- **Total:** ~15 MB

### Comparison:
- **Before cleanup:** ~90 MB (9 model files)
- **After cleanup:** ~15 MB (5 model files)
- **Space saved:** ~75 MB (83% reduction)

---

## Training Information

### Main Models (Cell 79 in Final.ipynb)
The primary 4 models are saved by Cell 79, which includes intelligent feature detection:

```python
# Cell 79 (196a6458) - Active model saving
if X_train.shape[1] == 58:
    print("✅ Using 58-feature model (correct)")
    joblib.dump(lgb_model, '../models/model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(encoder, '../models/encoder.pkl')
    joblib.dump(target_encoder, '../models/target_encoder.pkl')
```

### UMAP Model (Cell 77 in Final.ipynb)
The UMAP reducer is saved by Cell 77 as part of the complete analysis artifacts:

```python
# Cell 77 (ab416c29) - save_model_for_streamlit function
with open('../models/umap_reducer.pkl', 'wb') as f:
    pickle.dump(best_result['umap_reducer'], f)
```

---

## Verification Commands

### Check which models exist:
```bash
ls -lh models/
```

### Expected output (after regenerating UMAP):
```
-rw-r--r--  encoder.pkl (1.5 KB)
-rw-r--r--  model.pkl (848 KB)
-rw-r--r--  scaler.pkl (593 B)
-rw-r--r--  target_encoder.pkl (412 B)
-rw-r--r--  umap_reducer.pkl (14 MB)
```

### Check model versions:
```python
import joblib
model = joblib.load('models/model.pkl')
print(f"Model type: {type(model)}")
print(f"Features: {model.n_features_}")
```

---

## Next Steps

1. **Regenerate UMAP Reducer:**
   - Open `notebooks/Final.ipynb`
   - Run Cell 77 (or relevant UMAP analysis cells)
   - Verify `models/umap_reducer.pkl` exists

2. **Verify All Models:**
   ```bash
   ls -lh models/
   # Should show 5 .pkl files (~15 MB total)
   ```

3. **Test Main App:**
   ```bash
   streamlit run lightfm_study_recommender.py
   # App should load without errors
   ```

4. **Update Git (if using version control):**
   ```bash
   git add models/umap_reducer.pkl
   git commit -m "Regenerate UMAP reducer model"
   ```

---

**Last Updated:** November 5, 2025
**Documentation:** All docs updated to reflect 5 model files
