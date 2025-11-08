# 🚀 Quick Start Guide

## Running the EducationCare Study Recommender

Your main application is now **`recommender_app.py`** (in the root folder) - a comprehensive ML-powered personalized study recommendation system.

### Quick Start

```bash
cd /Users/jonathanlee/Documents/GitHub/educationcare
streamlit run recommender_app.py
```

The app will open in your browser at `http://localhost:8501`

### What This App Does

**3-Step ML-Powered Recommendation System:**

1. **Step 1: English Assessment** (4 skills)
   - Vocabulary, Grammar, Reading, Writing questions
   - Predicts English proficiency using ML model
   - Output: Developing/Intermediate/Advanced level

2. **Step 2: Learning Profile** (Study habits)
   - Academic history, study hours, engagement
   - Predicts academic success using ML model (58 features)
   - Output: Pass/Fail/Distinction/Withdrawn prediction

3. **Step 3: Personalized Recommendations**
   - ML-powered resource matching
   - Content-based filtering
   - Weekly action plans
   - Progress tracking

### Features

✅ **Dual ML Models**: English proficiency + Academic success prediction  
✅ **Smart Feature Engineering**: 9 inputs → 58 ML features  
✅ **Personalized Resources**: 12+ study resources matched to student needs  
✅ **Risk Assessment**: High/Medium/Low intervention priority  
✅ **Interactive UI**: 3-step wizard with progress tracking  

### Models Required

The app looks for these files in the `models/` folder:
- `model.pkl` - Academic success classifier (58 features)
- `scaler.pkl` - Feature scaler
- `encoder.pkl` - Categorical encoder
- `target_encoder.pkl` - Target variable encoder

And in the `proficiency /` folder (sibling to educationcare):
- `english_proficiency_model.pkl` - English proficiency classifier (4 features)

### Training Models

If models don't exist, run `notebooks/Final.ipynb` to train and save them.

### Removed Files

The following old files have been removed:
- ❌ `app.py` (old main app)
- ❌ `scripts/app_english_learning.py` (English learning module)
- ❌ `docs/README_APP.md`
- ❌ `docs/ENGLISH_LEARNING_GUIDE.md`
- ❌ `docs/STREAMLIT_FEATURES.md`
- ❌ `docs/STREAMLIT_SUMMARY.md`

### Project Structure

```
educationcare/
├── recommender_app.py                ← MAIN APP ⭐
├── lightfm_integration.py            ← LightFM collaborative filtering
├── categorical_mapping_utils.py      ← Category encoding utilities
├── models/                           ← ML models
├── config/                           ← Configuration
├── data/                             ← Training data
├── proficiency/                      ← English proficiency module
├── notebooks/                        ← Model training
├── scripts/                          ← Utilities
└── docs/                             ← Documentation
```

### Troubleshooting

**Models not found?**
- Run `notebooks/Final.ipynb` to train and save models to `models/` folder

**Import errors?**
- Install dependencies: `pip install -r requirements.txt`

**Port already in use?**
- Stop other Streamlit apps or use: `streamlit run scripts/recommender_app.py --server.port 8502`

## 📚 Documentation

See the `docs/` folder for:
- `ARCHITECTURE.md` - System architecture
- `FEATURES_EXPLAINED.md` - Feature descriptions
- `PROJECT_SUMMARY.md` - Project overview
- `SETUP_INSTRUCTIONS.md` - Setup guide
