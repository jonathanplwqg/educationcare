# 🎓 EducationCare LightFM Study Recommender - Setup Guide

## 📌 What Has Been Created

I've created a complete Streamlit web application for your EducationCare student success prediction project. Here's what you have:

### Files Created:
1. **lightfm_study_recommender.py** - Main LightFM-based study recommendation app
2. **notebooks/Final.ipynb** - Model training and feature engineering notebook
3. **proficiency/** - English proficiency testing module
   - `streamlit_english_test.py` - English proficiency test app
   - `train_english_proficiency_model.ipynb` - Proficiency model training
   - `english_proficiency_model.pkl` - Trained proficiency classifier
4. **requirements.txt** - Python package dependencies
5. **data/** - CSV datasets (7 files)
6. **models/** - Trained ML models (5 .pkl files)
7. **config/** - Configuration files (2 .json files)
8. **docs/** - Documentation including this file
9. **scripts/** - Utility scripts
10. **utils/** - Utility functions

## 🎯 Features

### Student Input Form (Sidebar)
The app collects comprehensive student information matching your project proposal:

**Personal Information:**
- Gender (M/F)
- Age Band (0-35, 35+)
- Region (13 UK regions)
- Highest Education (3 levels)
- IMD Band (deprivation index, 10 bands)
- Disability (Y/N)

**Academic Background:**
- Number of Previous Attempts (0-10)
- Studied Credits (0-300)
- Days Since Registration (0-365)

**Engagement & Activity:**
- Total Platform Clicks
- Number of Activities
- Primary Activity Type (20+ types from your data)

**Assessment Performance:**
- Average Score (0-100%)
- Submission Timeliness (-100 to +50 days)
- Banked Assessments count
- Total Assessments

**Learning Behavior:**
- Preferred Study Method
- Learning Pace
- Engagement Consistency

### Prediction Output

1. **Student Profile Summary**
   - Displays calculated metrics
   - Shows risk factor count
   - Real-time feature engineering

2. **Predicted Outcome**
   - Classification: Distinction/Pass/Fail/Withdrawn
   - Confidence score with gauge visualization
   - Probability distribution chart

3. **Student Persona (6 Clusters from your analysis)**
   - High Achievers
   - Struggling & Withdrawn
   - Engaged Achievers
   - Active but Struggling
   - Experienced Repeaters
   - Fast but Disengaged

4. **Personalized Feedback (4 Tabs)**
   - ✅ Strengths
   - ⚠️ Areas for Improvement
   - 🎯 Action Plan
   - 📚 Resources

5. **Visual Analytics**
   - Outcome probability bar chart
   - Key factors bar chart
   - Confidence gauge
   - Risk assessment metrics

## 🚀 Quick Start

### Step 1: Install Dependencies

Open terminal in your project directory and run:

```bash
pip install -r requirements.txt
```

This installs:
- streamlit
- pandas, numpy
- scikit-learn
- plotly (for visualizations)
- xgboost, lightgbm
- umap-learn

### Step 1b: Choose Your App

The project includes two Streamlit applications:

1. **Study Recommender** (Main App)
   ```bash
   streamlit run lightfm_study_recommender.py
   ```
   - Predicts student outcomes
   - Uses 58-feature LightGBM model
   - Provides study recommendations

2. **English Proficiency Test**
   ```bash
   streamlit run proficiency/streamlit_english_test.py
   ```
   - Tests English language proficiency
   - Adaptive difficulty testing
   - Provides detailed feedback

### Step 2: Save Your Model

You have two options:

**Option A: Quick Method (Recommended)**

1. Open `Final.ipynb` in Jupyter/VS Code
2. Copy the contents of `save_model_cell.py`
3. Paste into a new cell at the end of your notebook
4. Run the cell
5. Model files will be saved automatically

**Option B: Manual Method**

Add this to your notebook:

```python
import pickle

# Save your trained model (adjust variable names)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)  # or your model variable

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Create target encoder
from sklearn.preprocessing import LabelEncoder
target_encoder = LabelEncoder()
target_encoder.fit(['Pass', 'Fail', 'Distinction', 'Withdrawn'])

with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

print("✅ Model files saved!")
```

### Step 3: Run the App

```bash
streamlit run lightfm_study_recommender.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🧪 Testing (Optional)

Verify your folder structure:

```bash
educationcare/
├── lightfm_study_recommender.py  # Main study recommender app
├── data/                          # CSV files (7 datasets)
├── models/                        # ML models (5 .pkl files)
├── config/                        # JSON configs (2 files)
├── notebooks/                     # Final.ipynb
├── proficiency/                   # English proficiency module
│   ├── streamlit_english_test.py
│   ├── train_english_proficiency_model.ipynb
│   └── english_proficiency_model.pkl
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
└── utils/                         # Utility functions
```

This checks:
- ✅ All packages installed
- ✅ Required folders exist (data/, models/, config/, proficiency/)
- ✅ Model files present in models/ and proficiency/ folders

## 📖 How to Use the App

### For You (Developer):

1. **Demo Mode** (no model files):
   - App uses rule-based predictions
   - Good for testing UI/UX
   - All features work except ML predictions

2. **Full Mode** (with model files):
   - Real ML predictions
   - Actual probabilities
   - Cluster assignments from your model

### For End Users:

1. **Fill the Form** (left sidebar)
   - Complete all sections
   - Use realistic values
   - Hover over (?) icons for help

2. **Click "Predict"**
   - View outcome prediction
   - See confidence score
   - Read persona description

3. **Review Feedback**
   - Check strengths
   - Read improvement areas
   - Follow action plan
   - Explore resources

## 🎨 Customization

### Change Colors/Styling

Edit CSS in `lightfm_study_recommender.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #YOUR_COLOR;  # Change colors here
    }
</style>
""", unsafe_allow_html=True)
```

### Modify Model Parameters

Edit model loading and prediction sections:

```python
# Load different models
model = joblib.load('models/your_model.pkl')

# Adjust prediction thresholds
if prediction_proba > 0.75:
    outcome = "High Success"
```

### Update Feature Engineering

Modify the feature engineering pipeline in `notebooks/Final.ipynb` and re-save models:

```python
# Add new features
df['new_feature'] = calculate_new_metric(df)

# Retrain and save
joblib.dump(model, '../models/model.pkl')
```

## 🔧 Troubleshooting

### Problem: "No module named 'streamlit'"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Problem: "model.pkl not found"
**Solution:** Either:
1. Save model from notebook (Step 2 above)
2. Or use demo mode (app still works!)

### Problem: App won't start
**Solution:** Check you're in correct directory
```bash
cd /path/to/educationcare
streamlit run lightfm_study_recommender.py
```

### Problem: Import errors in app
**Solution:** Update packages
```bash
pip install --upgrade -r requirements.txt
```

### Problem: Predictions seem wrong
**Solution:** Verify feature alignment:
1. Check feature names match training data
2. Verify preprocessing steps match notebook
3. Test with known student examples

## 📊 Understanding the Predictions

### Confidence Score
- **75-100%**: High confidence
- **60-75%**: Medium confidence
- **Below 60%**: Low confidence (borderline cases)

### Risk Factors (0-5)
- **0**: Low risk, excellent profile
- **1-2**: Medium risk, needs monitoring
- **3-5**: High risk, immediate intervention needed

### Student Personas

| Persona | Risk | Characteristics |
|---------|------|----------------|
| High Achievers | Low | Strong performance, consistent |
| Engaged Achievers | Low | High engagement, first-timers |
| Fast but Disengaged | Medium | Quick but shallow learning |
| Active but Struggling | Medium | Lots of activity, poor timing |
| Experienced Repeaters | Medium | Multiple attempts, moderate |
| Struggling & Withdrawn | High | Low engagement, many drop |

## 🎯 Project Alignment

This app aligns with your project proposal:

✅ **Data Mining Techniques**: Uses your trained classification models
✅ **Student Success Prediction**: Predicts 4 outcomes (Distinction/Pass/Fail/Withdrawn)
✅ **Personalized Feedback**: Tailored recommendations based on student profile
✅ **Clustering Analysis**: 6 student personas from UMAP + K-means
✅ **Educational Support**: Action plans and resource recommendations
✅ **Visual Analytics**: Charts and gauges for interpretability

## 📝 Next Steps

### For Development:
1. ✅ Test app with various student profiles
2. ✅ Gather user feedback
3. ✅ Refine feedback messages
4. ✅ Add more visualizations
5. ✅ Deploy to cloud (Streamlit Cloud, Heroku, etc.)

### For Deployment:
1. **Streamlit Cloud** (easiest):
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Deploy with one click

2. **Local Network**:
   - Run on server
   - Share URL with team
   - Set up authentication if needed

## 📚 Additional Resources

### Streamlit Docs:
- [Official Docs](https://docs.streamlit.io)
- [Gallery Examples](https://streamlit.io/gallery)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)

### Your Project Files:
- `notebooks/Final.ipynb` - Model training and feature engineering
- `lightfm_study_recommender.py` - Main Streamlit app
- `docs/` - All documentation files
- `models/` - Trained models and artifacts
- `data/` - Source CSV datasets

## 💡 Tips

1. **Start with Demo Mode**: Test the app before saving model
2. **Use Test Data**: Create sample student profiles for testing
3. **Iterate Feedback**: Refine messages based on user testing
4. **Monitor Performance**: Track prediction accuracy over time
5. **Update Regularly**: Retrain model with new data periodically

## 🎉 Success Checklist

Before presenting/submitting:

- [ ] Tested app with demo data
- [ ] Saved model from notebook
- [ ] Tested app with real predictions
- [ ] Verified all feedback messages
- [ ] Tested all input combinations
- [ ] Checked visual elements render correctly
- [ ] Documented any customizations
- [ ] Prepared demo student profiles
- [ ] Screenshots for documentation
- [ ] Deployment (if required)

## 📧 Support

If you encounter issues:
1. Check this guide first
2. Review error messages carefully
3. Test with `test_app.py`
4. Verify model files exist
5. Check Python/package versions

## 🌟 Success!

You now have a fully functional Streamlit app for your EducationCare project!

**The app provides:**
- ✅ Real-time predictions
- ✅ Personalized feedback
- ✅ Visual analytics
- ✅ Educational support recommendations
- ✅ Professional UI/UX

**Ready to run:**
```bash
streamlit run lightfm_study_recommender.py
```

Good luck with your project! 🚀
