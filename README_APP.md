# 🎓 EducationCare Student Success Predictor

A Streamlit web application that predicts student outcomes and provides personalized feedback based on machine learning models trained on educational data.

## 📋 Overview

This application uses classification models to predict student outcomes (Distinction, Pass, Fail, or Withdrawn) based on:
- Demographic information
- Academic background
- Engagement metrics
- Assessment performance
- Learning behavior patterns

The app provides:
- **Outcome predictions** with confidence scores
- **Student personas** based on clustering analysis
- **Personalized feedback** and recommendations
- **Action plans** tailored to student needs
- **Visual analytics** of prediction factors

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Save Your Model

Before running the app, you need to save your trained model from the `Final.ipynb` notebook.

In your notebook, add this cell at the end:

```python
from save_model import save_model_artifacts

# Get your best model (adjust variable names as needed)
# From your results_comparison or classification_strategies
best_model = results_comparison[0]['Model']  # or your model variable

# Save all artifacts
save_model_artifacts(
    model=best_model,
    scaler=scaler,  # Your PowerTransformer
    encoder=encoder,  # Your OneHotEncoder
    feature_names=original_features,  # or selected_cluster_features
    target_encoder=LabelEncoder().fit(['Pass', 'Fail', 'Distinction', 'Withdrawn']),
    cluster_model=best_result['kmeans'],  # Optional: your KMeans model
    umap_reducer=best_result['umap_reducer']  # Optional: your UMAP model
)
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Features

### Input Form (Sidebar)
The app collects the following student information:

**Personal Information:**
- Gender
- Age Band
- Region
- Highest Education
- IMD Band (Deprivation Index)
- Disability status

**Academic Background:**
- Number of Previous Attempts
- Studied Credits
- Days Since Registration

**Engagement & Activity:**
- Total Platform Clicks
- Number of Activities
- Primary Activity Type

**Assessment Performance:**
- Average Assessment Score
- Submission Timeliness
- Banked Assessments
- Total Assessments

**Learning Behavior:**
- Preferred Study Method
- Learning Pace
- Engagement Consistency

### Prediction Output

1. **Student Profile Summary**
   - Quick overview of key metrics
   - Calculated engagement indicators
   - Risk factor assessment

2. **Predicted Outcome**
   - Classification result (Distinction/Pass/Fail/Withdrawn)
   - Confidence score with visual gauge
   - Probability distribution across all outcomes

3. **Student Persona**
   - Cluster assignment based on behavior patterns
   - 6 personas identified:
     - High Achievers
     - Struggling & Withdrawn
     - Engaged Achievers
     - Active but Struggling
     - Experienced Repeaters
     - Fast but Disengaged

4. **Personalized Feedback**
   - Strengths recognition
   - Areas for improvement
   - Customized action plan
   - Recommended resources

5. **Visual Analytics**
   - Outcome probability charts
   - Feature importance analysis
   - Risk factor indicators

## 🔧 Model Files

After running `save_model_artifacts()`, these files will be created:

```
├── model.pkl              # Trained classification model
├── scaler.pkl             # Feature scaler (PowerTransformer)
├── encoder.pkl            # Categorical encoder (OneHotEncoder)
├── target_encoder.pkl     # Target variable encoder
├── feature_names.json     # List of feature names
├── metadata.json          # Model metadata
├── cluster_model.pkl      # (Optional) KMeans clustering model
└── umap_reducer.pkl       # (Optional) UMAP dimensionality reducer
```

## 📁 Project Structure

```
educationcare/
├── app.py                     # Main Streamlit application
├── save_model.py             # Model saving utility
├── requirements.txt          # Python dependencies
├── README_APP.md             # This file
├── Final.ipynb               # Model training notebook
├── model.pkl                 # Saved model (generated)
├── scaler.pkl                # Saved scaler (generated)
├── encoder.pkl               # Saved encoder (generated)
└── ...                       # Other project files
```

## 🎯 Usage Example

1. **Fill in Student Information** in the sidebar
   - Start with demographic details
   - Add academic background
   - Input engagement metrics
   - Provide assessment scores

2. **Click "Predict Student Outcome"**
   - View predicted outcome
   - See confidence score
   - Identify student persona

3. **Review Feedback Tabs**
   - Strengths: What's working well
   - Areas for Improvement: What needs attention
   - Action Plan: Concrete steps to take
   - Resources: Helpful links and tools

## 🔍 Demo Mode

If model files are not found, the app runs in demo mode with rule-based predictions. This allows you to:
- Test the UI/UX
- Understand the input requirements
- Preview the feedback system

To use the full ML model, ensure all model files are saved (see step 2 above).

## 🎨 Customization

### Modify Cluster Interpretations

Edit the `CLUSTER_INTERPRETATIONS` dictionary in `app.py`:

```python
CLUSTER_INTERPRETATIONS = {
    0: {
        "name": "Your Cluster Name",
        "description": "Cluster description",
        "risk_level": "Low/Medium/High",
        "color": "success/warning/danger/info"
    },
    # ... add more clusters
}
```

### Add Custom Feedback Rules

Modify the feedback generation logic in the prediction section:

```python
# Around line 500-600 in app.py
if avg_score < 50:
    improvements.append("Your custom feedback here")
```

### Adjust Feature Engineering

Update the feature calculation logic around line 250:

```python
# Calculate derived features
activity_diversity = min(activity_count / 20, 1.0)
# Add your custom features
```

## 📈 Model Performance

Based on your `Final.ipynb` analysis:

- **Best Model**: RandomForest / XGBoost / LightGBM
- **Accuracy**: ~67-68%
- **Features Used**: 58 original features or enhanced cluster features
- **Clustering**: 6-7 optimal clusters (Silhouette score ~0.54)

## ⚠️ Important Notes

1. **Data Privacy**: This app processes student data. Ensure compliance with data protection regulations.

2. **Prediction Limitations**: The model provides guidance, not definitive assessments. Use predictions to inform support strategies, not as final evaluations.

3. **Model Updates**: Retrain the model periodically with new data to maintain accuracy.

4. **Feature Alignment**: Ensure input features match your training data preprocessing.

## 🐛 Troubleshooting

**Problem**: App shows "model files not found"
- **Solution**: Run `save_model_artifacts()` in your notebook first

**Problem**: Prediction errors
- **Solution**: Check that feature names match between app and saved model

**Problem**: Import errors
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Problem**: Streamlit won't start
- **Solution**: Ensure you're in the correct directory and Streamlit is installed

## 🤝 Contributing

To improve the app:
1. Test with various student profiles
2. Gather user feedback
3. Refine feedback messages
4. Add more personalized recommendations
5. Improve visualizations

## 📝 License

This project is part of the Data Mining coursework for educational purposes.

## 📧 Support

For issues or questions about the app, refer to your project documentation or reach out to your course instructor.

---

**Built with**: Streamlit, scikit-learn, Plotly, pandas, numpy

**Project**: EducationCare Student Success Prediction System

**Year**: 2025
