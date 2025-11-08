# 🏗️ System Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT                                   │
│  "I complete 5 lessons/week, score 70%, study consistently"     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│         STREAMLIT INTERFACE (recommender_app.py)       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Sidebar: Student Profile Input Form                      │  │
│  │  - Demographics (age, region, education)                 │  │
│  │  - Study patterns and engagement metrics                 │  │
│  │  - Performance indicators                                │  │
│  │  - Learning preferences                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                             │
│  Process student data through engineered features               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Engagement Metrics                                       │  │
│  │  - activity_count, sum_clicks, engagement_cv             │  │
│  │  - activity_diversity, weighted_engagement               │  │
│  │                                                            │  │
│  │  Performance Metrics                                      │  │
│  │  - score, score_per_weight, assessment_engagement        │  │
│  │                                                            │  │
│  │  Behavioral Patterns                                      │  │
│  │  - submission_timeliness, learning_pace                  │  │
│  │  - engagement_trend, score_momentum                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML MODEL (LightGBM Classifier)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Input: 58 technical features                            │  │
│  │  - engagement_cv: 0.4                                     │  │
│  │  - sum: 1200 (calculated from lessons × exercises)       │  │
│  │  - score: 70                                              │  │
│  │  - activity_diversity: 0.6                                │  │
│  │  ... (54 more features)                                   │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Output: Prediction + Confidence                         │  │
│  │  - Class: "Good Progress"                                │  │
│  │  - Confidence: 0.78                                       │  │
│  │  - Probabilities: [0.05, 0.15, 0.65, 0.15]              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│           FEEDBACK GENERATION ENGINE                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Analyze student profile                              │  │
│  │     - Calculate risk factors                              │  │
│  │     - Assign learner persona                              │  │
│  │                                                            │  │
│  │  2. Generate personalized feedback                        │  │
│  │     IF score < 60: Add "Improve Scores" section          │  │
│  │     IF lessons < 3: Add "Increase Practice" section      │  │
│  │     IF late assignments: Add "Time Management" section    │  │
│  │                                                            │  │
│  │  3. Create action plan                                    │  │
│  │     Based on persona ID (0-5)                            │  │
│  │     Custom 2-week plan with specific steps               │  │
│  │                                                            │  │
│  │  4. Select resources                                      │  │
│  │     Customize based on identified needs                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISPLAY RESULTS                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  📊 Prediction: "Good Progress (B/B+)"                   │  │
│  │  👥 Persona: "Engaged Achiever 📚"                       │  │
│  │  💡 Feedback Tabs:                                        │  │
│  │     [Strengths] [Focus Areas] [Action Steps] [Resources] │  │
│  │                                                            │  │
│  │  Strengths:                                               │  │
│  │  ✅ Strong Performance - 70% scores!                     │  │
│  │  ✅ Consistent Practice - Regular schedule!              │  │
│  │                                                            │  │
│  │  Focus Areas:                                             │  │
│  │  🎯 Increase Practice Frequency                          │  │
│  │     - Aim for 7+ lessons per week                        │  │
│  │     - Study 15-30 minutes daily                          │  │
│  │                                                            │  │
│  │  Action Plan:                                             │  │
│  │  Week 1-2: Complete 7 lessons per week                   │  │
│  │            Practice weakest skill daily                   │  │
│  │                                                            │  │
│  │  Resources:                                               │  │
│  │  - BBC Learning English                                  │  │
│  │  - Duolingo mobile app                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT                                   │
│  Receives actionable feedback, knows exactly what to improve    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Input Layer (Student Interface)
**File:** `recommender_app.py`
**Purpose:** Collect comprehensive student profile data
**Key Features:**
- Demographic information (age, region, education, etc.)
- Study behavior metrics (engagement, consistency)
- Performance indicators (scores, assessments)
- Learning preferences and patterns

### 2. Feature Engineering Layer
**File:** `recommender_app.py` + `notebooks/Final.ipynb`
**Functions:**
- Calculates 58+ features from raw inputs
- Combines engagement, performance, and behavioral metrics
- Applies domain knowledge from educational data science

**Key Transformations:**
```python
# Engagement metrics
total_engagement = sum_clicks × engagement_quality
activity_diversity = unique_activities / total_possible

# Performance metrics
normalized_score = score / weight
assessment_efficiency = score / time_taken

# Behavioral patterns
consistency_score = 1 - engagement_coefficient_of_variation
learning_velocity = credits_earned / days_active
```

### 3. ML Model Layer
**Files:** `models/model.pkl`, `models/scaler.pkl`, `models/encoder.pkl`
**Type:** LightGBM Classifier (58-feature model)
**Input:** 58 engineered features (numeric + encoded categorical)
**Output:** 4-class prediction (Distinction/Pass/Fail/Withdrawn)
**Accuracy:** ~90.8% on test data

### 4. Prediction & Recommendation Engine
**File:** `recommender_app.py`
**Components:**

**A. Outcome Prediction**
```python
# Load models from models/ folder
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

# Make prediction
prediction = model.predict(features)
probabilities = model.predict_proba(features)
```

**B. Study Recommendations**
```python
# Based on student profile and prediction
recommendations = generate_study_plan(
    student_profile=profile,
    predicted_outcome=prediction,
    weak_areas=identified_gaps
)
```

**C. Personalized Insights**
```python
insights = {
    'strengths': analyze_strong_features(profile),
    'focus_areas': identify_improvement_areas(profile),
    'action_steps': create_action_plan(profile, prediction),
    'resources': recommend_materials(weak_areas)
}
```

### 5. Display Layer
**File:** `recommender_app.py`
**Components:**
- Student Profile Summary
- Prediction Display (outcome + confidence)
- Feature Importance Visualization
- Personalized Recommendations
- Study Plan and Resources

## Data Flow Example

### Input:
```python
student = {
    'lessons_per_week': 5,
    'exercises_per_lesson': 10,
    'weeks_in_course': 8,
    'average_lesson_score': 70,
    'study_consistency': 'Fairly Consistent',
    'assignment_timeliness': 'Usually On Time',
    # ... more fields
}
```

### Translation:
```python
features = {
    'sum': 5 * 10 * 8 * 3 = 1200,
    'count': 5 * 8 = 40,
    'score': 70,
    'engagement_cv': 0.4,  # from consistency
    'submission_timeliness': 0,  # from timeliness
    # ... 53 more features
}
```

### Prediction:
```python
model.predict(features) → "Good Progress (B/B+)"
model.predict_proba(features) → [0.05, 0.15, 0.65, 0.15]
confidence = 0.78
persona_id = 2  # Engaged Achiever
```

### Feedback:
```python
strengths = ["Strong performance", "Consistent practice"]
focus_areas = ["Increase to 7+ lessons/week"]
action_plan = ["Week 1-2: 7 lessons/week", "Practice weak skill"]
resources = ["BBC Learning English", "Duolingo"]
```

## Technology Stack

```
Frontend:  Streamlit 1.30+
           - Interactive forms and widgets
           - Real-time predictions
           - Plotly visualizations
           
Backend:   Python 3.12
           - Feature engineering pipeline
           - ML model inference
           - Recommendation generation
           
ML Stack:  LightGBM Classifier
           - 58 engineered features
           - 4-class classification
           - ~90.8% accuracy
           
           Scikit-learn
           - StandardScaler for normalization
           - OneHotEncoder for categoricals
           - Model persistence
           
Data:      Pandas DataFrames
           - CSV data loading from data/
           - Feature transformation
           - Result formatting

Models:    Joblib persistence
           - model.pkl (LightGBM)
           - scaler.pkl (StandardScaler)
           - encoder.pkl (OneHotEncoder)
           - Additional models in models/
```

## File Structure

```
educationcare/
├── recommender_app.py  # Main Streamlit application
├── notebooks/
│   └── Final.ipynb                # Model training & feature engineering
├── data/                          # Source datasets (7 CSV files)
│   ├── studentRegistration.csv
│   ├── studentInfo.csv
│   ├── studentVle.csv
│   ├── studentAssessment.csv
│   ├── courses.csv
│   ├── vle.csv
│   └── assessments.csv
├── models/                        # Trained models (5 .pkl files)
│   ├── model.pkl                  # Main LightGBM 58-feature model
│   ├── scaler.pkl                 # StandardScaler for features
│   ├── encoder.pkl                # OneHotEncoder for categoricals
│   ├── target_encoder.pkl         # LabelEncoder for targets
│   └── umap_reducer.pkl           # UMAP dimensionality reduction
├── config/                        # Configuration files
│   ├── feature_names.json         # Feature list
│   └── metadata.json              # Model metadata
├── proficiency/                   # English proficiency testing module
│   ├── streamlit_english_test.py  # English test Streamlit app
│   ├── train_english_proficiency_model.ipynb  # Training notebook
│   └── english_proficiency_model.pkl  # Trained model
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # This file
│   ├── FEATURES_EXPLAINED.md      # Feature documentation
│   ├── PROJECT_SUMMARY.md         # Project overview
│   └── SETUP_INSTRUCTIONS.md      # Setup guide
├── scripts/                       # Utility scripts
│   ├── quick_start.py
│   └── save_model.py
├── utils/                         # Utility functions
├── requirements.txt               # Python dependencies
└── README.md                      # Main documentation
```

## Future Architecture Enhancements

### Phase 2: Recommender System

```
┌─────────────────────┐
│  Current System     │
│  (Prediction +      │
│   Feedback)         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Recommender System Layer                   │
│  ┌───────────────────────────────────────┐ │
│  │ Content-Based Filtering               │ │
│  │ - Recommend lessons for weak skills   │ │
│  │ - Suggest exercises at right level    │ │
│  └───────────────────────────────────────┘ │
│  ┌───────────────────────────────────────┐ │
│  │ Collaborative Filtering               │ │
│  │ - What helped similar students?       │ │
│  │ - Success patterns from peers         │ │
│  └───────────────────────────────────────┘ │
│  ┌───────────────────────────────────────┐ │
│  │ Learning Path Optimization            │ │
│  │ - Personalized curriculum             │ │
│  │ - Adaptive difficulty                 │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Phase 3: Progress Tracking

```
┌─────────────────────┐
│  Database Layer     │
│  - Student history  │
│  - Progress metrics │
│  - Intervention log │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Analytics Engine   │
│  - Trend analysis   │
│  - Effectiveness    │
│  - A/B testing      │
└─────────────────────┘
```

---

**This architecture prioritizes simplicity, maintainability, and immediate value delivery.** 🏗️
