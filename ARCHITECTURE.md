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
│              STREAMLIT INTERFACE (app_english_learning.py)       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Sidebar: Student-Friendly Input Form                     │  │
│  │  - How many lessons per week? (slider: 0-30)             │  │
│  │  - How consistent is your study? (dropdown)              │  │
│  │  - What's your average score? (slider: 0-100%)           │  │
│  │  - Which skills do you practice? (multi-select)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSLATION LAYER                               │
│  map_english_to_technical_features()                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  lessons_per_week (5) → count, sum                        │  │
│  │  study_consistency → engagement_cv                        │  │
│  │  average_score (70%) → score, score_per_weight           │  │
│  │  skills_practiced → activity_diversity                    │  │
│  │  primary_method → activity_type                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  map_categorical_features()                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  region (Asia) → East Anglian Region                     │  │
│  │  education → highest_education                            │  │
│  │  age_group → age_band                                     │  │
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
**File:** `app_english_learning.py` (lines 200-450)
**Purpose:** Collect natural student inputs
**Key Features:**
- Sliders, dropdowns, multi-select for easy input
- Grouped by category (Study Habits, Performance, etc.)
- Tooltips for clarity

### 2. Translation Layer
**File:** `app_english_learning.py` (lines 100-180)
**Functions:**
- `map_english_to_technical_features()` - Converts learning activities to metrics
- `map_categorical_features()` - Maps demographics to model categories

**Key Mappings:**
```python
# Study activity → Engagement metrics
total_clicks = lessons * exercises * weeks * 3
activity_count = lessons * weeks
engagement_cv = consistency_score

# Performance → Assessment metrics
score = average_lesson_score
submission_timeliness = timeliness_mapping[student_input]
```

### 3. ML Model Layer
**File:** `model.pkl` (saved from `Final.ipynb`)
**Type:** LightGBM Classifier
**Input:** 58 features (20 numeric + 38 categorical)
**Output:** 4-class prediction (Distinction/Pass/Fail/Withdrawn)
**Note:** Currently in demo mode; can be connected

### 4. Feedback Engine
**File:** `app_english_learning.py` (lines 600-900)
**Components:**

**A. Risk Assessment**
```python
risk_factors = 0
if previous_attempts > 0: risk_factors += 1
if avg_score < 50: risk_factors += 1
if late_submissions: risk_factors += 1
if inconsistent_study: risk_factors += 1
if low_practice: risk_factors += 1
```

**B. Persona Assignment**
```python
6 learner personas based on:
- Previous attempts
- Score levels
- Engagement consistency
- Practice frequency
```

**C. Feedback Generation**
```python
FOR each potential issue:
    IF condition_met:
        ADD specific feedback with:
            - Problem identification
            - Targeted tips (4-6 items)
            - Related resources
```

**D. Action Plan**
```python
persona_based_plans = {
    0: ["Continue excellence", "Challenge yourself"],
    1: ["Urgent intervention", "Basic mastery"],
    2: ["Maintain consistency", "Optimize"],
    3: ["Quality over quantity", "Review"],
    4: ["New strategy", "Get help"],
    5: ["Slow down", "Deep learning"]
}
```

### 5. Display Layer
**File:** `app_english_learning.py` (lines 500-900)
**Components:**
- Profile Summary (metrics cards)
- Quick Assessment (risk indicators)
- Prediction Box (outcome + confidence)
- Persona Card (learner type)
- 4-Tab Feedback System

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
Frontend:  Streamlit 1.50.0
           - Forms, widgets, layouts
           - Plotly for visualizations
           
Backend:   Python 3.12
           - Feature engineering
           - Rule-based feedback
           
ML Model:  LightGBM (from Final.ipynb)
           - 58 features
           - 4-class classification
           
Data:      Pandas DataFrames
           - Feature transformation
           - Result formatting
```

## File Structure

```
educationcare/
├── app_english_learning.py      # Main application (NEW)
├── app.py                        # Original technical version
├── Final.ipynb                   # Model training notebook
├── model.pkl                     # Trained model (optional)
├── scaler.pkl                    # Feature scaler (optional)
├── encoder.pkl                   # Categorical encoder (optional)
├── ENGLISH_LEARNING_GUIDE.md    # Documentation (NEW)
├── PROJECT_SUMMARY.md            # Summary (NEW)
├── ARCHITECTURE.md               # This file (NEW)
└── requirements.txt              # Dependencies
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
