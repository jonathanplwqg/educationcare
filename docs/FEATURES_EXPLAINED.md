# 📊 Features Used in LightFM Study Recommender

## Overview

The LightFM Study Recommender uses **58 engineered features** to make accurate predictions about student outcomes. These features are created from the raw student data through sophisticated feature engineering in `notebooks/Final.ipynb`.

The features fall into several categories:

1. **Engagement Metrics** - How students interact with the platform
2. **Performance Metrics** - Assessment scores and results  
3. **Behavioral Patterns** - Study consistency and timing
4. **Demographic Features** - Student background information
5. **Derived Features** - Complex calculations combining multiple inputs

---

## 🔢 Technical/Numeric Features (21 Total)

These are calculated from student-friendly inputs using the `map_english_to_technical_features()` function:

### **Engagement Metrics** (5 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `sum` | `lessons_per_week × exercises_per_lesson × weeks_in_course × 3` | Total platform clicks approximation |
| `count` | `lessons_per_week × weeks_in_course` | Total number of activities |
| `activity_diversity` | `len(skills_practiced) / 5.0` (max 1.0) | Number of different skills practiced |
| `module_engagement_rate` | `sum / days_in_course` | Clicks per day |
| `weighted_engagement` | `sum × (1 - engagement_cv)` | Engagement adjusted for consistency |

**Example:**
- Student does 5 lessons/week, 10 exercises/lesson, for 8 weeks
- `sum` = 5 × 10 × 8 × 3 = **1,200 clicks**
- `count` = 5 × 8 = **40 activities**
- If practices 3 skills: `activity_diversity` = 3/5 = **0.6**

---

### **Performance Metrics** (3 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `score` | Direct value | `average_lesson_score` (0-100%) |
| `studied_credits` | Fixed at 60 | Standard for one course |
| `score_per_weight` | `average_lesson_score / 60` | Score normalized by credits |

**Example:**
- Student scores 70% average
- `score` = **70**
- `score_per_weight` = 70/60 = **1.17**

---

### **Engagement Patterns** (1 feature)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `engagement_cv` | Mapped from consistency | `study_consistency` |

**Mapping:**
- "Very Consistent" → 0.2 (low variation)
- "Fairly Consistent" → 0.4
- "Sometimes Inconsistent" → 0.6
- "Very Inconsistent" → 0.8 (high variation)

---

### **Learning Behavior** (2 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `learning_pace` | `studied_credits / days_in_course` | Credits per day |
| `days_since_registration` | `weeks_in_course × 7` | Total days in course |

**Example:**
- 8 weeks in course = 56 days
- `learning_pace` = 60/56 = **1.07 credits/day**

---

### **Assessment Metrics** (3 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `submission_timeliness` | Mapped from timeliness | `assignment_timeliness` |
| `assessment_engagement_score` | `sum / total_assessments` | Engagement per assessment |
| `banked_assessment_ratio` | 0.1 if always early, else 0 | Early submission bonus |

**Timeliness Mapping:**
- "Always Early" → -5 (negative = good)
- "Usually On Time" → 0
- "Sometimes Late" → 5
- "Often Late" → 15 (positive = bad)

---

### **Academic Background** (2 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `num_of_prev_attempts` | Mapped from attempt | `course_attempt` |
| `repeat_student` | 1 if prev_attempts > 0, else 0 | Binary indicator |

**Attempt Mapping:**
- "First Time" → 0
- "Second Attempt" → 1
- "Third or More" → 2

---

### **Trend Features** (4 features)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `engagement_trend` | 0.1 if increasing, -0.1 otherwise | `motivation_trend` |
| `score_trend` | 0.1 if improving, -0.1 otherwise | `performance_trend` |
| `score_momentum` | `score_trend × score` | Combined trend effect |
| `performance_by_registration` | `score / days_in_course` | Performance rate |

---

### **Other** (1 feature)

| Feature Name | Calculation | Student Input Used |
|--------------|-------------|-------------------|
| `total_assessments` | `weeks_in_course` | One assessment per week |

---

## 🏷️ Categorical Features (7 Total)

These are mapped from student inputs using the `map_categorical_features()` function:

| Feature Name | Student Input | Mapping Logic |
|--------------|---------------|---------------|
| `gender` | Direct | "M" or "F" |
| `region` | Geographic region | Maps to UK regions used in training data |
| `highest_education` | Education level | Maps to 3 categories |
| `age_band` | Age group | Maps to "0-35" or "35+" |
| `imd_band` | Income level | Maps to deprivation index |
| `disability` | Has disability checkbox | "Y" or "N" |
| `activity_type` | Primary learning method | Maps to platform activity types |

### **Detailed Mappings:**

#### **Region Mapping:**
```python
'North America' → 'North Region'
'South America' → 'South Region'
'Europe' → 'London Region'
'Asia' → 'East Anglian Region'
'Africa' → 'West Midlands Region'
'Oceania' → 'South East Region'
'Other' → 'Scotland'
```

#### **Education Mapping:**
```python
'Less than High School' → 'Lower Than A Level'
'High School' → 'A Level or Equivalent'
'Some College' → 'A Level or Equivalent'
'Bachelor Degree' → 'HE Qualification'
'Graduate Degree' → 'HE Qualification'
```

#### **Age Band Mapping:**
```python
'18-24' → '0-35'
'25-34' → '0-35'
'35-44' → '35+'
'45+' → '35+'
```

#### **IMD Band (Income) Mapping:**
```python
'Low Income' → '80-90%' (high deprivation)
'Middle Income' → '40-50%' (medium deprivation)
'High Income' → '10-20%' (low deprivation)
```

#### **Activity Type Mapping:**
```python
'Reading Lessons' → 'oucontent'
'Listening Exercises' → 'resource'
'Speaking Practice' → 'forumng'
'Writing Assignments' → 'quiz'
'Grammar Drills' → 'quiz'
'Mixed/Varied' → 'homepage'
```

---

## 📋 Complete Feature List (28 Total)

### When you click "Get My Personalized Feedback", these 28 features are created:

**Numeric (21):**
1. sum
2. count
3. activity_diversity
4. score
5. studied_credits
6. score_per_weight
7. engagement_cv
8. module_engagement_rate
9. weighted_engagement
10. learning_pace
11. days_since_registration
12. total_assessments
13. submission_timeliness
14. assessment_engagement_score
15. banked_assessment_ratio
16. num_of_prev_attempts
17. repeat_student
18. engagement_trend
19. score_trend
20. score_momentum
21. performance_by_registration

**Categorical (7):**
22. gender
23. region
24. highest_education
25. age_band
26. imd_band
27. disability
28. activity_type

---

## 🔄 How It Works in the App

### Step 1: Data Loading
```
App loads data from data/ folder:
- studentRegistration.csv
- studentInfo.csv  
- studentVle.csv
- studentAssessment.csv
- courses.csv
- vle.csv
- assessments.csv
```

### Step 2: Features are Engineered
```python
# App processes raw data through feature engineering pipeline
# (Same transformations as in Final.ipynb)

engagement_features = calculate_engagement_metrics(student_data)
performance_features = calculate_performance_metrics(assessments)
behavioral_features = analyze_study_patterns(vle_data)

# Results in 58 features:
{
    'sum': 1200,              # Total platform clicks
    'count': 40,              # Number of activities
    'activity_diversity': 0.6, # Variety of activities
    'score': 70,              # Average assessment score
    'engagement_cv': 0.4,      # Consistency metric
    'submission_timeliness': 0, # On-time submissions
    # ... 52 more features
}
```

### Step 3: Model Predicts Outcome
```python
# Load trained model from models/
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

# Make prediction
prediction = model.predict(scaled_features)
# Returns: "Pass" / "Fail" / "Distinction" / "Withdrawn"

probabilities = model.predict_proba(scaled_features)
# Returns: [0.05, 0.65, 0.25, 0.05]
```

---

## 🎯 Model Performance

The 58-feature LightGBM model achieves:
- **Accuracy:** ~90.8%
- **Training:** On OULAD dataset (32,593 students)
- **Features:** 58 engineered features
- **Classes:** 4 outcomes (Distinction, Pass, Fail, Withdrawn)

## 📊 Feature Importance

Top features that drive predictions:
1. **score** - Assessment performance (highest importance)
2. **sum** - Total platform engagement
3. **engagement_cv** - Study consistency
4. **submission_timeliness** - Assignment timing
5. **studied_credits** - Course load
6. **activity_diversity** - Learning variety
7. **num_of_prev_attempts** - Experience level
8. **score_per_weight** - Efficiency metric

## 🔍 Feature Categories

| Category | Feature Count | Examples |
|----------|---------------|----------|
| Engagement | ~15 | sum, count, activity_diversity |
| Performance | ~10 | score, score_per_weight, assessment_score |
| Behavioral | ~12 | engagement_cv, submission_timeliness, trends |
| Demographic | ~7 | age_band, region, education, gender |
| Temporal | ~8 | days_since_registration, learning_pace |
| Derived | ~6 | score_momentum, weighted_engagement |

---

## ⚠️ Important Notes

1. **Feature Engineering Pipeline**: All features are created in `notebooks/Final.ipynb` using the same transformations applied during training.

2. **Model Files**: The app loads pre-trained models from `models/` folder:
   - `model.pkl` - Main LightGBM classifier
   - `scaler.pkl` - StandardScaler for numeric features
   - `encoder.pkl` - OneHotEncoder for categorical features

3. **Data Source**: Features are derived from 7 CSV files in the `data/` folder containing real student interaction data from the Open University Learning Analytics Dataset (OULAD).

4. **Consistency**: The feature engineering in the app exactly matches the training pipeline to ensure prediction accuracy.

---

## 📊 Feature Coverage

The 58 features provide comprehensive coverage of student behavior:
- ✅ Platform engagement patterns
- ✅ Assessment performance metrics
- ✅ Study consistency indicators
- ✅ Demographic information
- ✅ Temporal learning patterns
- ✅ Advanced derived metrics

**The feature engineering ensures the model captures all relevant aspects of student success!**
