# 📊 Features Used in Streamlit App (app_english_learning.py)

## Overview

The Streamlit app uses **TWO types of features** to make predictions:

1. **Technical/Numeric Features** (21 features) - Derived from student inputs
2. **Categorical Features** (7 features) - Demographics and preferences

These match the features your LightGBM model was trained on from `Final.ipynb`.

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

### Step 1: Student Fills Form
```
Student inputs (friendly):
- Lessons per week: 5
- Exercises per lesson: 10
- Weeks in course: 8
- Average score: 70%
- Study consistency: "Fairly Consistent"
- Assignment timeliness: "Usually On Time"
- Course attempt: "First Time"
- ... (more inputs)
```

### Step 2: Features are Calculated
```python
# App calls these functions:
technical_features = map_english_to_technical_features(user_inputs)
categorical_features = map_categorical_features(user_inputs)

# Results in:
{
    'sum': 1200,
    'count': 40,
    'activity_diversity': 0.6,
    'score': 70,
    'engagement_cv': 0.4,
    'submission_timeliness': 0,
    'num_of_prev_attempts': 0,
    'gender': 'M',
    'region': 'London Region',
    # ... all 28 features
}
```

### Step 3: Model Uses Features
```python
# These features would be passed to your LightGBM model:
model.predict(feature_vector)
# Returns: "Good Progress (B/B+)"
```

---

## 🎯 Comparison: Student View vs Model View

| What Student Sees | What Model Receives |
|-------------------|---------------------|
| "I complete 5 lessons per week" | `count = 40, sum = 1200` |
| "I'm fairly consistent" | `engagement_cv = 0.4` |
| "My average score is 70%" | `score = 70, score_per_weight = 1.17` |
| "I usually submit on time" | `submission_timeliness = 0` |
| "This is my first attempt" | `num_of_prev_attempts = 0, repeat_student = 0` |
| "I practice 3 skills" | `activity_diversity = 0.6` |

---

## 🔍 Feature Source Summary

| Feature Category | Count | Derived From |
|------------------|-------|--------------|
| Engagement Metrics | 5 | Lessons, exercises, weeks, skills |
| Performance Metrics | 3 | Average score, credits |
| Engagement Patterns | 1 | Study consistency |
| Learning Behavior | 2 | Weeks in course, pace |
| Assessment Metrics | 3 | Timeliness, engagement |
| Academic Background | 2 | Course attempts |
| Trends | 4 | Motivation & performance trends |
| Other | 1 | Total assessments |
| **Categorical** | 7 | Demographics & preferences |
| **TOTAL** | **28** | |

---

## ⚠️ Important Notes

1. **Currently in Demo Mode**: The app calculates all 28 features but doesn't send them to a model yet. Instead, it uses a simple rule-based prediction.

2. **To Use Your Real Model**: You need to:
   - Save your trained model from `Final.ipynb`
   - Load it in the app
   - Create a proper feature vector with all 28 features
   - Apply any necessary scaling/encoding
   - Get prediction from the model

3. **Feature Engineering**: The app's feature calculations match your notebook's feature engineering, ensuring consistency between training and prediction.

---

## 📊 Feature Coverage

These 28 features cover the same feature space as your trained model expects:
- ✅ All numeric features from your model
- ✅ All categorical features from your model
- ✅ Proper encoding and scaling (when model is connected)
- ✅ Derived features matching your training pipeline

**The translation layer ensures students provide natural inputs while the model receives the technical features it expects!**
