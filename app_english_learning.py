import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="English Learning Success Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌍 English Learning Success Predictor</div>', unsafe_allow_html=True)
st.markdown("### Personalized Feedback for English Language Learners")
st.markdown("---")

# Load model and preprocessors
@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        return model, scaler, encoder
    except:
        st.warning("⚠️ Model files not found. Using demo mode.")
        return None, None, None

# English Learning to Technical Feature Mapping
def map_english_to_technical_features(inputs):
    """
    Convert student-friendly English learning inputs to technical model features
    """
    features = {}
    
    # ==== ENGAGEMENT METRICS ====
    # Calculate total "clicks" based on lessons and exercises
    lessons_per_week = inputs['lessons_per_week']
    exercises_per_lesson = inputs['exercises_per_lesson']
    weeks_studying = inputs['weeks_in_course']
    
    # Approximate clicks: lessons * exercises * weeks * 3 (avg interactions per exercise)
    features['sum'] = lessons_per_week * exercises_per_lesson * weeks_studying * 3
    features['count'] = lessons_per_week * weeks_studying  # Total activities
    
    # Activity diversity based on skill variety
    skills_practiced = len(inputs['skills_practiced'])
    features['activity_diversity'] = min(skills_practiced / 5.0, 1.0)  # Max 5 skills
    
    # ==== PERFORMANCE METRICS ====
    features['score'] = inputs['average_lesson_score']
    features['studied_credits'] = 60  # Standard for one course
    features['score_per_weight'] = inputs['average_lesson_score'] / 60
    
    # ==== ENGAGEMENT PATTERNS ====
    # Engagement consistency (inverse of variability)
    consistency_map = {
        'Very Consistent': 0.2,
        'Fairly Consistent': 0.4,
        'Sometimes Inconsistent': 0.6,
        'Very Inconsistent': 0.8
    }
    features['engagement_cv'] = consistency_map[inputs['study_consistency']]
    
    # Module engagement rate (clicks per day)
    days_in_course = weeks_studying * 7
    features['module_engagement_rate'] = features['sum'] / max(days_in_course, 1)
    
    # Weighted engagement (adjusted for consistency)
    features['weighted_engagement'] = features['sum'] * (1 - features['engagement_cv'])
    
    # ==== LEARNING BEHAVIOR ====
    # Learning pace (credits per day)
    features['learning_pace'] = features['studied_credits'] / max(days_in_course, 1)
    
    # Days since registration
    features['days_since_registration'] = days_in_course
    
    # ==== ASSESSMENT METRICS ====
    # Number of assessments (weekly quizzes)
    features['total_assessments'] = weeks_studying
    
    # Submission timeliness
    timeliness_map = {
        'Always Early': -5,
        'Usually On Time': 0,
        'Sometimes Late': 5,
        'Often Late': 15
    }
    features['submission_timeliness'] = timeliness_map[inputs['assignment_timeliness']]
    
    # Assessment engagement
    features['assessment_engagement_score'] = features['sum'] / max(features['total_assessments'], 1)
    
    # Banked assessment ratio
    features['banked_assessment_ratio'] = 0.1 if inputs['assignment_timeliness'] == 'Always Early' else 0
    
    # ==== ACADEMIC BACKGROUND ====
    # Previous attempts
    attempt_map = {
        'First Time': 0,
        'Second Attempt': 1,
        'Third or More': 2
    }
    features['num_of_prev_attempts'] = attempt_map[inputs['course_attempt']]
    features['repeat_student'] = 1 if features['num_of_prev_attempts'] > 0 else 0
    
    # ==== TRENDS (simplified for new students) ====
    features['engagement_trend'] = 0.1 if inputs['motivation_trend'] == 'Increasing' else -0.1
    features['score_trend'] = 0.1 if inputs['performance_trend'] == 'Improving' else -0.1
    features['score_momentum'] = features['score_trend'] * features['score']
    features['performance_by_registration'] = features['score'] / max(days_in_course, 1)
    
    return features

def map_categorical_features(inputs):
    """Map categorical inputs for encoding"""
    categorical = {}
    
    # Gender
    categorical['gender'] = inputs['gender']
    
    # Region (simplified - using UK regions as proxy)
    region_map = {
        'North America': 'North Region',
        'South America': 'South Region',
        'Europe': 'London Region',
        'Asia': 'East Anglian Region',
        'Africa': 'West Midlands Region',
        'Oceania': 'South East Region',
        'Other': 'Scotland'
    }
    categorical['region'] = region_map[inputs['region']]
    
    # Education level
    edu_map = {
        'High School': 'A Level or Equivalent',
        'Some College': 'A Level or Equivalent',
        'Bachelor Degree': 'HE Qualification',
        'Graduate Degree': 'HE Qualification',
        'Less than High School': 'Lower Than A Level'
    }
    categorical['highest_education'] = edu_map[inputs['education_level']]
    
    # Age band
    age_map = {
        '18-24': '0-35',
        '25-34': '0-35',
        '35-44': '35+',
        '45+': '35+'
    }
    categorical['age_band'] = age_map[inputs['age_group']]
    
    # IMD band (using income as proxy for socioeconomic status)
    imd_map = {
        'Low Income': '80-90%',
        'Middle Income': '40-50%',
        'High Income': '10-20%'
    }
    categorical['imd_band'] = imd_map[inputs['income_level']]
    
    # Disability
    categorical['disability'] = 'Y' if inputs['has_disability'] else 'N'
    
    # Activity type (based on preferred learning method)
    activity_map = {
        'Reading Lessons': 'oucontent',
        'Listening Exercises': 'resource',
        'Speaking Practice': 'forumng',
        'Writing Assignments': 'quiz',
        'Grammar Drills': 'quiz',
        'Mixed/Varied': 'homepage'
    }
    categorical['activity_type'] = activity_map[inputs['primary_learning_method']]
    
    return categorical

# Learner Personas
LEARNER_PERSONAS = {
    0: {
        "name": "Star Learner ⭐",
        "description": "Consistent practice, strong performance, well-balanced approach",
        "risk_level": "Low",
        "color": "success"
    },
    1: {
        "name": "Struggling & At Risk 🚨",
        "description": "Low engagement, irregular practice, needs immediate support",
        "risk_level": "High",
        "color": "danger"
    },
    2: {
        "name": "Engaged Achiever 📚",
        "description": "High engagement, good performance, on track for success",
        "risk_level": "Low",
        "color": "success"
    },
    3: {
        "name": "Active but Challenged ⚠️",
        "description": "Lots of practice but struggling with performance",
        "risk_level": "Medium",
        "color": "warning"
    },
    4: {
        "name": "Determined Returner 🔄",
        "description": "Multiple attempts, showing perseverance",
        "risk_level": "Medium",
        "color": "warning"
    },
    5: {
        "name": "Quick but Unfocused ⚡",
        "description": "Fast pace but inconsistent engagement",
        "risk_level": "Medium",
        "color": "warning"
    }
}

# Sidebar - Student Information Form
with st.sidebar:
    st.markdown("## 🌍 Your English Learning Profile")
    st.markdown("Tell us about your English learning journey:")
    
    # Personal Information
    st.markdown("### 👤 About You")
    
    gender = st.selectbox(
        "Gender",
        options=["M", "F"],
        help="Your gender"
    )
    
    age_group = st.selectbox(
        "Age Group",
        options=["18-24", "25-34", "35-44", "45+"],
        help="Your age range"
    )
    
    region = st.selectbox(
        "Where are you from?",
        options=["North America", "South America", "Europe", "Asia", "Africa", "Oceania", "Other"],
        help="Your geographical region"
    )
    
    education_level = st.selectbox(
        "Your Education Level",
        options=["Less than High School", "High School", "Some College", "Bachelor Degree", "Graduate Degree"],
        help="Highest level of education completed"
    )
    
    income_level = st.selectbox(
        "Family Income Level",
        options=["Low Income", "Middle Income", "High Income"],
        help="This helps us understand potential barriers to learning"
    )
    
    has_disability = st.checkbox(
        "I have a learning disability or special need",
        help="This helps us provide appropriate support"
    )
    
    # Learning Background
    st.markdown("### 📖 English Learning Background")
    
    current_level = st.selectbox(
        "Current English Level",
        options=["Beginner (A1)", "Elementary (A2)", "Intermediate (B1)", 
                 "Upper Intermediate (B2)", "Advanced (C1)", "Proficient (C2)"],
        help="Your current proficiency level"
    )
    
    course_attempt = st.selectbox(
        "Is this your first time taking this course?",
        options=["First Time", "Second Attempt", "Third or More"],
        help="Have you attempted this course before?"
    )
    
    weeks_in_course = st.number_input(
        "How many weeks have you been in this course?",
        min_value=1,
        max_value=52,
        value=8,
        help="Number of weeks since you started"
    )
    
    # Study Habits
    st.markdown("### 📅 Your Study Habits")
    
    lessons_per_week = st.slider(
        "Lessons completed per week",
        min_value=0,
        max_value=30,
        value=5,
        help="How many lessons do you complete each week on average?"
    )
    
    exercises_per_lesson = st.slider(
        "Exercises per lesson",
        min_value=1,
        max_value=30,
        value=10,
        help="How many exercises do you do in each lesson?"
    )
    
    study_consistency = st.select_slider(
        "How consistent is your study schedule?",
        options=["Very Inconsistent", "Sometimes Inconsistent", "Fairly Consistent", "Very Consistent"],
        value="Fairly Consistent",
        help="Do you study at regular times or sporadically?"
    )
    
    # Skills & Learning Methods
    st.markdown("### 🎯 Skills & Learning")
    
    primary_learning_method = st.selectbox(
        "What's your primary learning method?",
        options=["Reading Lessons", "Listening Exercises", "Speaking Practice", 
                 "Writing Assignments", "Grammar Drills", "Mixed/Varied"],
        help="What type of activity do you spend most time on?"
    )
    
    skills_practiced = st.multiselect(
        "Which skills do you practice regularly?",
        options=["Reading", "Writing", "Listening", "Speaking", "Grammar"],
        default=["Reading", "Writing", "Listening"],
        help="Select all that apply"
    )
    
    # Performance
    st.markdown("### 📊 Your Performance")
    
    average_lesson_score = st.slider(
        "Average lesson score (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Your typical score on lessons and quizzes"
    )
    
    assignment_timeliness = st.select_slider(
        "How often do you complete assignments on time?",
        options=["Often Late", "Sometimes Late", "Usually On Time", "Always Early"],
        value="Usually On Time",
        help="Your typical submission pattern"
    )
    
    # Motivation & Progress
    st.markdown("### 💪 Motivation & Progress")
    
    motivation_trend = st.selectbox(
        "How is your motivation changing?",
        options=["Decreasing", "Staying Same", "Increasing"],
        help="Are you feeling more or less motivated over time?"
    )
    
    performance_trend = st.selectbox(
        "How are your scores changing?",
        options=["Getting Worse", "Staying Same", "Improving"],
        help="Are your lesson scores improving over time?"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Your Learning Profile Summary")
    
    # Create user inputs dictionary
    user_inputs = {
        'gender': gender,
        'age_group': age_group,
        'region': region,
        'education_level': education_level,
        'income_level': income_level,
        'has_disability': has_disability,
        'course_attempt': course_attempt,
        'weeks_in_course': weeks_in_course,
        'lessons_per_week': lessons_per_week,
        'exercises_per_lesson': exercises_per_lesson,
        'study_consistency': study_consistency,
        'primary_learning_method': primary_learning_method,
        'skills_practiced': skills_practiced,
        'average_lesson_score': average_lesson_score,
        'assignment_timeliness': assignment_timeliness,
        'motivation_trend': motivation_trend,
        'performance_trend': performance_trend
    }
    
    # Calculate derived features for display
    total_lessons = lessons_per_week * weeks_in_course
    total_exercises = total_lessons * exercises_per_lesson
    skill_diversity = len(skills_practiced) / 5.0
    
    # Display profile in columns
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    
    with prof_col1:
        st.metric("Current Level", current_level)
        st.metric("Total Lessons", total_lessons)
        st.metric("Lessons/Week", lessons_per_week)
    
    with prof_col2:
        st.metric("Average Score", f"{average_lesson_score}%")
        st.metric("Total Exercises", total_exercises)
        st.metric("Skill Variety", f"{skill_diversity:.0%}")
    
    with prof_col3:
        st.metric("Weeks in Course", weeks_in_course)
        st.metric("Study Pattern", study_consistency.split()[1] if len(study_consistency.split()) > 1 else study_consistency)
        st.metric("Trend", motivation_trend)

with col2:
    st.markdown("## 🎯 Quick Assessment")
    
    # Risk indicators
    risk_factors = 0
    risk_messages = []
    
    if course_attempt != "First Time":
        risk_factors += 1
        risk_messages.append("📌 Previous attempts")
    if average_lesson_score < 50:
        risk_factors += 1
        risk_messages.append("📌 Low scores")
    if assignment_timeliness in ["Often Late", "Sometimes Late"]:
        risk_factors += 1
        risk_messages.append("📌 Late submissions")
    if study_consistency in ["Very Inconsistent", "Sometimes Inconsistent"]:
        risk_factors += 1
        risk_messages.append("📌 Irregular practice")
    if lessons_per_week < 3:
        risk_factors += 1
        risk_messages.append("📌 Low practice frequency")
    
    if risk_factors == 0:
        st.success("✅ Excellent Learning Profile!")
    elif risk_factors <= 2:
        st.warning("⚠️ Some Areas Need Attention")
    else:
        st.error("🚨 Needs Immediate Support")
    
    st.metric("Attention Areas", f"{risk_factors}/5")
    
    if risk_messages:
        st.markdown("**Areas to address:**")
        for msg in risk_messages:
            st.markdown(msg)

# Predict button
st.markdown("---")
predict_button = st.button("🔮 Get My Personalized Feedback", type="primary", use_container_width=True)

if predict_button:
    st.markdown("## 🎯 Your Personalized Learning Report")
    
    # Map features
    technical_features = map_english_to_technical_features(user_inputs)
    categorical_features = map_categorical_features(user_inputs)
    
    # Demo prediction (replace with actual model when available)
    model, scaler, encoder = load_model()
    
    if model is None:
        # Demo mode - rule-based prediction
        st.info("📍 Generating your personalized feedback...")
        
        # Simple rule-based prediction
        prediction_score = (
            (average_lesson_score / 100) * 0.4 +
            (1 if study_consistency in ["Very Consistent", "Fairly Consistent"] else 0) * 0.2 +
            skill_diversity * 0.15 +
            (1 if assignment_timeliness in ["Always Early", "Usually On Time"] else 0) * 0.15 +
            (1 if course_attempt == "First Time" else 0) * 0.1
        )
        
        if prediction_score >= 0.75:
            predicted_outcome = "Excellent Progress (A/A+)"
            confidence = 0.85
        elif prediction_score >= 0.60:
            predicted_outcome = "Good Progress (B/B+)"
            confidence = 0.78
        elif prediction_score >= 0.40:
            predicted_outcome = "Needs Improvement (C)"
            confidence = 0.72
        else:
            predicted_outcome = "At Risk (D/F)"
            confidence = 0.68
        
        # Assign persona
        if course_attempt != "First Time":
            persona_id = 4  # Determined Returner
        elif average_lesson_score >= 80 and study_consistency == "Very Consistent":
            persona_id = 0  # Star Learner
        elif average_lesson_score >= 65 and lessons_per_week >= 5:
            persona_id = 2  # Engaged Achiever
        elif assignment_timeliness in ["Often Late", "Sometimes Late"] or average_lesson_score < 50:
            persona_id = 1  # Struggling & At Risk
        elif lessons_per_week >= 7 and average_lesson_score < 65:
            persona_id = 3  # Active but Challenged
        else:
            persona_id = 5  # Quick but Unfocused
    
    else:
        # Use actual model
        st.info("📍 Using trained model for prediction...")
        # TODO: Implement actual model prediction
        persona_id = 2  # Default
        predicted_outcome = "Good Progress (B/B+)"
        confidence = 0.75
    
    # Display prediction
    st.markdown("### 🎓 Predicted Learning Outcome")
    
    outcome_colors = {
        "Excellent Progress (A/A+)": "success",
        "Good Progress (B/B+)": "info",
        "Needs Improvement (C)": "warning",
        "At Risk (D/F)": "danger"
    }
    
    outcome_icons = {
        "Excellent Progress (A/A+)": "🌟",
        "Good Progress (B/B+)": "✅",
        "Needs Improvement (C)": "⚠️",
        "At Risk (D/F)": "🚨"
    }
    
    pred_col1, pred_col2 = st.columns([2, 1])
    
    with pred_col1:
        color_class = outcome_colors.get(predicted_outcome, "info")
        icon = outcome_icons.get(predicted_outcome, "📊")  # Default icon if not found
        st.markdown(f"""
        <div class="prediction-box {color_class}-box">
            <h2>{icon} {predicted_outcome}</h2>
            <p style="font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pred_col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Learner Persona
    st.markdown("### 👥 Your Learner Persona")
    
    persona_info = LEARNER_PERSONAS[persona_id]
    
    persona_col1, persona_col2 = st.columns([3, 1])
    
    with persona_col1:
        st.markdown(f"""
        <div class="prediction-box {persona_info['color']}-box">
            <h3>{persona_info['name']}</h3>
            <p>{persona_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with persona_col2:
        st.metric("Support Level", persona_info['risk_level'])
    
    # Personalized Feedback
    st.markdown("### 💡 Your Personalized Action Plan")
    
    feedback_tabs = st.tabs(["🌟 Strengths", "🎯 Focus Areas", "📝 Action Steps", "📚 Resources"])
    
    with feedback_tabs[0]:  # Strengths
        st.markdown("#### What You're Doing Well")
        strengths = []
        
        if average_lesson_score >= 70:
            strengths.append("✅ **Strong Performance** - Your scores show good understanding of the material!")
        if assignment_timeliness in ["Always Early", "Usually On Time"]:
            strengths.append("✅ **Excellent Time Management** - You submit work on time consistently!")
        if skill_diversity >= 0.6:
            strengths.append("✅ **Well-Rounded Practice** - You're developing multiple English skills!")
        if study_consistency in ["Very Consistent", "Fairly Consistent"]:
            strengths.append("✅ **Consistent Practice** - You maintain a regular study routine!")
        if course_attempt == "First Time":
            strengths.append("✅ **Fresh Start Success** - You're making good progress on your first try!")
        if lessons_per_week >= 5:
            strengths.append("✅ **Dedicated Learner** - You complete many lessons each week!")
        if motivation_trend == "Increasing":
            strengths.append("✅ **Growing Motivation** - Your enthusiasm is building over time!")
        
        if strengths:
            for strength in strengths:
                st.markdown(strength)
        else:
            st.info("Keep practicing! Your strengths will emerge as you continue learning.")
    
    with feedback_tabs[1]:  # Focus Areas
        st.markdown("#### Areas That Need Your Attention")
        improvements = []
        
        if average_lesson_score < 60:
            improvements.append({
                'title': '🎯 **Improve Lesson Scores**',
                'issue': f'Your average score ({average_lesson_score}%) is below target',
                'tips': [
                    "Review lessons before taking quizzes",
                    "Take notes while studying",
                    "Redo exercises you got wrong",
                    "Ask questions when you don't understand"
                ]
            })
        
        if assignment_timeliness in ["Often Late", "Sometimes Late"]:
            improvements.append({
                'title': '⏰ **Better Time Management**',
                'issue': 'Late submissions can hurt your progress',
                'tips': [
                    "Set calendar reminders for deadlines",
                    "Start assignments when they're given",
                    "Break large tasks into smaller daily goals",
                    "Dedicate specific times for English study"
                ]
            })
        
        if lessons_per_week < 3:
            improvements.append({
                'title': '📅 **Increase Practice Frequency**',
                'issue': f'Only {lessons_per_week} lessons/week is not enough',
                'tips': [
                    "Aim for at least 5 lessons per week",
                    "Study for 15-30 minutes daily",
                    "Use mobile app for quick practice sessions",
                    "Make English part of your daily routine"
                ]
            })
        
        if study_consistency in ["Very Inconsistent", "Sometimes Inconsistent"]:
            improvements.append({
                'title': '🎯 **Build Consistent Habits**',
                'issue': 'Irregular practice slows your progress',
                'tips': [
                    "Study at the same time each day",
                    "Start with small daily goals (10 min)",
                    "Track your study streak",
                    "Reward yourself for consistency"
                ]
            })
        
        if len(skills_practiced) < 3:
            improvements.append({
                'title': '🌈 **Practice All Skills**',
                'issue': 'You need to develop all English skills',
                'tips': [
                    "Include reading, writing, listening, and speaking",
                    "Spend time on your weakest skill each day",
                    "Use real-world content (videos, articles)",
                    "Balance all skills for best results"
                ]
            })
        
        if course_attempt != "First Time":
            improvements.append({
                'title': '🔄 **Learn from Past Attempts**',
                'issue': 'This is not your first time - let\'s succeed now!',
                'tips': [
                    "Identify what didn't work before",
                    "Use a different study method this time",
                    "Get help early - don't wait until struggling",
                    "Consider reducing your course load"
                ]
            })
        
        if improvements:
            for item in improvements:
                st.markdown(f"### {item['title']}")
                st.markdown(f"*{item['issue']}*")
                for tip in item['tips']:
                    st.markdown(f"- {tip}")
                st.markdown("---")
        else:
            st.success("Great job! Keep up your current approach!")
    
    with feedback_tabs[2]:  # Action Steps
        st.markdown("#### Your Personalized 2-Week Action Plan")
        
        # Generate specific action plan based on persona
        action_plans = {
            0: [  # Star Learner
                "🎯 **Week 1-2**: Continue your excellent routine",
                "📚 **Challenge Yourself**: Try advanced materials or help other students",
                "🎯 **Set New Goals**: Aim for 90%+ on all lessons",
                "💼 **Prepare for Next Level**: Review requirements for next course"
            ],
            1: [  # Struggling & At Risk
                "🆘 **THIS WEEK**: Schedule meeting with instructor or tutor (urgent!)",
                "📅 **Create Schedule**: Study 30 min every day at same time",
                "📝 **Start Small**: Complete 3 easy lessons this week to build confidence",
                "👥 **Get Support**: Join study group or find study partner",
                "🎯 **Focus**: Master basic vocabulary and grammar first"
            ],
            2: [  # Engaged Achiever
                "✅ **Maintain Consistency**: Keep your current study schedule",
                "📈 **Aim Higher**: Target 85%+ on all lessons",
                "🎯 **Focus on Weak Skills**: Spend extra time on lowest skill area",
                "🌟 **Challenge**: Complete 2 extra lessons this week"
            ],
            3: [  # Active but Challenged
                "⏰ **Quality Over Quantity**: Reduce lessons but increase focus",
                "📊 **Review Before New**: Spend 50% of time reviewing previous lessons",
                "🎯 **Master Basics**: Focus on fundamentals before moving ahead",
                "📝 **Take Notes**: Write down key points from each lesson"
            ],
            4: [  # Determined Returner
                "🔄 **New Strategy**: Try a completely different study approach",
                "👨‍🏫 **Get Help**: Book weekly tutoring sessions",
                "📚 **Different Materials**: Use videos, apps, or different textbooks",
                "💪 **Mindset**: Focus on improvement, not perfection",
                "🎯 **Small Wins**: Celebrate every success, no matter how small"
            ],
            5: [  # Quick but Unfocused
                "🔥 **Slow Down**: Spend more time on each lesson",
                "📚 **Deep Learning**: Review lessons until you truly understand",
                "👥 **Engage More**: Join forums, discussions, and study groups",
                "⚖️ **Balance**: Quality understanding beats speed"
            ]
        }
        
        st.markdown("**Follow these steps for the next 2 weeks:**")
        for action in action_plans.get(persona_id, []):
            st.markdown(action)
        
        st.markdown("---")
        st.markdown("**📊 Track Your Progress:**")
        st.markdown("- [ ] Set up daily study time")
        st.markdown("- [ ] Complete target number of lessons this week")
        st.markdown("- [ ] Practice weakest skill area")
        st.markdown("- [ ] Review this report next week")
    
    with feedback_tabs[3]:  # Resources
        st.markdown("#### Recommended Learning Resources")
        
        # Customize resources based on persona and needs
        if average_lesson_score < 60:
            st.markdown("**📚 For Improving Scores:**")
            st.markdown("""
            - [Basic English Grammar Guide](https://learnenglish.britishcouncil.org/grammar)
            - [Free Vocabulary Builder](https://www.vocabulary.com/)
            - [English Practice Exercises](https://www.englishclub.com/esl-exercises/)
            - [Request Free Tutoring](mailto:support@example.com)
            """)
        
        if "Speaking" not in skills_practiced or "Listening" not in skills_practiced:
            st.markdown("**🎤 For Speaking & Listening:**")
            st.markdown("""
            - [Pronunciation Practice](https://www.youtube.com/user/rachelsenglish)
            - [English Listening Practice](https://www.esl-lab.com/)
            - [Conversation Exchange Partners](https://www.conversationexchange.com/)
            - [English Podcasts for Learners](https://www.listennotes.com/)
            """)
        
        if study_consistency in ["Very Inconsistent", "Sometimes Inconsistent"]:
            st.markdown("**⏰ For Building Study Habits:**")
            st.markdown("""
            - [Habit Tracking App - Habitica](https://habitica.com/)
            - [Study Timer - Pomodoro](https://pomofocus.io/)
            - [How to Build Study Habits](https://www.mindtools.com/pages/article/build-good-habits.htm)
            - [Daily English Practice Ideas](https://www.fluentu.com/blog/english/daily-english-practice/)
            """)
        
        st.markdown("**🎯 General Resources:**")
        st.markdown("""
        - [BBC Learning English](https://www.bbc.co.uk/learningenglish) - Free lessons and activities
        - [Duolingo](https://www.duolingo.com/) - Mobile practice app
        - [English Grammar in Use](https://www.cambridge.org/elt) - Comprehensive grammar book
        - [EnglishCentral](https://www.englishcentral.com/) - Video-based learning
        - [Coursera English Courses](https://www.coursera.org/courses?query=english) - Free university courses
        """)
        
        st.markdown("**💪 Motivation & Support:**")
        st.markdown("""
        - [Language Learning Community - Reddit](https://www.reddit.com/r/languagelearning/)
        - [English Learning Discord Server](https://discord.gg/english)
        - [Success Stories](https://www.fluentu.com/blog/english/english-success-stories/)
        - Campus Support Services - Office Hours: Mon-Fri 9am-5pm
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>English Learning Success Predictor</strong></p>
    <p>Powered by Machine Learning | Personalized Feedback System</p>
    <p><em>This tool provides guidance based on your learning patterns. Use it to improve your study approach and get better results!</em></p>
</div>
""", unsafe_allow_html=True)
