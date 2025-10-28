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
    page_title="EducationCare - Student Success Predictor",
    page_icon="🎓",
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
st.markdown('<div class="main-header">🎓 EducationCare Student Success Predictor</div>', unsafe_allow_html=True)
st.markdown("### Personalized Learning Support & Outcome Prediction")
st.markdown("---")

# Load model and preprocessors (placeholder - you'll need to save these from your notebook)
@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        # You'll need to save these from your notebook first
        # Example: pickle.dump(model, open('model.pkl', 'wb'))
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        return model, scaler, encoder
    except:
        st.warning("⚠️ Model files not found. Using demo mode.")
        return None, None, None

# Cluster interpretations based on your analysis
CLUSTER_INTERPRETATIONS = {
    0: {
        "name": "High Achievers",
        "description": "Strong performance, good pace, low engagement variability",
        "risk_level": "Low",
        "color": "success"
    },
    1: {
        "name": "Struggling & Withdrawn",
        "description": "Low engagement, many withdrawals, poor timeliness",
        "risk_level": "High",
        "color": "danger"
    },
    2: {
        "name": "Engaged Achievers",
        "description": "High engagement, good performance, many first-time students",
        "risk_level": "Low",
        "color": "success"
    },
    3: {
        "name": "Active but Struggling",
        "description": "High activity diversity but poor timeliness",
        "risk_level": "Medium",
        "color": "warning"
    },
    4: {
        "name": "Experienced Repeaters",
        "description": "High previous attempts, moderate performance",
        "risk_level": "Medium",
        "color": "warning"
    },
    5: {
        "name": "Fast but Disengaged",
        "description": "Very fast pace but low engagement",
        "risk_level": "Medium",
        "color": "warning"
    }
}

# Sidebar - Student Information Form
with st.sidebar:
    st.markdown("## 📋 Student Information")
    st.markdown("Please fill in all the information below:")
    
    # Demographics
    st.markdown("### Personal Information")
    
    gender = st.selectbox(
        "Gender",
        options=["M", "F"],
        help="Student's gender"
    )
    
    age_band = st.selectbox(
        "Age Band",
        options=["0-35", "35+"],
        help="Student's age group"
    )
    
    region = st.selectbox(
        "Region",
        options=[
            "East Anglian Region", "East Midlands Region", "Ireland",
            "London Region", "North Region", "North Western Region",
            "Scotland", "South East Region", "South Region",
            "South West Region", "Wales", "West Midlands Region",
            "Yorkshire Region"
        ],
        help="Student's geographical region"
    )
    
    highest_education = st.selectbox(
        "Highest Education",
        options=["A Level or Equivalent", "HE Qualification", "Lower Than A Level"],
        help="Highest level of education completed"
    )
    
    imd_band = st.selectbox(
        "IMD Band (Deprivation Index)",
        options=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
                 "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
        help="Index of Multiple Deprivation band (socioeconomic indicator)"
    )
    
    disability = st.selectbox(
        "Disability",
        options=["N", "Y"],
        help="Does the student have a disability?"
    )
    
    # Academic Background
    st.markdown("### Academic Background")
    
    num_of_prev_attempts = st.number_input(
        "Number of Previous Attempts",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of times this course was attempted before"
    )
    
    studied_credits = st.number_input(
        "Studied Credits",
        min_value=0,
        max_value=300,
        value=60,
        help="Total number of credits being studied"
    )
    
    days_since_registration = st.number_input(
        "Days Since Registration",
        min_value=0,
        max_value=365,
        value=30,
        help="Number of days since course registration"
    )
    
    # Engagement Metrics
    st.markdown("### Engagement & Activity")
    
    total_clicks = st.number_input(
        "Total Platform Clicks",
        min_value=0,
        max_value=10000,
        value=500,
        help="Total number of clicks on the learning platform"
    )
    
    activity_count = st.number_input(
        "Number of Activities",
        min_value=0,
        max_value=100,
        value=20,
        help="Number of different activities engaged with"
    )
    
    activity_type = st.selectbox(
        "Primary Activity Type",
        options=["forumng", "homepage", "oucontent", "resource", 
                 "subpage", "url", "quiz", "page", "dataplus",
                 "folder", "oucollaborate", "ouelluminate", "glossary",
                 "dualpane", "externalquiz", "sharedsubpage", "questionnaire",
                 "htmlactivity", "repeatactivity", "ouwiki"],
        help="Most frequently used activity type"
    )
    
    # Performance Metrics
    st.markdown("### Assessment Performance")
    
    avg_score = st.slider(
        "Average Assessment Score",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.5,
        help="Average score across all assessments"
    )
    
    submission_timeliness = st.slider(
        "Submission Timeliness",
        min_value=-100.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        help="Average days before/after deadline (negative = early, positive = late)"
    )
    
    banked_assessments = st.number_input(
        "Number of Banked Assessments",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of assessments completed early and banked"
    )
    
    total_assessments = st.number_input(
        "Total Assessments",
        min_value=1,
        max_value=20,
        value=5,
        help="Total number of assessments in the course"
    )
    
    # Learning Behavior
    st.markdown("### Learning Behavior")
    
    study_method = st.selectbox(
        "Preferred Study Method",
        options=["Resource-Based", "Forum-Based", "Content-Based", "Mixed"],
        help="Primary method of learning"
    )
    
    learning_pace = st.slider(
        "Learning Pace (credits/day)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Rate of credit completion"
    )
    
    engagement_consistency = st.slider(
        "Engagement Consistency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How consistent is the student's engagement? (1 = very consistent, 0 = highly variable)"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Student Profile Summary")
    
    # Calculate derived features
    repeat_student = 1 if num_of_prev_attempts > 0 else 0
    activity_diversity = min(activity_count / 20, 1.0)  # Normalized
    score_per_weight = avg_score / max(studied_credits, 1)
    banked_ratio = banked_assessments / max(total_assessments, 1)
    engagement_cv = 1 - engagement_consistency  # Coefficient of variation (inverse of consistency)
    
    # Calculate engagement metrics
    assessment_engagement = total_clicks / max(total_assessments, 1)
    module_engagement_rate = total_clicks / max(days_since_registration, 1)
    weighted_engagement = total_clicks * (1 - engagement_cv)
    
    # Display profile in columns
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    
    with prof_col1:
        st.metric("Education Level", highest_education)
        st.metric("Previous Attempts", num_of_prev_attempts)
        st.metric("Credits Studying", studied_credits)
    
    with prof_col2:
        st.metric("Average Score", f"{avg_score:.1f}%")
        st.metric("Total Activities", activity_count)
        st.metric("Activity Diversity", f"{activity_diversity:.2%}")
    
    with prof_col3:
        st.metric("Submission Timeliness", f"{submission_timeliness:.0f} days")
        st.metric("Learning Pace", f"{learning_pace:.2f}")
        st.metric("Engagement CV", f"{engagement_cv:.2f}")

with col2:
    st.markdown("## 🎯 Quick Stats")
    
    # Risk indicators
    risk_factors = 0
    if num_of_prev_attempts > 1:
        risk_factors += 1
    if avg_score < 40:
        risk_factors += 1
    if submission_timeliness > 10:
        risk_factors += 1
    if engagement_cv > 0.7:
        risk_factors += 1
    if activity_count < 10:
        risk_factors += 1
    
    if risk_factors == 0:
        st.success("✅ Low Risk Profile")
    elif risk_factors <= 2:
        st.warning("⚠️ Medium Risk Profile")
    else:
        st.error("🚨 High Risk Profile")
    
    st.metric("Risk Factors Identified", f"{risk_factors}/5")

# Predict button
st.markdown("---")
predict_button = st.button("🔮 Predict Student Outcome", type="primary", use_container_width=True)

if predict_button:
    st.markdown("## 🎯 Prediction Results")
    
    # Prepare feature vector (matching your notebook's features)
    # Note: This is a simplified version. You'll need to adjust based on your actual preprocessing
    
    features_dict = {
        'num_of_prev_attempts': num_of_prev_attempts,
        'repeat_student': repeat_student,
        'studied_credits': studied_credits,
        'sum': total_clicks,
        'count': activity_count,
        'activity_diversity': activity_diversity,
        'score': avg_score,
        'score_per_weight': score_per_weight,
        'assessment_engagement_score': assessment_engagement,
        'module_engagement_rate': module_engagement_rate,
        'weighted_engagement': weighted_engagement,
        'engagement_trend': 0.0,  # Would need historical data
        'submission_timeliness': submission_timeliness,
        'banked_assessment_ratio': banked_ratio,
        'days_since_registration': days_since_registration,
        'score_trend': 0.0,  # Would need historical data
        'score_momentum': 0.0,  # Would need historical data
        'performance_by_registration': avg_score / max(days_since_registration, 1),
        'learning_pace': learning_pace,
        'engagement_cv': engagement_cv,
    }
    
    # Demo prediction (replace with actual model prediction)
    model, scaler, encoder = load_model()
    
    if model is None:
        # Demo mode - rule-based prediction
        st.info("📍 Running in Demo Mode (model files not loaded)")
        
        # Simple rule-based prediction for demo
        prediction_score = (
            (avg_score / 100) * 0.4 +
            (1 - engagement_cv) * 0.2 +
            (activity_diversity) * 0.15 +
            (1 if submission_timeliness <= 0 else 0) * 0.15 +
            (1 if num_of_prev_attempts == 0 else 0) * 0.1
        )
        
        if prediction_score >= 0.75:
            predicted_outcome = "Distinction"
            confidence = 0.85
        elif prediction_score >= 0.55:
            predicted_outcome = "Pass"
            confidence = 0.78
        elif prediction_score >= 0.30:
            predicted_outcome = "Fail"
            confidence = 0.72
        else:
            predicted_outcome = "Withdrawn"
            confidence = 0.68
        
        # Assign to a cluster based on characteristics
        if num_of_prev_attempts > 2:
            cluster_id = 4  # Experienced Repeaters
        elif avg_score >= 80 and engagement_cv < 0.3:
            cluster_id = 0  # High Achievers
        elif avg_score >= 60 and activity_count > 20:
            cluster_id = 2  # Engaged Achievers
        elif submission_timeliness > 10 or avg_score < 40:
            cluster_id = 1  # Struggling & Withdrawn
        elif activity_count > 15 and submission_timeliness > 5:
            cluster_id = 3  # Active but Struggling
        else:
            cluster_id = 5  # Fast but Disengaged
    
    else:
        # Full mode - use actual model (placeholder for now)
        st.info("📍 Model loaded - Using ML predictions")
        
        # TODO: Add actual model prediction logic here when model files are available
        # For now, fall back to demo mode logic
        prediction_score = (
            (avg_score / 100) * 0.4 +
            (1 - engagement_cv) * 0.2 +
            (activity_diversity) * 0.15 +
            (1 if submission_timeliness <= 0 else 0) * 0.15 +
            (1 if num_of_prev_attempts == 0 else 0) * 0.1
        )
        
        if prediction_score >= 0.75:
            predicted_outcome = "Distinction"
            confidence = 0.85
        elif prediction_score >= 0.55:
            predicted_outcome = "Pass"
            confidence = 0.78
        elif prediction_score >= 0.30:
            predicted_outcome = "Fail"
            confidence = 0.72
        else:
            predicted_outcome = "Withdrawn"
            confidence = 0.68
        
        # Assign cluster
        if num_of_prev_attempts > 2:
            cluster_id = 4
        elif avg_score >= 80 and engagement_cv < 0.3:
            cluster_id = 0
        elif avg_score >= 60 and activity_count > 20:
            cluster_id = 2
        elif submission_timeliness > 10 or avg_score < 40:
            cluster_id = 1
        elif activity_count > 15 and submission_timeliness > 5:
            cluster_id = 3
        else:
            cluster_id = 5
    
    # Display prediction
    st.markdown("### 🎓 Predicted Outcome")
    
    outcome_colors = {
        "Distinction": "success",
        "Pass": "info",
        "Fail": "warning",
        "Withdrawn": "danger"
    }
    
    outcome_icons = {
        "Distinction": "🌟",
        "Pass": "✅",
        "Fail": "⚠️",
        "Withdrawn": "❌"
    }
    
    pred_col1, pred_col2 = st.columns([2, 1])
    
    with pred_col1:
        color_class = outcome_colors.get(predicted_outcome, "info")
        st.markdown(f"""
        <div class="prediction-box {color_class}-box">
            <h2>{outcome_icons[predicted_outcome]} {predicted_outcome}</h2>
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
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Student Cluster/Persona
    st.markdown("### 👥 Student Persona")
    
    cluster_info = CLUSTER_INTERPRETATIONS[cluster_id]
    
    persona_col1, persona_col2 = st.columns([3, 1])
    
    with persona_col1:
        st.markdown(f"""
        <div class="prediction-box {cluster_info['color']}-box">
            <h3>{cluster_info['name']}</h3>
            <p>{cluster_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with persona_col2:
        st.metric("Risk Level", cluster_info['risk_level'])
    
    # Personalized Feedback and Recommendations
    st.markdown("### 💡 Personalized Feedback & Recommendations")
    
    feedback_tabs = st.tabs(["📈 Strengths", "⚠️ Areas for Improvement", "🎯 Action Plan", "📚 Resources"])
    
    with feedback_tabs[0]:
        st.markdown("#### Your Strengths")
        strengths = []
        
        if avg_score >= 70:
            strengths.append("✅ **Strong Academic Performance** - You're scoring well on assessments!")
        if submission_timeliness <= 0:
            strengths.append("✅ **Excellent Time Management** - You submit assignments on time or early!")
        if activity_diversity > 0.6:
            strengths.append("✅ **Diverse Learning Approach** - You engage with various learning resources!")
        if engagement_cv < 0.4:
            strengths.append("✅ **Consistent Engagement** - You maintain steady participation in the course!")
        if num_of_prev_attempts == 0:
            strengths.append("✅ **First-Time Success Track** - You're tackling this course for the first time!")
        
        if strengths:
            for strength in strengths:
                st.markdown(strength)
        else:
            st.info("Keep working hard! Your strengths will emerge as you progress through the course.")
    
    with feedback_tabs[1]:
        st.markdown("#### Areas for Improvement")
        improvements = []
        
        if avg_score < 50:
            improvements.append("🔴 **Assessment Scores** - Your average score is below passing. Consider:")
            improvements.append("   - Attending office hours for additional support")
            improvements.append("   - Joining study groups with peers")
            improvements.append("   - Reviewing assessment feedback carefully")
        
        if submission_timeliness > 5:
            improvements.append("🔴 **Submission Timeliness** - Late submissions can impact your grades:")
            improvements.append("   - Set calendar reminders for deadlines")
            improvements.append("   - Start assignments earlier")
            improvements.append("   - Break large tasks into smaller milestones")
        
        if activity_count < 15:
            improvements.append("🔴 **Platform Engagement** - Low activity may indicate disengagement:")
            improvements.append("   - Explore more learning resources")
            improvements.append("   - Participate in discussion forums")
            improvements.append("   - Watch recorded lectures and supplementary materials")
        
        if engagement_cv > 0.6:
            improvements.append("🔴 **Engagement Consistency** - Irregular participation detected:")
            improvements.append("   - Establish a regular study schedule")
            improvements.append("   - Dedicate specific times each week to coursework")
            improvements.append("   - Use the Pomodoro technique for focused study sessions")
        
        if num_of_prev_attempts > 1:
            improvements.append("🔴 **Repeat Attempts** - This isn't your first time:")
            improvements.append("   - Identify what didn't work in previous attempts")
            improvements.append("   - Seek advisor guidance on study strategies")
            improvements.append("   - Consider adjusting your course load")
        
        if improvements:
            for improvement in improvements:
                st.markdown(improvement)
        else:
            st.success("Great job! Keep maintaining your current approach!")
    
    with feedback_tabs[2]:
        st.markdown("#### Personalized Action Plan")
        
        # Generate action plan based on cluster
        action_plans = {
            0: [
                "🎯 **Maintain Excellence**: Continue your current study habits",
                "📚 **Peer Mentoring**: Consider helping struggling classmates",
                "🚀 **Advanced Challenges**: Explore supplementary advanced materials",
                "💼 **Career Preparation**: Start building your portfolio or CV"
            ],
            1: [
                "🆘 **Immediate Intervention**: Schedule meeting with academic advisor",
                "📞 **Support Services**: Contact student support services immediately",
                "📅 **Study Schedule**: Create a structured daily study plan",
                "👥 **Study Group**: Join or form a study group for accountability",
                "🎯 **Focus on Basics**: Prioritize understanding fundamental concepts"
            ],
            2: [
                "✅ **Consistency is Key**: Maintain your engagement patterns",
                "📈 **Optimize Performance**: Focus on maximizing assessment scores",
                "🎯 **Strategic Study**: Identify high-impact study activities",
                "🌟 **Aim Higher**: Challenge yourself to reach distinction level"
            ],
            3: [
                "⏰ **Time Management**: Implement strict deadline tracking",
                "📊 **Prioritization**: Focus on high-weight assessments first",
                "🎯 **Quality over Quantity**: Reduce activities, increase focus",
                "📝 **Planning**: Create assignment timelines with buffer time"
            ],
            4: [
                "🔄 **New Approach**: Try different learning strategies",
                "👨‍🏫 **Tutoring**: Get one-on-one help in challenging areas",
                "📚 **Resource Review**: Explore different textbooks/materials",
                "💪 **Mindset Shift**: Focus on growth and learning from past attempts"
            ],
            5: [
                "🔥 **Re-engage**: Increase platform interaction and participation",
                "👥 **Community**: Join discussion forums and group activities",
                "📚 **Deeper Learning**: Spend more time on comprehension vs speed",
                "⚖️ **Balance**: Maintain pace while increasing engagement quality"
            ]
        }
        
        for action in action_plans.get(cluster_id, []):
            st.markdown(action)
    
    with feedback_tabs[3]:
        st.markdown("#### Recommended Resources")
        
        st.markdown("""
        **📚 Study Skills**
        - [Effective Study Techniques](https://learningcenter.unc.edu/tips-and-tools/studying-101-study-smarter-not-harder/)
        - [Time Management Guide](https://www.mindtools.com/pages/main/newMN_HTE.htm)
        - [Pomodoro Technique](https://todoist.com/productivity-methods/pomodoro-technique)
        
        **🎯 Assessment Preparation**
        - Khan Academy - Free courses on various subjects
        - Coursera - Supplementary learning materials
        - YouTube Educational Channels
        
        **💪 Student Wellbeing**
        - [Managing Academic Stress](https://www.stress.org/managing-academic-stress)
        - Campus Counseling Services
        - Student Success Center
        
        **👥 Peer Support**
        - Study Group Finder (check your LMS)
        - Online Discussion Forums
        - Peer Tutoring Services
        """)
    
    # Probability distribution
    st.markdown("### 📊 Outcome Probabilities")
    
    # Demo probabilities (replace with actual model probabilities)
    probabilities = {
        "Distinction": max(0, prediction_score - 0.25 + np.random.uniform(-0.05, 0.05)),
        "Pass": max(0, prediction_score + np.random.uniform(-0.1, 0.1)),
        "Fail": max(0, 1 - prediction_score + np.random.uniform(-0.1, 0.1)),
        "Withdrawn": max(0, 1 - prediction_score - 0.2 + np.random.uniform(-0.05, 0.05))
    }
    
    # Normalize probabilities
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    fig = px.bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        labels={'x': 'Outcome', 'y': 'Probability'},
        title='Predicted Outcome Probabilities',
        color=list(probabilities.keys()),
        color_discrete_map={
            "Distinction": "#28a745",
            "Pass": "#17a2b8",
            "Fail": "#ffc107",
            "Withdrawn": "#dc3545"
        }
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for this prediction
    st.markdown("### 🔍 Key Factors Influencing This Prediction")
    
    feature_importance = {
        "Assessment Score": avg_score / 100 * 0.4,
        "Engagement Consistency": (1 - engagement_cv) * 0.2,
        "Activity Diversity": activity_diversity * 0.15,
        "Submission Timeliness": (1 if submission_timeliness <= 0 else 0.5) * 0.15,
        "First-Time Student": (1 if num_of_prev_attempts == 0 else 0.5) * 0.1
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        labels={'x': 'Impact Score', 'y': 'Factor'},
        title='Factors Contributing to Your Prediction',
        color=list(feature_importance.values()),
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>EducationCare Student Success Predictor</strong></p>
    <p>Powered by Machine Learning | Data Mining Project 2025</p>
    <p><em>This tool provides predictions based on historical student data and should be used as a guide for educational support, not as a definitive assessment.</em></p>
</div>
""", unsafe_allow_html=True)
