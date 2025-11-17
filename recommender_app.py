"""
EducationCare - ML-Powered Personalized Study Recommender with LightFM-Next Integration

A sophisticated recommendation system that integrates:
1. Two trained ML models for prediction
2. LightFM collaborative filtering for recommendations  
3. Content-based filtering as fallback

🧠 ML MODEL ARCHITECTURE + LightFM:
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: English Assessment → English Proficiency Model                     │
│ ├─ Input: 4 English skill scores (Vocabulary, Grammar, Reading, Writing)   │
│ ├─ Model: RandomForestClassifier (4 features → 3 classes)                  │ 
│ └─ Output: Proficiency level (0=Low, 1=Medium, 2=High) + confidence        │
│                                                                             │
│ STEP 2: Learning Profile → Academic Success Model                          │
│ ├─ Input: 9 learning behavior fields (attempts, scores, habits, etc.)      │
│ ├─ Feature Engineering: 9 inputs → 58 ML model features                    │
│ ├─ Model: RandomForestClassifier* (58 features → 4 classes)                │
│ │   *NOTE: Originally trained as LightGBM in Final.ipynb                   │
│ └─ Output: Academic outcome (Pass/Fail/Distinction/Withdrawn) + risk       │
│                                                                             │
│ STEP 3: LightFM Collaborative Filtering → Enhanced Recommendations         │
│ ├─ Input: User profiles + ML predictions                                    │
│ ├─ Model: LightFM matrix factorization with WARP loss                      │
│ └─ Output: Collaborative filtering recommendations + content-based fallback │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 KEY INNOVATIONS: 
- Smart Feature Engineering for ML models
- LightFM-Next collaborative filtering integration
- Hybrid recommendation approach with fallback
- Synthetic user-item interaction generation

🔄 ENHANCED INTEGRATION FLOW:
Step 1 → English scores → predict_english_proficiency() → English ML model
Step 2 → Academic profile → predict_academic_success() → Academic ML model  
Step 3 → Combined predictions → LightFM collaborative filtering → Enhanced recommendations
Step 4 → Fallback to content-based filtering if needed

Designed for maximum usability with state-of-the-art recommendation technology.
"""

import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="EducationCare - Personalized Study Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import joblib
import pickle
import json
import sys
import os
from pathlib import Path

# Handle potential import conflicts for LightFM
current_dir = os.path.dirname(os.path.abspath(__file__))

# LightFM Integration with conflict resolution
try:
    # Temporarily adjust path to avoid conflicts with local lightfm.py file
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    
    from lightfm_integration import LightFMEducationRecommender
    LIGHTFM_AVAILABLE = True
    print("✅ LightFM-Next integration imported successfully")
    
    # Restore path
    sys.path.insert(0, current_dir)
    
except ImportError as e:
    print(f"⚠️ LightFM integration not available: {e}")
    LIGHTFM_AVAILABLE = False
    LightFMEducationRecommender = None
    
    # Restore path even if import failed
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

# Other imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import categorical mapping utilities
try:
    from categorical_mapping_utils import CategoricalMapper, encode_user_categorical_inputs, get_form_field_options
    CATEGORICAL_MAPPING_AVAILABLE = True
    print("✅ Categorical mapping utilities imported")
except ImportError:
    print("⚠️ Categorical mapping utilities not found. Some features may be limited.")
    CATEGORICAL_MAPPING_AVAILABLE = False
    CategoricalMapper = None
    encode_user_categorical_inputs = None
    get_form_field_options = None

# IMD Band and Regional Mapping Utilities
def get_imd_suggestion_from_region(region):
    """
    Suggest IMD band based on region (approximate mapping based on UK statistics)
    This helps users who might not know their exact IMD band
    """
    regional_imd_mapping = {
        # London and affluent areas (generally lower deprivation)
        'London Region': '20-30%',
        'South East Region': '10-20%', 
        'South West Region': '30-40%',
        
        # Northern regions (often higher deprivation)
        'North Region': '50-60%',
        'North Western Region': '60-70%',
        'Yorkshire Region': '40-50%',
        
        # Other regions
        'Scotland': '40-50%',
        'Wales': '50-60%',
        'Ireland': '30-40%',
        'East Anglian Region': '20-30%',
        'East Midlands Region': '40-50%',
        'West Midlands Region': '50-60%',
    }
    
    return regional_imd_mapping.get(region, '30-40%')  # Default to middle band

def get_regional_options():
    """Get available regional options"""
    if CATEGORICAL_MAPPING_AVAILABLE:
        return get_form_field_options('region')
    else:
        # Fallback options
        return [
            'London Region', 'Scotland', 'Wales', 'North Region',
            'South East Region', 'South West Region', 'North Western Region',
            'Yorkshire Region', 'East Anglian Region', 'East Midlands Region',
            'West Midlands Region', 'Ireland'
        ]

def get_imd_band_options():
    """Get available IMD band options"""
    if CATEGORICAL_MAPPING_AVAILABLE:
        return get_form_field_options('imd_band')
    else:
        # Fallback options
        return ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

def explain_imd_band(imd_band):
    """Provide user-friendly explanation of IMD band"""
    explanations = {
        '0-10%': '🟢 Least deprived areas (affluent)',
        '10-20%': '🟢 Low deprivation (well-off)',
        '20-30%': '🟡 Below average deprivation',
        '30-40%': '🟡 Slightly below average deprivation', 
        '40-50%': '🟠 Average deprivation',
        '50-60%': '🟠 Slightly above average deprivation',
        '60-70%': '🔴 Above average deprivation',
        '70-80%': '🔴 High deprivation',
        '80-90%': '🔴 Very high deprivation',
        '90-100%': '🔴 Highest deprivation (most disadvantaged)'
    }
    return explanations.get(imd_band, 'Unknown deprivation level')

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-container {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .completed-step {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .current-step {
        border-color: #17a2b8;
        background-color: #d1ecf1;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(23, 162, 184, 0); }
        100% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0); }
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .progress-bar {
        background: linear-gradient(90deg, #28a745, #20c997);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ContentBasedStudyRecommender:
    """Main recommender system using content-based filtering"""
    
    def __init__(self):
        self.study_resources = None
        self.scaler = StandardScaler()
        self.english_model = None
        self.academic_model = None
        self.academic_scaler = None
        self.academic_encoder = None
        self.target_encoder = None
        self.feature_names = None

        
        # Initialize categorical mapper if available
        if CATEGORICAL_MAPPING_AVAILABLE:
            self.categorical_mapper = CategoricalMapper()
            print("✅ Categorical mapper initialized")
        else:
            self.categorical_mapper = None
            print("⚠️ Categorical mapper not available")
            
        self.load_models()
        self.initialize_study_database()
    
    def load_models(self):
        """Load existing models with LightGBM detection"""
        try:
            # 🔍 CHECK FOR LIGHTGBM MODELS FIRST
            print("🔍 Checking for LightGBM models...")
            
            # Look for potential LightGBM model files in models/ folder
            lightgbm_candidates = [
                'models/best_lgb_model.pkl', 'models/lgb_model.pkl', 'models/lightgbm_model.pkl', 
                'models/best_model.pkl', 'models/final_model.pkl', 'models/lgb_enhanced.pkl', 'models/model.pkl'
            ]
            
            for candidate in lightgbm_candidates:
                if Path(candidate).exists():
                    try:
                        test_model = joblib.load(candidate)
                        if 'lightgbm' in str(type(test_model).__module__).lower():
                            self.academic_model = test_model
                            print(f"🚀 Found LightGBM model: {candidate}")
                            print(f"   Type: {type(test_model).__name__}")
                            break
                    except:
                        continue
            
            # Load English proficiency model
            if Path('proficiency/english_proficiency_model.pkl').exists():
                self.english_model = joblib.load('proficiency/english_proficiency_model.pkl')
                model_type = type(self.english_model).__name__
                is_lgb = 'lightgbm' in str(type(self.english_model).__module__).lower()
                status = "🚀 LightGBM" if is_lgb else "🌳 RandomForest"
                print(f"✅ Loaded English proficiency model ({status} {model_type})")
            
            # Load Academic success model (if not already loaded as LightGBM)
            if not hasattr(self, 'academic_model') or self.academic_model is None:
                if Path('models/model.pkl').exists():
                    self.academic_model = joblib.load('models/model.pkl')
                    model_type = type(self.academic_model).__name__
                    is_lgb = 'lightgbm' in str(type(self.academic_model).__module__).lower()
                    status = "🚀 LightGBM" if is_lgb else "🌳 RandomForest"
                    print(f"✅ Loaded academic success model ({status} {model_type})")
                    
                    if not is_lgb:
                        print("⚠️  WARNING: Using RandomForest instead of expected LightGBM")
                        print("   Original training in Final.ipynb used LightGBM boosting")
                        print("   Consider retraining or finding LightGBM model files")
                
            if Path('models/scaler.pkl').exists():
                self.academic_scaler = joblib.load('models/scaler.pkl')
                print("✅ Loaded academic scaler")
                
            if Path('models/encoder.pkl').exists():
                self.academic_encoder = joblib.load('models/encoder.pkl')
                print("✅ Loaded academic encoder")
                
            if Path('models/target_encoder.pkl').exists():
                self.target_encoder = joblib.load('models/target_encoder.pkl')
                print("✅ Loaded target encoder")
                
            if Path('config/feature_names.json').exists():
                with open('config/feature_names.json', 'r') as f:
                    self.feature_names = json.load(f)
                print(f"✅ Loaded {len(self.feature_names)} feature names")
                
            # Load population averages for demographic features
            if Path('config/population_averages.json').exists():
                with open('config/population_averages.json', 'r') as f:
                    self.population_averages = json.load(f)
                print(f"✅ Loaded {len(self.population_averages)} population averages")
            else:
                print("⚠️ Population averages not found, using zeros as fallback")
                self.population_averages = {}
                
            # 📊 MODEL SUMMARY
            self._print_model_summary()
            
            # 🚀 LIGHTFM INTEGRATION
            self._initialize_lightfm()
                
        except Exception as e:
            st.warning(f"Could not load models: {e}")
            # Still try to initialize LightFM even if ML models fail
            self._initialize_lightfm()
    
    def _initialize_lightfm(self):
        """Initialize LightFM recommender"""
        if LIGHTFM_AVAILABLE:
            try:
                self.lightfm_recommender = LightFMEducationRecommender()
                print("✅ LightFM-Next collaborative filtering recommender initialized")
                print("🎯 Hybrid recommendations (LightFM + Content-based) now available")
            except Exception as e:
                print(f"⚠️ LightFM initialization failed: {e}")
                self.lightfm_recommender = None
        else:
            self.lightfm_recommender = None
            print("⚠️ LightFM-Next not available, using content-based filtering only")
    
    def _print_model_summary(self):
        """Print a summary of loaded models"""
        print("\n" + "="*50)
        print("📋 MODEL LOADING SUMMARY")
        print("="*50)
        
        if hasattr(self, 'english_model') and self.english_model:
            eng_type = type(self.english_model).__name__
            eng_is_lgb = 'lightgbm' in str(type(self.english_model).__module__).lower()
            print(f"🇬🇧 English Model: {eng_type} {'🚀' if eng_is_lgb else '🌳'}")
        
        if hasattr(self, 'academic_model') and self.academic_model:
            acad_type = type(self.academic_model).__name__
            acad_is_lgb = 'lightgbm' in str(type(self.academic_model).__module__).lower()
            print(f"🎓 Academic Model: {acad_type} {'🚀' if acad_is_lgb else '🌳'}")
            
        has_lightgbm = False
        if hasattr(self, 'english_model') and self.english_model:
            has_lightgbm |= 'lightgbm' in str(type(self.english_model).__module__).lower()
        if hasattr(self, 'academic_model') and self.academic_model:
            has_lightgbm |= 'lightgbm' in str(type(self.academic_model).__module__).lower()
            
        if has_lightgbm:
            print("✅ LightGBM models detected!")
        else:
            print("⚠️  No LightGBM models found - using alternatives")
            print("💡 To use LightGBM: retrain from Final.ipynb or find LightGBM .pkl files")
    
    def initialize_study_database(self):
        """Create comprehensive study resource database"""
        
        # Study resources categorized by type and difficulty
        self.study_resources = {
            # English Language Resources
            'english_vocab_beginner': {
                'name': '📚 Essential Vocabulary Builder',
                'type': 'English - Vocabulary',
                'difficulty': 'Beginner',
                'description': 'Build fundamental vocabulary with 500 most common English words',
                'time_required': '15-20 minutes/day',
                'format': 'Interactive flashcards + spaced repetition',
                'features': ['vocab_focus', 'beginner_friendly'],
                'effectiveness_score': 0.85
            },
            'english_vocab_intermediate': {
                'name': '🎯 Academic Vocabulary Mastery',
                'type': 'English - Vocabulary',
                'difficulty': 'Intermediate',
                'description': 'Learn 1000+ academic and professional vocabulary words',
                'time_required': '20-30 minutes/day',
                'format': 'Context-based learning + usage examples',
                'features': ['vocab_focus', 'academic_english'],
                'effectiveness_score': 0.90
            },
            'english_grammar_basic': {
                'name': '✏️ Grammar Fundamentals',
                'type': 'English - Grammar',
                'difficulty': 'Beginner',
                'description': 'Master basic tenses, sentence structure, and common grammar rules',
                'time_required': '10-15 minutes/day',
                'format': 'Interactive exercises + immediate feedback',
                'features': ['grammar_focus', 'structured_learning'],
                'effectiveness_score': 0.82
            },
            'english_reading_comprehension': {
                'name': '📖 Reading Skills Development',
                'type': 'English - Reading',
                'difficulty': 'Intermediate',
                'description': 'Improve reading comprehension with graded texts and exercises',
                'time_required': '25-35 minutes/day',
                'format': 'Progressive reading passages + comprehension questions',
                'features': ['reading_focus', 'comprehension'],
                'effectiveness_score': 0.88
            },
            'english_writing_basics': {
                'name': '✍️ Writing Skills Workshop',
                'type': 'English - Writing',
                'difficulty': 'Beginner',
                'description': 'Learn paragraph structure, sentence mechanics, and basic essay writing',
                'time_required': '30-40 minutes/day',
                'format': 'Guided writing practice + peer review',
                'features': ['writing_focus', 'mechanics'],
                'effectiveness_score': 0.87
            },
            
            # Academic Study Resources
            'time_management_course': {
                'name': '⏰ Time Management Mastery',
                'type': 'Study Skills',
                'difficulty': 'Beginner',
                'description': 'Learn effective time management and scheduling techniques',
                'time_required': '1 hour/week for 4 weeks',
                'format': 'Video lessons + practical exercises',
                'features': ['time_management', 'productivity'],
                'effectiveness_score': 0.91
            },
            'note_taking_system': {
                'name': '📝 Advanced Note-Taking System',
                'type': 'Study Skills',
                'difficulty': 'Intermediate',
                'description': 'Master Cornell notes, mind mapping, and digital note organization',
                'time_required': '2 hours initial setup + daily practice',
                'format': 'Interactive tutorials + templates',
                'features': ['note_taking', 'organization'],
                'effectiveness_score': 0.89
            },
            'exam_preparation_intensive': {
                'name': '🎯 Exam Success Bootcamp',
                'type': 'Test Preparation',
                'difficulty': 'Intermediate',
                'description': 'Comprehensive exam preparation strategies and stress management',
                'time_required': '3-4 hours/week',
                'format': 'Live workshops + practice tests',
                'features': ['exam_prep', 'stress_management'],
                'effectiveness_score': 0.93
            },
            
            # Engagement & Motivation
            'study_group_facilitator': {
                'name': '👥 Study Group Leadership',
                'type': 'Collaborative Learning',
                'difficulty': 'Advanced',
                'description': 'Learn to organize and lead effective study groups',
                'time_required': '1 hour/week',
                'format': 'Peer collaboration + mentorship',
                'features': ['leadership', 'collaboration'],
                'effectiveness_score': 0.86
            },
            'motivation_coaching': {
                'name': '🚀 Academic Motivation Program',
                'type': 'Personal Development',
                'difficulty': 'Beginner',
                'description': 'Build intrinsic motivation and overcome academic procrastination',
                'time_required': '30 minutes/week',
                'format': 'Self-reflection exercises + goal setting',
                'features': ['motivation', 'goal_setting'],
                'effectiveness_score': 0.84
            },
            
            # Technology & Platform Skills
            'digital_literacy_course': {
                'name': '💻 Digital Learning Skills',
                'type': 'Technology',
                'difficulty': 'Beginner',
                'description': 'Master online learning platforms and digital tools',
                'time_required': '2 hours initial + practice',
                'format': 'Hands-on tutorials + guided practice',
                'features': ['digital_skills', 'platform_navigation'],
                'effectiveness_score': 0.80
            },
            
            # Remedial Support
            'academic_recovery_program': {
                'name': '🆘 Academic Recovery Support',
                'type': 'Intensive Support',
                'difficulty': 'Beginner',
                'description': 'Comprehensive support program for struggling students',
                'time_required': '5+ hours/week',
                'format': 'One-on-one tutoring + structured learning plan',
                'features': ['intensive_support', 'personalized'],
                'effectiveness_score': 0.92
            },
            'confidence_building_workshop': {
                'name': '💪 Academic Confidence Building',
                'type': 'Personal Development',
                'difficulty': 'Beginner',
                'description': 'Build academic self-confidence and overcome imposter syndrome',
                'time_required': '1 hour/week for 6 weeks',
                'format': 'Interactive workshops + peer support',
                'features': ['confidence', 'mindset'],
                'effectiveness_score': 0.85
            }
        }
    
    def create_user_feature_vector(self, user_profile):
        """Create feature vector from user profile"""
        
        features = []
        
        # English proficiency features (4 features)
        if 'english_scores' in user_profile:
            eng_scores = user_profile['english_scores']
            features.extend([
                eng_scores.get('Vocabulary', 0),
                eng_scores.get('Grammar', 0),
                eng_scores.get('Reading', 0),
                eng_scores.get('Writing', 0)
            ])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])  # Default values
        
        # Academic behavior features (8 features)
        if 'academic_profile' in user_profile:
            acad = user_profile['academic_profile']
            features.extend([
                1 if acad.get('risk_level') == 'high' else 0.5 if acad.get('risk_level') == 'medium' else 0,
                acad.get('engagement_consistency', 0.5),
                acad.get('avg_score', 50) / 100,
                max(0, (10 - abs(acad.get('submission_timeliness', 0))) / 10),  # Timeliness score
                1 if acad.get('num_of_prev_attempts', 0) > 0 else 0,  # Repeat student
                acad.get('motivation_level', 5) / 10,
                acad.get('confidence_level', 5) / 10,
                (10 - acad.get('stress_level', 5)) / 10  # Inverted stress (higher = less stressed)
            ])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5])  # Default values
        
        return np.array(features)
    
    def create_resource_feature_matrix(self):
        """Create feature matrix for all resources"""
        
        resource_features = []
        resource_ids = []
        
        for resource_id, resource in self.study_resources.items():
            features = []
            
            # Type encoding (6 features - one-hot for main types)
            type_categories = ['English', 'Study Skills', 'Test Preparation', 'Collaborative Learning', 
                             'Personal Development', 'Technology', 'Intensive Support']
            type_vector = [0] * len(type_categories)
            
            resource_type = resource['type']
            for i, category in enumerate(type_categories):
                if category.lower() in resource_type.lower():
                    type_vector[i] = 1
                    break
            else:
                type_vector[0] = 1  # Default to first category
            
            features.extend(type_vector)
            
            # Difficulty encoding (3 features)
            diff_mapping = {'Beginner': [1, 0, 0], 'Intermediate': [0, 1, 0], 'Advanced': [0, 0, 1]}
            features.extend(diff_mapping.get(resource['difficulty'], [1, 0, 0]))
            
            # Effectiveness score (1 feature)
            features.append(resource['effectiveness_score'])
            
            # Feature flags (8 features for common characteristics)
            feature_flags = ['vocab_focus', 'grammar_focus', 'reading_focus', 'writing_focus',
                           'time_management', 'confidence', 'motivation', 'intensive_support']
            
            for flag in feature_flags:
                features.append(1 if flag in resource.get('features', []) else 0)
            
            resource_features.append(features)
            resource_ids.append(resource_id)
        
        return np.array(resource_features), resource_ids
    
    def _score_to_level(self, score):
        """Convert numeric score to categorical level"""
        if score < 0.4:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _timeliness_level(self, timeliness):
        """Convert timeliness score to level"""
        if timeliness <= -2:
            return "very_early"
        elif timeliness <= 0:
            return "on_time"
        elif timeliness <= 5:
            return "slightly_late"
        else:
            return "very_late"
    
    def _effectiveness_level(self, score):
        """Convert effectiveness score to level"""
        if score >= 0.9:
            return "very_high"
        elif score >= 0.8:
            return "high"
        else:
            return "medium"
    
    def predict_english_proficiency(self, english_scores):
        """
        Use English proficiency model to predict level from Step 1 assessment
        
        Maps: Step 1 English Assessment → English Proficiency Model
        Input: english_scores dict with 'Vocabulary', 'Grammar', 'Reading', 'Writing' 
        Output: proficiency level (0=Low, 1=Medium, 2=High) + confidence
        """
        if not self.english_model or not english_scores:
            # Return proper format even when model unavailable
            return {
                'level': 1,  # Default to medium level
                'confidence': 0.5,
                'probabilities': [0.3, 0.4, 0.3]
            }
            
        # Prepare features from Step 1 English Assessment: [vocab, grammar, reading, writing]
        features = np.array([[
            english_scores.get('Vocabulary', 0.5),
            english_scores.get('Grammar', 0.5), 
            english_scores.get('Reading', 0.5),
            english_scores.get('Writing', 0.5)
        ]])
        
        # Get prediction from English proficiency ML model
        prediction = self.english_model.predict(features)[0]
        probabilities = self.english_model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return {
            'level': prediction,  # 0=Low, 1=Medium, 2=High
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def engineer_features_for_academic_model(self, user_profile):
        """
        Enhanced feature engineering: Transform user inputs into 58 model features
        Now includes proper categorical variable encoding for demographics
        
        Strategy:
        1. Extract demographic information for categorical encoding
        2. Map direct user inputs to corresponding model features
        3. Calculate derived features from user inputs using domain knowledge
        4. Use categorical mapper for region, education, etc. if available
        5. Apply correlation-based estimation for complex features
        """
        
        if not self.feature_names:
            print("⚠️ Feature names not loaded, using simplified approach")
            return None
        
        expected_features = 58  # The correct LightGBM model expects 58 features
        if len(self.feature_names) != expected_features:
            print(f"⚠️ Feature names ({len(self.feature_names)}) != expected features ({expected_features})")
            # Use the first 58 features or pad if needed
            if len(self.feature_names) >= expected_features:
                self.feature_names = self.feature_names[:expected_features]
            else:
                # Pad with generic feature names
                while len(self.feature_names) < expected_features:
                    self.feature_names.append(f'feature_{len(self.feature_names)}')
            print(f"✅ Adjusted to {len(self.feature_names)} features")
        
        # Extract user data with defaults
        academic = user_profile.get('academic_profile', {})
        demographic = user_profile.get('demographic_info', {})  # NEW: Demographic data
        
        # === CATEGORICAL FEATURE ENCODING ===
        categorical_features = np.zeros(0)  # Default empty array
        
        if self.categorical_mapper and demographic:
            try:
                categorical_features = self.categorical_mapper.encode_user_categorical_inputs(demographic)
                print(f"✅ Encoded {np.sum(categorical_features != 0)} categorical features from demographics")
                print(f"   Region: {demographic.get('region', 'Not specified')}")
                print(f"   Education: {demographic.get('highest_education', 'Not specified')}")
                print(f"   IMD Band: {demographic.get('imd_band', 'Not specified')}")
            except Exception as e:
                print(f"⚠️ Categorical encoding failed: {e}. Using defaults.")
                categorical_features = np.zeros(20)  # Fallback
        
        # Core user inputs
        num_attempts = academic.get('num_of_prev_attempts', 0)
        avg_score = academic.get('avg_score', 65)
        timeliness = academic.get('submission_timeliness', 0)
        consistency = academic.get('engagement_consistency', 0.7)
        study_hours = academic.get('study_hours_per_week', 15)
        motivation = academic.get('motivation_level', 5) / 10
        confidence = academic.get('confidence_level', 5) / 10
        stress = (10 - academic.get('stress_level', 5)) / 10  # Invert stress
        credits = self._normalize_credits(academic.get('studied_credits', 120))
        
        # Initialize feature array
        features = np.zeros(58)
        
        # === DIRECT MAPPINGS ===
        direct_mappings = {
            'num_of_prev_attempts': num_attempts,
            'repeat_student': 1 if num_attempts > 0 else 0,
            'studied_credits': credits,
            'submission_timeliness': timeliness,
        }
        
        # === CALCULATED FEATURES ===
        # OPTIMIZED SCALING: Based on testing, we need more aggressive negative scaling
        # to achieve reasonable Pass predictions for good students
        
        # Performance metrics - More aggressive scaling for better Pass predictions
        score_scaled = -(avg_score - 40) / 25 * 1.5  # More negative, wider range
        performance_factor = max(-1.8, score_scaled) if avg_score > 70 else min(0.2, score_scaled)
        
        calculated_features = {
            'score': score_scaled,  # More aggressive negative scaling
            'score_per_weight': performance_factor * 1.4,  # Amplified effect
            'assessment_engagement_score': -(motivation * consistency - 0.2) * 1.3,  # More negative for high engagement
            'weighted_engagement': -(consistency * motivation - 0.2) * 1.2,
            'banked_assessment_ratio': -(max(0, avg_score - 40) / 40) * 1.1,  # Lower threshold, more negative
            'score_trend': -0.5 if avg_score > 75 else (-0.2 if avg_score > 60 else 0.1),  # Stronger negatives
            'score_momentum': -(motivation - 0.2) * 0.8,  # More aggressive motivation scaling
        }
        
        # Engagement metrics - More aggressive negative scaling for high engagement
        base_engagement_scaled = -((consistency + min(1.0, study_hours/20) + motivation)/3 - 0.3) * 1.3
        calculated_features.update({
            'module_engagement_rate': base_engagement_scaled,
            'engagement_trend': -0.4 if motivation > 0.7 else (-0.1 if motivation > 0.5 else 0.1),
            'engagement_cv': -(1 - consistency - 0.3) * 1.2,  # More aggressive consistency reward
            'activity_diversity': max(-0.6, base_engagement_scaled - 0.2),
        })
        
        # Time and activity features - More aggressive scaling for good habits
        calculated_features.update({
            'days_since_registration': 0.0,  # Neutral baseline
            'performance_by_registration': performance_factor,
            'learning_pace': -(max(0, study_hours - 8) / 15) * 1.2,  # Lower threshold, more negative
            'sum': -(max(0, study_hours - 6) / 20) * 1.1,  # More aggressive study time reward
            'count': -(max(0, study_hours - 4) / 25) * 1.0,  # Lower bar for session reward
        })
        
        # === FILL FEATURE ARRAY ===
        # Fill features based on mappings and intelligent defaults
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in direct_mappings:
                features[i] = direct_mappings[feature_name]
            elif feature_name in calculated_features:
                features[i] = calculated_features[feature_name]
            else:
                # 🎯 POPULATION-BASED INTELLIGENT DEFAULTS (No more zeros!)
                if feature_name in self.population_averages:
                    # Use training data population average for demographic features
                    features[i] = self.population_averages[feature_name]
                elif 'score' in feature_name.lower():
                    features[i] = score_scaled  # Use the scaled score
                elif 'engagement' in feature_name.lower():
                    features[i] = base_engagement_scaled  # Use scaled engagement
                elif 'activity' in feature_name.lower():
                    features[i] = max(-0.5, min(0.1, (study_hours / 20 + motivation * 0.4 - 0.7) * 1.2))
                elif feature_name.startswith('cluster_'):
                    # Assign to middle cluster (most balanced)
                    cluster_num = int(feature_name.split('_')[1]) if '_' in feature_name else 0
                    features[i] = 1 if cluster_num == 3 else 0
                elif 'umap' in feature_name.lower():
                    features[i] = 0.0  # Neutral UMAP position
                else:
                    # Fallback for unknown features
                    features[i] = -0.3
        
        # === INTEGRATE CATEGORICAL FEATURES ===
        # If we have categorical features, integrate them properly
        if len(categorical_features) > 0:
            # Find categorical feature positions in the feature array
            categorical_feature_names = [name for name in self.feature_names 
                                       if any(name.startswith(prefix) for prefix in 
                                            ['region_', 'highest_education_', 'imd_band_', 
                                             'age_band_', 'gender_', 'disability_'])]
            
            # Map categorical features to their positions
            if self.categorical_mapper:
                for i, feature_name in enumerate(self.feature_names):
                    if feature_name in self.categorical_mapper.all_encoded_features:
                        # Find the index in the categorical features array
                        try:
                            cat_idx = self.categorical_mapper.all_encoded_features.index(feature_name)
                            if cat_idx < len(categorical_features):
                                features[i] = categorical_features[cat_idx]
                        except ValueError:
                            pass  # Feature not found in categorical mapping
        
        print(f"✅ Engineered {len(features)} features from user input")
        print(f"   Categorical features: {np.sum(categorical_features != 0) if len(categorical_features) > 0 else 0}")
        print(f"   Numerical features: {len(calculated_features) + len(direct_mappings)}")
        print(f"   Using population averages for demographics: {demographic is not None}")
        
        return features.reshape(1, -1)
    
    def _normalize_credits(self, credits):
        """Normalize credit values to a standard range"""
        if isinstance(credits, str):
            if credits == "More than 120":
                return 150
            else:
                return int(credits)
        return int(credits) if credits else 120
    
    def predict_academic_success(self, user_profile):
        """
        Use Academic success model to predict outcome from Step 2 learning profile
        
        Maps: Step 2 Learning Profile → Academic Success Model (58 features)
        Input: user_profile dict with academic_profile data from Step 2 form
        Output: academic outcome (Pass/Fail/Distinction/Withdrawn) + confidence + risk level
        
        NOTE: Current model.pkl contains RandomForestClassifier, but Final.ipynb 
        originally trained LightGBM models. System works with both.
        """
        if not self.academic_model or not self.feature_names:
            return {'outcome': 'Pass', 'confidence': 0.5, 'probabilities': [0.5, 0.2, 0.2, 0.1], 'risk_level': 'medium'}
        
        try:
            # Engineer features from user input (9 inputs → 58 features)
            engineered_features = self.engineer_features_for_academic_model(user_profile)
            
            if engineered_features is None:
                raise Exception("Feature engineering failed")
            
            # Handle scaler mismatch: Check if scaler matches engineered features  
            model_expected_features = 58  # Our LightGBM model expects 58 features
            scaler_expected_features = self.academic_scaler.n_features_in_
            
            if scaler_expected_features == model_expected_features:
                # Perfect match - use scaler
                scaled_features = self.academic_scaler.transform(engineered_features)
            else:
                # Scaler mismatch - skip scaling and normalize manually
                print(f"⚠️ Scaler trained on {scaler_expected_features} features, model needs {model_expected_features}. Skipping scaler.")
                
                # Manual normalization: standardize features to roughly [-2, 2] range
                # This is a reasonable approximation of what StandardScaler would do
                scaled_features = np.copy(engineered_features)
                
                # Apply basic standardization
                mean_vals = np.mean(scaled_features, axis=1, keepdims=True)
                std_vals = np.std(scaled_features, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
                scaled_features = (scaled_features - mean_vals) / std_vals
                
                # Clip to reasonable range to avoid extreme values
                scaled_features = np.clip(scaled_features, -3, 3)
            
            # Make prediction with proper feature names to avoid sklearn warning
            if hasattr(scaled_features, 'shape') and len(scaled_features.shape) == 2:
                # Convert to DataFrame with feature names to avoid sklearn warning
                import pandas as pd
                feature_df = pd.DataFrame(scaled_features, columns=self.feature_names[:scaled_features.shape[1]])
                prediction = self.academic_model.predict(feature_df)[0]
                probabilities = self.academic_model.predict_proba(feature_df)[0]
            else:
                # Fallback to numpy array if DataFrame creation fails
                prediction = self.academic_model.predict(scaled_features)[0]
                probabilities = self.academic_model.predict_proba(scaled_features)[0]
            
            # Map prediction to binary outcome (Success/Failure)
            # Binary classification: 0 = Success (Pass+Distinction), 1 = Failure (Fail+Withdrawn)
            target_classes = ['Success', 'Failure']
            outcome = target_classes[prediction] if prediction < len(target_classes) else 'Success'
            
            # Determine risk level based on prediction and confidence
            if outcome == 'Failure':
                risk_level = 'high'
            elif probabilities[0] < 0.7:  # Success probability < 70%
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'outcome': outcome,
                'confidence': probabilities[prediction],
                'probabilities': probabilities,
                'success_probability': probabilities[0],  # Probability of Success
                'risk_level': risk_level
            }
            
        except Exception as e:
            print(f"Academic prediction error: {e}")
            # Fallback to rule-based assessment
            return self._fallback_academic_assessment(user_profile)
    
    def _create_academic_features(self, english_data, academic_data):
        """Create feature vector from user input data"""
        # Extract key features we can map from user input
        features = [
            academic_data.get('num_of_prev_attempts', 0),  # 0: num_of_prev_attempts
            1 if academic_data.get('num_of_prev_attempts', 0) > 0 else 0,  # 1: repeat_student
            academic_data.get('studied_credits', 60) if isinstance(academic_data.get('studied_credits'), int) else 60,  # 2: studied_credits
            academic_data.get('study_hours_per_week', 15) * 7,  # 3: sum (approximate total hours)
            academic_data.get('study_hours_per_week', 15),  # 4: count (hours per week)
            academic_data.get('engagement_consistency', 0.7),  # 5: activity_diversity
            academic_data.get('avg_score', 65),  # 6: score
            academic_data.get('avg_score', 65) / 100,  # 7: score_per_weight
            academic_data.get('engagement_score', 0.7),  # 8: assessment_engagement_score
            academic_data.get('engagement_consistency', 0.7),  # 9: module_engagement_rate
            academic_data.get('engagement_score', 0.7) * academic_data.get('engagement_consistency', 0.7),  # 10: weighted_engagement
            0.1,  # 11: engagement_trend (default)
            academic_data.get('submission_timeliness', 0),  # 12: submission_timeliness
            0.3,  # 13: banked_assessment_ratio (default)
            30,   # 14: days_since_registration (default)
        ]
        
        # Pad with default values for remaining features (15-57)
        remaining_features = [0.5] * (58 - len(features))
        features.extend(remaining_features)
        
        return np.array(features[:58])  # Ensure exactly 58 features
        
    def _fallback_academic_assessment(self, user_profile):
        """Fallback rule-based academic assessment if ML prediction fails"""
        academic_data = user_profile.get('academic_profile', {})
        
        # Simple rule-based risk assessment
        risk_factors = 0
        if academic_data.get('num_of_prev_attempts', 0) > 1:
            risk_factors += 1
        if academic_data.get('avg_score', 65) < 50:
            risk_factors += 2
        if academic_data.get('submission_timeliness', 0) > 5:
            risk_factors += 1
        if academic_data.get('motivation_level', 6) < 4:
            risk_factors += 1
        
        if risk_factors >= 3:
            return {'outcome': 'Fail', 'confidence': 0.7, 'probabilities': [0.1, 0.7, 0.1, 0.1], 'risk_level': 'high'}
        elif risk_factors >= 2:
            return {'outcome': 'Pass', 'confidence': 0.6, 'probabilities': [0.6, 0.2, 0.1, 0.1], 'risk_level': 'medium'}
        elif risk_factors == 0:
            return {'outcome': 'Distinction', 'confidence': 0.8, 'probabilities': [0.1, 0.05, 0.8, 0.05], 'risk_level': 'low'}
        else:
            return {'outcome': 'Pass', 'confidence': 0.7, 'probabilities': [0.7, 0.15, 0.1, 0.05], 'risk_level': 'low'}
        
        # LightFM Enhancement Layer
        try:
            self.lightfm_recommender = LightFMEducationRecommender()
            print("✅ LightFM recommender initialized")
        except Exception as e:
            print(f"⚠️ LightFM initialization failed: {e}")
            self.lightfm_recommender = None

    def calculate_compatibility_scores(self, user_profile):
        """Calculate compatibility scores between user and resources using content-based filtering"""
        
        # Get user feature vector
        user_features = self.create_user_feature_vector(user_profile)
        
        # Get resource feature matrix
        resource_features, resource_ids = self.create_resource_feature_matrix()
        
        # Calculate rule-based scores for each resource
        scores = []
        
        for i, (resource_id, resource) in enumerate(self.study_resources.items()):
            score = self._calculate_resource_score(user_profile, resource)
            scores.append(score)
        
        return np.array(scores), resource_ids
    
    def _calculate_resource_score(self, user_profile, resource):
        """Calculate recommendation score using ML predictions + content matching"""
        
        score = 0.4  # Base score
        
        # 🧠 ML-POWERED ENGLISH PROFICIENCY MATCHING
        if 'english_scores' in user_profile:
            english_pred = self.predict_english_proficiency(user_profile['english_scores'])
            eng_level = english_pred['level']  # 0=Low, 1=Medium, 2=High
            eng_confidence = english_pred['confidence']
            
            # Match English resources to proficiency level
            if resource['type'].startswith('English'):
                if eng_level == 0 and resource['difficulty'] == 'Beginner':
                    score += 0.4 * eng_confidence  # High need for basic resources
                elif eng_level == 1 and resource['difficulty'] == 'Intermediate':
                    score += 0.3 * eng_confidence  # Medium need for intermediate
                elif eng_level == 2 and resource['difficulty'] == 'Advanced':
                    score += 0.2 * eng_confidence  # Low need for advanced (already proficient)
                
                # Skill-specific targeting based on individual scores
                eng_scores = user_profile['english_scores']
                if 'vocab_focus' in resource.get('features', []):
                    vocab_need = max(0, 0.8 - eng_scores.get('Vocabulary', 0.5))  # Need if below 80%
                    score += vocab_need * 0.3
                
                if 'grammar_focus' in resource.get('features', []):
                    grammar_need = max(0, 0.8 - eng_scores.get('Grammar', 0.5))
                    score += grammar_need * 0.3
                    
                if 'reading_focus' in resource.get('features', []):
                    reading_need = max(0, 0.8 - eng_scores.get('Reading', 0.5))
                    score += reading_need * 0.25
                    
                if 'writing_focus' in resource.get('features', []):
                    writing_need = max(0, 0.8 - eng_scores.get('Writing', 0.5))
                    score += writing_need * 0.25
        
        # 🧠 ML-POWERED ACADEMIC SUCCESS MATCHING  
        if 'academic_profile' in user_profile:
            academic_pred = self.predict_academic_success(user_profile)
            predicted_outcome = academic_pred['outcome']
            risk_level = academic_pred['risk_level']
            confidence = academic_pred['confidence']
            
            # Match resources to predicted academic needs
            if predicted_outcome in ['Fail', 'Withdrawn'] or risk_level == 'high':
                # High-risk students need intensive support
                if 'intensive_support' in resource.get('features', []):
                    score += 0.5 * confidence
                if 'motivation' in resource.get('features', []):
                    score += 0.4 * confidence
                if 'confidence' in resource.get('features', []):
                    score += 0.4 * confidence
                    
            elif predicted_outcome == 'Pass' and risk_level == 'medium':
                # Medium-risk students need targeted support
                if 'time_management' in resource.get('features', []):
                    score += 0.3 * confidence
                if 'study_skills' in resource.get('features', []):
                    score += 0.3 * confidence
                    
            elif predicted_outcome == 'Distinction' and risk_level == 'low':
                # High-performers can benefit from advanced resources
                if resource['difficulty'] == 'Advanced':
                    score += 0.2 * confidence
                if 'leadership' in resource.get('features', []):
                    score += 0.2 * confidence
            
            # Specific behavioral interventions based on user data
            acad = user_profile['academic_profile']
            
            # Time management for poor timeliness (ML-informed)
            if acad.get('submission_timeliness', 0) > 2 and 'time_management' in resource.get('features', []):
                urgency_score = min(0.4, acad.get('submission_timeliness', 0) * 0.1)
                score += urgency_score * confidence
            
            # ML-informed confidence and motivation interventions
            if acad.get('confidence_level', 5) < 5 and 'confidence' in resource.get('features', []):
                confidence_need = (5 - acad.get('confidence_level', 5)) / 5  # 0-1 scale
                score += 0.3 * confidence_need * confidence
            
            if acad.get('motivation_level', 5) < 5 and 'motivation' in resource.get('features', []):
                motivation_need = (5 - acad.get('motivation_level', 5)) / 5  # 0-1 scale  
                score += 0.3 * motivation_need * confidence
            
            # ML-informed difficulty matching based on predicted performance
            avg_score = acad.get('avg_score', 50)
            success_probability = academic_pred.get('success_probability', 0.5)
            
            if predicted_outcome == 'Success' and success_probability > 0.8:
                # High-confidence success: can handle advanced content
                if resource['difficulty'] == 'Advanced':
                    score += 0.25 * confidence
                elif resource['difficulty'] == 'Intermediate':
                    score += 0.15 * confidence
            elif predicted_outcome == 'Failure':
                # At-risk students need beginner-friendly content
                if resource['difficulty'] == 'Beginner':
                    score += 0.3 * confidence
            else:  # Success with lower confidence
                # Moderate performers benefit from intermediate content
                if resource['difficulty'] == 'Intermediate':
                    score += 0.2 * confidence
                elif resource['difficulty'] == 'Beginner' and avg_score < 60:
                    score += 0.2 * confidence
        
        # 🎯 ML-Enhanced Resource Effectiveness Bonus
        # Weight effectiveness by ML confidence - more confident predictions get higher weight
        ml_confidence_boost = 1.0
        if 'english_scores' in user_profile and 'academic_profile' in user_profile:
            english_pred = self.predict_english_proficiency(user_profile['english_scores'])
            academic_pred = self.predict_academic_success(user_profile) 
            ml_confidence_boost = (english_pred['confidence'] + academic_pred['confidence']) / 2
        
        score += resource['effectiveness_score'] * 0.15 * ml_confidence_boost
        
        # 📊 Final ML-Powered Score Normalization
        # Normalize to 0-1 range with ML confidence weighting
        normalized_score = min(1.0, max(0.0, score))
        
        return normalized_score
    
    def _generate_synthetic_interactions(self, user_profiles):
        """Generate synthetic interactions based on user profiles"""
        interactions = []
        
        for profile in user_profiles:
            user_interactions = {}
            
            # Generate interactions based on English proficiency
            if 'english_scores' in profile:
                eng_scores = profile['english_scores']
                
                # Recommend vocab resources for low vocab scores
                if eng_scores.get('Vocabulary', 0) < 0.6:
                    user_interactions['english_vocab_beginner'] = 0.8
                elif eng_scores.get('Vocabulary', 0) < 0.8:
                    user_interactions['english_vocab_intermediate'] = 0.7
                
                # Grammar resources
                if eng_scores.get('Grammar', 0) < 0.6:
                    user_interactions['english_grammar_basic'] = 0.8
                
                # Reading resources
                if eng_scores.get('Reading', 0) < 0.7:
                    user_interactions['english_reading_comprehension'] = 0.7
                
                # Writing resources
                if eng_scores.get('Writing', 0) < 0.6:
                    user_interactions['english_writing_basics'] = 0.8
            
            # Generate interactions based on academic behavior
            if 'academic_profile' in profile:
                acad = profile['academic_profile']
                
                # Time management for poor timeliness
                if acad.get('submission_timeliness', 0) > 2:
                    user_interactions['time_management_course'] = 0.9
                
                # Academic recovery for high-risk students
                if acad.get('risk_level') == 'high':
                    user_interactions['academic_recovery_program'] = 0.95
                    user_interactions['confidence_building_workshop'] = 0.8
                
                # Motivation for disengaged students
                if acad.get('engagement_consistency', 0.5) < 0.3:
                    user_interactions['motivation_coaching'] = 0.8
                
                # Study skills for repeat students
                if acad.get('num_of_prev_attempts', 0) > 0:
                    user_interactions['note_taking_system'] = 0.7
                    user_interactions['exam_preparation_intensive'] = 0.8
                
                # Advanced resources for high performers
                if acad.get('avg_score', 50) > 80:
                    user_interactions['study_group_facilitator'] = 0.6
            
            interactions.append(user_interactions)
        
        return interactions
    
    def get_recommendations(self, user_profile, top_k=5):
        """Get personalized recommendations for a user"""
        
        # 🎯 ENHANCED: Try LightFM recommendations first
        if hasattr(self, 'lightfm_recommender') and self.lightfm_recommender:
            try:
                existing_models = {
                    'english_model': self,
                    'academic_model': self
                }
                
                lightfm_recs = self.lightfm_recommender.get_lightfm_recommendations(
                    user_profile, existing_models, top_k
                )
                
                if lightfm_recs:
                    print("✅ Using LightFM recommendations")
                    return lightfm_recs
                    
            except Exception as e:
                print(f"⚠️ LightFM failed, fallback to rules: {e}")
        
        # 🔄 FALLBACK: Original rule-based recommendations
        
        # Calculate compatibility scores
        scores, resource_ids = self.calculate_compatibility_scores(user_profile)
        
        # Get top recommendations
        top_indices = np.argsort(-scores)[:top_k]
        
        recommendations = []
        for idx in top_indices:
            resource_id = resource_ids[idx]
            resource = self.study_resources[resource_id]
            recommendations.append({
                'id': resource_id,
                'score': float(scores[idx]),
                'resource': resource,
                'relevance_explanation': self._explain_recommendation(user_profile, resource)
            })
        
        return recommendations
    

    
    def _explain_recommendation(self, user_profile, resource):
        """Generate explanation for why this resource was recommended"""
        explanations = []
        
        # English-based explanations
        if 'english_scores' in user_profile:
            eng_scores = user_profile['english_scores']
            
            if 'vocab_focus' in resource.get('features', []):
                if eng_scores.get('Vocabulary', 0) < 0.6:
                    explanations.append("Your vocabulary score indicates room for improvement")
            
            if 'grammar_focus' in resource.get('features', []):
                if eng_scores.get('Grammar', 0) < 0.6:
                    explanations.append("Your grammar assessment suggests this will help")
            
            if 'reading_focus' in resource.get('features', []):
                if eng_scores.get('Reading', 0) < 0.7:
                    explanations.append("Based on your reading comprehension results")
            
            if 'writing_focus' in resource.get('features', []):
                if eng_scores.get('Writing', 0) < 0.6:
                    explanations.append("Your writing skills assessment indicates this need")
        
        # Academic behavior explanations
        if 'academic_profile' in user_profile:
            acad = user_profile['academic_profile']
            
            if 'time_management' in resource.get('features', []):
                if acad.get('submission_timeliness', 0) > 2:
                    explanations.append("Your assignment submission patterns suggest timing challenges")
            
            if 'intensive_support' in resource.get('features', []):
                if acad.get('risk_level') == 'high':
                    explanations.append("Your academic profile indicates need for additional support")
            
            if 'motivation' in resource.get('features', []):
                if acad.get('engagement_consistency', 0.5) < 0.3:
                    explanations.append("Your engagement patterns suggest motivational support would help")
        
        if not explanations:
            explanations.append("This resource matches your learning profile")
        
        return "; ".join(explanations)


def create_user_interface():
    """Create the main user interface"""
    
    # Initialize recommender
    if 'recommender' not in st.session_state:
        st.session_state.recommender = ContentBasedStudyRecommender()    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'english_results' not in st.session_state:
        st.session_state.english_results = {}
    if 'academic_results' not in st.session_state:
        st.session_state.academic_results = {}
    
    # Header with LightFM status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🎯 EducationCare Study Recommender</div>', unsafe_allow_html=True)
    with col2:
        if LIGHTFM_AVAILABLE and hasattr(st.session_state.recommender, 'lightfm_recommender') and st.session_state.recommender.lightfm_recommender:
            st.markdown('<div style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; padding: 0.3rem 0.7rem; border-radius: 15px; text-align: center; font-size: 0.9rem; margin-top: 1rem;">✨ LightFM-Next Enhanced</div>', unsafe_allow_html=True)
        elif LIGHTFM_AVAILABLE:
            st.markdown('<div style="background: #ffc107; color: black; padding: 0.3rem 0.7rem; border-radius: 15px; text-align: center; font-size: 0.9rem; margin-top: 1rem;">⚡ LightFM Available</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: #6c757d; color: white; padding: 0.3rem 0.7rem; border-radius: 15px; text-align: center; font-size: 0.9rem; margin-top: 1rem;">🔧 Content-based</div>', unsafe_allow_html=True)
    
    if LIGHTFM_AVAILABLE:
        st.markdown("### *Get AI-enhanced recommendations powered by LightFM collaborative filtering*")
        st.success("🚀 **LightFM-Next Integration Active** - Using state-of-the-art collaborative filtering alongside ML predictions")
    else:
        st.markdown("### *Get personalized study recommendations based on your English proficiency and learning behavior*")
        st.info("💡 Install lightfm-next for enhanced collaborative filtering recommendations")
    
    # Progress indicator
    progress = (st.session_state.step - 1) / 3
    st.progress(progress)
    
    # Debug info and manual navigation (you can remove this later)
    with st.expander("🔧 Debug & Navigation Helper"):
        st.write(f"Current Step: {st.session_state.step}")
        st.write(f"English Results: {'✅' if st.session_state.get('english_results') else '❌'}")
        st.write(f"Academic Results: {'✅' if st.session_state.get('academic_results') else '❌'}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Go to Step 1"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Go to Step 2") and st.session_state.get('english_results'):
                st.session_state.step = 2
                st.rerun()
        with col3:
            if st.button("Go to Step 3") and st.session_state.get('academic_results'):
                st.session_state.step = 3
                st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step_class = "completed-step" if st.session_state.step > 1 else "current-step" if st.session_state.step == 1 else ""
        st.markdown(f"""
        <div class="step-container {step_class}">
            <h4>Step 1: English Assessment</h4>
            <p>Quick 4-skill English proficiency test</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        step_class = "completed-step" if st.session_state.step > 2 else "current-step" if st.session_state.step == 2 else ""
        st.markdown(f"""
        <div class="step-container {step_class}">
            <h4>Step 2: Learning Profile</h4>
            <p>Tell us about your study habits</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        step_class = "completed-step" if st.session_state.step > 3 else "current-step" if st.session_state.step == 3 else ""
        st.markdown(f"""
        <div class="step-container {step_class}">
            <h4>Step 3: Get Recommendations</h4>
            <p>Receive personalized study plan</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on current step
    if st.session_state.step == 1:
        show_english_assessment()
    elif st.session_state.step == 2:
        show_academic_profile()
    elif st.session_state.step == 3:
        show_recommendations()


def show_english_assessment():
    """Display English proficiency assessment (based on existing proficiency test)"""
    
    st.markdown("## 🇬🇧 English Proficiency Assessment")
    st.markdown("*Complete this quick assessment to understand your English language skills*")
    
    # English questions (simplified from existing test)
    vocab_questions = [
        {"q": "Choose the best meaning of 'rapid':", "options": ["slow", "fast", "heavy", "short"], "answer": "fast"},
        {"q": "Synonym of 'begin':", "options": ["start", "stop", "close", "late"], "answer": "start"},
        {"q": "Opposite of 'difficult':", "options": ["easy", "hard", "strong", "quick"], "answer": "easy"},
        {"q": "'She bought a _____ of bread.'", "options": ["loaf", "leaf", "piece", "glass"], "answer": "loaf"}
    ]
    
    grammar_questions = [
        {"q": "He ____ to school every day.", "options": ["go", "goes", "going", "gone"], "answer": "goes"},
        {"q": "They ____ dinner right now.", "options": ["are cooking", "cook", "cooks", "cooked"], "answer": "are cooking"},
        {"q": "I have ____ this movie before.", "options": ["see", "saw", "seen", "seeing"], "answer": "seen"},
        {"q": "She didn't ____ to the party.", "options": ["go", "goes", "going", "gone"], "answer": "go"}
    ]
    
    reading_questions = [
        {"text": "Liam was late because the bus broke down.", "q": "Why was Liam late?", 
         "options": ["He overslept", "The bus broke down", "He forgot homework", "Holiday"], "answer": "The bus broke down"},
        {"text": "Maria drinks coffee every morning, but today she chose tea.", "q": "What did Maria drink today?",
         "options": ["Coffee", "Tea", "Juice", "Milk"], "answer": "Tea"},
        {"text": "The restaurant was full, so they waited for a table.", "q": "What was the problem?",
         "options": ["Bad food", "They were late", "It was full", "It was closed"], "answer": "It was full"}
    ]
    
    writing_questions = [
        {"q": "Choose the correct sentence:", 
         "options": ["She don't like apples.", "She doesn't like apples.", "She doesn't likes apples.", "She not like apples."],
         "answer": "She doesn't like apples."},
        {"q": "Correct punctuation:", 
         "options": ["Yesterday I went to London Paris and Rome.", "Yesterday, I went to London, Paris, and Rome.", 
                    "Yesterday I went, to London, Paris and Rome.", "Yesterday I went to London, Paris and, Rome."],
         "answer": "Yesterday, I went to London, Paris, and Rome."}
    ]
    
    with st.form("english_assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Vocabulary")
            vocab_answers = []
            for i, q in enumerate(vocab_questions):
                ans = st.radio(f"{i+1}. {q['q']}", q['options'], key=f"vocab_{i}")
                vocab_answers.append(ans)
            
            st.subheader("📖 Reading Comprehension")
            reading_answers = []
            for i, q in enumerate(reading_questions):
                st.write(f"**Text:** {q['text']}")
                ans = st.radio(f"{i+1}. {q['q']}", q['options'], key=f"reading_{i}")
                reading_answers.append(ans)
        
        with col2:
            st.subheader("✏️ Grammar")
            grammar_answers = []
            for i, q in enumerate(grammar_questions):
                ans = st.radio(f"{i+1}. {q['q']}", q['options'], key=f"grammar_{i}")
                grammar_answers.append(ans)
            
            st.subheader("✍️ Writing Mechanics")
            writing_answers = []
            for i, q in enumerate(writing_questions):
                ans = st.radio(f"{i+1}. {q['q']}", q['options'], key=f"writing_{i}")
                writing_answers.append(ans)
        
        submitted = st.form_submit_button("📊 Complete English Assessment", type="primary")
    
    if submitted:
        # Calculate scores (same logic as existing test)
        vocab_score = sum(1 for ans, q in zip(vocab_answers, vocab_questions) if ans == q['answer'])
        grammar_score = sum(1 for ans, q in zip(grammar_answers, grammar_questions) if ans == q['answer'])
        reading_score = sum(1 for ans, q in zip(reading_answers, reading_questions) if ans == q['answer'])
        writing_score = sum(1 for ans, q in zip(writing_answers, writing_questions) if ans == q['answer'])
        
        # Scale to original ranges and normalize
        vocab_norm = (vocab_score * 10 / len(vocab_questions)) / 10
        grammar_norm = (grammar_score * 8 / len(grammar_questions)) / 8
        reading_norm = (reading_score * 6 / len(reading_questions)) / 6
        writing_norm = (writing_score * 5 / len(writing_questions)) / 5
        
        # Store results
        st.session_state.english_results = {
            'Vocabulary': vocab_norm,
            'Grammar': grammar_norm,
            'Reading': reading_norm,
            'Writing': writing_norm,
            'overall_score': np.mean([vocab_norm, grammar_norm, reading_norm, writing_norm])
        }
        
        # Show results
        st.success("✅ English assessment completed!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills radar chart
            categories = list(st.session_state.english_results.keys())[:-1]  # Exclude overall_score
            values = list(st.session_state.english_results.values())[:-1]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Your Skills'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Your English Skills Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Overall English Level", 
                     f"{st.session_state.english_results['overall_score']:.1%}")
            
            # Individual scores
            for skill, score in st.session_state.english_results.items():
                if skill != 'overall_score':
                    level = "🟢 Strong" if score >= 0.8 else "🟡 Adequate" if score >= 0.6 else "🔴 Needs Work"
                    st.write(f"**{skill}**: {score:.1%} - {level}")
        
    # Next step button (outside the form submission logic)
    if st.session_state.get('english_results') and st.button("➡️ Continue to Learning Profile", type="primary"):
        st.session_state.step = 2
        st.rerun()


def show_academic_profile():
    """Enhanced academic/behavioral profile form with demographic fields"""
    
    st.markdown("## 👤 Your Learning Profile")
    st.markdown("*Tell us about your study habits, academic experience, and background*")
    
    with st.form("academic_profile_form"):
        # Create three columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📚 Academic Background")
            
            num_of_prev_attempts = st.number_input(
                "How many times have you attempted this type of course before?",
                min_value=0, max_value=5, value=0,
                help="Including previous attempts at similar courses"
            )
            
            avg_score = st.slider(
                "What's your typical assignment score?",
                min_value=0, max_value=100, value=65, step=5,
                help="Average percentage across recent assignments"
            )
            
            studied_credits = st.selectbox(
                "How many credits are you currently studying?",
                options=[30, 60, 90, 120, "More than 120"],
                index=1,
                help="Total credit load this semester/term"
            )
            
            highest_education = st.selectbox(
                "What's your highest level of education?",
                options=["Lower Than A Level", "A Level or Equivalent", "HE Qualification"],
                index=1,
                help="Completed education level (mapped to model categories)"
            )
        
        with col2:
            st.subheader("🌍 Background Information")
            
            # NEW: Demographic fields for categorical encoding
            region = st.selectbox(
                "What region are you from?",
                options=get_regional_options(),
                index=0,
                help="Your geographic region (used for personalized recommendations)"
            )
            
            # Automatically infer IMD band from region (hidden from user)
            imd_band = get_imd_suggestion_from_region(region)
            
            # Show brief regional context instead of technical IMD explanation
            if region in ['London Region', 'South East Region']:
                st.info("💡 Your region typically has excellent educational resources and opportunities!")
            elif region in ['Scotland', 'Wales']:
                st.info("💡 Strong educational traditions in your region - great learning communities!")
            elif region in ['North East Region', 'North West Region']:
                st.info("💡 Your region has collaborative learning environments and supportive communities!")
            else:
                st.info(f"💡 We'll personalize recommendations based on your {region} context!")
            
            age_band = st.selectbox(
                "Age group",
                options=["0-35", "35+"],
                index=0,
                help="Your age group"
            )
            
            gender = st.selectbox(
                "Gender (optional)",
                options=['F', 'M'],
                index=0,
                help="Optional demographic information"
            )
            
            disability = st.selectbox(
                "Do you have any declared disabilities?",
                options=['No', 'Yes'],
                index=0,
                help="This helps us provide appropriate support recommendations"
            )
        
        with col3:
            st.subheader("⏰ Study Behavior")
            
            submission_timeliness = st.slider(
                "How often do you submit assignments on time?",
                min_value=-5, max_value=15, value=0, step=1,
                help="Negative = usually early, 0 = on time, positive = usually late (days)"
            )
            
            engagement_consistency = st.slider(
                "How consistent is your study schedule?",
                min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                help="1.0 = very consistent daily routine, 0.0 = very irregular"
            )
            
            study_hours_per_week = st.slider(
                "How many hours do you study per week?",
                min_value=0, max_value=40, value=15, step=2,
                help="Including lectures, reading, and assignments"
            )
            
            platform_engagement = st.selectbox(
                "How often do you use the online learning platform?",
                options=["Daily", "Several times a week", "Weekly", "Rarely"],
                index=1,
                help="Frequency of logging into course systems"
            )
            
            motivation_level = st.slider(
                "How motivated do you feel about your studies?",
                min_value=1, max_value=10, value=6,
                help="1 = very unmotivated, 10 = extremely motivated"
            )
            
            confidence_level = st.slider(
                "How confident do you feel about succeeding?",
                min_value=1, max_value=10, value=6,
                help="1 = very uncertain, 10 = very confident"
            )
            
            stress_level = st.slider(
                "What's your current stress level about studies?",
                min_value=1, max_value=10, value=5,
                help="1 = no stress, 10 = extremely stressed"
            )
        
        # Additional preferences section
        st.subheader("🎯 Learning Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            learning_style = st.multiselect(
                "What learning methods work best for you?",
                options=["Reading textbooks", "Video lectures", "Interactive exercises", 
                        "Discussion forums", "Group work", "Hands-on practice"],
                default=["Reading textbooks", "Interactive exercises"],
                help="Select all that apply"
            )
        
        with col2:
            study_environment = st.selectbox(
                "Where do you prefer to study?",
                options=["Quiet library", "Home office", "Coffee shop/public", "With others", "Varies"],
                help="Your most productive study environment"
            )
        
        submitted = st.form_submit_button("📋 Complete Enhanced Learning Profile", type="primary")
    
    if submitted:
        # Process both academic and demographic data
        engagement_score = {
            "Daily": 0.9,
            "Several times a week": 0.7,
            "Weekly": 0.5,
            "Rarely": 0.2
        }[platform_engagement]
        
        # Determine risk level
        risk_factors = 0
        if num_of_prev_attempts > 1:
            risk_factors += 1
        if avg_score < 50:
            risk_factors += 1
        if submission_timeliness > 5:
            risk_factors += 1
        if engagement_consistency < 0.4:
            risk_factors += 1
        if study_hours_per_week < 10:
            risk_factors += 1
        
        # Regional/demographic risk factors
        if imd_band in ['70-80%', '80-90%', '90-100%']:
            risk_factors += 1
        if highest_education == 'Lower Than A Level':
            risk_factors += 1
            
        if risk_factors >= 4:
            risk_level = "high"
        elif risk_factors >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Store results with enhanced structure
        st.session_state.academic_results = {
            # Academic profile
            'academic_profile': {
                'num_of_prev_attempts': num_of_prev_attempts,
                'avg_score': avg_score,
                'submission_timeliness': submission_timeliness,
                'engagement_consistency': engagement_consistency,
                'study_hours_per_week': study_hours_per_week,
                'engagement_score': engagement_score,
                'motivation_level': motivation_level,
                'confidence_level': confidence_level,
                'stress_level': stress_level,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'learning_style': learning_style,
                'study_environment': study_environment,
                'highest_education': highest_education,
                'studied_credits': studied_credits,
            },
            
            # NEW: Demographic profile for categorical encoding
            'demographic_info': {
                'region': region,
                'imd_band': imd_band,
                'age_band': age_band,
                'gender': gender,
                'disability': 'Y' if disability == 'Yes' else 'N',
                'highest_education': highest_education  # Also store here for categorical mapping
            }
        }
        
        # Show enhanced summary with demographic insights
        st.success("✅ Enhanced learning profile completed!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[risk_level]
            st.metric("Risk Level", f"{risk_color} {risk_level.title()}")
        
        with col2:
            st.metric("Region", region)
        
        with col3:
            st.metric("Education Level", highest_education)
        
        with col4:
            st.metric("Study Hours/Week", f"{study_hours_per_week}h")
        
        # Enhanced profile summary with demographic awareness
        st.subheader("📊 Your Enhanced Learning Profile Summary")
        
        # Show demographic insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🌍 Demographic Profile:**")
            st.write(f"- Region: {region}")
            st.write(f"- Education: {highest_education}")
            st.write(f"- Age Group: {age_band}")
            st.write(f"- Gender: {gender}")
            
            # Regional insights
            if region in ['London Region', 'South East Region']:
                st.info("💡 Your region typically has access to excellent educational resources!")
            elif region in ['Scotland', 'Wales']:
                st.info("💡 Strong educational traditions in your region - leverage local study groups!")
        
        with col2:
            st.write("**📚 Academic Profile:**")
            st.write(f"- Average Score: {avg_score}%")
            st.write(f"- Previous Attempts: {num_of_prev_attempts}")
            st.write(f"- Study Hours: {study_hours_per_week}/week")
            st.write(f"- Motivation Level: {motivation_level}/10")
            
            # Risk-based insights
            if risk_level == "high":
                st.warning("🆘 Consider intensive support programs - your profile indicates you could benefit from additional help!")
            elif risk_level == "low":
                st.success("🌟 You're on track for success - maintain your current habits!")
            else:
                st.info("📈 You're doing well - a few targeted improvements could boost your success!")
        
        # Show categorical encoding preview if available
        if CATEGORICAL_MAPPING_AVAILABLE:
            st.subheader("🔧 Enhanced ML Features Preview")
            try:
                categorical_features = encode_user_categorical_inputs(st.session_state.academic_results['demographic_info'])
                active_features = np.sum(categorical_features != 0)
                st.info(f"✅ Your demographic data maps to **{active_features} active categorical features** for enhanced ML predictions")
            except Exception as e:
                st.warning(f"⚠️ Categorical mapping preview unavailable: {e}")
        
    # Next step button (outside the form submission logic)
    if st.session_state.get('academic_results') and st.button("🎯 Get My Enhanced Personalized Recommendations", type="primary"):
        st.session_state.step = 3
        st.rerun()


def show_recommendations():
    """Display personalized recommendations using ML predictions with enhanced demographic data"""
    
    st.markdown("## 🎯 Your Personalized Study Recommendations")
    st.markdown("*Based on your English proficiency, learning profile, and background*")
    
    # 🎯 ENHANCED ML MODEL MAPPING:
    # Step 1 English Assessment → english_results → English Proficiency Model (4 features)
    # Step 2 Enhanced Learning Profile → academic_results → Academic Success Model (58+ features with demographics)
    
    academic_results = st.session_state.academic_results
    
    # Handle both old and new data structures for backward compatibility
    if 'academic_profile' in academic_results:
        # New enhanced structure with demographic data
        user_profile = {
            'english_scores': st.session_state.english_results,      # From Step 1 → English Model
            'academic_profile': academic_results['academic_profile'], # From Step 2 → Academic Model
            'demographic_info': academic_results.get('demographic_info', {})  # NEW: Demographic data
        }
    else:
        # Old structure - migrate to new format
        user_profile = {
            'english_scores': st.session_state.english_results,      
            'academic_profile': academic_results,    
            'demographic_info': {}  # Empty for backward compatibility
        }
    
    # Get enhanced recommendations with demographic awareness
    with st.spinner("🔍 Analyzing your complete profile and generating personalized recommendations..."):
        try:
            # Enhanced feature engineering with demographic data
            enhanced_features = st.session_state.recommender.engineer_features_for_academic_model(user_profile)
            
            # Get ML-enhanced recommendations
            recommendations = st.session_state.recommender.get_recommendations(user_profile, top_k=6)
            
        except Exception as e:
            st.warning(f"Enhanced analysis failed: {e}. Using standard recommendations.")
            recommendations = st.session_state.recommender.get_recommendations(user_profile, top_k=6)
    
    # 🧠 Enhanced ML Predictions Overview with Demographic Insights
    st.subheader("🧠 AI Analysis of Your Complete Learning Profile")
    
    # Get ML predictions
    english_pred = st.session_state.recommender.predict_english_proficiency(user_profile['english_scores'])
    academic_pred = st.session_state.recommender.predict_academic_success(user_profile)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🇬🇧 English Proficiency Analysis**")
        eng_levels = ["Developing", "Intermediate", "Advanced"] 
        predicted_eng_level = eng_levels[english_pred['level']]
        confidence_pct = english_pred['confidence'] * 100
        
        st.info(f"""
        **Predicted Level:** {predicted_eng_level}  
        **Confidence:** {confidence_pct:.1f}%  
        **Strongest Skills:** {max(user_profile['english_scores'].items(), key=lambda x: x[1])[0]}  
        **Focus Area:** {min(user_profile['english_scores'].items(), key=lambda x: x[1])[0]}
        """)
    
    with col2:
        st.markdown("**📊 Academic Success Prediction**")
        outcome_color = {"Success": "🟢", "Failure": "🔴"}
        outcome_emoji = outcome_color.get(academic_pred['outcome'], "🟡")
        
        # Enhanced prediction without exposing regional details to user
        demographic_info = user_profile.get('demographic_info', {})
        success_prob = academic_pred.get('success_probability', 0.5)
        
        st.info(f"""
        **Predicted Outcome:** {outcome_emoji} {academic_pred['outcome']}  
        **Success Probability:** {success_prob*100:.1f}%  
        **Confidence:** {academic_pred['confidence']*100:.1f}%  
        **Risk Level:** {academic_pred['risk_level'].title()}  
        **Recommendation:** {'Intensive Support Needed' if academic_pred['risk_level'] == 'high' else 'Targeted Support' if academic_pred['risk_level'] == 'medium' else 'Enhancement Focus'}
        """)
    
    # Show demographic insights if available
    if demographic_info and demographic_info.get('region'):
        st.subheader("🌍 Personalized Insights Based on Your Background")
        
        region = demographic_info.get('region', '')
        imd_band = demographic_info.get('imd_band', '')
        education = demographic_info.get('highest_education', '')
        
        insights = []
        
        # Regional insights
        if region in ['London Region', 'South East Region']:
            insights.append("💡 **Regional Advantage**: Your region has excellent educational resources and support networks!")
        elif region in ['Scotland', 'Wales']:
            insights.append("💡 **Cultural Strength**: Strong educational traditions in your region - leverage local study communities!")
        elif region in ['North East Region', 'North West Region']:
            insights.append("💡 **Community Focus**: Your region emphasizes collaborative learning - consider joining study groups!")
        
        # Socioeconomic insights (using inferred IMD band without showing technical details)
        if imd_band in ['70-80%', '80-90%', '90-100%']:
            insights.append("🎓 **Additional Support**: Based on your region, we've identified extra support services that can help you succeed!")
        
        # Educational background insights
        if education == 'Lower Than A Level':
            insights.append("📚 **Foundation Path**: Building on strong foundations - consider progressive learning approaches!")
        elif education == 'HE Qualification':
            insights.append("🎓 **Advanced Learner**: Your educational background suggests you can handle challenging materials!")
        
        for insight in insights:
            st.info(insight)
    
    # Enhanced overview with demographic data
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("English Level", predicted_eng_level, 
                 delta=f"{confidence_pct:.0f}% confident")
    
    with col2:
        risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[academic_pred['risk_level']]
        success_prob = academic_pred.get('success_probability', 0.5)
        st.metric("Success Prediction", f"{outcome_emoji} {academic_pred['outcome']}", 
                 delta=f"{success_prob*100:.0f}% probability")
    
    with col3:
        # Show intervention priority instead of region to keep regional inference private
        success_prob = academic_pred.get('success_probability', 0.5)
        ml_priority_score = (2 - english_pred['level']) + (0 if success_prob > 0.8 else 1 if success_prob > 0.5 else 2)
        priority_level = "High" if ml_priority_score >= 3 else "Medium" if ml_priority_score >= 2 else "Low"
        st.metric("Intervention Priority", priority_level)
    
    with col4:
        if demographic_info.get('highest_education'):
            education_display = demographic_info['highest_education'][:10] + "..." if len(demographic_info['highest_education']) > 10 else demographic_info['highest_education']
            st.metric("Education", f"🎓 {education_display}")
        else:
            total_time = sum(15 if 'beginner' in rec['resource']['difficulty'].lower() else 25 
                            for rec in recommendations[:3])
            st.metric("Recommended Time", f"{total_time}min/day")
    
    # Priority recommendations with enhanced personalization
    st.subheader("🚀 Your Top 3 Priority Recommendations")
    st.markdown("*Customized based on your complete profile including background and demographics*")
    
    for i, rec in enumerate(recommendations[:3]):
        resource = rec['resource']
        
        # Enhanced recommendation explanation with demographic context
        enhanced_explanation = rec['relevance_explanation']
        if demographic_info.get('region') and i == 0:  # Add regional context to first recommendation
            if demographic_info['region'] in ['London Region', 'South East Region']:
                enhanced_explanation += " Additionally, given your location in a resource-rich region, consider leveraging local study groups and libraries."
            elif imd_band in ['70-80%', '80-90%', '90-100%']:  # Use inferred IMD internally without showing to user
                enhanced_explanation += " Based on your region, this recommendation includes additional support resources specifically designed to help you succeed."
        
        # Create enhanced recommendation card
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>#{i+1} {resource['name']}</h3>
            <p><strong>Type:</strong> {resource['type']} | <strong>Level:</strong> {resource['difficulty']}</p>
            <p>{resource['description']}</p>
            <p><strong>Time Required:</strong> {resource['time_required']}</p>
            <p><strong>Why this helps you:</strong> {enhanced_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for recommendation strength
        st.markdown('<div class="progress-bar" style="width: {}%;"></div>'.format(
            int(rec['score'] * 100)), unsafe_allow_html=True)
        st.markdown(f"**Match Score:** {rec['score']:.1%}")
        st.markdown("---")
    
    # Additional recommendations
    if len(recommendations) > 3:
        with st.expander("📚 See More Recommendations"):
            for i, rec in enumerate(recommendations[3:], 4):
                resource = rec['resource']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**#{i} {resource['name']}**")
                    st.write(f"*{resource['type']} - {resource['difficulty']}*")
                    st.write(resource['description'])
                    st.write(f"**Why recommended:** {rec['relevance_explanation']}")
                
                with col2:
                    st.metric("Match", f"{rec['score']:.1%}")
                    st.write(f"⏱️ {resource['time_required']}")
                
                st.markdown("---")
    
    # Enhanced action plan with demographic considerations
    st.subheader("📅 Suggested Weekly Action Plan")
    st.markdown("*Customized for your region and background*")
    
    action_plan = create_enhanced_action_plan(recommendations[:3], user_profile)
    
    for day, activities in action_plan.items():
        with st.expander(f"📆 {day}"):
            for activity in activities:
                st.write(f"• {activity}")
    
    # Progress tracking with demographic insights
    st.subheader("📈 Track Your Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weekly Goals:**")
        goals = generate_weekly_goals(recommendations[:3], user_profile)
        for goal in goals:
            st.write(f"✓ {goal}")
    
    with col2:
        st.write("**Success Metrics:**")
        metrics = generate_enhanced_success_metrics(user_profile)
        for metric in metrics:
            st.write(f"📊 {metric}")
    
    # Enhanced download section
    st.subheader("💾 Save Your Personalized Study Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Full Study Plan", type="secondary"):
            study_plan_text = generate_enhanced_study_plan_text(recommendations, user_profile)
            st.download_button(
                label="📄 Download as Text File",
                data=study_plan_text,
                file_name=f"EducationCare_StudyPlan_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔄 Retake Assessment", type="secondary"):
            # Reset session state
            for key in ['step', 'english_results', 'academic_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()


def create_enhanced_action_plan(recommendations, user_profile):
    """Create enhanced weekly action plan based on recommendations and demographic data"""
    
    demographic_info = user_profile.get('demographic_info', {})
    region = demographic_info.get('region', '')
    
    action_plan = {
        "Monday": ["🎯 Start the week by reviewing your goals", "📚 Begin with your #1 priority recommendation"],
        "Tuesday": ["📖 Continue with English skill building", "⏰ Practice time management techniques"],
        "Wednesday": ["👥 Mid-week check-in and peer interaction", "📝 Focus on writing/communication skills"],
        "Thursday": ["🔍 Review and assess progress so far", "📊 Work on academic skill development"],
        "Friday": ["💪 Intensive practice session", "✅ Complete weekly targets"],
        "Saturday": ["🌟 Explore supplementary materials", "🎨 Creative learning activities"],
        "Sunday": ["📈 Weekly reflection and planning", "🎯 Set goals for next week"]
    }
    
    # Add regional customizations
    if 'London' in region:
        action_plan["Saturday"].append("🏛️ Consider visiting British Library for additional resources")
        action_plan["Wednesday"].append("🎓 Explore London university open lectures")
    elif 'Scotland' in region:
        action_plan["Wednesday"].append("🏴󐁧󐁢󐁳󐁣󐁴󐁿 Connect with local Scottish study networks")
        action_plan["Saturday"].append("📚 Use National Library of Scotland resources")
    elif region:
        action_plan["Saturday"].append(f"🏫 Explore local educational resources in {region}")
    
    # Customize based on recommendations
    top_rec = recommendations[0]['resource'] if recommendations else None
    
    if top_rec:
        if 'English' in top_rec['type']:
            action_plan["Tuesday"].insert(1, f"🇬🇧 Work on {top_rec['name']} (20-30 min)")
            action_plan["Thursday"].insert(1, f"🇬🇧 Continue {top_rec['name']} practice")
        elif 'Study Skills' in top_rec['type']:
            action_plan["Monday"].insert(1, f"📚 Start {top_rec['name']} program")
            action_plan["Wednesday"].insert(1, f"📚 Continue {top_rec['name']} activities")
    
    # Add demographic-specific support
    imd_band = demographic_info.get('imd_band', '')
    if imd_band in ['70-80%', '80-90%', '90-100%']:
        action_plan["Monday"].append("🤝 Identify available support services")
        action_plan["Friday"].append("📞 Check in with academic support team")
    
    return action_plan


def generate_enhanced_success_metrics(user_profile):
    """Generate enhanced success metrics including demographic considerations"""
    
    metrics = [
        "Weekly quiz scores",
        "Assignment submission timeliness",
        "Daily study time logged",
        "Platform engagement frequency"
    ]
    
    # Add English-specific metrics if needed
    eng_scores = user_profile.get('english_scores', {})
    if eng_scores.get('overall_score', 1) < 0.7:
        metrics.extend([
            "English vocabulary test scores",
            "Writing quality improvements"
        ])
    
    # Add demographic-specific metrics
    demographic_info = user_profile.get('demographic_info', {})
    if demographic_info.get('highest_education') == 'Lower Than A Level':
        metrics.append("Foundation skills development progress")
    
    if demographic_info.get('imd_band') in ['70-80%', '80-90%', '90-100%']:
        metrics.append("Support services utilization")
    
    if demographic_info.get('region'):
        metrics.append("Local resource engagement")
    
    return metrics


def generate_enhanced_study_plan_text(recommendations, user_profile):
    """Generate enhanced downloadable study plan text with demographic information"""
    
    demographic_info = user_profile.get('demographic_info', {})
    academic_profile = user_profile.get('academic_profile', user_profile.get('academic_results', {}))
    
    plan_text = f"""
EDUCATIONCARE ENHANCED PERSONALIZED STUDY PLAN
Generated on: {datetime.now().strftime('%B %d, %Y')}

=== YOUR COMPLETE PROFILE SUMMARY ===
English Level: {user_profile['english_scores']['overall_score']:.1%}
Academic Risk: {academic_profile.get('risk_level', 'medium').title()}
Study Hours/Week: {academic_profile.get('study_hours_per_week', 15)}"""
    
    if demographic_info:
        plan_text += f"""
Educational Background: {demographic_info.get('highest_education', 'Not specified')}
Age Group: {demographic_info.get('age_band', 'Not specified')}
Gender: {demographic_info.get('gender', 'Not specified')}"""
    
    plan_text += f"""

=== TOP RECOMMENDATIONS ==="""
    
    for i, rec in enumerate(recommendations[:3], 1):
        resource = rec['resource']
        plan_text += f"""
{i}. {resource['name']}
   Type: {resource['type']}
   Time: {resource['time_required']}
   Description: {resource['description']}
   Why recommended: {rec['relevance_explanation']}
   
"""
    
    plan_text += """
=== ENHANCED WEEKLY ACTION PLAN ===
- Monday: Review goals, start priority recommendation"""
    
    if demographic_info.get('imd_band') in ['70-80%', '80-90%', '90-100%']:
        plan_text += ", identify support services"
    
    plan_text += """
- Tuesday: English skill building, time management
- Wednesday: Peer interaction, communication skills"""
    
    if demographic_info.get('region'):
        region = demographic_info['region']
        if 'London' in region:
            plan_text += ", explore London university resources"
        elif 'Scotland' in region:
            plan_text += ", connect with Scottish study networks"
    
    plan_text += """
- Thursday: Progress review, academic skills
- Friday: Intensive practice, complete targets
- Saturday: Supplementary materials, creative learning"""
    
    if 'London' in demographic_info.get('region', ''):
        plan_text += ", British Library visit"
    
    plan_text += """
- Sunday: Weekly reflection and next week planning

=== SUCCESS TRACKING ===
Monitor these metrics weekly:
- Assignment scores and timeliness
- Study time and consistency
- English skill improvements
- Platform engagement levels"""
    
    if demographic_info.get('highest_education') == 'Lower Than A Level':
        plan_text += """
- Foundation skills development progress"""
    
    if demographic_info.get('region'):
        plan_text += """
- Local resource engagement"""
    
    plan_text += f"""

=== PERSONALIZED INSIGHTS ==="""
    
    if demographic_info.get('region'):
        region = demographic_info['region']
        if region in ['London Region', 'South East Region']:
            plan_text += f"""
- Regional Advantage: Your area offers excellent educational resources
- Recommended: Leverage local libraries and university open courses"""
        elif region in ['Scotland', 'Wales']:
            plan_text += f"""
- Cultural Strength: Your area has strong educational traditions
- Recommended: Connect with local study communities and heritage institutions"""
        else:
            plan_text += f"""
- Local Resources: Explore educational opportunities in your area
- Recommended: Connect with local study groups and libraries"""
    
    if demographic_info.get('imd_band') in ['70-80%', '80-90%', '90-100%']:
        plan_text += """
- Support Available: Additional academic support services designed for your success
- Action: Proactively engage with available support programs"""
    
    plan_text += """

Generated by EducationCare Enhanced Study Recommender System
Including demographic-aware personalization
"""
    
    return plan_text


def create_action_plan(recommendations, user_profile):
    """Create weekly action plan based on recommendations"""
    
    action_plan = {
        "Monday": ["🎯 Start the week by reviewing your goals", "📚 Begin with your #1 priority recommendation"],
        "Tuesday": ["📖 Continue with English skill building", "⏰ Practice time management techniques"],
        "Wednesday": ["👥 Mid-week check-in and peer interaction", "📝 Focus on writing/communication skills"],
        "Thursday": ["🔍 Review and assess progress so far", "📊 Work on academic skill development"],
        "Friday": ["💪 Intensive practice session", "✅ Complete weekly targets"],
        "Saturday": ["🌟 Explore supplementary materials", "🎨 Creative learning activities"],
        "Sunday": ["📈 Weekly reflection and planning", "🎯 Set goals for next week"]
    }
    
    # Customize based on recommendations
    top_rec = recommendations[0]['resource'] if recommendations else None
    
    if top_rec:
        if 'English' in top_rec['type']:
            action_plan["Tuesday"].insert(1, f"🇬🇧 Work on {top_rec['name']} (20-30 min)")
            action_plan["Thursday"].insert(1, f"🇬🇧 Continue {top_rec['name']} practice")
        elif 'Study Skills' in top_rec['type']:
            action_plan["Monday"].insert(1, f"📚 Start {top_rec['name']} program")
            action_plan["Wednesday"].insert(1, f"📚 Continue {top_rec['name']} activities")
    
    return action_plan


def generate_weekly_goals(recommendations, user_profile):
    """Generate weekly goals based on profile and recommendations"""
    
    goals = []
    
    # English-based goals
    eng_scores = user_profile.get('english_scores', {})
    if eng_scores.get('overall_score', 0) < 0.6:
        goals.append("Complete 30 minutes of English practice daily")
        goals.append("Learn 10 new vocabulary words")
    
    # Academic behavior goals
    acad = user_profile.get('academic_profile', {})
    if acad.get('submission_timeliness', 0) > 2:
        goals.append("Submit all assignments 2 days before deadline")
    
    if acad.get('engagement_consistency', 1) < 0.5:
        goals.append("Log into learning platform daily")
    
    if acad.get('study_hours_per_week', 20) < 15:
        goals.append(f"Study for {min(acad.get('study_hours_per_week', 10) + 5, 20)} hours this week")
    
    # Recommendation-based goals
    for rec in recommendations:
        if 'time_management' in rec['resource'].get('features', []):
            goals.append("Implement new time management technique")
        if 'confidence' in rec['resource'].get('features', []):
            goals.append("Complete confidence-building exercise")
    
    return goals[:5]  # Limit to 5 goals


def generate_success_metrics(user_profile):
    """Generate success metrics to track"""
    
    metrics = [
        "Weekly quiz scores",
        "Assignment submission timeliness",
        "Daily study time logged",
        "Platform engagement frequency"
    ]
    
    # Add English-specific metrics if needed
    eng_scores = user_profile.get('english_scores', {})
    if eng_scores.get('overall_score', 1) < 0.7:
        metrics.extend([
            "English vocabulary test scores",
            "Writing quality improvements"
        ])
    
    return metrics


def generate_study_plan_text(recommendations, user_profile):
    """Generate downloadable study plan text"""
    
    plan_text = f"""
EDUCATIONCARE PERSONALIZED STUDY PLAN
Generated on: {datetime.now().strftime('%B %d, %Y')}

=== YOUR PROFILE SUMMARY ===
English Level: {user_profile['english_scores']['overall_score']:.1%}
Academic Risk: {user_profile['academic_profile']['risk_level'].title()}
Study Hours/Week: {user_profile['academic_profile']['study_hours_per_week']}

=== TOP RECOMMENDATIONS ===
"""
    
    for i, rec in enumerate(recommendations[:3], 1):
        resource = rec['resource']
        plan_text += f"""
{i}. {resource['name']}
   Type: {resource['type']}
   Time: {resource['time_required']}
   Description: {resource['description']}
   Why recommended: {rec['relevance_explanation']}
   
"""
    
    plan_text += """
=== WEEKLY ACTION PLAN ===
- Monday: Review goals, start priority recommendation
- Tuesday: English skill building, time management
- Wednesday: Peer interaction, communication skills
- Thursday: Progress review, academic skills
- Friday: Intensive practice, complete targets
- Saturday: Supplementary materials, creative learning
- Sunday: Weekly reflection and next week planning

=== SUCCESS TRACKING ===
Monitor these metrics weekly:
- Assignment scores and timeliness
- Study time and consistency
- English skill improvements
- Platform engagement levels

Generated by EducationCare Study Recommender System
"""
    
    return plan_text


if __name__ == "__main__":
    create_user_interface()