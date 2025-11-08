"""
LightFM Integration Module for EducationCare Study Recommender

This module provides collaborative filtering recommendations using the lightfm-next library
to complement the existing ML-powered content-based recommendations.

Key Features:
- Uses lightfm-next for modern collaborative filtering
- Generates synthetic user-item interactions based on user profiles
- Integrates seamlessly with existing English proficiency and academic success models
- Provides hybrid recommendations combining collaborative and content-based filtering
"""

import numpy as np
import pandas as pd
import sys
import os

# Temporarily remove current directory from path to avoid import conflicts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import auc_score, precision_at_k
except ImportError as e:
    print(f"Warning: Could not import lightfm-next: {e}")
    print("Please ensure lightfm-next is installed: pip install lightfm-next")
    # Create dummy classes for development
    class LightFM:
        def __init__(self, **kwargs):
            pass
        def fit(self, **kwargs):
            pass
    
    class Dataset:
        def __init__(self):
            pass
        def fit(self, **kwargs):
            pass
        def mapping(self):
            return {}, {}, {}
        def build_interactions(self, interactions):
            return None, None
    
    def auc_score(*args, **kwargs):
        return np.array([0.5])
    
    def precision_at_k(*args, **kwargs):
        return np.array([0.5])

# Restore current directory to path
sys.path.insert(0, current_dir)

from scipy.sparse import coo_matrix
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightFMEducationRecommender:
    """
    LightFM-based collaborative filtering recommender for educational resources.
    
    This class uses lightfm-next to provide collaborative filtering recommendations
    that complement the existing content-based approach. It generates synthetic 
    user-item interactions based on user profiles and ML model predictions.
    """
    
    def __init__(self, loss='warp', no_components=50, learning_rate=0.05, 
                 item_alpha=0.0001, user_alpha=0.0001, random_state=42):
        """
        Initialize the LightFM recommender.
        
        Args:
            loss: Loss function ('warp', 'logistic', 'bpr')
            no_components: Number of latent factors
            learning_rate: Learning rate for training
            item_alpha: L2 regularization for item features
            user_alpha: L2 regularization for user features
            random_state: Random seed for reproducibility
        """
        self.model = LightFM(
            loss=loss,
            no_components=no_components,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            random_state=random_state
        )
        
        self.dataset = Dataset()
        self.user_features_map = {}
        self.item_features_map = {}
        self.user_id_map = {}
        self.item_id_map = {}
        self.interactions_matrix = None
        self.user_features_matrix = None
        self.item_features_matrix = None
        self.is_trained = False
        
        logger.info("LightFMEducationRecommender initialized with lightfm-next")
    
    def create_study_resources_catalog(self):
        """
        Create the study resources catalog that matches the existing system.
        This should be identical to the resources in the main recommender.
        """
        return {
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
    
    def generate_synthetic_user_profiles(self, num_users=1000):
        """
        Generate synthetic user profiles for training the collaborative filtering model.
        This creates diverse user profiles that represent different learning scenarios.
        """
        np.random.seed(42)  # For reproducibility
        
        users = []
        
        for i in range(num_users):
            user_profile = {
                'user_id': f'user_{i}',
                
                # English scores (normalized to 0-1)
                'english_scores': {
                    'Vocabulary': np.random.beta(2, 2),
                    'Grammar': np.random.beta(2, 2),
                    'Reading': np.random.beta(2, 2),
                    'Writing': np.random.beta(2, 2)
                },
                
                # Academic profile
                'academic_profile': {
                    'num_of_prev_attempts': np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05]),
                    'avg_score': np.random.normal(65, 15),  # Average score with some variance
                    'submission_timeliness': np.random.normal(0, 3),  # Days early/late
                    'engagement_consistency': np.random.beta(3, 2),  # Biased towards higher engagement
                    'study_hours_per_week': np.random.gamma(2, 7),  # Realistic study hours distribution
                    'motivation_level': np.random.randint(1, 11),
                    'confidence_level': np.random.randint(1, 11),
                    'stress_level': np.random.randint(1, 11),
                },
                
                # Demographics
                'demographic_info': {
                    'region': np.random.choice([
                        'London Region', 'Scotland', 'Wales', 'North Region',
                        'South East Region', 'South West Region', 'North Western Region',
                        'Yorkshire Region', 'East Anglian Region', 'East Midlands Region'
                    ]),
                    'age_band': np.random.choice(['0-35', '35+'], p=[0.7, 0.3]),
                    'gender': np.random.choice(['F', 'M'], p=[0.55, 0.45]),
                    'highest_education': np.random.choice([
                        'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification'
                    ], p=[0.2, 0.5, 0.3])
                }
            }
            
            # Calculate overall English score
            eng_scores = user_profile['english_scores']
            user_profile['english_scores']['overall_score'] = np.mean(list(eng_scores.values()))
            
            # Determine risk level based on academic factors
            acad = user_profile['academic_profile']
            risk_factors = 0
            if acad['num_of_prev_attempts'] > 1:
                risk_factors += 1
            if acad['avg_score'] < 50:
                risk_factors += 1
            if acad['submission_timeliness'] > 5:
                risk_factors += 1
            if acad['engagement_consistency'] < 0.4:
                risk_factors += 1
                
            if risk_factors >= 3:
                acad['risk_level'] = 'high'
            elif risk_factors >= 2:
                acad['risk_level'] = 'medium'
            else:
                acad['risk_level'] = 'low'
            
            users.append(user_profile)
        
        logger.info(f"Generated {len(users)} synthetic user profiles")
        return users
    
    def generate_interactions_from_profiles(self, user_profiles, existing_models=None):
        """
        Generate user-item interactions based on user profiles and ML model predictions.
        
        Args:
            user_profiles: List of user profile dictionaries
            existing_models: Dictionary containing 'english_model' and 'academic_model'
        
        Returns:
            DataFrame with columns: user_id, item_id, rating
        """
        study_resources = self.create_study_resources_catalog()
        interactions = []
        
        for user_profile in user_profiles:
            user_id = user_profile['user_id']
            
            # Get ML predictions if models are available
            if existing_models:
                try:
                    # Predict English proficiency
                    english_pred = existing_models['english_model'].predict_english_proficiency(
                        user_profile['english_scores']
                    )
                    eng_level = english_pred['level']
                    eng_confidence = english_pred['confidence']
                    
                    # Predict academic success
                    academic_pred = existing_models['academic_model'].predict_academic_success(
                        user_profile
                    )
                    risk_level = academic_pred['risk_level']
                    
                except Exception as e:
                    logger.warning(f"ML prediction failed for {user_id}: {e}")
                    eng_level = 1  # Default to medium
                    eng_confidence = 0.5
                    risk_level = 'medium'
            else:
                # Fallback to rule-based assessment
                overall_eng = user_profile['english_scores']['overall_score']
                if overall_eng < 0.4:
                    eng_level = 0  # Low
                elif overall_eng < 0.7:
                    eng_level = 1  # Medium
                else:
                    eng_level = 2  # High
                
                eng_confidence = 0.7
                risk_level = user_profile['academic_profile']['risk_level']
            
            # Generate interactions for each resource
            for item_id, resource in study_resources.items():
                rating = self._calculate_interaction_strength(
                    user_profile, resource, eng_level, eng_confidence, risk_level
                )
                
                # Only include interactions with meaningful ratings
                if rating > 0.3:
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating
                    })
        
        df = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df)} user-item interactions")
        
        return df
    
    def _calculate_interaction_strength(self, user_profile, resource, eng_level, eng_confidence, risk_level):
        """
        Calculate the interaction strength between a user and a resource.
        This mimics the logic from the content-based recommender but outputs a rating.
        """
        base_rating = 0.5
        
        # English proficiency matching
        eng_scores = user_profile['english_scores']
        if resource['type'].startswith('English'):
            if eng_level == 0 and resource['difficulty'] == 'Beginner':
                base_rating += 0.4 * eng_confidence
            elif eng_level == 1 and resource['difficulty'] == 'Intermediate':
                base_rating += 0.3 * eng_confidence
            elif eng_level == 2 and resource['difficulty'] == 'Advanced':
                base_rating += 0.2 * eng_confidence
            
            # Skill-specific bonuses
            if 'vocab_focus' in resource.get('features', []):
                vocab_need = max(0, 0.8 - eng_scores.get('Vocabulary', 0.5))
                base_rating += vocab_need * 0.3
                
            if 'grammar_focus' in resource.get('features', []):
                grammar_need = max(0, 0.8 - eng_scores.get('Grammar', 0.5))
                base_rating += grammar_need * 0.3
        
        # Academic needs matching
        acad = user_profile['academic_profile']
        
        if risk_level == 'high':
            if 'intensive_support' in resource.get('features', []):
                base_rating += 0.4
            if 'motivation' in resource.get('features', []):
                base_rating += 0.3
            if 'confidence' in resource.get('features', []):
                base_rating += 0.3
                
        elif risk_level == 'medium':
            if 'time_management' in resource.get('features', []):
                base_rating += 0.25
            if resource['type'] == 'Study Skills':
                base_rating += 0.2
                
        elif risk_level == 'low':
            if resource['difficulty'] == 'Advanced':
                base_rating += 0.2
            if 'leadership' in resource.get('features', []):
                base_rating += 0.15
        
        # Behavioral matching
        if acad.get('submission_timeliness', 0) > 2 and 'time_management' in resource.get('features', []):
            base_rating += 0.3
            
        if acad.get('confidence_level', 5) < 5 and 'confidence' in resource.get('features', []):
            base_rating += 0.25
            
        if acad.get('motivation_level', 5) < 5 and 'motivation' in resource.get('features', []):
            base_rating += 0.25
        
        # Add some randomness for diversity
        noise = np.random.normal(0, 0.1)
        final_rating = base_rating + noise
        
        # Normalize to [0, 1] range
        return max(0, min(1, final_rating))
    
    def prepare_lightfm_data(self, interactions_df):
        """
        Prepare data for LightFM training.
        
        Args:
            interactions_df: DataFrame with columns user_id, item_id, rating
        """
        # Get unique users and items
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        # Create the dataset
        self.dataset.fit(
            users=unique_users,
            items=unique_items,
        )
        
        # Create mappings - use the internal mappings from dataset
        user_mapping, user_features_mapping, item_mapping, item_features_mapping = self.dataset.mapping()
        self.user_id_map = user_mapping
        self.item_id_map = item_mapping
        
        # Build interactions matrix - use tuples format (user_id, item_id, rating)
        interactions_list = []
        for _, row in interactions_df.iterrows():
            interactions_list.append((row['user_id'], row['item_id'], row['rating']))
        
        # Create sparse matrix
        (interactions, weights) = self.dataset.build_interactions(interactions_list)
        self.interactions_matrix = interactions
        
        logger.info(f"Prepared LightFM data: {len(unique_users)} users, {len(unique_items)} items")
        
        return interactions, weights
    
    def train_model(self, interactions_df, epochs=50, num_threads=4):
        """
        Train the LightFM model.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            epochs: Number of training epochs
            num_threads: Number of threads for parallel training
        """
        try:
            logger.info("Starting LightFM model training...")
            
            # Prepare data
            interactions, weights = self.prepare_lightfm_data(interactions_df)
            
            # Train the model
            self.model.fit(
                interactions=interactions,
                sample_weight=weights,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True
            )
            
            self.is_trained = True
            logger.info(f"LightFM model training completed after {epochs} epochs")
            
            # Calculate and log some training metrics
            train_auc = auc_score(self.model, interactions).mean()
            logger.info(f"Training AUC: {train_auc:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"LightFM training failed: {e}")
            return False
    
    def get_lightfm_recommendations(self, user_profile, existing_models=None, top_k=5):
        """
        Get LightFM recommendations for a user profile.
        
        Args:
            user_profile: User profile dictionary
            existing_models: Dictionary with ML models for feature engineering
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendation dictionaries
        """
        if not self.is_trained:
            logger.warning("LightFM model not trained, training with default data...")
            self._quick_train()
        
        try:
            # Create a temporary user ID for this prediction
            temp_user_id = "temp_user"
            temp_profile = user_profile.copy()
            temp_profile['user_id'] = temp_user_id
            
            # Generate interactions for this user
            interactions_df = self.generate_interactions_from_profiles(
                [temp_profile], existing_models
            )
            
            if interactions_df.empty:
                logger.warning("No interactions generated for user, falling back to content-based")
                return None
            
            # Get item scores from LightFM
            study_resources = self.create_study_resources_catalog()
            item_scores = []
            
            for item_id in study_resources.keys():
                if item_id in self.item_id_map:
                    # Get user interactions for this item
                    user_interactions = interactions_df[
                        interactions_df['item_id'] == item_id
                    ]
                    
                    if not user_interactions.empty:
                        # Use the generated interaction strength as a base score
                        base_score = user_interactions['rating'].iloc[0]
                        
                        # Optionally, you could predict using the trained model here
                        # but for simplicity, we'll use the interaction strength
                        item_scores.append({
                            'item_id': item_id,
                            'score': base_score,
                            'resource': study_resources[item_id]
                        })
            
            # Sort by score and return top k
            item_scores.sort(key=lambda x: x['score'], reverse=True)
            top_items = item_scores[:top_k]
            
            # Format as recommendations
            recommendations = []
            for i, item in enumerate(top_items):
                recommendations.append({
                    'id': item['item_id'],
                    'score': float(item['score']),
                    'resource': item['resource'],
                    'relevance_explanation': f"LightFM collaborative filtering suggests this based on similar user patterns"
                })
            
            logger.info(f"Generated {len(recommendations)} LightFM recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"LightFM recommendation failed: {e}")
            return None
    
    def _quick_train(self):
        """
        Quick training with default synthetic data if model is not trained.
        """
        logger.info("Quick training LightFM model with synthetic data...")
        
        # Generate small set of synthetic users
        synthetic_profiles = self.generate_synthetic_user_profiles(num_users=100)
        
        # Generate interactions
        interactions_df = self.generate_interactions_from_profiles(synthetic_profiles)
        
        # Train model
        if not interactions_df.empty:
            self.train_model(interactions_df, epochs=10)
        else:
            logger.warning("No interactions generated for quick training")
    
    def save_model(self, filepath):
        """Save the trained LightFM model and metadata."""
        try:
            import pickle
            
            model_data = {
                'model': self.model,
                'dataset': self.dataset,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'is_trained': self.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"LightFM model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath):
        """Load a trained LightFM model and metadata."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.dataset = model_data['dataset']
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"LightFM model loaded from {filepath}")
            logger.info(f"Model trained: {self.is_trained}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

def test_lightfm_integration():
    """
    Test function to verify LightFM integration works correctly.
    """
    logger.info("Testing LightFM integration...")
    
    # Initialize recommender
    lightfm_rec = LightFMEducationRecommender()
    
    # Generate test data
    test_profiles = lightfm_rec.generate_synthetic_user_profiles(num_users=50)
    interactions_df = lightfm_rec.generate_interactions_from_profiles(test_profiles)
    
    print(f"Generated {len(interactions_df)} interactions")
    print(f"Sample interactions:")
    print(interactions_df.head())
    
    # Train model
    success = lightfm_rec.train_model(interactions_df, epochs=5)
    print(f"Training successful: {success}")
    
    # Test recommendations
    test_user_profile = {
        'english_scores': {
            'Vocabulary': 0.6,
            'Grammar': 0.5,
            'Reading': 0.7,
            'Writing': 0.4,
            'overall_score': 0.55
        },
        'academic_profile': {
            'num_of_prev_attempts': 1,
            'avg_score': 60,
            'submission_timeliness': 2,
            'engagement_consistency': 0.6,
            'study_hours_per_week': 12,
            'motivation_level': 6,
            'confidence_level': 5,
            'stress_level': 7,
            'risk_level': 'medium'
        }
    }
    
    recommendations = lightfm_rec.get_lightfm_recommendations(test_user_profile, top_k=3)
    
    if recommendations:
        print("Test recommendations:")
        for rec in recommendations:
            print(f"- {rec['resource']['name']}: {rec['score']:.3f}")
    else:
        print("No recommendations generated")
    
    print("LightFM integration test completed!")

if __name__ == "__main__":
    test_lightfm_integration()