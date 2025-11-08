"""
Categorical Mapping Utilities for EducationCare ML Models
=========================================================

This module provides utilities for handling categorical variables in the ML pipeline:
1. Maps user-friendly form inputs to one-hot encoded features expected by ML models
2. Handles demographic variables: region, IMD band, education level, age band, gender, disability
3. Provides form field options for Streamlit UI components

Key Challenge Solved:
- Users fill simple categorical dropdowns
- ML models expect specific one-hot encoded feature vectors
- This module bridges that gap with proper encoding/decoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from pathlib import Path

class CategoricalMapper:
    """
    Handles categorical variable encoding/decoding for the EducationCare ML pipeline
    
    This class manages the mapping between:
    - User form inputs (friendly categorical values)
    - ML model features (one-hot encoded vectors)
    """
    
    def __init__(self, data_file='data/studentInfo.csv'):
        """Initialize the categorical mapper with dataset statistics"""
        self.data_file = data_file
        self.categorical_columns = [
            'gender', 'region', 'highest_education', 'imd_band', 
            'age_band', 'disability'
        ]
        
        # Load and process categorical data
        self._load_categorical_mappings()
        
    def _load_categorical_mappings(self):
        """Load categorical mappings from the dataset"""
        try:
            # Load student data to extract categorical values
            df = pd.read_csv(self.data_file)
            
            # Extract unique values for each categorical column
            self.categorical_values = {}
            for col in self.categorical_columns:
                if col in df.columns:
                    unique_vals = sorted(df[col].dropna().unique())
                    self.categorical_values[col] = unique_vals
                else:
                    # Fallback values if column not found
                    self.categorical_values[col] = self._get_fallback_values(col)
            
            # Create one-hot encoded feature names
            self.all_encoded_features = []
            self.feature_mappings = {}
            
            for col in self.categorical_columns:
                col_features = []
                for value in self.categorical_values[col]:
                    feature_name = f"{col}_{value.replace(' ', '_').replace('-', '_').replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}"
                    col_features.append(feature_name)
                    self.all_encoded_features.append(feature_name)
                
                self.feature_mappings[col] = {
                    'values': self.categorical_values[col],
                    'features': col_features
                }
            
            print(f"✅ Categorical mapper initialized with {len(self.all_encoded_features)} encoded features")
            
        except Exception as e:
            print(f"⚠️ Error loading categorical mappings: {e}")
            self._use_fallback_mappings()
    
    def _get_fallback_values(self, column):
        """Provide fallback values if data file is not available"""
        fallback_mappings = {
            'gender': ['F', 'M'],
            'region': [
                'East Anglian Region', 'East Midlands Region', 'Ireland',
                'London Region', 'North Region', 'North Western Region',
                'Scotland', 'South East Region', 'South West Region',
                'Wales', 'West Midlands Region', 'Yorkshire Region'
            ],
            'highest_education': [
                'A Level or Equivalent', 'HE Qualification', 
                'Lower Than A Level', 'No Formal quals', 'Post Graduate Qualification'
            ],
            'imd_band': [
                '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
            ],
            'age_band': ['0-35', '35-55', '55<='],
            'disability': ['N', 'Y']
        }
        return fallback_mappings.get(column, [])
    
    def _use_fallback_mappings(self):
        """Use hardcoded fallback mappings if data loading fails"""
        self.categorical_values = {}
        for col in self.categorical_columns:
            self.categorical_values[col] = self._get_fallback_values(col)
        
        # Create feature mappings
        self.all_encoded_features = []
        self.feature_mappings = {}
        
        for col in self.categorical_columns:
            col_features = []
            for value in self.categorical_values[col]:
                feature_name = f"{col}_{value.replace(' ', '_').replace('-', '_').replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}"
                col_features.append(feature_name)
                self.all_encoded_features.append(feature_name)
            
            self.feature_mappings[col] = {
                'values': self.categorical_values[col],
                'features': col_features
            }
    
    def encode_user_categorical_inputs(self, user_inputs):
        """
        Convert user form inputs to one-hot encoded features
        
        Args:
            user_inputs (dict): User selections from form
            
        Returns:
            np.array: One-hot encoded feature vector
        """
        # Initialize feature vector with zeros
        feature_vector = np.zeros(len(self.all_encoded_features))
        
        # Encode each categorical input
        for col, user_value in user_inputs.items():
            if col in self.feature_mappings and user_value is not None:
                # Find the correct feature index
                try:
                    value_idx = self.categorical_values[col].index(user_value)
                    feature_name = self.feature_mappings[col]['features'][value_idx]
                    feature_idx = self.all_encoded_features.index(feature_name)
                    feature_vector[feature_idx] = 1
                except (ValueError, IndexError):
                    print(f"⚠️ Unknown value '{user_value}' for category '{col}'")
        
        return feature_vector
    
    def get_feature_names(self):
        """Get all one-hot encoded feature names"""
        return self.all_encoded_features
    
    def get_categorical_info(self):
        """Get information about all categorical variables"""
        return self.categorical_values

def encode_user_categorical_inputs(user_inputs, mapper=None):
    """
    Standalone function to encode user inputs
    
    This function provides a simpler interface for encoding categorical inputs
    without needing to create a CategoricalMapper instance each time.
    """
    if mapper is None:
        mapper = CategoricalMapper()
    
    return mapper.encode_user_categorical_inputs(user_inputs)

def get_form_field_options(field_name):
    """
    Get available options for a specific form field
    
    Args:
        field_name (str): Name of the categorical field
        
    Returns:
        list: Available options for that field
    """
    try:
        mapper = CategoricalMapper()
        return mapper.categorical_values.get(field_name, [])
    except Exception as e:
        print(f"⚠️ Error getting options for {field_name}: {e}")
        # Return fallback options
        fallback_options = {
            'region': [
                'East Anglian Region', 'East Midlands Region', 'Ireland',
                'London Region', 'North Region', 'North Western Region',
                'Scotland', 'South East Region', 'South West Region',
                'Wales', 'West Midlands Region', 'Yorkshire Region'
            ],
            'imd_band': [
                '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
            ],
            'highest_education': [
                'A Level or Equivalent', 'HE Qualification', 
                'Lower Than A Level', 'No Formal quals', 'Post Graduate Qualification'
            ],
            'age_band': ['0-35', '35-55', '55<='],
            'gender': ['F', 'M'],
            'disability': ['N', 'Y']
        }
        return fallback_options.get(field_name, [])

def get_demographic_defaults():
    """
    Get reasonable default values for demographic variables
    
    These are used when users don't provide demographic information
    but ML models still need demographic features.
    """
    return {
        'gender': 'M',  # Most common in dataset
        'region': 'London Region',  # Neutral choice
        'highest_education': 'A Level or Equivalent',  # Most common level
        'imd_band': '30-40%',  # Middle deprivation band
        'age_band': '35-55',  # Most common age range
        'disability': 'N'  # Most common
    }

def create_demographic_feature_vector(user_demographics=None):
    """
    Create a complete demographic feature vector for ML models
    
    Args:
        user_demographics (dict): User's demographic information
        
    Returns:
        np.array: Complete one-hot encoded demographic features
    """
    mapper = CategoricalMapper()
    
    # Use defaults for missing information
    defaults = get_demographic_defaults()
    if user_demographics:
        # Merge user input with defaults
        complete_demographics = {**defaults, **user_demographics}
    else:
        complete_demographics = defaults
    
    return mapper.encode_user_categorical_inputs(complete_demographics)

# Convenience functions for specific use cases
def get_region_imd_mapping():
    """Get region to IMD band mapping for suggestions"""
    return {
        'London Region': '20-30%',
        'South East Region': '10-20%', 
        'South West Region': '30-40%',
        'North Region': '50-60%',
        'North Western Region': '60-70%',
        'Yorkshire Region': '40-50%',
        'Scotland': '40-50%',
        'Wales': '50-60%',
        'Ireland': '30-40%',
        'East Anglian Region': '20-30%',
        'East Midlands Region': '40-50%',
        'West Midlands Region': '50-60%',
    }

def validate_categorical_input(field_name, value):
    """
    Validate that a categorical input value is acceptable
    
    Args:
        field_name (str): Name of the categorical field
        value: Value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    valid_options = get_form_field_options(field_name)
    return value in valid_options

if __name__ == "__main__":
    # Test the categorical mapper
    print("🔄 Testing Categorical Mapper...")
    
    try:
        mapper = CategoricalMapper()
        
        # Test encoding
        test_inputs = {
            'gender': 'M',
            'region': 'London Region',
            'highest_education': 'A Level or Equivalent',
            'imd_band': '30-40%',
            'age_band': '35-55',
            'disability': 'N'
        }
        
        encoded = mapper.encode_user_categorical_inputs(test_inputs)
        print(f"✅ Encoded {len(test_inputs)} categorical inputs to {len(encoded)} features")
        print(f"   Non-zero features: {np.sum(encoded > 0)}")
        
        # Test form options
        regions = get_form_field_options('region')
        print(f"✅ Available regions: {len(regions)}")
        
        imd_bands = get_form_field_options('imd_band')
        print(f"✅ Available IMD bands: {len(imd_bands)}")
        
        print("✅ Categorical mapping utilities working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing categorical mapper: {e}")
        import traceback
        traceback.print_exc()