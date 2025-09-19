



import pandas as pd

# Load the dataset
df = pd.read_csv('data2.csv')

# Basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe(include='all'))

# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Value counts for categorical columns
categorical_cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass']
for col in categorical_cols:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())

# Correlation matrix for numeric columns
print("\nCorrelation matrix:\n", df.corr())

