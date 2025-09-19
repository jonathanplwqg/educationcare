import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data1.csv')

# Basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())

# Descriptive statistics
print("Summary statistics:\n", df.describe())

# Value counts for categorical columns
categorical_cols = ['Resources', 'Extracurricular', 'Motivation', 'Internet', 'Gender', 'LearningStyle', 'OnlineCourses', 'Discussions', 'EduTech', 'StressLevel', 'FinalGrade']
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Distribution of FinalGrade
plt.figure(figsize=(6,4))
sns.countplot(x='FinalGrade', data=df)
plt.title("Distribution of FinalGrade")
plt.show()

# Distribution of ExamScore
plt.figure(figsize=(6,4))
sns.histplot(df['ExamScore'], bins=20, kde=True)
plt.title("Distribution of ExamScore")
plt.show()



