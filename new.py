
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
studentRegistration = pd.read_csv('studentRegistration.csv')
studentInfo = pd.read_csv('studentInfo.csv')
studentVle = pd.read_csv('vle.csv')
studentAssessment = pd.read_csv('studentAssessment.csv')
courses = pd.read_csv('courses.csv')
vle = pd.read_csv('vle.csv')
assessments = pd.read_csv('assessments.csv')



student_data = pd.merge(studentRegistration, studentInfo, on=['id_student', 'code_module', 'code_presentation'], how='inner')
student_data = pd.merge(student_data, courses, on=['code_module', 'code_presentation'], how='inner')
print(student_data.head())

student_assesment_data = assessments.merge(studentAssessment,on=['id_assessment'],how="inner")

student_assesment_data["score"].fillna(0, inplace=True)
student_assesment_data.dropna(inplace=True)
# print(student_assesment_data.weight.max())
# print(student_assesment_data.weight.min())
# print(student_assesment_data.isna().sum())
student_data_final = student_assesment_data.merge(student_data,on=['code_module','code_presentation','id_student'],how="inner")
# print(student_data_final)
df = student_data_final.copy()
print(df)