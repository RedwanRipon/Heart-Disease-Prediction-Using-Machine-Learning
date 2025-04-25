# Heart-Disease-Prediction-Using-Machine-Learning
We're going to build a model to predict whether or not they have heart disease.  It is intended to be an end-to-end example of what a data science and machine learning proof of concept might look like.



# Predicting heart disease using machine learning
This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.

# We're going to take the following approach:

Problem definition
Data
Evaluation
Feature
Modelling
Experimentation
1. Problem Defination
In a statement,

Given clinical parameter about a patient, can we predict whether or not they have heart disease

2. Data
The orginal data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

3. Evaluation
If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

4. Features
Create data dictionary

The following are the features we'll use to predict our target variable (heart disease or no heart disease).

Feature # Description # Example Values
age Age in years 29, 45, 60

sex 1 = male; 0 = female 0, 1

cp Chest pain type 0: Typical angina (chest pain), 1: Atypical angina (chest pain not related to heart), 2: Non- anginal pain (typically esophageal spasms (non heart related), 3: Asymptomatic (chest pain not showing signs of disease)

trestbps Resting blood pressure 120,140,150 (in mm Hg on admission to the hospital)

chol Serum cholesterol in mg/dl 180, 220, 250

fbs Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) 0, 1

restecg Resting electrocardiographic results 0: Nothing to note, 1: ST-T Wave abnormality, 2: Left ventricular hypertrophy

thalach Maximum heart rate achieved 160, 180, 190

exang Exercise induced angina 0, 1 (1 = yes; 0 = no)

oldpeak ST depression (heart potentially not 0.5, 1.0, 2.0 getting enough oxygen) induced by exercise relative to rest

slope The slope of the peak exercise ST segment 0: Upsloping, 1: Flatsloping, 2: Downsloping

ca Number of major vessels (0-3) colored 0, 1, 2, 3 by fluoroscopy

thal Thalium stress result 1: Normal, 3: Normal, 6: Fixed defect, 7: Reversible defect

target Have disease or not (1 = yes; 0 = no) 0, 1

Preparing the tools
We're going to use pandas, matplotlib and numoy for data analysis and manipulation.

# import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
%matplotlib inline

 

# Models from Scikit-Learn
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay


# Load Data
df = pd.read_csv("heart-disease.csv")
df.shape 
