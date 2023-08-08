# Heart Disease Prediction using Machine Learning

This repository contains a machine learning project that focuses on predicting heart disease using various classifiers. The project includes data preprocessing, feature engineering, model building, and evaluation steps. The dataset used for this project is sourced from 'heart.csv', containing various features related to heart health.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Exploring the Dataset](#exploring-the-dataset)
- [Feature Engineering](#feature-engineering)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
  - [KNeighbors Classifier](#kneighbors-classifier)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
- [Conclusion](#conclusion)

## Introduction
The goal of this project is to build and evaluate machine learning models for predicting the presence of heart disease based on various health-related features. We will explore and analyze the dataset, preprocess the data, and experiment with three different classifiers: KNeighbors Classifier, Decision Tree Classifier, and Random Forest Classifier.

## Dataset
The dataset used in this project is 'heart.csv'. It contains various columns such as age, sex, chest pain type, resting blood pressure, serum cholesterol level, maximum heart rate, exercise-induced angina, and more.

## Getting Started
To get started with this project, you need to have Python installed along with the required libraries such as numpy, pandas, matplotlib, seaborn, and scikit-learn. Clone this repository to your local machine and ensure you have the 'heart.csv' file in the same directory.

## Exploring the Dataset
We begin by exploring the dataset using pandas and matplotlib. The code snippet provided in the repository includes commands to view the dataset's shape, column names, data types, sample records, and basic statistics.

## Feature Engineering
Before training the models, we perform feature engineering. Categorical variables ('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal') are converted into dummy variables using the 'get_dummies()' function to ensure compatibility with machine learning algorithms.

## Data Preprocessing
We standardize the numeric features ('age', 'trestbps', 'chol', 'thalach', 'oldpeak') using StandardScaler from scikit-learn to bring them to a similar scale.

## Model Building
We experiment with three classifiers and evaluate their performance using cross-validation.

### KNeighbors Classifier
We iterate over different values of 'k' (number of neighbors) and calculate the mean accuracy using cross-validation. The best 'k' value is selected based on the highest accuracy.

### Decision Tree Classifier
We experiment with different depths of decision trees and evaluate their accuracy using cross-validation. The depth that yields the best accuracy is chosen.

### Random Forest Classifier
We explore various numbers of estimators (trees) in the random forest ensemble and assess their performance using cross-validation. The number of estimators leading to the highest accuracy is selected.

## Conclusion
In this project, we've demonstrated the process of building and evaluating machine learning models for heart disease prediction. By exploring the dataset, performing feature engineering, and experimenting with different classifiers, we aim to find the most accurate model for this specific task.

Feel free to fork and modify this repository to experiment with other classifiers, fine-tune hyperparameters, and further enhance the predictive performance.
