# Copper_Industrial_Modelling
Copper Industrial Modelling project is aimed at predicting selling price and transaction status of Industrial copper by utilising Machine Learning Algorithms

**Problem Statement:**
1) Exploring skewness and outliers in the dataset.
2) Transform the data into a suitable format and perform any necessary cleaning
and pre-processing steps.
3) ML Regression model which predicts continuous variable ‘Selling_Price’.
4) ML Classification model which predicts Status: WON or LOST.
5) Creating a streamlit page for Selling_Price prediction value and Status(Won/Lost)

**Approach :**
Industrial copper dataset contains 181673 rows and 14 columns. Exploratory Data Analysis like  univariate Analysis, Bi-variate Analysis done which displayed no significant correlation for the selected numerical features.New Feature including month_year,year,month of item_date and delivery_date was extracted and feature selection done using pearson correlation.Only date column had strong positive correlation hence month column of item_date was included for analysis.Data Cleaning and outliers were handled using Winsorizing Technique as abnormal selling price for few items in inventory is normal and is better than dropping outliers.

