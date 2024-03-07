# Copper Industrial Modelling
Copper Industrial Modelling project is aimed at predicting selling price and transaction status of Industrial copper by utilising Machine Learning Algorithms

**Problem Statement:**
1) Exploring skewness and outliers in the dataset.
2) Transform the data into a suitable format and perform any necessary cleaning
and pre-processing steps.
3) ML Regression model which predicts continuous variable ‘Selling_Price’.
4) ML Classification model which predicts Status: WON or LOST.
5) Creating a streamlit page for Selling_Price prediction value and Status(Won/Lost)

**Tools and Technologies Used :** Python - Pandas,Numpy,Sklearn,matplotlib,seaborn,xgboost,pickle, Jupyter, Visual Studio Code and Streamlit 

**Approach :**

Industrial copper dataset contains 181673 rows and 14 columns.[**Exploratory Data Analysis**](https://github.com/KiruthikaParanthaman/Copper_Industrial_Modelling/blob/main/Copper%20Modelling%20jupyter.ipynb) like  univariate Analysis, Bi-variate Analysis done which displayed no significant correlation for the selected numerical features.New Feature including month_year,year,month of item_date and delivery_date was extracted and feature selection done using **pearson correlation**.Only date column had strong positive correlation hence month column of item_date was included for analysis.**Data Cleaning** and outliers were handled using **Winsorizing Technique** as abnormal selling price for few items in inventory is normal and is better than dropping outliers.Categorical data was converted to numerical by label encoding and item_month using **One-Hot Encoding** to aid predictions in real time as some months were misisng in dataset 

Data was standardized using **Standard Scaler** and data was split into train,Validation and test dataset. Selling Price was predicted using **Random Forest Regressor** as it had least absolute mean of 45.88 in comparison with other regression models like Linear Regression,Decision Tree,XGBoostRegressor,KnnRegressor.For Copper Transaction status prediction dataset 150447 rows with  Won and Lost status alone was selected.**Balanced Bagging Classifier** model was selected for status prediction as it had highest accuracy of 90% in comparison with Logistic Regression,XGBclassifier.Finally the selected models and standard scaler was pickled for predicting real time data

**Preview of Application :**

![app_screenshot](https://github.com/KiruthikaParanthaman/Copper_Industrial_Modelling/assets/141828622/f7a034ac-102e-4fc1-95b0-e0231239c524)


