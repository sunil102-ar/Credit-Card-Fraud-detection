# Credit-Card-Fraud-detection Web App

![image](https://user-images.githubusercontent.com/69753319/117111104-c1a52700-ada4-11eb-835f-5d1f5d36d9ee.png)

This repository contains the procedure we followed to deploy our web app of Credit Card Fraud detection on Heroku.

Since the data for credit card fraud is not available in real form(due to confidentiality), and is availbale only in dimensionality reduced form, we will be sharing some of the test cases here

In this project I build machine learning models to identify fraud in European credit card transactions. I also make several data visualizations to reveal patterns and structure in the data.

The dataset, hosted on Kaggle, includes credit card transactions made in September 2013 by European cardholders. The data contains 284,807 transactions that occurred over a two-day period, of which 492 (0.17%) are fraudulent. Each transaction has 30 features, all of which are numerical. The features V1, V2, ..., V28 are the result of a PCA transformation. To protect confidentiality, background information on these features is not available. The Time feature contains the time elapsed since the first transaction, and the Amount feature contains the transaction amount. The response variable, Class, is 1 in the case of fraud, and 0 otherwise.

Model Training
Architecture

XGboost model on balanced dataset.
Inference Results

Accuracy: 0.99
Recall: 0.76
F1 score: 0.84

Web App Production
1. Procfile - Contains the type of app.
2. Requirements - Libraries needed to run the app.
3. Templates - Files required for rendering purpose
4. App - Main file which will run our Web App.

![image](https://user-images.githubusercontent.com/69753319/118075699-304e3a00-b3ce-11eb-9309-3c2eab7a188f.png)

![image](https://user-images.githubusercontent.com/69753319/118075741-465bfa80-b3ce-11eb-9c3f-c3d3f0399b40.png)

![image](https://user-images.githubusercontent.com/69753319/118075820-74d9d580-b3ce-11eb-8b78-ccdf92dfbaa2.png)

![image](https://user-images.githubusercontent.com/69753319/118075859-8a4eff80-b3ce-11eb-9f3c-589348eb6f38.png)
