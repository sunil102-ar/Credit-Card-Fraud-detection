#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection
# 
# In this project you will predict fraudulent credit card transactions with the help of Machine learning models. Please import the following libraries to get started.

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[3]:


#Import some more important library
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import statsmodels.api as sm
import pickle
import gc 
from sklearn import svm
from xgboost import XGBClassifier
import xgboost as xgb


# ## Exploratory data analysis

# In[4]:


df = pd.read_csv("C:/Users/HP/Documents/creditcard.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


df.Time.tail(15)


# In[9]:


df.describe()


# In[10]:


df.Class.value_counts()


# In[11]:


sns.countplot(x=df.Class, hue=df.Class)


# In[12]:


plt.figure(figsize=(10, 5))
sns.distplot(df.Amount)


# In[13]:


df.head()


# # Spliting data into train and test

# In[14]:


X = df.drop(labels='Class', axis=1)
y = df['Class']

X.shape, y.shape


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit  
# Splitting the data into Train and Test set in proper proportion
kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)
for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc, X.iloc
        y_train, y_test = y[train_index], y[test_index]


# In[16]:


# Checking Skewness of data
plt.rcParams['figure.figsize'] = [10,8]
plt.hist(df['Amount'],edgecolor='k',bins = 5)
plt.title('Transaction Amount')
plt.xlabel('Amount in USD') 
plt.ylabel('Count')


# # If there is skewness present in the distribution use:
# - <b>Power Transformer</b> package present in the <b>preprocessing library provided by sklearn</b> to make distribution more gaussian

# In[17]:


from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer


# In[18]:


pt = preprocessing.PowerTransformer(copy=False)
PWTR_X = pt.fit_transform(X)


# # Model Building

# # Imbalanced dataset
#Logistic Regression on imbalanced dataset
# # Model 1

# In[19]:


# Splitting dataset into test and train sets in 70:30 ratio after applying Power Transform
kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)
for train_index, test_index in sss.split(PWTR_X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = PWTR_X[train_index], PWTR_X[test_index]
        y_train, y_test = y[train_index], y[test_index]


# In[20]:


from sklearn.linear_model import LogisticRegression

# Fit a logistic regression model to train data
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


# In[21]:


# Predict on test data
y_predicted = model_lr.predict(X_test)


# In[22]:


# Evaluation matric
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


# In[23]:


# Evaluation Metrics
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[24]:


# Function for roc_curve
def plot_roc_curve(fpr,tpr,roc_auc):
    plt.plot(fpr, tpr, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='upper right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[25]:


# tpr and fpr
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)


# In[26]:


# Plotting the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1. Precision 0.85
# 2. Recall    0.59
# 3. Accuracy  0.85
# 4. F1 Score  0.70
# 5. ROC Curve 0.79

# # Model 2

# # logistic regression on imbalanced dataset with K-fold and hypertuning

# In[27]:


# logistic regression on imbalanced dataset with K-fold and hypertuning
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
n_folds = 5

# parameters 
params ={'C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'penalty': ['l1', 'l2']}
lrh = LogisticRegression()
model_lrh = GridSearchCV(estimator=lrh, cv=n_folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[28]:


#fit the model
model_lrh.fit(X_train,y_train)


# In[29]:


pd.DataFrame(model_lrh.cv_results_)


# In[30]:


#Best score and best parameters
print("Logistic Regression with PCA Best AUC : ", model_lrh.best_score_)
print("Logistic Regression with PCA Best hyperparameters: ", model_lrh.best_params_)


# In[31]:


# Passing the best parameteres
model_lrh_tuned = LogisticRegression(penalty='l2',C=0.1)


# In[32]:


# Predicting on test data
model_lrh_tuned.fit(X_train,y_train)
y_predicted = model_lrh_tuned.predict(X_test)


# #Evaluation Metrices
# print('Classification report:\n', classification_report(y_test, y_predicted))
# print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
# print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
# print('ROC AUC : ', roc_auc_score(y_test, y_predicted))

# In[34]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.85
# 2.Recall    0.58
# 3.Acurracy  0.85
# 4.F1 Score  0.69
# 5.ROC AUC   0.79

# # Model 3

# In[35]:


# Random Forest on Imbalanced dataset


# In[36]:


#Initializing Random forest and creating model
from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier(n_jobs=-1, 
                             random_state=2018,
                             criterion='gini',
                             n_estimators=100,
                             verbose=False)


# In[37]:


# Fitting the model on Train data and Predicting on Test data
model_rfc.fit(X_train,y_train)
y_predicted = model_rfc.predict(X_test)


# In[38]:


# Evaluation Metrics
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[39]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.95
# 2.Recall    0.71
# 3.Acurracy  0.91
# 4.F1 Score  0.81
# 5.ROC AUC   0.85

# # Model 4

# In[40]:


# Random forest on imbalanced dataset with hypertuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# In[41]:


# Defining Parameters
params = { 
    'n_estimators': [200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[42]:


# Stratified K Fold
cross_val = StratifiedKFold(n_splits=3)
index_iterator = cross_val.split(X_train, y_train)
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[43]:


# Passing the best parameteres based on Randomized Search CV
model_rfc_tuned = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud
                               criterion='gini',
                               max_depth=5,
                               max_features='sqrt',
                               min_samples_leaf=10,
                               n_estimators=200,
                               n_jobs=-1, 
                               random_state=5)


# In[44]:


# Fitting the model on Train data and Predicting on Test Data
model_rfc_tuned.fit(X_train,y_train)
y_predicted = model_rfc_tuned.predict(X_test)


# In[45]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[46]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.82
# 2.Recall    0.72
# 3.Acurracy  0.88
# 4.F1 Score  0.76
# 5.ROC AUC   0.85

# # Model 5 

# In[47]:


# XGboost on imbalanced dataset


# In[48]:


#Initializing XGBOOST and creating model
model_xgb = XGBClassifier()


# In[49]:


# Fitting the model on Train data and Predicting on Test data
model_xgb.fit(X_train,y_train)
y_predicted = model_xgb.predict(X_test)


# In[50]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[51]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.95
# 2.Recall    0.84
# 3.Acurracy  0.92
# 4.F1 Score  0.73
# 5.ROC AUC   0.87

# # Model 6

# # XGB on Imbalanced Data with K-Fold and Hyperparamater Tuning

# In[52]:


#XGboost on imbalanced dataset with hypertuning
# Defining parameters
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[53]:


# Stratified K Fold
cross_val = StratifiedKFold(n_splits=5)
index_iterator = cross_val.split(X_train, y_train)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[54]:


# Passing the best parameteres based on Randomized Search CV
model_xgb_tuned = XGBClassifier(min_child_weight= 5,
        gamma= 1.5,
        subsample= 1.0,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[55]:


# Fitting the model on Train data and Predicting on Test data
model_xgb_tuned.fit(X_train,y_train)
y_predicted = model_xgb_tuned.predict(X_test)


# In[56]:


# Evaluation metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[57]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.95
# 2.Recall    0.73
# 3.Acurracy  0.91
# 4.F1 Score  0.82
# 5.ROC AUC   0.86

# # Balanced dataset

# In[58]:


# Balancing the dataset with different technique
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# Resample training data
ros = RandomOverSampler()
smote = SMOTE(random_state=5)
adasyn = ADASYN(random_state=5)

X_train_ros, y_train_ros = ros.fit_sample(X_train,y_train)
X_train_smote, y_train_smote = smote.fit_sample(X_train,y_train)
X_train_adasyn, y_train_adasyn =adasyn.fit_sample(X_train,y_train)


# # Model 7

# # Running logistic model on balanced dataset with randomover sample

# In[59]:


# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_ros, y_train_ros)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[60]:


# Evaluation Metrics
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[61]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.06
# 2.Recall    0.91
# 3.Acurracy  0.98
# 4.F1 Score  0.12
# 5.ROC AUC   0.94

# # Model 8

# # Running logistic algo. on dataset balanced via SMOTE technique

# In[62]:


# logistic with SMOTE balanced
# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_smote, y_train_smote)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[63]:


# Evaluation Metrics
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# In[64]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.06
# 2.Recall    0.91
# 3.Acurracy  0.97
# 4.F1 Score  0.11
# 5.ROC AUC   0.93

# # Model 9

# # Logistic Regression on ADASYN Balanced Data

# In[65]:


# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_adasyn, y_train_adasyn)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[66]:


# Evaluation Metrics
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# #Conclusion
# 1.Precision : 0.02
# 2.Recall : 0.91
# 3.F1-score : 0.04
# 4.Accuracy : 0.91
# 5.ROC AUC : 0.91

# # Model 10

# # Running random forest algo. on balanced dataset via Random over sampler technique

# In[67]:


# Random forest on ROS balanced dataset
# Insantiate Model
model_rfc = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud
                               criterion='entropy',
                               max_depth=10, # Change depth of model
                               min_samples_leaf=10, # Change the number of samples in leaf nodes
                               n_estimators=20, # Change the number of trees to use
                               n_jobs=-1, 
                               random_state=5)


# In[68]:


# Fit the model on train data and predict on test data 
model_rfc.fit(X_train_ros,y_train_ros)
y_predicted = model_rfc.predict(X_test)


# In[69]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))

# Conclusion
1.Precision 0.43
2.Recall    0.78
3.Acurracy  0.99
4.F1 Score  0.55
5.ROC AUC   0.88
# In[70]:


#RF on smote dataset
# Fit the model on train data and predict on test data 
model_rfc.fit(X_train_smote,y_train_smote)
y_predicted = model_rfc.predict(X_test)


# In[71]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# # Conclusion
# 1.Precision 0.06
# 2.Recall    0.86
# 3.Acurracy  0.97
# 4.F1 Score  0.12
# 5.ROC AUC   0.92

# # Model 11

# # Tune random forest model on balanced dataset

# In[72]:


#Hyper Tuning model Random Forest on ROS Balanced Data
params = { 
    'n_estimators': [200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[73]:


cross_val = StratifiedKFold(n_splits=3)
index_iterator = cross_val.split(X_train_ros, y_train_ros)
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[74]:


# Insanitiate Model on best params
model_rfc_tuned = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, 
                               criterion='entropy',
                               max_depth=8, 
                               max_features='auto',
                               n_estimators=200,
                               n_jobs=-1)


# In[75]:


#Fit the model on train data and predict the model on test data
model_rfc_tuned.fit(X_train_ros,y_train_ros)
y_predicted = model_rfc_tuned.predict(X_test)


# In[76]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# # Conclusion
# 1.Precision 0.22
# 2.Recall    0.81
# 3.Acurracy  0.99
# 4.F1 Score  0.35
# 5.ROC AUC   0.90

# # Model 12

# # Running XGBoost algo. on balanced dataset

# In[77]:


# XGBOOST on balanced datset
model_xgb_ros = XGBClassifier()
#Fit the model on train data and predict the model on test data
model_xgb_ros.fit(X_train_ros,y_train_ros)
y_predicted = model_xgb_ros.predict(X_test)


# In[78]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# # Conclusion
# 1.Precision 0.92
# 2.Recall    0.75
# 3.Acurracy  0.99
# 4.F1 Score  0.83
# 5.ROC AUC   0.87

# # Model 13

# # Hypertune  XGboost model on balanced dataset

# In[79]:


# hypertuning on xgboost
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[80]:


cross_val = StratifiedKFold(n_splits=4)
index_iterator = cross_val.split(X_train_ros, y_train_ros)

xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 
xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[81]:


model_xgb_tuned_ros = XGBClassifier(min_child_weight= 5,
        gamma= 1.5,
        subsample= 1.0,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[82]:


#Fit the model on train data and predict the model on test data
model_xgb_tuned_ros.fit(X_train_ros,y_train_ros)
y_predicted = model_xgb_tuned_ros.predict(X_test)


# In[83]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# # Conclusion
# 1.Precision 0.93
# 2.Recall    0.76
# 3.Acurracy  0.99
# 4.F1 Score  0.84
# 5.ROC AUC   0.88

# # Model 14

# # XGBoost model on balanced dataset via SMOTE technique

# In[84]:


#XGBOOST on SMOTE balanced data
model_xgb_smote = XGBClassifier()

#Fit the model on train data and predict the model on test data
model_xgb_smote.fit(X_train_smote,y_train_smote)
y_predicted = model_xgb_smote.predict(X_test)


# In[85]:


# Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# # Conclusion
# 1.Precision 0.82
# 2.Recall    0.76
# 3.Acurracy  0.99
# 4.F1 Score  0.79
# 5.ROC AUC   0.88

# # Model 15

# # Hypertune XGBoost model on balanced dataset

# In[93]:


# XGBOOST on SMOTE balanced data
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10,15],
        'gamma': [0.5, 1, 1.5, 2, 5,8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0,1.2],
        'max_depth': [3, 4, 5,6,7]
        }


# In[94]:


cross_val = StratifiedKFold(n_splits=5)
index_iterator = cross_val.split(X_train_smote, y_train_smote)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =40 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[95]:


model_xgb_tuned_smote = XGBClassifier(min_child_weight= 10,
        gamma= 1.5,
        subsample= 0.6,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[96]:


#Fit the model on train data and predict the model on test data
model_xgb_tuned_smote.fit(X_train_smote,y_train_smote)
y_predicted = model_xgb_tuned.predict(X_test)


# In[97]:


#Evaluation Metrices
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))


# Here we will observe the distribution of our classes
# Conclusion
1.Precision 0.95
2.Recall    0.73
3.Acurracy  0.99
4.F1 Score  0.82
5.ROC AUC   0.86
# # Final Model Selection

# In[100]:


#Predicting on the test data using the best model
y_predicted = model_xgb_smote.predict(X_test)


# In[86]:


# Create true and false positive rates
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)


# In[87]:


# Printing Evaluation Metrices
print('Classification report for XGBoost Smote:\n', classification_report(y_test, y_predicted))
print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))
print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
threshold = thresholds[np.argmax(tpr-fpr)]
print("Threshold:",threshold)


# In[88]:


# Plotting the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# # Conclusion
# 1.Precision 0.82
# 2.Recall    0.76
# 3.Acurracy  0.99
# 4.F1 Score  0.79
# 5.ROC AUC   0.88

# # Important Feature 

# In[89]:


target = 'Class'
pca_comp = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',       'Amount', 'Time']


# In[90]:


tmp = pd.DataFrame({'Feature': pca_comp, 'Feature importance': model_xgb_smote.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()


# # Conclusion
#We found out that PCA converted variables like V14, V4 and V12 are able to explain the maximum variance and hence we can #target these variables to detect a fraud.# Final Statement
#In above scenario Accuracy was not a concerning Evaluation criteria and we focussed more on Recall and AUC.
#We finally able to build a proper logistic model and predicted on test data and the results were satisfying.
#We were also able to figure out the variables which will be important in detecting any fraud transactions.
# In[91]:


import pickle
pickle.dump(model_xgb_smote,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:




