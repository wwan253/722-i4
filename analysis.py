# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
import math
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix
from pandas_profiling import ProfileReport
from sklearn.cluster import KMeans
from datetime import datetime, timedelta,date
import plotly.offline as pyoff
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff

df = pd.read_csv("C:/Users/GGPC/Desktop/722/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.columns)
print(df.shape)

print(df.isnull().sum())

df["MonthlyCharges"] = df["MonthlyCharges"].apply(float)
df["TotalCharges"] = df["TotalCharges"].replace(' ',None) # to ensure that TotalCharges can be successfully converted to a float column
df["TotalCharges"] = df["TotalCharges"].apply(float)

plt.hist(df['Churn'])


plt.figure()
sns.countplot(x="gender", data=df)
plt.figure()
sns.countplot(x="gender", hue="Churn", data=df)

plt.figure()
sns.countplot(x="Dependents", data=df)
plt.figure()
sns.countplot(x="Dependents", hue="Churn", data=df)

plt.figure()
sns.countplot(x="tenure", data=df)
plt.figure()
sns.countplot(x="tenure", hue="Churn", data=df)

plt.figure()
sns.scatterplot(x="MonthlyCharges", y="tenure", hue="Churn",
                     data=df)

plt.figure()
sns.scatterplot(x="TotalCharges", y="tenure", hue="Churn",
                     data=df)

outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(df['tenure'])
print(outlier_datapoints)
outlier_datapoints = detect_outlier(df['TotalCharges'])
print(outlier_datapoints)
outlier_datapoints = detect_outlier(df['MonthlyCharges'])
print(outlier_datapoints)

# df['TotalCharges'].fillna(899)
# =============================================================================
# df.fillna(df.mean())
# 
# #use the upper value
# df.fillna(method='pad')
# #use the next value
# df.fillna(method='bfill')
# 
# df.interpolate()
# =============================================================================


# =============================================================================
# =============================================================================
#  def cap(x,quantile=[0.01,0.99]):
#      Q01,Q99 = x.quantile(quantile).values.tolist()
#      if Q01>x.min():
#          x = x.copy()
#          x.loc[x<Q01] = Q01
#      if Q99<x.max():
#          x = x.copy()
#          x.loc[x>Q99] = Q99
#          
#      return(x)
#  
#  new_df = df.apply(cap,quantile=[0.01,0.99])
# =============================================================================



plt.figure()
df.dropna(inplace=True)
df = df.iloc[:,1:]
df['Churn'].replace(to_replace='Yes',value=1,inplace=True)
df['Churn'].replace(to_replace='No',value=0,inplace=True)
df = pd.get_dummies(df)

y = df['Churn'].values
X = df.drop(columns=['Churn'])

from sklearn.ensemble import RandomForestClassifier as rm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model_rf = rm(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)
importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')

plt.figure()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))
plt.figure()
print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))

from imblearn.over_sampling import SMOTE
shuffled_df = df.sample(frac=1,random_state=4)
fraud_df = shuffled_df.loc[shuffled_df['Churn']==1]
non_fraud_df = shuffled_df.loc[shuffled_df['Churn']==0].sample(n=1869,random_state=42)
normalized_df = pd.concat([fraud_df,non_fraud_df])
plt.figure()
plt.hist(normalized_df['Churn'])
# =============================================================================
# sm = SMOTE(sampling_strategy = 'minority',random_state = 7)
# oversample_trainX,oversample_trainY = sm.fit_sample(df.drop('Churn',axis=1), df['Churn'])
# oversample_train = pd.concat([pd.DataFrame(oversample_trainY)],pd.DataFrame(oversample_trainX),axis=1)
# oversample_train.columns = normalized_df.columns
# =============================================================================

# =============================================================================
# df_combine = pd.concat([df,normalized_df],ignore_index=True)
# =============================================================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

im_features = SelectKBest(score_func = f_classif, k=45)
im = im_features.fit(X,y)

scores = pd.DataFrame(im.scores_)
column = pd.DataFrame(X.columns)

feature_score = pd.concat([column,scores],axis=1)
feature_score.columns = ["Variable","Score"]
print(feature_score.nlargest(45, "Score"))
print(type(feature_score))

drop_list = []
for i in range(0,len(feature_score)):
    if feature_score['Score'].iloc[i]<90 :
        drop_list.append(feature_score["Variable"].iloc[i])
print(drop_list)

for i in drop_list:
    df=df.drop(i,axis=1)

print(len(df.columns))

new_tenure = []
for i in df["tenure"]:
    if i <=12:
        new_tenure.append(0)
    elif i <=48:
        new_tenure.append(1)
    else:
        new_tenure.append(2)

plt.hist(df['tenure'])
df["tenure"] = new_tenure
plt.figure()
plt.hist(df["tenure"])

from sklearn import tree
from sklearn.metrics import accuracy_score

y = df['Churn'].values
X = df.drop(columns=['Churn'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

clf1 = tree.DecisionTreeClassifier(criterion='entropy',
                                   random_state=30,splitter='random',
                                   max_depth=3,min_samples_leaf=5,min_samples_split=70)

clf2 = tree.DecisionTreeClassifier(criterion='entropy',
                                   random_state=30,splitter='random',
                                   max_depth=11,min_samples_leaf=10,min_samples_split=70) 

clf3 = tree.DecisionTreeClassifier(criterion='gini',
                                   random_state=30,splitter='random',
                                   max_depth=3,min_samples_leaf=5,min_samples_split=70) 

clf4 = tree.DecisionTreeClassifier(criterion='gini',
                                   random_state=30,splitter='random',
                                   max_depth=3,min_samples_leaf=10,min_samples_split=70) 
clf5 = tree.DecisionTreeClassifier(criterion='gini',
                                   random_state=30,splitter='random',
                                   max_depth=10,min_samples_leaf=10,min_samples_split=70) 
clf6 = tree.DecisionTreeClassifier(criterion='gini',
                                   random_state=30,splitter='random',
                                   max_depth=15,min_samples_leaf=10,min_samples_split=70) 

clf1.fit(X_train,y_train)

clf1_predict = clf1.predict(X_test)

acc1 = accuracy_score(y_test, clf1_predict)

print("Accuracy for clf1: " +str(acc1*100) + "%")

clf2.fit(X_train,y_train)

clf2_predict = clf2.predict(X_test)

acc2 = accuracy_score(y_test, clf2_predict)

print("Accuracy for clf2: " +str(acc2*100) + "%")

clf3.fit(X_train,y_train)

clf3_predict = clf3.predict(X_test)

acc3 = accuracy_score(y_test, clf3_predict)

print("Accuracy for clf3: " +str(acc3*100) + "%")

clf4.fit(X_train,y_train)

clf4_predict = clf4.predict(X_test)

acc4 = accuracy_score(y_test, clf4_predict)

print("Accuracy for clf4: " +str(acc4*100) + "%")

clf5.fit(X_train,y_train)

clf5_predict = clf5.predict(X_test)

acc5 = accuracy_score(y_test, clf5_predict)

print("Accuracy for clf5: " +str(acc5*100) + "%")

clf6.fit(X_train,y_train)

clf6_predict = clf6.predict(X_test)

acc6 = accuracy_score(y_test, clf6_predict)

print("Accuracy for clf6: " +str(acc6*100) + "%")

test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                   random_state=30,splitter='random',
                                   max_depth=i+1,min_samples_leaf=10,min_samples_split=70) 
    clf = clf.fit(X_train,y_train)
    clf_predict = clf.predict(X_test)
    acc = accuracy_score(y_test, clf_predict)
    test.append(acc)
    
plt.figure()

plt.plot(range(1,21),test,color="red",label="max_depth")
plt.legend()
plt.show()



feature_name = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No internet service',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No internet service',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
       'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check']

import graphviz

clf2=clf2.fit(X_train,y_train)

dot_data = tree.export_graphviz(clf2,feature_names=feature_name,
                                class_names=["NO_Churn","Churn"],
                                filled=True,
                                rounded=True)

graph = graphviz.Source(dot_data)
graph.view()

from sklearn.neural_network import MLPClassifier

mlp1 = MLPClassifier(hidden_layer_sizes=(5,),activation='logistic',max_iter=100)
mlp2 = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',max_iter=100)
mlp3 = MLPClassifier(hidden_layer_sizes=(5,),activation='relu',max_iter=100)
mlp4 = MLPClassifier(hidden_layer_sizes=(10,),activation='relu',max_iter=100)

mlp1.fit(X_train,y_train)

mlp1_predict = mlp1.predict(X_test)

acc = accuracy_score(y_test,mlp1_predict)

print("Accuracy for mlp1: " +str(acc*100) + "%")

mlp2.fit(X_train,y_train)

mlp2_predict = mlp2.predict(X_test)

acc = accuracy_score(y_test,mlp2_predict)

print("Accuracy for mlp2: " +str(acc*100) + "%")

mlp3.fit(X_train,y_train)

mlp3_predict = mlp3.predict(X_test)

acc = accuracy_score(y_test,mlp3_predict)

print("Accuracy for mlp3: " +str(acc*100) + "%")

mlp4.fit(X_train,y_train)

mlp4_predict = mlp4.predict(X_test)

acc = accuracy_score(y_test,mlp4_predict)

print("Accuracy for mlp4: " +str(acc*100) + "%")


from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(criterion='entropy',max_depth=10,max_features=20,n_estimators=50)
rf2 = RandomForestClassifier(criterion='entropy',max_depth=10,max_features=20,n_estimators=150)
rf3 = RandomForestClassifier(criterion='gini',max_depth=10,max_features=20,n_estimators=50)
rf4 = RandomForestClassifier(criterion='gini',max_depth=10,max_features=20,n_estimators=150)

rf1.fit(X_train,y_train)

rf_predict = rf1.predict(X_test)

acc = accuracy_score(y_test,rf_predict)

print("Accuracy for rf1: " +str(acc*100) + "%")   

rf2.fit(X_train,y_train)

rf_predict = rf2.predict(X_test)

acc = accuracy_score(y_test,rf_predict)

print("Accuracy for rf2: " +str(acc*100) + "%") 

rf3.fit(X_train,y_train)

rf_predict = rf3.predict(X_test)

acc = accuracy_score(y_test,rf_predict)

print("Accuracy for rf3: " +str(acc*100) + "%") 

rf3.fit(X_train,y_train)

rf_predict = rf3.predict(X_test)

acc = accuracy_score(y_test,rf_predict)

print("Accuracy for rf3: " +str(acc*100) + "%") 

plt.figure()
importances = rf1.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')

from sklearn.metrics import confusion_matrix

y_pred = clf2.predict(X_test)

cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

y_pred = mlp2.predict(X_test)

cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

y_pred = rf2.predict(X_test)

cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

from sklearn.ensemble import VotingClassifier
ensemble=VotingClassifier(estimators=[('Decision Tree', clf2), ('Random Forest', rf2)], 
                       voting='soft', weights=[2,1]).fit(X_train,y_train)
print('The accuracy for DecisionTree and Random Forest is:',ensemble.score(X_test,y_test))

from sklearn.ensemble import VotingClassifier
ensemble=VotingClassifier(estimators=[('Neural network', mlp2), ('Random Forest', rf2)], 
                       voting='soft', weights=[2,1]).fit(X_train,y_train)
print('The accuracy for Neural network and Random Forest is:',ensemble.score(X_test,y_test))