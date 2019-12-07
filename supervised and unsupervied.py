# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:27:27 2019

@author: zubai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os
os.getcwd()
# importing dataset
df=pd.read_csv('postoperative.csv')
# dataset Overview, infomation and dimension checking
print(df.head())
print(df.info())
print(df.shape)
# Statistical Description of dataset
print(df.describe)
# Checking Missing Values
print(df.isna().sum())
df['COMFORT'].replace('?',np.nan, inplace= True)
df['decision ADM-DECS (discharge decision)'].replace('I',np.nan, inplace= True)
# visualize of missing value
sns.heatmap(df.isnull(), cbar=False)
# handling the missing values with forward method and checking again
df.fillna(method ='bfill', inplace=True) 
print(df.isna().sum())

# Exploratory dataset Analysis
sns.countplot(df['decision ADM-DECS (discharge decision)'])
df['decision ADM-DECS (discharge decision)'].value_counts()
sns.countplot(df["%L-CORE (patient's internal temperature in C)"])
df["%L-CORE (patient's internal temperature in C)"].value_counts()
sns.countplot(df["L-SURF (patient's surface temperature in C)"])
df["L-SURF (patient's surface temperature in C)"].value_counts()
sns.countplot(df["L-O2 (oxygen saturation in %)"])
df["L-O2 (oxygen saturation in %)"].value_counts()
sns.countplot(df["L-BP (last measurement of blood pressure)"])
df["L-BP (last measurement of blood pressure)"].value_counts()
sns.countplot(df["SURF-STBL (stability of patient's surface temperature)"])
df["SURF-STBL (stability of patient's surface temperature)"].value_counts()
sns.countplot(df['CORE-STBL'])
df['CORE-STBL'].value_counts()
sns.countplot(df['BP-STBL'])
df['BP-STBL'].value_counts()
sns.countplot(df['COMFORT'])
df['COMFORT'].value_counts()
data=df.groupby(["decision ADM-DECS (discharge decision)","%L-CORE (patient's internal temperature in C)"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['red','black', 'green', 'red', 'green'], grid=False)

data=df.groupby(["decision ADM-DECS (discharge decision)","L-SURF (patient's surface temperature in C)"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['red','black', 'green', 'red', 'green'], grid=False)
data=df.groupby(["decision ADM-DECS (discharge decision)","L-O2 (oxygen saturation in %)"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['green','blue', 'green', 'red', 'green'], grid=False)
data=df.groupby(["decision ADM-DECS (discharge decision)","L-BP (last measurement of blood pressure)"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['pink','red', 'green', 'red', 'green'], grid=False)
data=df.groupby(["decision ADM-DECS (discharge decision)","SURF-STBL (stability of patient's surface temperature)"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['Green','Red', 'green'], grid=False)
data=df.groupby(["decision ADM-DECS (discharge decision)","CORE-STBL"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['pink','Green', 'Red'], grid=False)
data=df.groupby(["decision ADM-DECS (discharge decision)","BP-STBL"]).COMFORT.count()
data.unstack().plot(kind='bar',stacked=True,  color=['pink','Green', 'Red'], grid=False)
# Maping Function
df["L-O2 (oxygen saturation in %)"] = df["L-O2 (oxygen saturation in %)"].map({'excellent': 1, 'good': 0})
df["SURF-STBL (stability of patient's surface temperature)"] = df["SURF-STBL (stability of patient's surface temperature)"].map({'stable': 1, 'unstable': 0})
df["decision ADM-DECS (discharge decision)"] = df["decision ADM-DECS (discharge decision)"].map({'A': 1, 'S': 0})
print(df.head())
# Dummies method for one hot Encoding
cont = pd.get_dummies(df["%L-CORE (patient's internal temperature in C)"],prefix="%L-CORE (patient's internal temperature in C)",drop_first=True)
df = pd.concat([df,cont],axis=1)
cont = pd.get_dummies(df["L-SURF (patient's surface temperature in C)"],prefix="L-SURF (patient's surface temperature in C)",drop_first=True)
df = pd.concat([df,cont],axis=1)
cont1 = pd.get_dummies(df["L-BP (last measurement of blood pressure)"],prefix="L-BP (last measurement of blood pressure)",drop_first=True)
df = pd.concat([df,cont1],axis=1)
cont2 = pd.get_dummies(df["L-BP (last measurement of blood pressure)"],prefix="L-BP (last measurement of blood pressure)",drop_first=True)
df = pd.concat([df,cont2],axis=1)
cont3 = pd.get_dummies(df["CORE-STBL"],prefix="CORE-STBL",drop_first=True)
df = pd.concat([df,cont3],axis=1)
cont4 = pd.get_dummies(df["COMFORT"],prefix="COMFORT",drop_first=True)
df = pd.concat([df,cont4],axis=1)
cont6 = pd.get_dummies(df["BP-STBL"],prefix="BP-STBL",drop_first=True)
df = pd.concat([df,cont6],axis=1)
print(df.head())
#Dropping the repeated variables¶
# We have created dummies for the below variables, so we can drop them
df = df.drop(["%L-CORE (patient's internal temperature in C)","L-SURF (patient's surface temperature in C)","L-BP (last measurement of blood pressure)","CORE-STBL","BP-STBL", "COMFORT"], 1)
print(df.head())
# Split of Dataset

X=df.drop(["decision ADM-DECS (discharge decision)"], axis=1)
y=df.iloc[:,2:3]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test)
print(y_test)
# Machine Learning Model
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
lr = classification_report(y_pred, y_test)
print(lr)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Multilayer Perceptron Model

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
#Fitting the training data to the network
classifier.fit(X_train, y_train)
#Predicting y for X_val
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
CLR = classification_report(y_pred, y_test)
print(CLR)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Unsupervised Learning
# K-Means Clustering
from sklearn.cluster import KMeans


km = KMeans(n_clusters = 2, random_state=90)
km.fit(df.drop('decision ADM-DECS (discharge decision)', axis=1))
km.cluster_centers_
def converter(prvt):
    if prvt == 'A':
        return 1
    else:
        return 0
df['Cluster']= df['decision ADM-DECS (discharge decision)'].apply(converter)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(df['Cluster'],km.labels_))
print(classification_report(df['Cluster'],km.labels_))
# DBScan
from sklearn.cluster import DBSCAN
X=df.iloc[:,8:9].values
dbscan=DBSCAN(eps=3,min_samples=2)
model=dbscan.fit(X)
labels=model.labels_
print(labels)
from sklearn import metrics
sample_cores=np.zeros_like(labels,dtype=bool)
sample_cores[dbscan.core_sample_indices_]=True
n_clusters=len(set(labels))- (0 if  -1 in labels else 0)
print(n_clusters)
#The data point are small  and does no come under DBSCAN.So the number of cluster are 1, so it do not predict anything as the number of features are categorical and on converting the data type, they just remain in only in one cluster.

# Hierarchical Agglomerative
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10, 3))
plt.style.use('seaborn-whitegrid')
X=df.drop(["decision ADM-DECS (discharge decision)"], axis=1)
y=df.iloc[:,2:3]
# Using scipy to generate dendrograms
Z = linkage(X, 'ward')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()
#Generating hierarchical clusters

k=2

Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
Hclustering.fit(X)

print(sm.accuracy_score(y, Hclustering.labels_))
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
Hclustering.fit(X)

print(sm.accuracy_score(y, Hclustering.labels_))
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
Hclustering.fit(X)

print(sm.accuracy_score(y, Hclustering.labels_))
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')
Hclustering.fit(X)

print(sm.accuracy_score(y, Hclustering.labels_))
