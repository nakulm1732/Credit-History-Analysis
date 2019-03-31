# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:57:37 2019

@author: nakul
"""

import pandas as pd
import numpy as np
from AdvancedAnalytics import DecisionTree
df = pd.read_excel('CreditHistory_Clean.xlsx')
df.purpose.unique()
df.savings.unique()
df.head()
df.columns
df.shape
attribute_map = {
        'age':['I', (19, 120)],
        'amount':['I', (0, 20000)],
        'checking':['N',(1,2,3,4)],
        'coapp':['N',(1,2,3)],
        'depends':['B',(1,2)],
        'duration':['I',(1,72)],
        'employed':['N',(1,2,3,4,5)],
        'existcr':['N', (1,2,3,4)],
        'foreign':['B', (1,2)],
        'history':['N',(0,1,2,3,4)],
        'housing':['N',(1,2,3)],
        'installp':['N',(1,2,3,4)],
        'job':['N',(1,2,3,4)],
        'marital':['N', (1,2,3,4)],
        'other':['N',(1,2,3)],
        'property':['N',(1,2,3,4)],
        'resident':['N',(1,2,3,4)],
        'savings':['N',(1,2,3,4,5)],
        'good_bad':['B',('good','bad')],
        'purpose':['N',('3','6','2','0','1','9','4','5','X','8')],
        'telephon':['B',(1,2)]}

from AdvancedAnalytics import ReplaceImputeEncode
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
drop=False, display=True)
encoded_df = rie.fit_transform(df)
#Cross Validation
Y = np.asarray(encoded_df['good_bad'])
X = np.asarray(encoded_df.drop('good_bad', axis=1))
col = rie.col
len(col)
col.remove("good_bad")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
depths = [5,6,7,8,10,12,15,20,25]
for d in depths:
    classifier = DecisionTreeClassifier(max_depth=d, min_samples_split=5, min_samples_leaf=5)
    classifier.fit(X,Y)
    cvd=cross_val_score(classifier, X, Y, cv=10, scoring="f1")
    print("Tree Depth: ", d,\
          " F1 Avg:", cvd.mean(),\
          " Std:", cvd.std())
    DecisionTree.display_binary_metrics(classifier, X, Y)

#Best Results are obtained at a depth of 5

    
#Decision Trees after splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)
classifier.fit(X_train, Y_train)
DecisionTree.display_binary_metrics(classifier, X_train, Y_train)
DecisionTree.display_binary_metrics(classifier, X_test, Y_test)
DecisionTree.display_importance(classifier, col)

classes = ['1','0']
dot_data = export_graphviz(classifier, filled=True, rounded=True, class_names=classes, feature_names=col, out_file= None )
from pydotplus.graphviz import graph_from_dot_data 
graph = graph_from_dot_data(dot_data)
import graphviz
graph_pdf = graphviz.Source(dot_data)
graph_pdf.view()

