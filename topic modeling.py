# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 21:02:48 2019

@author: ZuBaiR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data ='blog_nature.xlsx'
dataset = pd.read_excel(data)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(dataset['article_content'])

print(dtm)

from sklearn.decomposition import NMF
nmf_model = NMF(n_components=20,random_state=42)

nmf_model.fit(dtm)

for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
dataset.head()

topic_results = nmf_model.transform(dtm)

topic_results.argmax(axis=1)

dataset['article_content'] = topic_results.argmax(axis=1)

dataset.head(10)