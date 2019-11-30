# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:04:15 2019

@author: ZuBaiR
"""

from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import nltk
nltk.download('punkt')
dataset=pd.read_csv('dataset.csv')
print(dataset.head())
df=dataset['title'].values
print(df)
newVec=[nltk.word_tokenize(title) for title in df]
print(newVec)
model=Word2Vec(newVec,min_count=1,size=32)
print(model.most_similar('man'))
vec=model.wv['King']-model.wv['man']+model.wv['woman']
print(model.wv['man'])