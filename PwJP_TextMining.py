# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:12:27 2021

@author: aroza
"""

# kod
import pandas
pandas.read_csv
df = pandas.read_csv('women_clothing_review.csv')
list_reviews = df.values.tolist()

list_reviews_text=df['review_text']