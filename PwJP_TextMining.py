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


#%%


list_id_clothes = []
list_age = [] 
list_titles_of_reviews = []
list_only_reviews = []
list_rating = []
list_if_recommended = []
list_count_of_positive_feedback = []
list_division_name = []
list_department_name = []
list_class_name = []


for row in list_reviews:
    list_id_clothes.append(row[1])
    list_age.append(row[2])
    list_titles_of_reviews.append(row[3])
    list_only_reviews.append(row[4])
    list_rating.append(row[5])
    list_if_recommended.append(row[6])
    list_count_of_positive_feedback.append(row[7])
    list_division_name.append(row[8])
    list_department_name.append(row[9])
    list_class_name.append(row[10])
   