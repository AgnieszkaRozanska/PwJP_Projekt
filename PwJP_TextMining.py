# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:12:27 2021

@author: aroza
"""

# imports
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Functions

def draw_plot(feature, title, df,  size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set1')
    g.set_title(title)
    plt.xticks(rotation=90, size=8)
    plt.xlabel(' ')
    plt.ylabel('count')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height,'{:1}'.format(height),ha="center") 
    plt.show()      
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                               Code
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read CSV

pandas.read_csv
df = pandas.read_csv('women_clothing_review.csv')
list_reviews = df.values.tolist()

list_reviews_text=df['review_text']


#%%
# Extracting data from the list

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
   
    
#%%
# Drawing graphs of distributions of variables
draw_plot("age", "Customer age distribution", df,4)
draw_plot("division_name", "The distribution of divisions name", df,4)
draw_plot("department_name", "The distribution of departments name", df,4)
draw_plot("class_name", "The distribution of classes name", df,4)









   
    
    