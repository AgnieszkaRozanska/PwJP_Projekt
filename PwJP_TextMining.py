# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:12:27 2021

@author: aroza
"""

# imports
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#to extract unique words
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
#download english stopwords
nltk.download('stopwords')
nltk.download('punkt')
#delete punctuation
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
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
 
    
def draw_wordCloud(feature, title, maxWords):
    text_data=df[feature]
    text = " ".join(t for t in text_data.dropna()) #generujemy jeden ciągły tekst, usuwamy brakujące wartosci
    stop_words = set(stopwords.words('english')) #wykorzystujemy stopwords (odrzucamy z analizy słowa ze stoplisty)
    
    # ustawiamy parametry dla chmury słów
    wordcloud = WordCloud(stopwords=stop_words, scale=4, max_font_size=50, max_words=maxWords,background_color="black", colormap="Blues").generate(text)
    fig = plt.figure(figsize=(20,20))
    plt.axis('off')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
 
def convert_to_lowercase(list_text):
    for idx,x in enumerate(list_text):
        for ele in x:
            if ele.isupper():
                x = x.replace(ele, ele.lower())    
        list_text[idx]  = x  
   
 
    

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


#%% Most frequent words in review_text
all_reviews = ','.join([str(i) for i in list_only_reviews])



stop_words = set(stopwords.words('english')) 
  
tokenizer = RegexpTokenizer(r'\w+')
word_tokens=tokenizer.tokenize(all_reviews)
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 

filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

counter = Counter(filtered_sentence)
most_occur = counter.most_common(10)
word=[]
word_counter=[]
for it in most_occur:
    word.append(it[0])
    word_counter.append(it[1])

pos = np.arange(len(word))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos)
ax.set_xticklabels(word)
ax.set_title('10 Most frequent words')

plt.bar(pos, word_counter, width=0.4, color=(0.3, 0.3, 0.3, 0.3), edgecolor='blue')
plt.show()

#%% Draw word cloud

draw_wordCloud('review_text', "WordCloud for the 200 most frequent words", 200)


#%% Word2Vec

# Conversion of lower case letters

list_reviews_text_cleaned = word_tokens.copy()
convert_to_lowercase(list_reviews_text_cleaned)


from gensim.models import Word2Vec

model = Word2Vec(sentences = list_reviews_text_cleaned, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")
model.train([["dress", "size"]], total_examples=1, epochs=100)

#Wyuczone wektory słów są przechowywane w KeyedVectorsinstancji jako model.wv

vector = model.wv['d'] # get numpy vector of a word
print("Get numpy vector of a word d")
print(vector)
sims = model.wv.most_similar('d', topn=10) # get other similar words 
print("Get other similar words to d")
print(sims) 


