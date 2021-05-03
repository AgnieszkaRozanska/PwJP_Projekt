# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:12:27 2021

@author: aroza
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse.linalg as sp

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

import gensim
from gensim import corpora


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
   
 
def topic_Modelling(corpus, num_of_topics, dictionary, num_of_words):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_of_topics, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=num_of_words)
    for topic in topics:
        print(topic)       

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                               Code
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read CSV

pd.read_csv
df = pd.read_csv('women_clothing_review.csv')
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
import spacy
import re 
spacy.cli.download("en_core_web_sm")
df = df.dropna().reset_index(drop=True)

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
    
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['review_text'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
#%%
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
#%%
from gensim.models.phrases import Phrases, Phraser

sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]
#%%
import multiprocessing

from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() 
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
#%%
most_sim=w2v_model.wv.most_similar(positive=["dress"])
#%% Modelowanie tematyczne

#tworzenie słownika (dictionary) i korpusu (corpus) potrzebnego do modelowania tematycznego
from gensim import corpora
dictionary = corpora.Dictionary(sentences) 
corpus = [dictionary.doc2bow(text) for text in sentences]


#zapis słownika i korpusu
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#%% Wyszukanie 5 tematów z 4 najważniejszymi słowami kluczowymi i ich wagami 

topic_Modelling(corpus, 5, dictionary, 4)





