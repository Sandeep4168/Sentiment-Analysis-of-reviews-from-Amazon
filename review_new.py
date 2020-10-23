# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:09:52 2020

@author: user
"""

# # To get the training data from the SST( Stanford Sentiment Tree),pytreebank was loaded 
# import pytreebank
# dataset = pytreebank.load_sst("../review_amazon/spiders") # loading SST dataset 


# #seperating the train, test, dev datasets
# for category in ['train', 'test', 'dev']:
#     with open('../data/sst/sst_{}.txt'.format(category), 'w') as outfile:
#         for item in dataset[category]:
#             outfile.write("__label__{}\t{}\n".format(
#                 item.to_labeled_lines()[0][0] + 1,
#                 item.to_labeled_lines()[0][1]
#             ))



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup

import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df_train = pd.read_csv('sst_train.txt', sep='\t', header=None,
                   names=['sentiment', 'review'],encoding='latin-1'
                  )
df_train['sentiment'] = df_train['sentiment'].str.replace('__label__', '')
df_train['sentiment'] = df_train['sentiment'].astype(int).astype('category')
df_train=df_train.reindex(columns=['review','sentiment'])
print(df_train.dtypes)
print(df_train.head())


df_test = pd.read_csv('sst_test.txt', sep='\t', header=None,
                   names=['sentiment', 'review'],encoding='latin-1'
                  )
df_test['sentiment'] = df_test['sentiment'].str.replace('__label__', '')
df_test['sentiment'] = df_test['sentiment'].astype(int).astype('category')
df_test=df_test.reindex(columns=['review','sentiment'])
print(df_test.dtypes)
print(df_test.head())                   
                     
df_review=df_train.append(df_test)                    
                     
                     
# CLeaning the data from SST

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub("\d+", "", text)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


#Apply function on review column
df_review['review']=df_review['review'].apply(clean_text)                  

X=df_review['review']
y=df_review['sentiment']

cv=CountVectorizer()
X=cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

def model(model):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
    print('Classification Report:\n',classification_report(y_test,y_pred))
    
lr=LogisticRegression(solver='liblinear',multi_class='auto')
svm=SGDClassifier()
mnb=MultinomialNB()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()

models=[lr,svm,mnb,dt,rf]

for i in models:
    model(i)


# for our analysis we will be using the Logistic Regression

# Predicting the sentiment of Mobile reviews

#Cleaning the reviews

df_mobile=pd.read_csv('amazon.csv',encoding='latin-1')

oneplus=df_mobile.review[:99].apply(clean_text)
oppo_f11=df_mobile.review[99:198].apply(clean_text)
redmi_note8=df_mobile.review[198:296].apply(clean_text)
redmi_y2=df_mobile.review[296:395].apply(clean_text)
redmi_y3=df_mobile.review[395:494].apply(clean_text)
samsung_m21=df_mobile.review[494:594].apply(clean_text)
samsung_m31=df_mobile.review[594:693].apply(clean_text)
samsung_m31s=df_mobile.review[693:792].apply(clean_text)
samsung_m51=df_mobile.review[792:898].apply(clean_text)
vivo_u10=df_mobile.review[898:997].apply(clean_text)


mobiles=[('oneplus',oneplus),('oppo_f11',oppo_f11),('redmi_note8',redmi_note8),('redmi_y2',redmi_y2)
         ,('redmi_y3',redmi_y3),('samsung_m21',samsung_m21),('smasung_m31',samsung_m31),('samsung_m31s',samsung_m31s)
         ,('samsung_m51',samsung_m51),('vivo_u10',vivo_u10)]


    
oneplus=cv.transform(oneplus)
lr.fit(X_train,y_train)
pred=lr.predict(oneplus)
oneplus_df=pd.DataFrame(pred)
    
df_rev=pd.DataFrame()
for name,raw in mobiles:
    raw=cv.transform(raw)
    lr.fit(X_train,y_train)
    pred=lr.predict(raw)
    df=pd.DataFrame(pred)
    df.to_csv(name+'.csv')

# EDA of Sentiment analysis of reviews for all the mobiles
review_df=pd.read_csv('amazon_mobile_reviews.csv',encoding='latin-1')
review_df.prediction=review_df.prediction.map({1:'very negative',2:'negative',3:'neutral',4:'positive',5:'very positive'})
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30,20))
sns.set_context("paper", font_scale=1.5)   
sns.countplot(x='prediction',data=review_df,hue='name',order=['very negative','negative','neutral','positive','very positive'])
plt.show()

# from the Graph it is sure that most of the reviews were predicted neutral
review_df.prediction.value_counts().plot(kind='bar')
plt.show()

                     
                     
                     
                     
                     
              