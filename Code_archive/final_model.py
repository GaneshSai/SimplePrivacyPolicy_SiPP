

import os
import pandas as pd
from bs4 import BeautifulSoup
import codecs
import nltk
import spacy
model = spacy.load('en_core_web_lg')
from nltk.tokenize import sent_tokenize
pd.set_option('display.max_columns', 500)
import ast
import bs4
from operator import itemgetter


os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115\annotations')
files = os.listdir()

def extract_text_annotation(row):
 d = ast.literal_eval(row['Attribute Value Pairs'])
 text = []
 for key in d.keys():
  try :
   text.append(d[key]['selectedText'])
  except :
   pass

 return text



final_annotations = pd.DataFrame(columns=['Category Name', 'Segment ID', 'extracted text','policy_file'], dtype= 'object')


original_annotations = pd.DataFrame(columns= ['Annotation ID', 'Batch ID', 'Annotator ID', 'Policy ID', 'Segment ID', 'Category Name', 'Attribute Value Pairs', 'Policy URL', 'Date','extracted text','policy_file'], dtype= 'object')



for i in files:
 print('doing ',i)
 sample = pd.read_csv(i, names = ['Annotation ID','Batch ID','Annotator ID','Policy ID','Segment ID','Category Name','Attribute Value Pairs','Policy URL',	'Date'])
 sample['extracted text'] = sample.apply(lambda x: extract_text_annotation(x), axis=1)
 #df = sample.groupby(['Category Name', 'Segment ID'])['extracted text'].agg(sum).reset_index()
 #df['extracted text'] = df['extracted text'].apply(lambda x: max(x, key=len) if len(x)>0 else ' ')
 sample['policy_file'] = i
 original_annotations = pd.concat([original_annotations,sample])
 print(i,'is done')

final_annotations.to_csv('final_annotated_opp115.csv')

original_annotations.to_csv('original_annotations.csv')

# sample attribute / annotation




# converting html files into text

os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115\original_policies')
files = os.listdir()

for i in files :
 file = codecs.open(i,"r", "utf-8")
 txt = file.read()
 soup = bs4.BeautifulSoup(txt, 'html.parser')
 text = soup.getText()
 with open(str(i)+'.txt','w') as f:
  f.write(text)

""" TFidf / distance based supervised model"""

import nltk
from nltk.tokenize import  word_tokenize
import spacy
from nltk.probability import FreqDist
model = spacy.load('en_core_web_lg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split




os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115')

data = pd.read_csv('final_annotated_opp115.csv')
data['extracted text'] = data['extracted text'].str.lower()
data['tokens'] = data['extracted text'].apply(lambda x : word_tokenize(x))

# nouns, adjectives, adverbs, verbs, pronouns

def extract_pos(row):
 txt = model(' '.join(row['extracted text']))
 nouns = [ent.text for ent in txt if ent.pos_ == 'NOUN']
 nouns = FreqDist(nouns).most_common(300)
 verbs = [ent.text for ent in txt if ent.pos_ == 'VERB']
 verbs= FreqDist(verbs).most_common(300)
 adjs = [ent.text for ent in txt if ent.pos_ == 'ADJ']
 adjs = FreqDist(adjs).most_common(300)
 adv = [ent.text for ent in txt if ent.pos_ == 'ADV']
 adv = FreqDist(adv).most_common(300)
 pron = [ent.text for ent in txt if ent.pos_ == 'PRON']
 pron = FreqDist(pron).most_common(300)
 return pd.Series([nouns,verbs,adjs,adv,pron])


data[['nouns','verbs','adjectives','adverbs','pronouns']] = data.apply(lambda x : extract_pos(x), axis= 1)


data['Category Name'].value_counts()

df = data.groupby(['Category Name'])['nouns','verbs','adjectives','adverbs','pronouns'].agg(sum)


# getting unique items



FreqDist(df.loc['User Access, Edit and Deletion']['nouns']).most_common(50)
FreqDist(df.loc['User Access, Edit and Deletion']['verbs']).most_common(50)
FreqDist(df.loc['User Access, Edit and Deletion']['adjectives']).most_common(50)
FreqDist(df.loc['User Access, Edit and Deletion']['adverbs']).most_common(50)
FreqDist(df.loc['User Access, Edit and Deletion']['pronouns']).most_common(50)



# getting vocabulary for each category

# catgeory wise all extracted text
df = original_annotations.groupby(['Category Name'])['extracted text'].agg('sum').reset_index()

df[['nouns','verbs','adjectives','adverbs','pronouns']] = df.apply(lambda x : extract_pos(x), axis= 1)

df.drop('extracted text', inplace = True , axis = 1)
df.to_csv('frequent_words.csv')








# data retention




"""  
'Data Retention', 'Data Security', 'Do Not Track',
       'First Party Collection/Use', 'International and Specific Audiences',
       'Other', 'Policy Change', 'Third Party Sharing/Collection',
       'User Access, Edit and Deletion', 'User Choice/Control'
   """

l = FreqDist(df.loc['User Choice/Control']['verbs']).most_common(40)

# building vocabulary for TF IDf from most common words

categories = ['Data Retention', 'Data Security', 'Do Not Track','First Party Collection/Use', 'International and Specific Audiences', 'Other', 'Policy Change', 'Third Party Sharing/Collection','User Access, Edit and Deletion', 'User Choice/Control']

pos = ['nouns', 'verbs', 'adjectives', 'adverbs', 'pronouns']
vocab= []

for i in categories:
 for j in pos :
  l_common = FreqDist(df.loc[i][j]).most_common(60)
  vocab.extend([k[0] for k in l_common])

vocab = list(set(vocab))

# simplifying the corpus before training

# delete others, do not track, international and specific audience
# user choice / control and user access edit merge

data = data[data['Category Name'].isin(['Data Retention', 'Data Security', 'First Party Collection/Use', 'Policy Change', 'Third Party Sharing/Collection','User Access, Edit and Deletion', 'User Choice/Control'])]

data['Category Name'] = data['Category Name'].replace(to_replace= ['User Access, Edit and Deletion','User Choice/Control'], value = ['user rights','user rights'])

train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(data['extracted text'], data['Category Name'], test_size=0.2)


tv = TfidfVectorizer(use_idf=True,max_features= 1500)
train_features = tv.fit_transform(train_corpus)
test_features = tv.transform(test_corpus)


# modelling

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

model = LogisticRegression()
model= DecisionTreeClassifier(criterion= 'gini', max_depth= 20,max_features= 50)
model = MultinomialNB(alpha= 0.5)



model.fit(train_features,train_label_names)
preds = model.predict(train_features)
#print(classification_report(train_label_names,preds))
test_preds = model.predict(test_features)
print(classification_report(test_label_names,test_preds))


## unsupervised model


def sim_score(d,closest_nouns, closest_verbs,closest_adjectives):
 score = 0
 n_score, v_score, p_score = 0, 0, 0
 n_count, v_count, p_count = 0, 0, 0

 #d = list(set(d))
 for i in closest_nouns:
  for j in d:
   if i.similarity(j) > 0.4 and j.pos_ == 'NOUN':
    n_score = n_score + i.similarity(j)
    n_count = n_count + 1
 for i in closest_verbs:
  for j in d:
   if i.similarity(j) > 0.4 and j.pos_ == 'VERB':
    v_score = v_score + i.similarity(j)
    v_count = v_count + 1

 for i in closest_adjectives:
  for j in d:
   if i.similarity(j) > 0.4 and j.pos_ == 'ADJ':
    p_score = p_score + i.similarity(j)
    p_count = p_count + 1

 if n_score > 0 and v_score > 0:
  score = (n_score + v_score + p_score) / (n_count + p_count + v_count)

 return score

dict_words = pd.read_csv('privacy words.csv')
dict_words.set_index('categories',inplace= True)


def get_scores(row):
 sent = model(row['extracted text'])
 cats = ['User rights','Data collection','Data Security']
 scores =[]
 for i in cats :
  closest_nouns = model(dict_words.loc[i]['Nouns'])
  closest_verbs = model(dict_words.loc[i]['Verbs'])
  closest_adjectives = model(dict_words.loc[i]['Adjectives'])
  scores.append((i,sim_score(sent,closest_nouns,closest_verbs,closest_adjectives)))
  scores = sorted(scores,key=lambda x: x[1], reverse=True)

 return scores[0][0]

data['predicted categories'] = data.apply(lambda x : get_scores(x), axis=1)


data2 = data[data['Category Name'].isin(['Data Security', 'First Party Collection/Use', 'Third Party Sharing/Collection','User Access, Edit and Deletion', 'User Choice/Control'])]

data2['Category Name'] = data2['Category Name'].replace(to_replace= ['User Access, Edit and Deletion','User Choice/Control','First Party Collection/Use', 'Third Party Sharing/Collection'], value = ['User rights','User rights','Data collection','Data collection'])

print(classification_report(data2['Category Name'],data2['predicted categories']))