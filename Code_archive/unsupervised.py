

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
import gensim
from gensim.models.word2vec import Word2Vec
from gensim import downloader
w2v = downloader.load('word2vec-google-news-300')
import itertools
import string


""" Reading original files with segments into one single file"""

os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115\original policies txt')
os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115')

files = os.listdir()

all_policy_text = pd.DataFrame(columns=['Segment ID', 'extracted text','policy_file'], dtype= 'object')

for i in files:
 print('doing ',i)
 with open(i, 'r') as txt:
  file_text = txt.read()
 segments = file_text.split('|||')
 policy_text = pd.DataFrame(columns=['Segment ID', 'extracted text', 'policy_file'], dtype='object')
 policy_text['Segment ID'] = list(range(0, len(segments)))
 policy_text['extracted text'] = segments
 policy_text['policy_file'] = i[:-9]
 all_policy_text = pd.concat([all_policy_text,policy_text])
 print(i,'is done')

all_policy_text.to_csv('all_policy_text.csv')


""" Getting annotation categories for each segment across all policies"""

os.chdir(r'C:\Users\prudh\OneDrive\Documents\repos\SimplePrivacyPolicy_SiPP\OPP-115\annotations')
files = os.listdir()

annotated_catgs= pd.DataFrame(columns=['Category Name', 'Segment ID','policy_file'], dtype= 'object')

for i in files:
 print('doing ',i)
 sample = pd.read_csv(i, names = ['Annotation ID','Batch ID','Annotator ID','Policy ID','Segment ID','Category Name','Attribute Value Pairs','Policy URL',	'Date'])
 df = sample.groupby(['Segment ID'])['Category Name'].unique().reset_index()
 df['policy_file'] = i[:-4]
 annotated_catgs = pd.concat([annotated_catgs,df])
 print(i,'is done')
annotated_catgs.to_csv('annotated_categories.csv')

text_with_annotations = pd.merge(all_policy_text,annotated_catgs,how= 'inner', on=['Segment ID','policy_file'])

text_with_annotations.to_csv('text_with_annotations.csv')
text_with_annotations['num_of_cats'] = text_with_annotations['Category Name'].apply(lambda x: len(x))

text_with_annotations['num_of_cats'].value_counts()


""" Predicting using unsupervised model """


def sim_score(d,ref_nouns, ref_verbs,ref_adjectives,ref_pronouns):

 n_count, v_count,a_count, p_count = 8,8,8,1
 d = d.lower()
 d = model(d)
 tokens = [(i, i.pos_) for i in d if i.text not in string.punctuation]

 nouns = list(set([i[0].text for i in tokens if i[1] == 'NOUN']))
 adjectives = list(set([i[0].text for i in tokens if i[1] == 'ADJ']))
 verbs = list(set([i[0].text for i in tokens if i[1] == 'VERB']))
 pronouns = list(set([i[0].text for i in tokens if i[1] == 'PRON']))

 n_score = distance(nouns,ref_nouns,n_count) / n_count
 v_score = distance(verbs,ref_verbs,v_count) / v_count
 a_score = distance(adjectives,ref_adjectives,a_count) / a_count
 p_score = distance(pronouns,ref_pronouns,p_count) / p_count
 score = (n_score + v_score + a_score +p_score) / 4

 return score

def distance(first_group,second_group,count):
 pairs = list(itertools.product(first_group, second_group))
 try :
  distances = [w2v.similarity(k[0],k[1]) for k in pairs]
  distances.sort(reverse=True)
 except :
  distances = [0] * count
 return sum(distances[:count])


ref_words =  pd.read_csv('privacy words2.csv')
ref_words = ref_words.set_index('categories')




def get_scores(row):
 sent = row['extracted text']
 cats = ['Data Retention', 'Data Security',
  'First Party Collection/Use', 'Third Party Sharing/Collection',
  'User Access, Edit and Deletion', 'User Choice/Control']
 scores =[]
 for i in cats :
  ref_nouns = ref_words.loc[i]['Nouns'].split(',')
  ref_verbs = ref_words.loc[i]['Verbs'].split(',')
  ref_adjectives = ref_words.loc[i]['Adjectives'].split(',')
  ref_pronouns = ref_words.loc[i]['Pronouns'].split(',')
  scores.append((i,sim_score(sent,ref_nouns,ref_verbs,ref_adjectives,ref_pronouns)))
  scores = sorted(scores,key=lambda x: x[1], reverse=True)
 return scores

row = text_with_annotations.iloc[1]
k = get_scores(row)
d = text_with_annotations['extracted text'][1]


text_with_annotations['predicted categories'] = text_with_annotations.apply(lambda x : get_scores(x), axis=1)

start_time = time.time()
get_scores(text_with_annotations.iloc[10])
print("--- %s seconds ---" % (time.time() - start_time))


text_with_annotations['preds'] = text_with_annotations['predicted categories'].apply(lambda x : [p[0] for p in x][:3])

text_with_annotations['correct'] = text_with_annotations.apply(lambda x : len(list(set(list(x['Category Name'])).intersection(x['preds']))), axis = 1)

text_with_annotations.correct.value_counts()

text_with_annotations.to_csv('preds.csv')