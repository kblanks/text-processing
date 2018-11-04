'''
Created on Oct 27, 2018

@author: Kyle
'''
#import libraries
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from pandas import DataFrame
from numpy import asarray
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.text.tsne import TSNEVisualizer

def read_text(text_name):
    with open(text_name,'r') as f: book = f.read()
    f.close()
    return book

def doVisualizer(featNames,vector,numTerms=10):
    visualizer = FreqDistVisualizer(features=featNames,n=numTerms)
    visualizer.fit(vector)
    visualizer.poof()

text_1=read_text('LOTR1.txt')
text_2=read_text('LOTR2.txt')
text_3=read_text('LOTR3.txt')
text = [text_1,text_2,text_3]

trainDF = DataFrame()

#basic text level features
trainDF['text'] = text
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count']=trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density']=trainDF['char_count']/(trainDF['word_count']+1)

# count and tfidf
count_vec = CountVectorizer(stop_words='english', analyzer='word')
tfidf_vec = TfidfVectorizer(analyzer='word',norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=False)
count_fit = count_vec.fit(trainDF['text'])
tfidf_fit = tfidf_vec.fit(trainDF['text'])
vector_count=count_fit.transform(trainDF['text'])
vector_tfidf=tfidf_fit.transform(trainDF['text'])
count_feat=count_vec.get_feature_names() 
tfidf_feat=tfidf_vec.get_feature_names()
count_freqs=zip(count_feat,vector_count.sum(axis=0).tolist()[0])
tfidf_freqs=zip(tfidf_feat,tfidf_vec.idf_)

# output
print(sorted(tfidf_freqs, key=lambda x: -x[1])[:5])
print(sorted(count_freqs, key=lambda x: -x[1])[:5])
print(trainDF['char_count'])
print(trainDF['word_count'])
print(trainDF['word_density'])
#doVisualizer(count_feat, vector_count, 20)