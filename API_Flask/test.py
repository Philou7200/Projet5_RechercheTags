
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re 

from gensim import models
from gensim.models import Phrases
from gensim import corpora

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#!pip install wordcloud


def tags(text):
    phrase = "HEHE CA FONCTIONNE"
    return phrase

def prepare(text):
    return text.split(' ')


def most_com(text):  
    words=prepare(text)
    most_common_words= [word for word, word_count in Counter(words).most_common(3)]
    return " ".join(most_common_words)




def recherche_tag_lda(document, lda_model, dictionary_LDA,num_topics):
    print(document)

    tokens = word_tokenize(document)
    topics = lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10)
    doc = pd.DataFrame([(el[0], round(el[1],2), topics[el[0]][1]) \
                  for el in lda_model[dictionary_LDA.doc2bow(tokens)]],\
                 columns=['topic #', 'weight', 'words in topic'])
    doc = doc[doc['weight']== max(doc['weight'])]['words in topic']
    
    #mise en forme des tags
    doc = doc.values[0]
    doc = doc.replace('"',' ')
    doc = doc.replace('*','')
    doc = doc.replace(' + ','')
    #table = str.maketrans('','',digits)
    #doc = doc.translate(table)
    doc = doc.replace('. ','')
    doc = re.split(' ',doc)
    tags = []
    i=0
    for t in doc:
        if i==0:
            i=1
        else :
            tags.append(t)
            i=0
    return " ".join(tags)

def tags_lda(document):
    data = pd.read_csv('data.csv')
    tokens = data['question'].apply(nltk.word_tokenize) #.tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])
    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                    id2word=dictionary_LDA, \
                                    passes=4, alpha=[0.01]*num_topics, \
                                    eta=[0.01]*len(dictionary_LDA.keys()))

    tags = recherche_tag_lda(document, lda_model, dictionary_LDA,num_topics)
    return tags

#document = "voici ma question elle est trop stylée !!!!"
#doc = recherche_tag_lda(lda_model, document)
#print(doc)