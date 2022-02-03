#conda install nltk
#conda install bs4 
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re 
#from datetime import datetime
#import time


from collections import Counter
from string import punctuation
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import nltk

#import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#import matplotlib.pyplot as plt


# Fonction pour le pré traitement des questions
token=ToktokTokenizer()
lemma=WordNetLemmatizer()
stopwords.words("english")
dfT = pd.read_csv('tags.csv')
dfT.columns = ['Id', 'Tag']
liste = dfT['Tag']
words = liste.values

def retire_html(text): # retire les balises HTML
    soup = BeautifulSoup(text).get_text()
    return soup

def clean_punck(text): # retire la ponctuation et les majuscules
    #punct = list(punctuation)
    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punct = list('!"%&\'()*,./:;<=>?@[\\]^_`{|}~')
    for c in punct:
        text = text.replace(c, "")
    text = text.lower()
    return text

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def retire_punck(text): 
    #print("RETIRE PONCTUATION :")
    #print(text)
    text = text.lower()
    punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in dfT['Tag'].to_list():
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
    print(filtered_list)   
    return ' '.join(map(str, filtered_list))

def lemitize_words(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def tokenize(text): # passe de text à liste de mots
    token=ToktokTokenizer()
    words = token.tokenize(text)
    return words

def retire_stopwords(text): # retire les mots dénués de sens fondamental EN MINUSCULE
    text = token.tokenize(text)
    filtre = set(stopwords.words("english"))
    filtered = [w for w in text if not w in filtre]
    return ' '.join(map(str, filtered))


def nettoyage_df(df, col):
    df[col] = df[col].apply(retire_html)
    #print(df[col][0])
    df[col] = df[col].apply(retire_punck)
    #print(df[col][0])
    df[col] = df[col].apply(lemitize_words)
    #print(df[col][0])
    df[col] = df[col].apply(retire_stopwords)
    #print(df[col][0])
    return df

def nettoyage_text(text):
    text = retire_html(text)
    text = retire_punck(text)
    text = lemitize_words(text)
    text = retire_stopwords(text)
    return text 

# Fonction de traitement de la table TAGS
def most_common_tags(dfT,nbr):
    # recupératoin des 500 tags les plus communs
    dfT.columns = ['Id', 'Tag']
    words = dfT['Tag'].values
    mc = Counter(words).most_common(nbr)
    list_tags = [e[0] for e in mc]
    return list_tags

def nettoyage_tag(dfT, list_tags):
    # suppresion des tags les moins communs
    #dfT['Tag'] = dfT['Tag'].astype(str)
    dfT.columns = ['Id', 'Tag']
    print(dfT.shape)
    new_dfT = dfT[dfT['Tag'].isin(list_tags)]
    print(new_dfT.shape)
    
    #création de la table tags
    tags = new_dfT.groupby(["Id"])['Tag'].apply(lambda new_dfT: ' '.join(new_dfT))
    #tags.columns = ['Id', 'Tag']
    return tags

# Fonction de créatoin de notre data
def create_data(dfT, dfQR, nbr):
    list_tags = most_common_tags(dfT,nbr)
    tags = nettoyage_tag(dfT, list_tags)
    
    dfQR.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
    data = dfQR.merge(tags, on='Id')
    data.drop_duplicates(inplace = True)
    data.drop(columns=['Id', 'Score'], inplace=True)
    
    tags_features = Counter(dfT['Tag'].values).most_common(nbr)
    data = nettoyage_df(data, 'Body')
    data = nettoyage_df(data, 'Title')
    data['question'] = data['Title'] +' '+ data['Title'] +' '+ data['Title'] + ' '+ data['Body']
    data.drop(columns = ['Title','Body'], inplace = True)
    return data

def preprocessing():
    dfQR = pd.read_csv('questions.csv')
    dfT = pd.read_csv('tags.csv')
    data = create_data(dfT, dfQR, 20)
    data.to_csv('data.csv')

#preprocessing()