from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import streamlit as st

from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re

from cleanTweets import CleanTweets
#from textprocessing import st.session_state.df_train
CT = CleanTweets()

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter",
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)
def remove_auxilary(text,wordlist):

    text = re.sub(r'[^\w\s]','',text)
    text = text.replace('user', '').replace('hashtag', '').replace('allcaps', '').replace('url', '')
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    tokens_without_sw2=[]
    if wordlist !='':
        for word in tokens_without_sw:
            if word not in wordlist:
                tokens_without_sw2.append(word)
        return tokens_without_sw2

        #tokens_without_sw = [word for word in tokens_without_sw if not word in wordlist]
    return tokens_without_sw


def pre_processing(df_train, df_column):

    st.caption('Progress of processing tweet data')
    my_bar1 = st.progress(0)
    for idx in df_train.index:
        sent = df_train.loc[idx,df_column]
        sent = sent.replace('‘', '\'').replace('’', '\'').replace('“', '"').replace('”', '"')
        #print(sent)
        sent = ' '.join(text_processor.pre_process_doc(sent))
        #print(sent)
        df_train.loc[idx,'text_PP'] = sent
        my_bar1.progress(idx/df_train.index[-1])
    
    return df_train
    
def text_tokenize(df_train,df_column,wordlist):

    if wordlist !='':
        wordlist= wordlist.split(',')   

    text_wa = []
    st.caption('Progress of tokenizing and removing stopwords')
    my_bar2 = st.progress(0)
    for idx in df_train.index:
        sent = df_train.loc[idx,df_column]
        sent = remove_auxilary(sent,wordlist)
        sent = ' '.join(sent)
        text_wa.append(sent)
        my_bar2.progress(idx/df_train.index[-1])
        if idx%1000==0:
            print('current index', idx)
            
    df_train['text_PP'] = text_wa

    return df_train


        
        
    