from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

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
def remove_auxilary(text):
    text = re.sub(r'[^\w\s]','',text)
    text = text.replace('user', '').replace('hashtag', '').replace('allcaps', '').replace('url', '')
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw

def pre_processing(df_train, df_column):

    for idx in df_train.index:
        sent = df_train.loc[idx,df_column]
        sent = sent.replace('‘', '\'').replace('’', '\'').replace('“', '"').replace('”', '"')
        #print(sent)
        sent = ' '.join(text_processor.pre_process_doc(sent))
        #print(sent)
        df_train.loc[idx,'text'] = sent
    
    return df_train
    
def text_tokenize(df_train,df_column):
    text_wa = []
    for idx in df_train.index:
        sent = df_train.loc[idx,df_column]
        sent = remove_auxilary(sent)
        sent = ' '.join(sent)
        text_wa.append(sent)
        if idx%1000==0:
            print('current index', idx)
            
    df_train['text_PP'] = text_wa

    return df_train


        
        
    