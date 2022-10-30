import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
from text_processing_code import *
from cleanTweets import CleanTweets
import pandas as pd
from numpy import empty
from attr import define
import streamlit as st
from collections import Counter


def app():

    st.subheader('Please upload a file to do the pre-processing')
    CT = CleanTweets()


    def confirm_checkbox(process_tweets, tweets, tokenize,wordlist, clean_NonAscii, df_train, df_column):
        df_column_name = df_column
        if process_tweets:
            df_train = CT.clean_process(df_train, df_column)

        if tweets:
            df_train = pre_processing(df_train, df_column)

        if tokenize:
            
            
            df_train = text_tokenize(df_train, df_column, wordlist)


        if clean_NonAscii:
            df_train = CT.clean_NA_process(df_train, df_column_name)
            df_column_name = 'text_PP'

        message = True

        return message, df_train


    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    st.session_state.uploaded_file = st.file_uploader(
        "Choose a file", type=['csv'], accept_multiple_files=False)


    if st.session_state.uploaded_file is not None:
        st.session_state.processed = False
        # read the uploaded file

        st.session_state.df_train = pd.read_csv(st.session_state.uploaded_file)
        # save an orignial version
        df_train_orgin = st.session_state.df_train
        # give an option when user would like to display the data
        display = st.checkbox('Display data frame')
        if display:
            st.write(df_train_orgin)

        if df_train_orgin.columns is not None:

            # get the column which contains the data the user would like to use for data processing
            st.write(
                'Please select the column name that contains the data you would like to process, click enter to confirm')

            df_column = st.selectbox('select a column', df_train_orgin.columns)
            wordlist=""

            st.write('Please select the type of text processing you would like to used')
            st.session_state.process_tweets = st.checkbox('Clean Tweets',help="This function will do basic processing to the tweet data including replacing html character")
            st.session_state.tweets = st.checkbox('Process tweets data',help='This function will repalce irregular punctuation')
            st.session_state.tokenize = st.checkbox('Tokenize and remove stopwords',help='This function will tokenize the data and remove stopwords in NLTK')
            if st.session_state.tokenize:
                wordlist= st.text_input('You can customize a words list to remove unwanted word. Please enter the word that you would like to remove, if you have more than one word, please end one word with a comma',value = "",help = 'please follow the format of: apple, pear,stawberry')
            #st.session_state.stopword = st.checkbox('Remove high frequent words',help='You can customize a words list to remove unwanted word')
            st.session_state.clean_NonAscii = st.checkbox('Remove Ascii code',help='This function will remove all non-ASCII characters')

            if st.button('Process'):
                st.session_state.processed = True

                if st.session_state.processed:
                    #try:
                    st.session_state.message, st.session_state.df_train = confirm_checkbox(
                        st.session_state.process_tweets, st.session_state.tweets, st.session_state.tokenize,wordlist, st.session_state.clean_NonAscii, st.session_state.df_train, df_column)
                    st.session_state.df_train_final = st.session_state.df_train['text_PP']
                    if st.session_state.message:
                        st.success('process is done')
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption('Processed data')
                            st.write(st.session_state.df_train_final)

                        with col2:
                            top30 = Counter(" ".join(st.session_state.df_train['text_PP']).split()).most_common(30)
                            top30= pd.DataFrame(top30)
                            top30.rename(columns={0: "word", 1: "freq"}, inplace=True)
                            st.caption('Top 30 frequent word')
                            st.write(top30)

                        csv = convert_df(st.session_state.df_train_final)
                        csv2 = convert_df(st.session_state.df_train)

                        col3, col4 = st.columns(2)
                        with col3:

                            st.download_button(
                                label="Download processed data as CSV",
                                data=csv,
                                file_name='pre-processed_data.csv',
                                mime='text/csv',
                            )
                        with col4:
                            st.download_button(
                                label="Download original and processed data as CSV",
                                data=csv2,
                                file_name='pre-processed_data_all.csv',
                                mime='text/csv',
                            )

                                
                    # except:
                    #     st.warning(
                    #         'The selected column is unable to process, please re-select agian')
