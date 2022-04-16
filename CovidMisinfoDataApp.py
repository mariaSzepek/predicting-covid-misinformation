from this import d
import streamlit as st

import pandas as pd
import numpy as np

import joblib
import re
import nltk
import nltk.data

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import sqlite3

m_jlib = joblib.load('ff')
# pipe_lr = joblib.load(open("./fff.pkl", "rb"))
raw_data = pd.read_csv("raw_data.csv")#, sep=";", encoding="ISO 8859-1")

stemmer = SnowballStemmer('english')

from PIL import Image
image = Image.open('dataproject_image.jpeg')

def clean_text(each_text):
    # remove URL from text
    each_text_no_url = re.sub(r"http\S+", "", each_text)

    # remove numbers from text
    text_no_num = re.sub(r'\d+', '', each_text_no_url)

    # tokenize each text
    word_tokens = word_tokenize(text_no_num)

    # remove sptial character
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))

    # remove lower
    clean_text_lowered = [w.lower() for w in clean_text]

    # do stemming
    stemmed_text = [stemmer.stem(w) for w in clean_text_lowered]
    a = " ".join(" ".join(stemmed_text).split())

    return a


def new_text_input(new_text, raw_data=raw_data):
    emoji_exist = 0
    link_exist = 0
    hashtag_exist = 0
    tag_exist = 0

    if "<U+" in new_text:
        emoji_exist = 1

    if "https" in new_text:
        link_exist = 1

    for hast_tag in re.findall('#(\w+)', new_text):
        hashtag_exist += 1

    for tag in re.findall('@(\w+)', new_text):
        tag_exist += 1

    vectorizer = CountVectorizer()
    clean_text_docmatrix = vectorizer.fit_transform(raw_data["clean_text"])

    clean_text_docmatrix = vectorizer.transform([new_text])
    docmatrix_arrayed = clean_text_docmatrix.toarray()
    columns = vectorizer.get_feature_names()
    docmatrix_df = pd.DataFrame(docmatrix_arrayed, columns=columns)

    docmatrix_df = pd.concat([docmatrix_df, pd.Series(emoji_exist), pd.Series(link_exist), pd.Series(hashtag_exist),
                                pd.Series(tag_exist)], axis=1)
    return docmatrix_df


def predict_misinformation(docx):
    results = m_jlib.predict(new_text_input(docx))
    return results[0]


def main():
    st.title("Vaccine Misinformation Detection")
    st.image(image, caption=None, width=None)
    st.subheader("Does it contain vaccine misinformation?")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type message here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # raw_text = clean_text
        prediction = predict_misinformation(raw_text)

        # probability = get_prediction_proba(raw_text)
        with col1:
            st.success("Original Text")
            st.write(raw_text)

        with col2:
            st.success("Prediction")
            st.write(prediction)

            if prediction == 0:
                st.balloons()

            else:
                st.warning("It's a misinformation!!")

if __name__ == "__main__":
    main()
