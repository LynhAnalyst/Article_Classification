import base64, html5lib, re, requests
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nlp import text_preprocess
from time import sleep

st.set_page_config('Article Classifier')

with open('style.css', 'r') as f:
    background = f.read()

f.close()
st.markdown(background, unsafe_allow_html = True)

placeholder = 'Maximum 10 links at a time - Only accept VnExpress link\n- Link 1\n- Link 2\n- Link 3'
user_input = st.text_area('Insert links below:', height = 240, placeholder = placeholder)

col1, col2, col3 = st.columns(3)
with col2:
    button = st.button('Classify', use_container_width = True)

tokenizer = pickle.load(open('tokenize_text.pkl', 'rb'))
labels = pickle.load(open('dict_label.pkl', 'rb'))
if button:
    model = load_model('model_cnn_news')
    urls = re.findall('http.+', user_input)
    if len(urls) > 10 or len(urls) < 1:
        st.warning('You\'ve just put in %s link(s)' %len(urls))
    else:
        title_and_subject = []
        for url in urls:
            try:
                html = requests.get(url).text
                parser = BeautifulSoup(html, 'html.parser')
                sleep(1)
                title = parser.select_one('.title-detail').text.strip()
                description = parser.find('p', {'class':'description'}).text.strip()
                detail = ' '.join(p.text.strip() for p in parser.select('p.Normal:not([style])'))
                subject = parser.select_one('ul.breadcrumb').li.text.strip()
                title_ = text_preprocess(title + ' ' + description + ' ' + detail)
                predict = model.predict(pad_sequences(tokenizer.texts_to_sequences([title_]), maxlen = 1600, padding = 'post'))
                title_and_subject.append((title,labels[str(np.argmax(predict[0]))]))
            except:
                st.warning('%s is not an eligible link' %url)
        df = pd.DataFrame(title_and_subject, columns = ['Title', 'Predict subject'])
        st.table(df)