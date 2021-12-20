import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import SVC

#===========================================#
#        Loads Model                        #
#===========================================#

vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('classifier.sav', 'rb'))


#===========================================#
#              Streamlit Code               #
#===========================================#

st.title('Duygu Analizi')

user_input = st.text_input('Text')

if st.button('Duygu Analizi Yap!'):
    cumle_vector = vectorizer.transform([user_input])
    predict = classifier.predict(cumle_vector)
    result = ['Pozitif' if lbl == 1 else 'Negatif' for lbl in predict]  
    st.warning(f"Metnin Duygusu: {result[0]}")