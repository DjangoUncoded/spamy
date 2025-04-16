import streamlit as st
import pickle
import spacy
import nltk
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemer=PorterStemmer()

tfid=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms=st.text_input("Enter a message")



def transform_text(text):
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)  # Tokenize with SpaCy
    y = []
    #nlp = spacy.load("en_core_web_sm") to use spacy library
    #nlp-> appends the words to the list 
    #nlp.sents  ->appends senteneces to the lisy

    for token in doc:
        if token.is_alpha and not token.is_stop:  # Keep only words (no numbers or punctuation)
            y.append(stemer.stem(token.text))
  
    return " ".join(y)

if st.button("Check Spam"):
    if input_sms:  # Ensure input is not empty
        transformed_text = transform_text(input_sms)
        vector_input = tfid.transform([transformed_text])  # Corrected input format
        result = model.predict(vector_input)[0]

        if result == 0:  # Assuming 1 means Spam, 0 means Not Spam
            st.header("🚨 SPAM 🚨")
        else:
            st.header("✅ NOT SPAM ✅")
    else:
        st.warning("Please enter a message before classifying.")