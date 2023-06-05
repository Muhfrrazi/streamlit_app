import streamlit as st
import tensorflow as tf
import numpy as np
import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import re
import string
import preprocessor as p
from tensorflow import keras

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Define the maximum sequence length
max_seq = 38

# Function to preprocess the data
def preprocess_data(data):
    data = data.tolist()  # Convert numpy array to list
    processed_data = []
    for sentence in data:
        sentence = text_preprocess(sentence)
        encoded_data = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_seq,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        processed_data.append((encoded_data['input_ids'], encoded_data['attention_mask']))
    return processed_data

# Function to preprocess the sentence
def text_preprocess(sentence):
    pattern = r'[0-9]'
    for punctuation in string.punctuation:
        sentence = p.clean(sentence)
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
        sentence = re.sub(r'http[s]?://\S+', '', sentence)
        sentence = sentence.replace(punctuation, '')
        sentence = re.sub(pattern, '', sentence)
        sentence = re.sub(r'\r?\n|\r', '', sentence)
        sentence = sentence.encode('ascii', 'ignore').decode('ascii')
        sentence = sentence.lower()
    return sentence

# Function to perform sentiment prediction
def predict_sentiment(sentence):
    preprocessed_sentence = preprocess_data(np.array([sentence]))
    input_ids, attention_mask = preprocessed_sentence[0]
    prediction = model.predict([input_ids, attention_mask])
    predicted_label = np.argmax(prediction)
    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_label = label_mapping[predicted_label]
    return predicted_label

# Streamlit app
def main():
    st.title("Analisis Sentimen Berbahasa Indonesia")
    sentence = st.text_input("Masukkan teks disini:")
    if st.button("Cek Kalimat"):
        st.write("Hasil Klasifikasi:")
        sentiment = predict_sentiment(sentence)
        if sentiment == "positive":
            st.markdown('<div style="background-color: green; padding: 10px; color:white;">Sentiment: positive</div>', unsafe_allow_html=True)
        elif sentiment == "negative":
            st.markdown('<div style="background-color: #FE4365; padding: 10px; color:white;">Sentiment: negative</div>', unsafe_allow_html=True)
        elif sentiment == "neutral":
            st.markdown('<div style="background-color: #FDFD96; padding: 10px; color: black;">Sentiment: neutral</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    # Register the custom objects using custom_object_scope
    with keras.utils.custom_object_scope({'TFBertForSequenceClassification': transformers.TFAutoModelForSequenceClassification}):
        # Memuat model yang telah disimpan
        model_path= TFAutoModelForSequenceClassification.from_pretrained('muhfrrazi/IndoBERT-Sentiment-Analysist_Dataset-Indonesia')
        model = keras.models.load_model(model_path)

        main()
