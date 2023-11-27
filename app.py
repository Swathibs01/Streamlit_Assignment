import streamlit as st
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from BackPropogation import BackPropogation
from sklearn.model_selection import train_test_split





def cnn_tumor(img):
    img=Image.open(img)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_array = cv2.medianBlur(img, 5)
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    img=img.resize((128,128))
    input_img = np.expand_dims(img, axis=0)
    st.image(input_img, caption='Processed Image', use_column_width=True)
    with open('cnn_tumor_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        predictions = loaded_model.predict(input_img)    
    if predictions:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")

def perceptron():
    with open('imdb_perceptron.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    user_input = st.text_input("Enter your movie review:")

    if st.button("Predict"):

        if user_input:
            st.write("Review:", user_input)

            user_input_sequence = [word_index.get(word, 0) for word in user_input.split()]
            processed_input = tf.keras.preprocessing.sequence.pad_sequences([user_input_sequence], maxlen=500, padding='post', truncating='post')

            prediction = loaded_model.predict(processed_input)

            sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
            st.write("Predicted Sentiment:", sentiment)
        else:
            st.warning("Please enter a movie review.")

def backprop():
    with open('BackP.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    user_input = st.text_input("Enter your movie review:")

    if st.button("Predict"):

        if user_input:
            st.write("Review:", user_input)

            user_input_sequence = [word_index.get(word, 0) for word in user_input.split()]
            processed_input = tf.keras.preprocessing.sequence.pad_sequences([user_input_sequence], maxlen=500, padding='post', truncating='post')

            prediction = loaded_model.predict(processed_input)

            sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
            st.write("Predicted Sentiment:", sentiment)
        else:
            st.warning("Please enter a movie review.")

def rnn_model():
    with open('rnn_spam_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        user_input_sequence = st.text_area("Enter your text message:")
        if st.button('Predict'):
            if user_input_sequence:
                tokenizer = Tokenizer(num_words=5000)
                tokenizer.fit_on_texts([user_input_sequence])
                sequences = tokenizer.texts_to_sequences([user_input_sequence])
                processed_input = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
        
                prediction = loaded_model.predict(np.array(processed_input))

                is_spam = 'Spam' if prediction[0] > 0.5 else 'Not Spam'
                st.write("Predicted Label:", is_spam)
        else:
                st.warning("Please enter a text message.")
                

def lstm_model():
    user_input = st.text_input('Enter a sentence:', 'I love this movie!')
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([user_input])
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
    with open('lstm_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    if st.button('Predict Sentiment'):
            prediction = loaded_model.predict(np.array(padded_sequence))
            sentiment = 'Positive' if prediction > 0.5 else 'Negative'
            st.success(f'Sentiment: {sentiment}, Confidence: {prediction[0][0]:.4f}')  

def dnn_model():
    user_input = st.text_input('Enter a sentence:', 'I love this movie!')
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([user_input])
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
    with open('imdb_dnn.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    if st.button('Predict Sentiment'):
            prediction = loaded_model.predict(np.array(padded_sequence))
            sentiment = 'Positive' if prediction > 0.5 else 'Negative'
            st.success(f'Sentiment: {sentiment}, Confidence: {prediction[0][0]:.4f}') 



st.title('Model Prediction')
option = st.selectbox("Choose One",['Tumor Detection','Sentiment Classification'])

if option=='Tumor Detection':
    st.title('CNN Tumor Detection Model')
    img=st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    cnn_tumor(img)
else:
    opt=st.radio("Select your prediction",key="visibility",options=["Perceptron",'Backpropogation','DNN','RNN','LSTM'])
    if opt=="Perceptron":
        st.title('Perceptron Model')
        perceptron()
    elif opt=="Backpropogation":
        st.title('Backpropogation Model')
        backprop()
    elif opt=='RNN':
        st.title('RNN Spam Detection')
        rnn_model()
    elif opt=='LSTM':
        st.title('LSTM Model')
        lstm_model()       
    else:
        st.title('DNN Model')
        dnn_model()