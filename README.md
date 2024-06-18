# Phishing Email Detector

This repository contains a machine learning application that detects phishing emails. The application is built using a deep learning model and a web interface for user interaction.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Application Structure](#application-structure)
5. [Reference](#Reference)

## Overview

The Phishing Email Detector is a web application that allows users to input email text and receive a prediction on whether the email is a phishing attempt. The backend model is a deep learning model built using TensorFlow and Keras, while the web interface is built using Flask.

## Model Architecture

The deep learning model consists of the following layers:

1. **Embedding Layer**: Converts input text into dense vectors of fixed size.
2. **SpatialDropout1D Layer**: Regularizes the model by randomly dropping a fraction of the input units.
3. **Algorithmic Layer**: Captures long-term dependencies in the text data. This layer is either LSTM, RNN or GRU layer.
4. **Dense Layer**: Outputs a probability value indicating whether the email is a phishing email.

## Dataset
The dataset can be downloaded from kaggle. Here is the link to the [phishing dataset]("https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv.")
We train the model on the "phishing_email.csv" dataset. 

## Application Structure

phishing-email-detector/

├── app.py # Flask application

├── phishing_email_detector.keras # weights of Trained LSTM model

├── phishing_email_detector2.keras # weights of Trained RNN model

├── phishing_email_detector3.keras # weights of Trained GRU model

├── LSTM.ipynb # trained LSTM

├── RNN.ipynb # trained RNN

├── GRU.ipynb # trained GRU

├── requirements.txt # Required packages

└── templates/

└── index.html # HTML template for the web interface

## Reference

Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619*
