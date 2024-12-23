# Amazon Sentiment Analysis on Food Reviews

## Overview

This project focuses on sentiment analysis of Amazon food reviews to classify them as positive, negative or neutral. Sentiment analysis helps in understanding customer feedback and improving product quality and services.

## Table of Contents
- [Libraries and Dataset](#libraries-and-dataset)
- [Data preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#explorator-data-analysis)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Model Evaluation & Conclusion](#model-evaluation)
- [Future Enhancement](#future-enhancement)

The following technologies/libraries were used in this project:
- Data Preprocessing - **Pandas**, **Numpy**
- Visualization - **Seaborn**, **Matplotlib**
- NLP - **NLTK**, **Tensorflow**, **keras**
- Machine Learning - **Scikit-learn**
- Machine Learning models - **Naive Bayes**, **Logistic Regression**, **Bidirectional LSTM**

## Data preprocessing
- Cleaned the dataset by :
- Removing punctuation, html tags, special characters.
- Converting text to lowercase
- Removing stopwords

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution and relationships between the features
- Created word collectors for positive, negative, neutral, and all reviews
- Identified the top 40 polar words in each category (positive, negative, and neutral) to Visualize word distributions and sentiment trends

## Feature Extraction
- Used Document-Term Matrix (DTM) and Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer for traditional models.
- Converted text into sequences using Keras Tokenizer for LSTM.

## Model Training
- Naive Bayes Classifier
     - Implemented using Scikit-learn.
     - Best suited for text classification tasks with its probabilistic approach.
- Logistic Regression
     - Simple and effective baseline model.
- Bidirectional LSTM
  - Used Keras library to build a deep learning model with:
     - An embedding layer to convert words into vectors.
     - Bidirectional LSTM for capturing context from both directions.
     - Dense layers for final classification.

## Model Evaluation
- Assessed model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC score.
- LSTM has the highest accuracy of 87% and this shows that the LSTM deep learning model with bidirectional layer is the best approach for analysing the sentiments of customer reviews.

## Future Enhancement
- Explore additional deep learning architectures like GRU or Transformer-based models.
- Integrate pre-trained word embeddings like GloVe or FastText.
- Deploy the model using Flask or Streamlit for real-time predictions.




