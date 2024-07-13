# sentiment-analysis
Analyzing User Sentiments and Performance of Threads
Overview
### This repository contains two main components:

Analyzing User Sentiments and Performance of Threads.ipynb: A Jupyter notebook for analyzing user sentiments and performance of Threads using various data analysis and visualization techniques.

sentiment_app.py: A Streamlit application for performing sentiment analysis on user reviews using a pre-trained Support Vector Machine (SVM) model.
![image](https://github.com/user-attachments/assets/97ac1280-c72c-407b-8e51-a773afde7b77)

#### Files and Directories
Analyzing User Sentiments and Performance of Threads.ipynb: Jupyter notebook for data analysis and visualization.
sentiment_app.py: Streamlit application for sentiment analysis.
svm.pkl: Serialized SVM model for sentiment prediction.
tfidf.pkl: Serialized TF-IDF vectorizer.
threads_reviews.csv: Dataset containing user reviews.
sentiment.jpg: Image used in the Streamlit app.
requirements.txt: List of dependencies.
Analyzing User Sentiments and Performance of Threads
Requirements
Python 3.7+
Jupyter Notebook
Required Python packages (listed in requirements.txt)

#### Notebook Content
Data loading and inspection
Data cleaning (handling duplicates, missing values)
Exploratory Data Analysis (EDA) using matplotlib, seaborn, and plotly
Text preprocessing for sentiment analysis
Visualization of review sources and ratings over time
Building and saving a TF-IDF vectorizer and SVM model
Model evaluation and metrics

#### Sentiment Analysis App
Requirements
Python 3.7+
Streamlit
Required Python packages (listed in requirements.txt)

#### App Features
Text input for user review
Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)
Sentiment prediction using the pre-trained SVM model
Display of sentiment prediction (Positive, Neutral, Negative)

#### Dependencies
List of main dependencies:
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
nltk
spacy
streamlit
wordcloud
