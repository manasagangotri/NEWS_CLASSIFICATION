# NEWS_CLASSIFICATION

# News Classification using Logistic Regression and Passive Aggressive Classifier

This project performs news classification using two datasets: `fake_news` and `true_news`. The objective is to classify news articles as either fake or true using machine learning models such as Logistic Regression and Passive Aggressive Classifier. The project achieves high accuracy for both models, with 98% accuracy for Logistic Regression and 99% accuracy for Passive Aggressive Classifier.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)


## Dataset Description

- `fake_news.csv`: Dataset containing fake news articles.
- `true_news.csv`: Dataset containing true news articles.

## Installation

1. Clone this repository:
   ```bash
   git clone <>
   cd news_classification


## Usage

1.Load both datasets (fake_news.csv and true_news.csv) and create a target variable.

2. Assign 0 as the target for fake_news.
   
3. Assign 1 as the target for true_news.
   
4. Combine both datasets into a single dataframe.

#  Perform data preprocessing, feature extraction, and model training by running:
    ```bash
    python src/train_model.py

## Data Preprocessing

# The preprocessing steps include:

1.  Tokenization: Splitting text into individual words.
   
2.  Stop Word Removal: Removing common words that do not add significant meaning.
  
3.  Stemming: Reducing words to their root form (e.g., "running" to "run").
  
4.  Text Joining: The processed tokens are joined back into a single string without commas.

## Feature Extraction

1.  TF-IDF Vectorization: Convert the text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
   
## Model Training

1.  Logistic Regression: A simple and effective classification algorithm. It achieved an accuracy of 98%.

2.  Passive Aggressive Classifier: A classifier designed for large-scale and streaming data, achieving 99% accuracy.

## Evaluation
  The models were evaluated on a held-out test set. Both models achieved high accuracy, indicating their effectiveness in distinguishing between fake and true news.
## Results
   Logistic Regression: 98% accuracy.
   Passive Aggressive Classifier: 99% accuracy.
## Conclusion

This project demonstrates effective methods for classifying news articles using Logistic Regression and Passive Aggressive Classifier. Both models provide high accuracy, with Passive Aggressive Classifier slightly outperforming Logistic Regression. The preprocessing steps and feature extraction techniques, such as TF-IDF, contribute to the models' success.





