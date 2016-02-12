#!usr/bin/env python
# -*- coding: utf-8 -*-

# Usage: python [files]
#
# Classify Wikipedia Disease Articles
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2016-02-11

import sys
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class Wiki_Disease_Analyser(object):
    
    def __init__(self):
        self.STOPS = set(stopwords.words("english")) # a set of stop words in English
        self.tokenizer = RegexpTokenizer(r'\w+') 
        self.count_vect = CountVectorizer(analyzer='word')
        self.tf_transformer = TfidfTransformer(use_idf=True)
        # Multinomial Naive Bayes classifier for text classification 
        self.clf = MultinomialNB()
        # Logistic Regression classifier 
        # self.clf = LogisticRegression()
    
    # Remove punctuation marks and stop words from a given string
    def clean_string(self, text):   
        # Remove punctuation marks
        tokens = self.tokenizer.tokenize(text)
        # Remove stop words
        meaningful_words = filter(lambda word: word.lower() not in self.STOPS, tokens)
        # Return a string of meaningful words
        return ' '.join(meaningful_words)
    
    # Remove punctuation marks and stop words from a text list
    def clean_text(self, flat_texts):
        return [self.clean_string(text) for text in flat_texts]

    # Reformat the 3-D noisy text data in to a 1-D list of strings
    # by flattening the text matrix corresponding to each wiki page
    def flatten(self, text_mat):
        return [' '.join(map(str, text)) for text in text_mat.flatten()]

    # Randomly split a dataset into training set (80%) and test set (20%)
    def split_data(self, text_mat):
        # Reformat the text matrix
        flat_texts = self.flatten(text_mat)
        # Clean the text data
        clean_flat_texts = self.clean_text(flat_texts)
        # Randomly split data into training set and test set
        train_texts, test_texts = train_test_split(clean_flat_texts, test_size=0.2, random_state=42)
        return train_texts, test_texts
    
    # Build vocabulary 
    def build_vocab(self, data):
        self.count_vect.fit(data)
    
    def create_feats(self, data):
        # Count occurrences
        counts = self.count_vect.transform(data)
        # Convert occurrences to term frequencies
        tf = self.tf_transformer.fit_transform(counts)
        # Use numpy format for learning
        features = tf.toarray()
        return features
    
    # Train the a Multinomial Naive Bayes classifier 
    def train(self, train_data, train_labels):
        # Features ready
        train_data_features = self.create_feats(train_data)
        # Train the classifier
        self.clf.fit(train_data_features, train_labels)
    
    # Test the learned classifer and calculate the accuracy
    def predict(self, test_data, test_labels):
        # Features ready
        test_data_features = self.create_feats(test_data)
        # Predict on the test set
        predicted = self.clf.predict(test_data_features)
        # Compute accuracy
        acc = np.mean(predicted == test_labels)
        return predicted, acc  
    
    # Save the classifier
    def save_clf(self):
        joblib.dump(self.clf, 'wikiclf.pkl') 
     
if __name__ == '__main__':
    reload(sys)  
    sys.setdefaultencoding('utf8')
    
    # Load extracted text of interest, where x=positive and y=negative
    npzfiles = np.load('wikiTOI.npz')
    pos_texts = npzfiles['x']
    neg_texts = npzfiles['y']
    
    # Load disease names for evaluation
    disease_names = np.load('disease_names.npz')['x']
    
    # Apply Multinomial Naive Bayes Method
    analyser = Wiki_Disease_Analyser()
      
    # Split data for training and testing
    pos_train, pos_test = analyser.split_data(pos_texts)
    neg_train, neg_test = analyser.split_data(neg_texts)
    all_train = pos_train + neg_train
    all_test = pos_test + neg_test
    
    # Build vocabulary using all available training data
    analyser.build_vocab(all_train)
    
    # vocab = multiNB.count_vect.get_feature_names()
    # print 'The vocabulary size is', len(vocab)
    
    # Construct labels (binary), 1 for positive, 0 for negative
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)

    # Train a Multinomial Naive Bayes classifier
    analyser.train(all_train, train_labels)
    predicted, acc = analyser.predict(all_test, test_labels)
    # analyser.save_clf()
    
    # Display the classification accuracy
    print 'Classification accuracy on the test set is %f\n' % round(acc, 4) 
    
    # Find the misclassified diseases, display the false positives
    print 'Misclassified diseases are:', 
    for i in range(len(pos_test)):
        if predicted[i] != test_labels[i]:
            print disease_names[i],
 