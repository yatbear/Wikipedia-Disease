# Classify Wikipedia Disease Articles


Classify articles taken from Wikipedia as positive (relevant to diseases) or negative (irrelevant). 


## Data

A collection of Wikipedia article html dumps.
 
 
## Functions

- Information Extraction
	- Extract text of interest from html dumps.	 
- Text Cleaning
	- Remove punctuation marks and stop words (English).

- Vocabulary Construction
	- Build vocabulary from clean training data.

- Vectorization 
	- Count Occurrences.
	- Calculate Term Frequencies.  
	 
- Classification
	- Multinomial Naive Bayes
	- Logistic Regression  

	
## Accuracy

	Multinomial Naive Bayes Classifier: 98.7%
	
	Logistic Regression Classifier: 98.6%


## Usage

	python extract_text_of_interest.py 
	
	python wiki_disease_clf.py


## Dependencies

	NLTK toolkit
	
	scikit-learn library
	 
	

