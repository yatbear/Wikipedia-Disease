#!usr/bin/env python
# -*- coding: utf-8 -*-

# Usage: python [files]
#
# Data Preparation
#       Preprocess the dataset 
#       Parse wiki page dumps
#       Extract text of interest for analysis 
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2016-02-10

import os
import sys
import re
import urllib
import numpy as np
from HTMLParser import HTMLParser

class WikiParser(HTMLParser):
    
    def __init__(self):
        HTMLParser.__init__(self)
        self.wikiText = list()
        self.disease_names = list()
        self.NOISY_PATTERN = '%E2%80%93' # noisy substring found in some of the file names
        self.IGNORED_WORDS = ['navigation', 'search', 'edit', 'See also']
        
    def extract_wiki_pages(self, folder):
        page_set = list()
        pos = True if 'positive' in folder else False # indicator of whether the current folder is positive
        for page_name in os.listdir(folder): 
            # Clean page names, ignore hidden files and txt files
            if page_name.startswith('.') or page_name.endswith('.txt'):
                continue
            if self.NOISY_PATTERN in page_name:
                old_name = page_name
                new_name = page_name.replace(self.NOISY_PATTERN, '') 
                old_path = os.path.join(folder, old_name)
                new_path = os.path.join(folder, new_name)
                os.rename(old_path, new_path)
                page_set.append(new_path)
            else:
                page_path = os.path.join(folder, page_name)
                page_set.append(page_path)
            # Extract disease names from the positive folder
            if pos:
                self.disease_names.append(page_name)
        return page_set
           
    def handle_data(self, data):
        if data.strip() != '':
            self.wikiText.append(data)
             
    def parse_page(self, page_path):
        self.feed(urllib.urlopen(page_path).read())
        
    def extract_text_of_interest(self):
        TOI = list()
        ifScratch = False # indicator of whether the data item is relevant and should be scratched
        # Extract data item of interest
        for item in self.wikiText:
            # This where to start
            if 'From Wikipedia, the free encyclopedia' in item: 
                ifScratch = True
                continue
            # This is where to stop 
            if 'References' in item:
                ifScratch = False
                break
            if ifScratch:
                # Skip meaningless words 
                if 'Jump to:' in item or item in self.IGNORED_WORDS:
                    continue
                # Eliminate unwanted punctuations and numbers
                if re.search('[a-zA-Z]', item):
                    TOI.append(item)
        return TOI
    
if __name__ == '__main__':
    reload(sys)  
    sys.setdefaultencoding('utf8')
    
    # A parser that preprocesses and extracts data from wiki articles
    parser = WikiParser()
    
    # Extract raw data 
    pos_folder, neg_folder = 'training/positive', 'training/negative'
    pos_pages = parser.extract_wiki_pages(pos_folder)
    neg_pages = parser.extract_wiki_pages(neg_folder)
    
    # Filter the raw data, extract text of interest
    def extract_data(page_set):
        TOIs = list() # a list of text of interest
        for page in page_set:
            parser.wikiText = []
            # Parse each page
            parser.parse_page(page)
            # Extract text of interest from the current page
            TOI = parser.extract_text_of_interest()
            TOIs.append(TOI)
        return TOIs
    
    pos_TOIs = extract_data(pos_pages)
    neg_TOIs = extract_data(neg_pages)

    # Save extracted text to file
    np.savez('wikiTOI', x=pos_TOIs, y=neg_TOIs)
    # Save extracted disease names to file
    np.savez('disease_names', x=parser.disease_names)
  
    parser.close()
    