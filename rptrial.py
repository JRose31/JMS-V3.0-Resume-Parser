# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:20:08 2021

@author: jrose
"""


import pdfplumber
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


with pdfplumber.open(r'support_docs\Jamaine_Roseborough_Resume.pdf') as pdf:
    first_page = pdf.pages[0]
    doc_as_text = first_page.extract_text()

#python -m textblob.download_corpora

def process_text(text):
    tb = TextBlob(text) # Make a textblob so that we can singularize the word
    singular = [x.singularize() for x in tb.words] # Singularize each word in the text
    clean = ' '.join(singular) # Join it together into a single string
    
    # Perform the count transformation
    vectorizer = CountVectorizer(stop_words='english')
    vec = vectorizer.fit_transform([clean])
    
    # Perform the TF-IDF transformation
    tf_idf_vec = TfidfTransformer()
    tf_idf_sen = tf_idf_vec.fit_transform(vec)
    
    # Print out results in a dataframe
    tf_df = pd.DataFrame(tf_idf_sen.toarray(), columns = vectorizer.get_feature_names())
    return tf_df

print(process_text(doc_as_text))