# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:20:08 2021

@author: jrose
"""


import pdfplumber
from textblob import TextBlob
from textblob import Word
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Read PDF in as str
with pdfplumber.open(r'support_docs\Jamaine_Roseborough_Resume.pdf') as pdf:
    first_page = pdf.pages[0]
    doc_as_text = first_page.extract_text()

#python -m textblob.download_corpora

# Clean text
def clean_text(text):
    tb = TextBlob(text)
    correction = [Word(x).spellcheck()[0][0] for x in tb.words]
    clean = ' '.join(correction)
    return clean

# Performs TF-IDF Transformation
def process_text(text):
    # Perform the count transformation
    vectorizer = CountVectorizer(stop_words='english')
    vec = vectorizer.fit_transform([text])
    
    # Perform the TF-IDF transformation
    tf_idf_vec = TfidfTransformer()
    tf_idf_sen = tf_idf_vec.fit_transform(vec)
    
    # Print out results in a dataframe
    tf_df = pd.DataFrame(tf_idf_sen.toarray(), columns = vectorizer.get_feature_names())
    return pd.DataFrame({'Word': list(tf_df.columns),
                         'Weight': [i for i in tf_df.iloc[0]]})

my_clean_text = clean_text(doc_as_text)
print(process_text(my_clean_text))