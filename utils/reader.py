import re
import os
import pandas as pd 
import pandas as pd 
import numpy as np 
from tika import parser
from utils.process_data import preprocess
from sklearn.externals import joblib

model = joblib.load('./data/document_prediction_model.sav')

def read(file, path, model=model):
    '''
    goes to each folder and iterates through the pdf files in a folder
    takes:
    -file: list of files in the directory
    -path: path to the data folders

    '''
    process_val = {'stopwords':'english',
               'custom_sw':'aaa',
               'stemmer':'yes',
               'ngrams':'bigrams'}
    
    regex = '\n\n(?=\u2028|[A-Z-0-9])' #removed one \n


    clean_file = re.split(regex, 
                    [parser.from_file(path + file)]
     [0]['content'].strip()
    )
    
        
    prep = lambda i: preprocess(pd.Series(i),**process_val).apply(lambda x:', '.join(map(str, x)))
    
    preprocessed = prep(clean_file)

    labels = list(model.predict(preprocessed))

    text = [i[:300] for i in clean_file]

    return pd.DataFrame({'label':labels, 'text': text})
   