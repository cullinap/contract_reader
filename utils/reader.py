import re
import os
import pandas as pd 
import pandas as pd 
import numpy as np 
from tika import parser
from utils.process_data import preprocess
from sklearn.externals import joblib

model = joblib.load('./data/svc_prediction_model.sav')

class Read():

    def __init__(self,path='./process_agreement/', file=None):
        self.path = path
        self.file = [f for f in os.listdir('./process_agreement/') if f.endswith('.pdf')][0]
        df = self.full_text
        print(f'your document has the following categories')
        for x in df()['label'].unique():
                print(x)

    def full_text(self, model=model):
        '''
        

        '''
        process_val = {'stopwords':'english',
                   'custom_sw':'aaa',
                   'stemmer':'yes',
                   'ngrams':'bigrams'}
        
        regex = '\n\n(?=\u2028|[A-Z-0-9])' #removed one \n


        clean_file = re.split(regex, 
                        [parser.from_file(self.path + self.file)]
         [0]['content'].strip()
        )
        
            
        prep = lambda i: preprocess(pd.Series(i),**process_val).apply(lambda x:', '.join(map(str, x)))
        
        preprocessed = prep(clean_file)

        labels = list(model.predict(preprocessed))

        text = [i[:300] for i in clean_file]

        return pd.DataFrame({'label':labels, 'text': text})

    def section(self, label):
        if os.path.exists(self.path + self.file) == True:
            df = self.full_text
            section = df()[df()['label']==label]

            # print('the labels in this agreement are...')
            # for x in df()['label'].unique():
            #     print(x)
            # print()

            for i,p in section.iterrows():
                print(p['text'])

        else:
            print('no file in directory')






   