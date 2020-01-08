import re
import os
import pickle
from tika import parser
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import bigrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib
import operator

class Process():

    def __init__(self, file, path, pkl_file_name):
        self.file = file
        self.path = path
        self.pkl_file_name = pkl_file_name

    def _strip(self, file, path):
        '''
	    goes to each folder and iterates through the pdf files in a folder
	    takes:
		-file: list of files in the directory
		-path: path to the data folders
	    '''
        regex = '\n\n(?=\u2028|[A-Z-0-9])'
    
        try:
            return re.split(regex, 
			[parser.from_file(path + file + '/' + doc) for doc in os.listdir(path + file)] 
		[0]['content'].strip()
	     )

        except:
            return ['no data'] #otherwise the program would return an error

    def _iterate(self):
        '''
        iterates through each folder and applies the _strip function
        '''
        return {f: self._strip(f, self.path) for f in self.file}


    def preprocess(self, text: pd.Series, **kwargs):
        '''
        Basic NLP preprocessor
        -removes stop words
        -n-grams
        -stem
        -simple gensim preprocessor
        '''

        stopwords_ = kwargs.get('stopwords')
        stemmer = kwargs.get('stemmer')
        ngrams = kwargs.get('ngrams')

        text = text.apply(gensim.utils.simple_preprocess, min_len=3)
        
        sw = set(stopwords.words(stopwords_))

        if stopwords_: text = text.apply(lambda s: [w for w in s if w not in sw]) 

        if stemmer=='yes': text = text.apply(lambda s: [SnowballStemmer("english", ignore_stopwords=True).stem(w) for w in s])

        if ngrams=='bigrams': 
            text = text.apply(lambda s: ['_'.join(x) for x in nltk.bigrams(s)] + s)

        elif ngrams=='trigrams':
            text = text.apply(lambda s: ['_'.join(x) for x in nltk.trigrams(s)] + s)

        return text


    def remove_lines(self, OVERWRITE, vectorizor, **kwargs):
        '''
        iterates through the documents and removes line delimeter and makes pkl file
        '''
        pkl_file = './data/' + self.pkl_file_name

        processor_val = {k: v for k,v in kwargs.items()}

        if os.path.exists(pkl_file) == False and OVERWRITE == True:

            print('cleaning pdf files...')
            documents = self._iterate()
            d = {d: [t.replace('\n', ' ').strip() for t in documents[d]] for d in documents.keys()}

            print('now preprocessing the data...')

            processed = {k: 
                      self.preprocess(pd.DataFrame(d[k])[0], **processor_val).apply(lambda x: ', '.join(map(str, x)))
                for k in d.keys()
              }


            print('dumping file to pkl...')

            # Store data (serialize)
            with open(pkl_file, 'wb') as handle:
                pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print('done')


            return processed

        else:

            print('file exists, loading data...')

            with open(pkl_file, 'rb') as handle:
                processed = pickle.load(handle)

            print('loaded...')

            return processed

    def vectorize(self, **kwargs):
        '''
        provide:
         -vectorizor
         -min and max df

        optional:
         -preferred pkl file
        '''

        pkl_file = './data/' + self.pkl_file_name
        #output_name = './data/' + OUTPUT_NAME

        print('loading data...')

        with open(pkl_file, 'rb') as handle:
            processed = pickle.load(handle)

        print('loaded...')

        #create df for processing

        df_ = pd.concat([processed[k] for k in processed.keys()],axis=0).reset_index().drop(['index'],axis=1)

        if kwargs.get('vectorizor') == 'tfidf':

            print('applying tfidf')

            tfidf = TfidfVectorizer(max_df=kwargs.get('max_df'), 
                                    min_df=kwargs.get('min_df'))

            tfidf_ = tfidf.fit_transform(df_[0])

            terms = tfidf.get_feature_names()

            joblib.dump((tfidf_,terms),'./data/tfidf-output.pkl')

            return tfidf_

            #make this so if file exists return file else process

        elif kwargs.get('vectorizor') == 'bow':

            print('applying bow')

            bow = CountVectorizer(max_df=kwargs.get('max_df'), 
                                  min_df=kwargs.get('min_df'))

            return bow.fit_transform(df_[0])


    
		    








