import pandas as pd 
import numpy as np 
from process_data import preprocess #?

model='./data/document_prediction_model.sav'


def read(file, model=):
    '''
    goes to each folder and iterates through the pdf files in a folder
    takes:
    -file: list of files in the directory
    -path: path to the data folders

    '''
    regex = '\n\n(?=\u2028|[A-Z-0-9])' #removed one \n

    try:

        clean_file = re.split(regex, 
            [parser.from_file(path + file + '/' + doc) for doc in os.listdir(path + file)] 
         [0]['content'].strip()
        )

        prep = lambda i: preprocess(pd.Series(i),**process_val).apply(lambda x:', '.join(map(str, x)))

        labels = [model.predict(prep(x))[0] for x in new_data['1st Century Bank']]
        text = [i[:300] for i in new_data['1st Century Bank']]

        return df = pd.DataFrame({'label':labels, 'text': text})


    except:
        print('no data')
        return ['no data']