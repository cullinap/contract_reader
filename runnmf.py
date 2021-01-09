from sys import argv
import pandas as pd
from utils.nmfTopicmodel import Topic_Model

data_path = argv[1]

df = pd.read_csv(data_path)

if __name__ == '__main__':
    topicModel = Topic_Model(data=df['text']).fit()
    topicModel
    print(topicModel.coh_score)

