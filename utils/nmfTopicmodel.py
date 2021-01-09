import nltk
import gensim
import operator
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from itertools import combinations
#import seaborn as sns 
import matplotlib.pyplot as plt


class Preprocess():
    '''
    Text preprocessor class for topic modeling
    Initiate in the following way:
    ----
    instance = preprocess.Preprocess(data=df['tweet'])
    instance.clean_text() # output is [this, is, some, text, that, needs, cleaning]
    ----
    Methods:
    -clean_text: cleans text 
    -vectorizor: turns text into numbers!
    '''

    def __init__(self, data):
        self.data = data

    def clean_text(self, stemmer=True, bigrams=False):
        '''
        takes a pandas series 
        outputs a pandas series with clean tokens in list form
        -gensim simple preprocess lower cases & removes accents (very good for tweets)
        -nltk remove stopwords - english
        -nltk stemmer: default set to true
        -nltk bigrams: default set to false (unigrams is default)
        '''

        text = self.data
        sw = set(stopwords.words('english'))

        text = text.apply(gensim.utils.simple_preprocess, deacc=True, min_len=2)
        text = text.apply(lambda s: [w for w in s if w not in sw]) 
        if stemmer: text = text.apply(lambda s: [SnowballStemmer("english", ignore_stopwords=True).stem(w) for w in s])
        if bigrams: text = text.apply(lambda s: ['_'.join(x) for x in nltk.bigrams(s)] + s)

        return text

class Embeddings:

    '''
    This is a utility class to store data during various
    steps in the data gathering and preprocessing phase.
    The Topic_Model class will popululate the variables
    automatically.
    '''

    def __init__(self):
        self.tfidf = None
        self.tfidf_feature_names = None
        self.w2v = None
        self.w2v_words = None
        self.fasttext = None
        self.nmf = None
        self.coh_score = None
        self.k_values = None
        self.original_text = None
        self.clean_text = None


class Topic_Model:
    """
    A class for taking preprocessed data transforming into 
    decomposed data. The NMF will automatically determine best
    K value based on coherence score using word2vec vs. top words
    from each cluster. Based on the words proximity to eachother
    in each vectorizing step, the model collects clustered 
    data in organized by K topics. 
    """
    
    def __init__(self, data = None, hashtag = None, pages = None, visualize = True, kmax = 14):
        self.root = Embeddings()
        self.preprocess = Preprocess(data=data)
        self.data = data
        self.hashtag = hashtag
        self.pages = pages
        self.kmin = 4
        self.kmax = kmax
        self.visualize = visualize

    def fit(self):
        self.root = Embeddings()
        self.add_original_text(self.root)
        self.clean_data(self.root)
        self.vectorize(self.root)
        self.NMF(self.root)
        self.word2vec(self.root)
        self.calc_coh(self.root)
        if self.visualize:
            self.visualize_coherences(self.root)

        return self.root

    def add_original_text(self, node):
        node.original_text = self.data

    def clean_data(self, node):
        print('cleaning text')
        node.data = self.preprocess.clean_text()

    def vectorize(self, node):
        print('vectorizing text')
        tfidf = TfidfVectorizer()
        
        node.tfidf = tfidf.fit_transform(node.data.apply(lambda x: ' '.join(x)))
        node.tfidf_feature_names = tfidf.get_feature_names

    def NMF(self, node):
        print('calculating NMF')
        kmin = self.kmin
        kmax = self.kmax
        V = node.tfidf
            
        d = {}
        for k in range(kmin,kmax+1):
            model = NMF(init='nndsvd', n_components=k, max_iter=500) 
            d[k] = model.fit_transform(V), model.components_ #collect W & H factors

        node.nmf = d
    
    def word2vec(self, node):
        print('calculating word2vec')

        node.w2v = gensim.models.Word2Vec(node.data, size=500, min_count=1, sg=1)
        node.w2v_words = node.w2v.wv.vocab
    
    def sort_tfidf(self):
        return sorted(
                {str(t): float(score[0,c]) for c,t in enumerate(terms())}.items(), 
                key=operator.itemgetter(1),
                reverse=True
            )

    def calc_coh(self, node):
        # future k_min & k_max
        k_values, coherences = [],[]
        d = node.nmf
        terms = node.tfidf_feature_names()
        w2v_m = node.w2v
        # calculate_coherence = .calculate_coherence()


        for k in range(4,15):
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = [top_term(terms, d[k][0], topic_index, 10) for topic_index in range(k)]

            # Now calculate the coherence based on our Word2vec model
            k_values.append(k)
            coherences.append(calculate_coherence(w2v_m, term_rankings))

        node.coh_score = coherences
        node.k_values = k_values

    def visualize_coherences(self, node):
        coherences = node.coh_score
        k_values = node.k_values

        plt.style.use("ggplot")
        plt.rcParams.update({"font.size": 14})

        fig = plt.figure(figsize=(13,7))
        # create the line plot
        ax = plt.plot(k_values, coherences)
        plt.xticks(k_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Mean Coherence")
        # add the points
        plt.scatter(k_values, coherences, s=120)
        # find and annotate the maximum point on the plot
        ymax = max(coherences)
        xpos = coherences.index(ymax)
        best_k = k_values[xpos]
        plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
        # show the plot
        plt.show()

    def process_best_text_cluster(self):
        pass
        
        #node = self.node
        # find index of best mean coherence score
        #best_k_index = np.argmax(np.array(node.coh_score))

        # gather data necessary to output top sentences/terms
        #W = node.nmf[best_k_index][0]
        #H = node.nmf[best_k_index][1]
        #df = node.original_text
        #terms = node.tfidf_feature_names

        # set 
         

        


### UTILS ###
"""
The utility functions are necessary for sorting words in 
topic clusters, linking words to topics, and calculating 
coherence.
"""
    
def get_top_snippets(all_snippets, W, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( W[:,topic_index] )[::-1]
    # now get the snippets corresponding to the top-ranked indices
    
    return [all_snippets[doc_index] for doc_index in top_indices[0:top]]


def calculate_coherence(w2v_model, term_rankings):
    '''
    takes:
    -w2v model
    -term rankings (calculated by top_term)
    returns:
    -ratio of overall coherence/number of term rankings
    '''
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = [w2v_model.wv.similarity(p[0],p[1]) for p in combinations(term_rankings[topic_index],2)]
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
        
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

#top terms for each topic

def top_term(all_terms, H, topic_index, top):
    '''
    takes: 
    -tf_idf feature names (terms)
    -NMF factorization matrix (H) - when multiplied by W returns V
    -the number of topics in the input
    -# of top terms to use
    returns:
    -top terms for specified parameters
    '''
    all_terms = all_terms
    H = H
    topic_index = topic_index
    top = top

    return [all_terms[term_index] for term_index in np.argsort(H[topic_index,:])[::-1][0:top]]


# Linking words to topics
def word_topic(vec_data, unsuper_method, terms):

    #terms = vectorizor.get_feature_names()
    
    # Loading scores for each word on each topic/component.
    words_by_topic=vec_data.T * unsuper_method

    # Linking the loadings to the words in an easy-to-read way.
    components=pd.DataFrame(words_by_topic,index=terms)
    
    return components

def top_words(components, n_top_words):
    n_topics = range(components.shape[1])
    index= np.repeat(n_topics, n_top_words, axis=0)
    topwords=pd.Series(index=index)
    for column in range(components.shape[1]):
        # Sort the column so that highest loadings are at the top.
        sortedwords=components.iloc[:,column].sort_values(ascending=False)
        # Choose the N highest loadings.
        chosen=sortedwords[:n_top_words]
        # Combine loading and index into a string.
        chosenlist=chosen.index +" "+round(chosen,2).map(str) 
        topwords.loc[column]=[x for x in chosenlist]
    return(topwords)
