#LSA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF

from itertools import combinations
import pandas as pd
import numpy as np


def get_top_snippets( all_snippets, W, topic_index, top ):
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


def lda_pipeline(vec_data, ntopics, n_top_words, vectorizor):
    '''
    takes:
        vec_data --> vectorized data ex: tfidf, bow
        ntopics --> integer, number of topics ex: 5
        n_top_words --> integer, number of words to look for in each topic ex: 10
        vectorizer --> instance, an instance of the vectorizer ex: tfidf, bow
    '''

    #instantiate LDA 
    lda = LDA(n_components=ntopics, 
          doc_topic_prior=None, # Prior = 1/n_documents
          topic_word_prior=1/ntopics,
          learning_decay=0.7, # Convergence rate.
          learning_offset=10.0, # Causes earlier iterations to have less influence on the learning
          max_iter=10, # when to stop even if the model is not converging (to prevent running forever)
          evaluate_every=-1, # Do not evaluate perplexity, as it slows training time.
          mean_change_tol=0.001, # Stop updating the document topic distribution in the E-step when mean change is < tol
          max_doc_update_iter=100, # When to stop updating the document topic distribution in the E-step even if tol is not reached
          n_jobs=-1, # Use all available CPUs to speed up processing time.
          verbose=0, # amount of output to give while iterating
          random_state=0
        )

    #fit transform the data
    data_lda = lda.fit_transform(vec_data)

    terms=vectorizor.get_feature_names()

    #link the words to topics
    components_lda = word_topic(vec_data, data_lda, terms)

    df = pd.DataFrame()

    #extract top N words and their loadings for each topic
    df['LDA']=top_words(components_lda, n_top_words)

    return df


#returns components 
def lsa_pipeline(vec_data, ntopics, n_top_words, vectorizor):
    '''
    takes in vectorized data, topic numbers, and a vectorizor instance
    returns dataframe with terms and scores for each topic
    '''
    svd = TruncatedSVD(ntopics)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    data_lsa = lsa.fit_transform(vec_data)

    #getting the word list
    terms = vectorizor.get_feature_names()

    #loading scores for each word on each topic/component
    components_lsa = word_topic(vec_data, data_lsa, terms)

    #linking the loadings to the words in an easy to read way

    df=pd.DataFrame()

    df['LSA']=top_words(components_lsa, n_top_words)

    return df 


def nmf_pipeline(vec_data, ntopics, n_top_words, vectorizor, init='custom'):
    '''
    takes:
        vec_data --> vectorized data ex: tfidf, bow
        ntopics --> integer, number of topics ex: 5
        n_top_words --> integer, number of words to look for in each topic ex: 10
        vectorizer --> instance, an instance of the vectorizer ex: tfidf, bow
    '''

    #instantiate LDA 
    nmf = NMF(init='nndsvdar', # how starting value are calculated
          l1_ratio=0.0, # Sets whether regularization is L2 (0), L1 (1), or a combination (values between 0 and 1)
          max_iter=200, # when to stop even if the model is not converging (to prevent running forever)
          n_components=ntopics, 
          random_state=0, 
          solver='cd', # Use Coordinate Descent to solve
          tol=0.0001, # model will stop if tfidf-WH <= tol
          verbose=0 # amount of output to give while iterating
         )

    #fit transform the data
    data_nmf = nmf.fit_transform(vec_data)

    terms=vectorizor.get_feature_names()

    #link the words to topics
    components_nmf = word_topic(vec_data, data_nmf, terms)

    df = pd.DataFrame()

    #extract top N words and their loadings for each topic
    df['NMF']=top_words(components_nmf, n_top_words)

    return df