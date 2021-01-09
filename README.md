### Topic Modeling with NMF

Quick start guide: 

##### Find optimal cluster k value using TC-W2V:

Use the following command to run NMF on your dataset:

'''python
python runnmf.py <filepath>
'''

The output will be a graph showing the optimal k value to cluster your text based on the NMF algorithm and topic coherence word2vec (TC-W2V)  

Ensure the data is in csv format and the target column is named 'text'.

##### Project Information

The purpose of this project is to take a legal document, like a contract, model the topics and create a pipeline to tag parts of the document with a relevant label. This notebook will focus on the the preprocessing of the data, the topic modeling and the creation of the training set. Ultimately the code in this repo will be useful for people who want to understand a complex legal document such as a credit card agreement more clearly.

The data comes from the following link: https://www.consumerfinance.gov/credit-cards/agreements/

The Consumer Financial Protection Bureau (CFPB) collects credit card agreements from creditors on a quarterly basis and posts them at the link above. The CFPB organizes the data by putting each participating company in a directory and then collecting all the statements in a directory for each company. For Q4 of 2018 there are 652 companies and each company has on average 2-4 agreements. 

For *most* people contract documents are not fun to read because they are usually written in complex legal jargon and the style of writing is purposely dry so as to spell out worst-case scenarios. That said it is important to understand what you or your business is getting into before signing any sort of agreement. Because it takes a certain type of expertise to understand these documents I feel it would be interesting to see if we can leverage natural language techniques to tag this these documents

This repo will enable you to insert a credit card agreement pdf and output labeled sections of the documents to make it easier to read the document. Please see example.ipynb for a walkthrough on how to use this repo. 

The notebook contract_reader.ipynb has further details on how the repo is constructed.
