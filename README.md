### Topic Modeling with NMF

Quick start guide: 

##### Find optimal cluster k value using TC-W2V:

Use the following command to run NMF on your dataset:

```
python runnmf.py <filepath>
```

The output will be a graph showing the optimal k value to cluster your text based on the NMF algorithm and topic coherence word2vec (TC-W2V)  

![alt text](http://example.png)

Ensure the data is in csv format and the target column is named 'text'.

### Project Information

The purpose of this project is to take text data like twitter data or from a legal document like a contract and make a general purpose clustering app using non-negative matrix factorization (NMF). 

Example data:

Contractual data comes from:
Consumer Finance credit card agreements: https://www.consumerfinance.gov/credit-cards/agreements/
Twitter data: pull from twitter using tweepy.

Example notebooks:
twitter_nmf.ipynb
TC-W2V walkthough.ipynb
contract_reader.ipynb

#### More info:

The Consumer Financial Protection Bureau (CFPB) collects credit card agreements from creditors on a quarterly basis and posts them at the link above. The CFPB organizes the data by putting each participating company in a directory and then collecting all the statements in a directory for each company. For Q4 of 2018 there are 652 companies and each company has on average 2-4 agreements.

For *most* people contract documents are not fun to read because they are usually written in complex legal jargon and the style of writing is purposely dry so as to spell out worst-case scenarios. That said it is important to understand what you or your business is getting into before signing any sort of agreement. Because it takes a certain type of expertise to understand these documents I feel it would be interesting to see if we can leverage natural language techniques to tag this these documents

This repo will enable you to insert a credit card agreement pdf and output labeled sections of the documents to make it easier to read the document. Please see example.ipynb for a walkthrough on how to use this repo. 

The notebook contract_reader.ipynb has further details on how the repo is constructed.


###### references:
An analysis of the coherence of descriptors in topic modeling Derek O’Callaghan et. al 
