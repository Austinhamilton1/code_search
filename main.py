from preprocessing import Preprocessor
from indexing import LSAIndexer
from summarizer import Summarize

import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

#Preprocessor(source_folder : str, language : str('python | js'), granularity : str('method | class | file'))
p = Preprocessor('.', 'python', 'method')

#fills p.documents_ with the documents at the given granularity
print('Preprocessing...', end='')
p.preprocess()
print('Done')

"""summarizer = Summarize(document_list=p.documents_)

print('Generating Summaries...', end='')
summarizer.generate(show_progress=True, save_file='summaries.log')
print('Done')"""

summarizer = Summarize.from_file('summaries.log')

for i in range(len(p.documents_)):
    summary = summarizer[str(p.documents_[i])]
    p.documents_[i].corpus = summary

#gets the optimal model for a set of documents (basically tunes n_components of the LSA model)
#p.documents_ -> needs to be this
#random_state -> random seed for the LSA model
#max_topics -> maximum value of n_components to be considered
#dropout_window -> if the model does not improve in dropout_window tries, the process will end early
#show_progress -> should the progress be shown in a progress bar
print('Building LSA Indexer...', end='')
umass, model = LSAIndexer.optimal_model(p.documents_, random_state=42, max_topics=150, dropout_window=7, show_progress=True)
print('Done')

print(f'Generated a model with an average umass score of {umass}')

#should give the __search method of the Preprocessor class
query = 'look for files with a certain extension'
result = model.query(query, top_n_results=5)

for document in result:
    print(document)