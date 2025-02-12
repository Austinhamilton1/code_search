from preprocessing import Preprocessor
from indexing import LSAIndexer
from summarizer import Summarize

#Preprocessor(source_folder : str, language : str('python | js'), granularity : str('method | class | file'))
p = Preprocessor('test_repo/', 'python', 'method')



#fills p.documents_ with the documents at the given granularity
p.preprocess()



documents_list = p.documents_

# for doc in documents_list:
#     print(doc)
    
suumr = Summarize(documents_list)

print(suumr.summary_generator())

'''
Add a LLM here to generate a corpus for p.documents_
You will need to access document.data for each document in p.documents_
    This will contain the code for each method/class/file in the source folder
Next you will need to add a LLM with a prompt like "Give a detailed summary of what this code does" or something similar
    Make sure the result of each prompt is stored in document.corpus, this will let the LSA model work
    As long as the results of each prompt is stored in document.corpus, everything should be good
'''

#gets the optimal model for a set of documents (basically tunes n_components of the LSA model)
#p.documents_ -> needs to be this
#random_state -> random seed for the LSA model
#max_topics -> maximum value of n_components to be considered
#dropout_window -> if the model does not improve in dropout_window tries, the process will end early
#show_progress -> should the progress be shown in a progress bar
try:
    model = LSAIndexer.optimal_model(p.documents_, random_state=42, max_topics=150, dropout_window=7, show_progress=True)
    query = ''
    result = model.query(query, top_n_results=5)

except:
    pass
#to query
