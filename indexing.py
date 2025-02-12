from preprocessing import LSATokenizer, VectorTokenizer

import pickle
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
import numpy as np

class Indexer:
    def __init__(self, documents):
        '''
        Base class for an indexing model

        Parameters:
        -----------
        documents: list[Document]
            - A list of documents to index
        '''
        self.documents_ = documents
        self.corpus_ = [document.corpus for document in documents]

    def query(self, query, top_n_results=None):
        '''
        Query our indexer to retrieve a document associated with a natural language query

        Parameters:
        -----------
        query: str
            - A natural language query to test for

        top_n_results: int | None
            - How many results to return (None = ALL)

        Returns:
        --------
        Document
            - The associated document that matches the query
        '''
        raise NotImplementedError()
    
    def to_file(self, filename):
        '''
        Save the model to a file for later querying

        Parameters:
        -----------
        filename: str
            - Path to a file
        '''
        with open(filename, 'wb+') as file:
            pickle.dump(obj=self, file=file)

    @staticmethod
    def from_file(filename):
        '''
        Instantiate a model from a saved file

        Parameters:
        -----------
        filename: str
            - The name of the saved model's file

        Returns:
        --------
        Indexer
            - An instance of an Indexer model
        '''
        with open(filename, 'rb') as file:
            return pickle.load(file=file)

class VectorIndexer(Indexer):
    def __init__(self, documents): 
        '''
        An indexing model with a word2vec backend

        Parameters:
        -----------
        documents: list[Document]
            - A list of documents to index
        '''
        super().__init__(documents)
        self.word2vec_ = gensim.downloader.load('word2vec-google-news-300')
        self.tokenizer_ = VectorTokenizer()
        self.model_ = np.zeros((len(self.corpus_), self.word2vec_.vector_size))
        for i in range(len(self.corpus_)):
            self.model_[i] += self.doc2vec(self.corpus_[i])

    def doc2vec(self, document):
        '''
        Convert a document to a vector

        Parameters:
        -----------
        document: str
            - Document to convert

        Returns:
        --------
        numpy.ndarray
            - Vector representation of the document
        '''
        doc_vec = np.zeros((self.word2vec_.vector_size))
        tokens = self.tokenizer_(document)
        if len(tokens) == 0:
            return doc_vec
        for token in tokens:
            if token not in self.word2vec_:
                continue
            doc_vec += self.word2vec_[token]
        return doc_vec / len(tokens)

    def query(self, query, top_n_results=None):
        '''
        Query our indexer to retrieve a document associated with a natural language query

        Parameters:
        -----------
        query: str
            - A natural language query to test for

        top_n_results: int | None
            - How many results to return (None = ALL)

        Returns:
        --------
        Document
            - The associated document that matches the query
        '''
        vector = self.doc2vec(query).reshape(1, -1)
        distance_matrix = pairwise_distances(
            vector,
            self.model_,
            metric='cosine',
            n_jobs=-1,
        )

        ordering = np.argsort(distance_matrix[0])
        return [self.documents_[i] for i in ordering][:top_n_results]

class LSAIndexer(Indexer):
    def __init__(self, documents, n_components=2, random_state=None):
        '''
        Indexes a (preprocessed) piece of text

        Parameters:
        -----------
        documents: list[Document]
            - A list of documents to index

        n_components: int
            - The number of topics

        random_state: float | None
            - The seed for the LSA randomization
        '''
        super().__init__(documents)
        self.vectorizer_ = TfidfVectorizer(tokenizer=LSATokenizer(), token_pattern=None)
        self.svd_model_ = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.transformer_ = Pipeline([
            ('tfidf', self.vectorizer_),
            ('svd', self.svd_model_),
        ])
        self.svd_matrix_ = self.transformer_.fit_transform(self.corpus_)

    def query(self, query, top_n_results=None):
        '''
        Query our indexer to retrieve a document associated with a natural language query

        Parameters:
        -----------
        query: str
            - A natural language query to test for

        top_n_results: int | None
            - How many results to return (None = ALL)

        Returns:
        --------
        Document
            - The associated document that matches the query
        '''
        query_vector = self.transformer_.transform([query])

        distance_matrix = pairwise_distances(
            query_vector,
            self.svd_matrix_,
            metric='cosine',
            n_jobs=-1,
        )

        ordering = np.argsort(distance_matrix[0])
        return [self.documents_[i] for i in ordering][:top_n_results]
    
    def get_top_n_words(self, n):
        '''
        Return the top N words for each topic saved in the indexer

        Parameters:
        -----------
        n: int
            - The number of top words to extract from each topic

        Returns:
        --------
        list[str]
            - A list of size m * n of the top n words asscociated with each of m topics
        '''
        top_words = []
        vocabulary = self.vectorizer_.get_feature_names_out()
        for topic in self.svd_model_.components_:
            top_words.append([vocabulary[i] for i in topic.argsort()[:-n-1:-1]])
        return np.array(top_words)
    
    def umass_(self, i, j):
        '''
        Get the umass score of two words in a topic

        Parameters:
        -----------
        i: int
            - Index of the main word in the tfidf
        
        j: int
            - Index of the secondary word in the tfidf

        Returns:
        --------
        float
            - UMASS of the two words
        '''
        zo_matrix = (self.svd_matrix_ > 0).astype(int)
        col_i, col_j = zo_matrix[:, i], zo_matrix[:, j]
        col_ij = col_i + col_j
        col_ij = (col_ij == 2).astype(int)
        D_i, D_ij = col_i.sum(), col_ij.sum()
        return np.log((D_ij + 1) / D_i)
    
    def topic_umass_(self, topic_idx, top_n_words):
        '''
        Return the umass of a particular topic

        Parameters:
        -----------
        topic_idx: int
            - The index of the topic

        top_n_words: int
            - How many words to consider

        Returns:
        --------
        float
            - UMASS score of topic at index topic_idx
        '''
        indexed_topic = zip(self.svd_model_.components_[topic_idx], range(0, len(self.svd_model_.components_)))
        topic_top = sorted(indexed_topic, key=lambda x: 1-x[0])[0:top_n_words]
        umass = 0
        for j_index in range(len(topic_top)):
            for i_index in range(j_index - 1):
                i = topic_top[i_index][1]
                j = topic_top[j_index][1]
                umass += self.umass_(i, j)
        return umass
    
    def average_umass_(self):
        '''
        Get the average umass score of the model

        Returns:
        --------
        float
            - The average umass score of the model
        '''
        umass = 0
        num_topics = len(self.svd_model_.components_)
        for i in range(num_topics):
            umass += self.topic_umass_(i, 10)
        return umass / num_topics
        
    @staticmethod
    def optimal_model(documents, random_state=None, max_topics=150, dropout_window=5, show_progress=False):
        '''
        Return the top scoring model (coherence score)

        Parameters:
        -----------
        documents: list[Document]
            - The preprocessed documents
        
        random_state: None | int
            - Random seed

        num_jobs: int
            - The number of processes to run concurrently or -1 to run the maximum available

        max_topics: int
            - Maximum number of topics to check up to

        dropout_window: int
            - If the score has not decreased in the last dropout_window iterations, drop
              out of the loop early 

        show_progress: bool
            - Show a progress bar or not

        Returns:
        --------
        (float, Indexer)
            - The optimal coherence score and optimal model
        '''
        optimal_scores = np.full((max_topics,), np.inf)
        models = [None for _ in range(max_topics)]
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=max_topics)

        #this is used for early dropout
        last_decrease = -1

        for i in range(max_topics):
            #create a model and get its average coherence score
            model = LSAIndexer(documents=documents, n_components=i+1, random_state=random_state)
            umass = model.average_umass_()
            models[i] = model
            optimal_scores[i] = umass

            #if the current score is less than the last, set the last_decrease
            if i > 0 and umass < optimal_scores[i-1]:
                last_decrease = i
            
            #if the score hasn't decreased in dropout_window iterations, stop the algorithm
            if i - last_decrease >= dropout_window:
                progress_bar.update(max_topics - i)
                break

            #if the progress bar exists, update it
            if progress_bar is not None:
                progress_bar.update(1)

        #get the optimal score and optimal model
        optimal_idx = optimal_scores.argmin()
        return optimal_scores[optimal_idx], models[optimal_idx]