import ollama
from tqdm import tqdm
import pickle

class Summarize:
    def __init__(self, document_list):
        self.documents_ = document_list
        self.summaries_ = {}

    def add_docs(self, document_list):
        for doc in document_list:
            self.documents_.append(doc)
    
    def generate(self, show_progress=False, save_file=None):
        generation_kwargs = {
            "max_tokens":250, # Max number of new tokens to generate
            # "stop":["<|endoftext|>", "</s>"], # Text sequences to stop generation on
            "echo":True, # Echo the prompt in the output
            "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
        }
       
        if show_progress:
            iterator = tqdm(self.documents_)
        else:
            iterator = self.documents_
     
        for document in iterator:
            if document in self.summaries_:
                continue

            response = ollama.chat(
                model="llama2",
                options=generation_kwargs,
                messages=[
                    {"role": "user", "content": f"Summarize this code:{document}"}
                ],
            )
            self.summaries_[str(document)] = response['message']['content']

            if save_file != None:
                self.to_file(save_file)
    
    def to_file(self, filename):
        with open(filename, 'wb+') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    
    def __getitem__(self, document):
        if document in self.summaries_:
            return self.summaries_[str(document)]
        return None