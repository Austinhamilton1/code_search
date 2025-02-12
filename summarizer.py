
import ollama



class Summarize:
    def __init__(self, document_list):
        self.document_list = document_list
        self.corpus = []
    
    def summary_generator(self):
        
        summarized_data = {}
        generation_kwargs = {
        "max_tokens":150, # Max number of new tokens to generate
        # "stop":["<|endoftext|>", "</s>"], # Text sequences to stop generation on
        "echo":True, # Echo the prompt in the output
        "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
            }
       
     
        for document in self.document_list:
            response = ollama.chat(
                        model="llama2",
                        options=generation_kwargs,
                        messages=[
                            {"role": "user", "content": f"Summarize this code:{document}"}
                        ],
                    )
            
            summarized_data[document] = response
            self.corpus.append(summarized_data)
        return self.corpus
