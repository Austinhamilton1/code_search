from preprocessing import Preprocessor

p = Preprocessor('.', 'python', 'class')
p.preprocess()

print(len(p.documents_))
#for document in p.documents_:
#    print(document)