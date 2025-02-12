class Document:
    def __init__(self, filename, title, data, span):
        '''
        Represents a document and holds metadata about the document

        Parameters:
        -----------
        filename: str
            - The name of the file that the document comes from
        title: str
            - The identifier of the document (e.g., class declaration for classes)
        data: str
            - The text representation of the document
        span: tuple[int]
            - The start of the document and the start of the target feature (e.g., method or class start)
        '''
        self.filename = filename
        self.title = title
        self.data = data
        self.span = span
        self.corpus = ''

    def __str__(self):
        '''
        Returns the string representation of a document

        Returns:
        str 
            - Representation of a document
        '''
        result = '-'*50 + '\n'
        result += f'In {self.filename}\n'
        result += '-'*50 + '\n'
        if self.title != self.filename:
            result += f'In {self.title} on line number: {self.span[1]+1}\n'
            result += '-'*50 + '\n'
        line_no = self.span[0]
        for line in self.data.split('\n'):
            result += f'{line_no+1} > {line}\n'
            line_no += 1
        result += '-'*50 + '\n'
        return result