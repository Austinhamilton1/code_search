from datastructures import Document

import re
import os
import string
import sys

from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk import word_tokenize

class Parser:
    def __init__(self, src_file):
        '''
        Base class to parse a source file into document(s)

        Parameters:
        -----------
        src_file: str
            - The path to the source file 
        '''
        self.src_file = src_file
        self.documents_ = []
        self.comment_regex = []

    @staticmethod
    def _locate_line(lines, offset):
        '''
        Find the line number that is associated with the file_offset

        Parameters:
        -----------
        lines: list[str]
            - Lines of a file

        offset: int
            - Character offset

        Returns:
        --------
        int: the line number at file_offset
        '''
        count = 0
        for i, line in enumerate(lines):
            count += len(line)
            if(count >= offset):
                return i
        return -1
    
    @staticmethod
    def _get_code_block(lines, start_line):
        '''
        Gets the code block associated with the method or class 
        '''
        raise NotImplementedError()

    def parse(self):
        '''
        Parse the comments from a document (should be called from one of the base models)
        '''
        raise NotImplementedError()
    
class PythonParser(Parser):
    def __init__(self, src_file):
        '''
        Parse a python file into documents

        Parameters:
        -----------
        src_file: str
            - The path to the source file 
        '''
        super().__init__(src_file)

    @staticmethod
    def _count_leading_whitespace(line):
        '''
        Count the number of leading whitespace for a line

        Parameters:
        -----------
        line: str
            - A line of text from a file
        '''
        p = re.compile(r'^(\s*)')
        m = re.match(p, line)
        if m != None:
            return m.end() - m.start()
        return 0
    
    @staticmethod
    def _get_code_block(lines, start_line):
        '''
        Get the associated code block starting at start line (same indentation scale)

        Parameters:
        lines: list[str]
            - The source code to search through
        start_line: int
            - The target line (e.g., method declaration line number)

        Returns:
        --------
        tuple[str, int, int]
            - The associated code and the start and target line
        '''
        document_data = lines[start_line]
        if start_line == len(lines)-1:
            return None

        whitespace = PythonParser._count_leading_whitespace(lines[start_line+1])

        for i in range(start_line+1, len(lines)):
            #get the current line
            current_line = lines[i]

            if current_line.strip() == '':
                #if we got a blank line, skip this line
                document_data += current_line
                continue

            #count the whitespace for the current line
            current_whitespace = PythonParser._count_leading_whitespace(current_line)
            if current_whitespace < whitespace:
                #if the whitespace for the current line is less than the whitespace
                #for the class, break out of the loop and set the end line to the 
                #previous line
                break
            
            #add the line to the current document
            document_data += current_line
        
        return (document_data, start_line)

    def parse(self, granularity='file'):
        '''
        Parse a Python file's code blocks

        Parameters:
        -----------
        granularity: str | None
            - The granularity of the parsing
        '''
        #if the granularity is None, parse the whole document
        if granularity == 'file':
            #since we have None granularity, the only document is the whole document
            with open(self.src_file, 'r') as file:
                data = file.read()
                line_count = len(file.readlines())
                self.documents_ = [Document(self.src_file, self.src_file, data, (0, line_count))]
        elif granularity == 'class':
            #look for classes
            class_regex = re.compile(r'^\s*class (.+?):.*$', re.MULTILINE)
            with open(self.src_file, 'r') as file:
                #we need to match regex and readlines so do both of these
                data = file.read()
                file.seek(0)
                lines = file.readlines()

                #look for classes
                matches = re.finditer(class_regex, data)
                for match in matches:
                    #grab all the data that is in each class
                    start_line = Parser._locate_line(lines, match.start(1))

                    #get the associated code block with the class
                    code_block = PythonParser._get_code_block(lines, start_line)
                    if code_block == None:
                        continue

                    #create a document for the class
                    self.documents_.append(Document(self.src_file, match.group(1), code_block[0], (code_block[1], code_block[1])))
        elif granularity == 'method':
            method_regex = re.compile(r'^\s*def (.+?):.*$', re.MULTILINE)
            with open(self.src_file, 'r') as file:
                #we need to match regex and read lines so do both of these
                data = file.read()
                file.seek(0)
                lines = file.readlines()

                #look for methods
                matches = re.finditer(method_regex, data)
                for match in matches:
                    #grab all the data that is in each method
                    start_line = Parser._locate_line(lines, match.start(1))
                    
                    code_block = PythonParser._get_code_block(lines, start_line)
                    if code_block == None:
                        continue

                    #create a document for the class
                    self.documents_.append(Document(self.src_file, match.group(1), code_block[0], (code_block[1], code_block[1])))

        else:
            raise ValueError(f'Invalid granularity: {granularity} (supported [None|class|method])')
    
class JSParser(Parser):
    def __init__(self, src_file):
        '''
        Parse a JavaScript file into documents

        Parameters:
        -----------
        src_file: str
            - The path to the source file 
        '''
        super().__init__(src_file)

    @staticmethod
    def _get_code_block(lines, start_line):
        '''
        Get the associated code block starting at start line (same indentation scale)

        Parameters:
        lines: list[str]
            - The source code to search through
        start_line: int
            - The target start line (e.g., method declaration line)

        Returns:
        --------
        tuple[str, int, int]
            - The associated code and the start and target line
        '''
        comment_line_counter = 0
        comment_lines = []
        single_line_comment = re.compile(r'^\s*//.+$')
        for i in range(start_line-1, 0, -1):
            #get all the single line comments above the function
            current_line = lines[i]
            if re.match(single_line_comment, current_line):
                #prepend the comment line and increment the comment_line_counter
                comment_lines.insert(0, current_line)
                comment_line_counter += 1
            else:
                break

        multiline_comment_end = re.compile(r'^.*?\*/')
        multiline_comment_start = re.compile(r'^.*?/\*')
        for i in range(start_line-1, -1, -1):
            #if there is a multiline comment above the function, grab it
            current_line = lines[i]
            if i == start_line - 1 and re.match(multiline_comment_end, current_line) == None:
                #if the line above the function is not a multiline comment end, break immediately
                break

            if re.match(multiline_comment_start, current_line) != None:
                #add the line, if the line is the start of a multiline comment, we can break
                comment_lines.insert(0, current_line)
                comment_line_counter += 1
                break

            #we've already tested the first line, so we can go ahead and insert the comment
            comment_lines.insert(0, current_line)
            comment_line_counter += 1

        #get the comments compiled together
        comments = ''.join(comment_lines)

        #this is the start of the code block
        document_start = start_line - comment_line_counter

        bracket_count = 0
        bracket_seen = False
        data = ''
        for i in range(start_line, len(lines)):
            #get the function body
            should_break = False
            current_line = lines[i]
            for char in current_line:
                if char == '{':
                    #increment the count if we find a bracket
                    bracket_seen = True
                    bracket_count += 1
                if char == '}':
                    #decrement the count if we find a closing bracket
                    bracket_count -= 1
                if bracket_seen == False:
                    #if the first bracket has not been seen, add the char to the data and don't worry about the brackets
                    data += char
                    continue
                #add to the data
                data += char
                if bracket_count == 0:
                    #if the bracket count equals 0, we have found and even number of opening and closing brackets
                    should_break = True
                    break

            if should_break:
                #breaking from the inner loop should break the outer loop
                break

        #the whole code block
        data = comments + data

        return (data, document_start, start_line)

    def parse(self, granularity='file'):
        '''
        Parse a JavaScript file's methods or classes

        Parameters:
        -----------
        granularity: str (not supported yet) | None
            - The granularity of the parsing
        '''
        if granularity == 'file':
            #the whole document
            with open(self.src_file, 'r') as file:
                data = file.read()
                file.seek(0)
                lines = file.readlines()
                doc = Document(self.src_file, self.src_file, data, (0, len(lines)))
                self.documents_ = [doc]
        elif granularity == 'class':
            anonymous_class_regex = re.compile(r'^.*?(\(|\s)(function\s*?\(.*?\)).*$', re.MULTILINE)
            named_class_regex = re.compile(r'^.*?(\(|\s)function\s*?(.+?\(.*?\)).*$', re.MULTILINE)
            with open(self.src_file, 'r') as file:
                #we need to match regex and read line so both of these
                data = file.read()
                file.seek(0)
                lines = file.readlines()

                #javascript incorporates anonymous classes so we need both of these
                anonymous_matches = re.finditer(anonymous_class_regex, data)
                named_matches = re.finditer(named_class_regex, data)

                #we can combine the two to make it easier to handle
                matches = []
                for match in anonymous_matches:
                    matches.append(match)
                for match in named_matches:
                    matches.append(match)

                for match in matches:
                    #grab all the data that is in each class, including previous comments
                    start_line = Parser._locate_line(lines, match.start(2))

                    code_block = JSParser._get_code_block(lines, start_line)
                    if code_block == None:
                        continue

                    #create a document for the method
                    self.documents_.append(Document(self.src_file, match.group(2), code_block[0], (code_block[1], code_block[2])))
        elif granularity == 'method':
            anonymous_function_regex = re.compile(r'^.*?(\(|\s|\{)(function\s*?\(.*?\)).*$', re.MULTILINE)
            named_function_regex = re.compile(r'^.*?(\(|\s|\{)function\s*?(.+?\(.*?\)).*$', re.MULTILINE)
            with open(self.src_file, 'r') as file:
                #we need to match regex and read line so both of these
                data = file.read()
                file.seek(0)
                lines = file.readlines()

                #javascript incorporates anonymous functions so we need both of these
                anonymous_matches = re.finditer(anonymous_function_regex, data)
                named_matches = re.finditer(named_function_regex, data)

                #we can combine the two to make it easier to handle
                matches = []
                for match in anonymous_matches:
                    matches.append(match)
                for match in named_matches:
                    matches.append(match)

                for match in matches:
                    #grab all the data that is in the each method, including previous comments
                    start_line = Parser._locate_line(lines, match.start(2))

                    code_block = JSParser._get_code_block(lines, start_line)
                    if code_block == None:
                        continue

                    #create a document for the method
                    self.documents_.append(Document(self.src_file, match.group(2), code_block[0], (code_block[1], code_block[2])))
        else:
            raise ValueError(f'Invalid granularity: {granularity} (supported [file|method])')

class Preprocessor:
    def __init__(self, source_folder, source_language, granularity='file'):
        '''
        Preprocessing class to clean and parse documents before passing
        on to the LSI module

        Parameters:
        -----------
        source_folder: str
            - A path name, location of the source corpus

        source_language: str
            - One or more languages to parse for

        granularity: str
            - How big the documents should be [file|class|method]
        '''
        self.src_folder = source_folder

        #parse source languages
        self.languages = []
        source_language = re.sub(r'\s', '', source_language)
        languages = source_language.split('|')
        for language in languages:
            if language == 'python':
                self.languages.append({
                    'lang': 'python',
                    'ext': '.py',
                })
            elif language == 'js':
                self.languages.append({
                    'lang': 'javascript',
                    'ext': '.js',
                })
            else:
                raise ValueError(f'Invalid source language: "{language}" (support for [python|javascript])')

        #parse granularity  
        if granularity == 'method' or granularity == 'class' or granularity == 'file':
            self.granularity = granularity
        else:
            raise ValueError(f'Invalid granularity: "{granularity}" (support for [None|class|method])')
                
    def __search(self, base_path):
        '''
        Depth first search to find source files within a nested directory

        Parameters:
        -----------
        base_path: str
            - A path name, location of the source corpus

        Returns:
        --------
        list[str]
            - A list of path names
        '''
        results = []
        extensions = set([language['ext'] for language in self.languages])
        search_files = os.listdir(base_path)

        #check each file and folder in the base path
        for filename in search_files:
            full_path = os.path.join(base_path, filename)
            if os.path.isdir(full_path):
                #recursive depth first
                for file in self.__search(full_path):
                    results.append(file)
            else:
                #check the file extension
                _, ext = os.path.splitext(full_path)
                if ext in extensions:
                    results.append(full_path)

        return results
    
    def preprocess(self):
        '''
        Preprocesses the data based on the configuration values
        '''
        src_files = self.__search(self.src_folder)
        self.documents_ = []
        for file in src_files:
            #check the type of file to get the correct parser
            _, ext = os.path.splitext(file)
            parser = None
            if ext == '.js':
                parser = JSParser(file)
            elif ext == '.py':
                parser = PythonParser(file)
            if parser == None:
                raise ValueError(f'Invalid file: {file} (support for [javascript|python])')
            
            #parse the file and add it to the corpus
            parser.parse(granularity=self.granularity)
            for document in parser.documents_:
                self.documents_.append(document)
                
class Tokenizer:
    def __init__(self):
        '''
        Tokenizes a corpus, removing stopwords and punctuation
        '''
        self.stemmer = LancasterStemmer()
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def is_punkt_(word):
        '''
        Returns true if the word is punctuation, false otherwise

        Parameters:
        -----------
        word: str
            - The word to check

        Returns:
        --------
        bool
        '''
        for char in word:
            if char not in string.punctuation:
                return False
        return True

    def __call__(self):
        '''
        Tokenizes the corpus
        '''
        raise NotImplementedError()
    
class LSATokenizer(Tokenizer):
    def __init__(self):
        '''
        Tokenizes a corpus for an LSA model
        '''
        super().__init__()

    def __call__(self, corpus):
        '''
        Tokenize the corpus

        Parameters:
        -----------
        corpus: str
            - The corpus to tokenize

        Returns:
        --------
        list[str]
            - The tokens
        '''
        return [self.stemmer.stem(token.lower()) for token in word_tokenize(corpus) if token.lower() not in self.stop_words and not Tokenizer.is_punkt_(token)]

class VectorTokenizer(Tokenizer):
    def __init__(self):
        '''
        Tokenizes a corpus for a word2vec model
        '''
        super().__init__()

    @staticmethod
    def remove_nonalpha_(token):
        '''
        Remove all non alpha characters from a token

        Parameters:
        -----------
        token: str
            - A token to clean

        Returns:
        --------
        str
            - A token with non alpha characters removed
        '''
        return re.sub(r'[^A-Za-z]', '', token)

    def __call__(self, corpus):
        '''
        Tokenize the corpus

        Parameters:
        -----------
        corpus: str
            - The corpus to tokenize

        Returns:
        --------
        list[str]
            - The tokens
        '''
        tokens = []
        for token in word_tokenize(corpus):
            clean_token = VectorTokenizer.remove_nonalpha_(token.lower())
            if len(clean_token) > 0 and clean_token not in self.stop_words:
                tokens.append(token)
        return tokens
