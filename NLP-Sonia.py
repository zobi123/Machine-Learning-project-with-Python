# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:05:10 2019

@author: ZuBaiR
"""

#Tokenization:It is the process for breaking the text data into smaller pieces for analysis. The pieces of words, sentence are called token.

from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out." 

# Sentence tokenizer
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))

# Stemming.: converting words to their base form like playing to play, branded to brand, kept to kept
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 
        'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# Create various stemmer objects
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

# Create a list of stemmer names for display
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names), 
        '\n', '='*68)

# Stem each word and display the output
for word in input_words:
    output = [word, porter.stem(word), 
            lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output))

#Chunks: Diving input data into chunks. it is not the same as tokenization while chunks is need to be meaningful
    import nltk
nltk.download('brown')
import numpy as np
from nltk.corpus import brown
# Split the input text into chunks, where
# each chunk contains N words
def chunker(input_data, N):
    input_words = input_data.split(' ')
    output = []

    cur_chunk = []
    count = 0
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []

    output.append(' '.join(cur_chunk))

    return output 

if __name__=='__main__':
    # Read the first 12000 words from the Brown corpus
    input_data = ' '.join(brown.words()[:12000])

    # Define the number of words in each chunk 
    chunk_size = 700

    chunks = chunker(input_data, chunk_size)
    print('\nNumber of text chunks =', len(chunks), '\n')
    for i, chunk in enumerate(chunks):
        print('Chunk', i+1, '==>', chunk[:50])
#POS (Parts of Speech)
#Example 1
import nltk
nltk.download('averaged_perceptron_tagger')
sent=" Timothy is a natural when it comes to drawing"
sent_tokens=word_tokenize(sent)
for token in sent_tokens:
    print(nltk.pos_tag([token]))

# Example 2
sent2="sonia is eating the delicious cake"
sent2_tokens=word_tokenize(sent2)
for token in sent2_tokens:
    print(nltk.pos_tag([token]))

# Named Entity Recognition

# Example 1
from nltk import ne_chunk
NE_Sent= 'The US President stays in the WHITE HOUSE'
NE_tokens=word_tokenize(NE_Sent)
import nltk
nltk.download('words')
nltk.download('maxent_ne_chunker')
NE_tags=nltk.pos_tag(NE_tokens)
NE_NER=ne_chunk(NE_tags)
print(NE_NER)

# Example 2
import numpy as np
new="The big cat late at the little mouse who was after the fresh cheese"
new_tokens=nltk.pos_tag(word_tokenize(new))
print(new_tokens)
grammar_np=r"NP:{<DT>?<JJ>*<NN>}"
chunk_parser=nltk.RegexpParser(grammar_np)
chunk_result=chunk_parser.parse(new_tokens)
print(chunk_result)

