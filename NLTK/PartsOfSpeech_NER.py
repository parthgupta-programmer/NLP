from nltk.tokenize import word_tokenize
import nltk

corpus = """
Mahendra Singh Dhoni was born in Ranchi, India, on July 7, 1981.
MS Dhoni played for the Indian cricket team and Chennai Super Kings.
He led India to victory in the 2011 Cricket World Cup.
Chennai Super Kings won the IPL title under his captaincy in 2010 and 2011.
"""

# Tokenize the corpus into words
words=word_tokenize(corpus)

# Perform Part-of-Speech (POS) tagging on the tokenized words
pos_tags=nltk.pos_tag(words)

print(f'\nPart-of-Speech (POS) Tags: \n{pos_tags}')

# Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

entities=nltk.ne_chunk(pos_tags).draw()  # Draw the named entities in a tree format