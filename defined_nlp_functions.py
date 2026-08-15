import string
import emoji
import re
import pandas
import spacy 

nlp=spacy.load('en_core_web_sm')

def clean_data(series):
    
    # Lowercasing
    output=series.str.lower()
    
    # Removing Punctuation
    output=output.str.translate(str.maketrans('','',string.punctuation))
    
    # Removing Spaces 
    output=output.apply(lambda x: re.sub(r'\s+',' ',x)).str.strip()
    
    # Converting emojis to text
    output=output.apply(lambda x: emoji.demojize(x)).str.replace(':','')

    return output

def token_lemma_nonstop(text):
    
    
    doc=nlp(text)
    output=[token.lemma_ for token in doc if not token.is_stop]
    output = ' '.join(output)
    return output

def pos(text,pos_list=['NOUN','PROPN','VERB','NUM','SYM',]):
    doc=nlp(text)
    output=[token.text for token in doc if token.pos_ in pos_list]
    output=' '.join(output)
    return output

def nlp_pipeline(series):
    
    output=clean_data(series)
    output=output.apply(token_lemma_nonstop)
    output=output.apply(pos)
    return output