from nltk.stem import PorterStemmer,RegexpStemmer,SnowballStemmer # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

corpus = "Studies studying studied relational conditional generously running happiness organization maximum"
words=word_tokenize(corpus)

# Stemming using PorterStemmer,RegexStemmer and SnowballStemmer

for word in words:
    print(f'\n{word} :-')
    print(f'PorterStemmer: {PorterStemmer().stem(word)}')
    print(f'RegexpStemmer: {RegexpStemmer("ing$|s$|e$|able$").stem(word)}')
    print(f'SnowballStemmer: {SnowballStemmer("english").stem(word)}')
