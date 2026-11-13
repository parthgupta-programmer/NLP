# Stop words are high-frequency, low-meaning words (such as "the," "is," "at," and "and").
# These are frequently filtered out during natural language processing (NLP) and information retrieval tasks.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer

corpus='Cricket is a popular bat-and-ball sport. It is played between two teams of eleven players. The game takes place on an oval field with a central rectangular pitch. Cricket originated in England and is now a global phenomenon. It is hugely popular in countries like India, Australia, and the United Kingdom. Matches are played in different formats, including five-day Test matches. Shorter formats include One Day Internationals and Twenty20 games. Teams take turns batting to score runs. The other team bowls and fields to get batters out and limit runs. Cricket teaches teamwork, patience, and sportsmanship. It unites millions of passionate fans around the world.'

sentences=sent_tokenize(corpus)

lemmatizer=WordNetLemmatizer()

print(f'\nSentences before removing StopWords: \n{[sentence for sentence in sentences]} ')


for i in range(len(sentences)):
    words=word_tokenize(sentences[i])
    words=[lemmatizer.lemmatize(word,pos='v') for word in words if word not in stopwords.words('english')]
    sentences[i]=' '.join(words)

print(f'\nSentences after removing StopWords: \n{[sentence for sentence in sentences]} ')
