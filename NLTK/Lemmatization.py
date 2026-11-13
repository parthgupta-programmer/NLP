# Lemmatization returns a root word rather than the root stem, the output of stemming.

from nltk.stem import WordNetLemmatizer # type: ignore

print(f'Going --> {WordNetLemmatizer().lemmatize("going")}')
print(f'Going --> {WordNetLemmatizer().lemmatize("going",pos="v")}')
