from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
corpus = """Hello, My name is Ishan Sharma. 
I am trying the complete generative AI course with Langchain and huggingface in udemy"""

print(sent_tokenize(corpus))
print(word_tokenize(corpus))


## Stemming:
## Stemming is a process of converting a word to its root form or stem.
## Disadvantages:
## For some of the words, the result may not be proper giving inaccurate results
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer

porter_stemmer = PorterStemmer() # less accurate, results in the formation of words which changes the meaning or are not valid 
snowball_stemmer = SnowballStemmer(language='english') # more accurate
regexp_stemmer = RegexpStemmer('ing$|s$|e$|able$') # works with regexp

words = ['eating', 'eaten', 'programming', 'falling', 'predictive']

for word in words:
    print(f"{word} ---- PorterStemmer ----> {porter_stemmer.stem(word)}")
    print(f"{word} ---- SnowballStemmer ----> {snowball_stemmer.stem(word)}")
    print(f"{word} --- RegexpStemmer ----> {regexp_stemmer.stem(word)}")

