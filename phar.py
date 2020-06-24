from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

import spacy
nlp = spacy.load('en', parser=False) 

def tokenize(text):
        sents = sent_tokenize(text)
        return [word_tokenize(sent) for sent in sents]

def removeStopWords(words):
        customStopWords = set(stopwords.words('english')+list(punctuation))
        return [word for word in words if word not in customStopWords]

def get_related(word):
    filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
    return similarity[:5]

def get_semantic_similar_phrases(sentence, n=5):
    synonyms = []
    for pos, word in enumerate(tokenize(sentence)[0]):
        word_syns = [w.lower_ for w in get_related(nlp.vocab[word.decode('utf-8')])]
        synonyms.append(word_syns)

    sentence_tokens = tokenize(sentence)[0]
    for token_index, token in enumerate(sentence_tokens):
        selected_syn_set = synonyms[token_index]
        for synonym in selected_syn_set:
            sentence_tokens[token_index] = synonym
            print(' '.join(sentence_tokens))
        sentence_tokens = tokenize(sentence)[0]
        print("")

get_semantic_similar_phrases('who is the founder of microsoft')