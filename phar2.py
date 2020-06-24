from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import random


def tag(sentence):
    words = word_tokenize(sentence)
    words = pos_tag(words)
    return words

def paraphraseable(tag):
    return  tag == 'VB' or tag == 'VBZ' or tag.startswith('JJ') #tag.startswith('NN') or

def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

def synonymIfExists(sentence):
    word_list = []
    for (word, t) in tag(sentence):
        print(word,t)
        if paraphraseable(t):
            syns = synonyms(word, t)
            print((list(syns)))
            if syns:
                print("T")

                if len(syns) > 1:
                    alt = random.choice(list(syns))
                    word_list.append(alt)
                    print(alt)
        else:
            word_list.append(word)
  

    print("s\n ",' '.join(word for word in word_list))
    return ' '.join(word for word in word_list)
        

file = "temp.txt" #"text.txt"
import os
os.path.isfile(file)
f = open(file ,encoding='utf-8' )
text = f.read()

synonymIfExists("Our medical strategies mirror our commitment to providing affordable quality care to member corporations before health becomes an issue.")