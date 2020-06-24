from nltk.corpus import wordnet as wn
sent = "Obama met Putin the previous week"

for i in sent.split():
    possible_senses = wn.synsets(i)
    print ((possible_senses), possible_senses)



for i in sent.split():
    possible_senses = wn.synsets(i)
    if possible_senses:
        print (i, possible_senses[0].lemma_names)
    else:
        print (i)