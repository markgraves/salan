#!/usr/bin/python
from __future__ import division, print_function
import nltk
import logging

#logger = logging.getLogger(__name__)
logger = logging.getLogger('salan')
log_filename = 'log.txt'

word_pos = {}
#Note: initializing normalize_word_cache supports the creation of wordsets for a collection of words
normalize_word_cache = {}

normalize_word_wnl = nltk.stem.wordnet.WordNetLemmatizer()
def normalize_word(word, lowercase=True, lemmatize=True):
    global normalize_word_cache
    global word_pos
    if lowercase:
        word_lower = word.lower()
    else:
        word_lower = word # needed for segment_sentence
    if word_lower in normalize_word_cache:
        return normalize_word_cache[word_lower]
    newword = word_lower
    if not lemmatize:
        return newword
    treebank_tag = nltk.pos_tag([word])[0][1]
    if ( len(newword) > 4 ) and ( treebank_tag == 'NNS' ):
        # Only lemmatize plural nouns, leave verbs alone
        wnpos = get_wordnet_pos(treebank_tag)
        if wnpos:
            newword = normalize_word_wnl.lemmatize(newword, wnpos)
        #if newword != word_lower:
            #print('Changing %s to %s' % (word_lower, newword))
        word_pos[newword] = nltk.pos_tag([newword])[0][1]
    else:
        word_pos[word_lower] = treebank_tag
    normalize_word_cache[word_lower] = newword
    return newword

def get_wordnet_pos(treebank_tag):
    key = treebank_tag[0:1]
    if key =='N':
        return nltk.corpus.wordnet.NOUN
    elif key =='V':
        return nltk.corpus.wordnet.VERB
    elif key =='J':
        return nltk.corpus.wordnet.ADJ
    elif key =='R':
        return nltk.corpus.wordnet.ADV
    else:
        return ''



