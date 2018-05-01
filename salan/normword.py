"""Normalize word"""

import logging
import nltk

LOGGER = logging.getLogger(__name__)

NORMWORD_WNL = None
NORMWORD_POS = {}
NORMWORD_CACHE = {}


def init_normword_wnl():
    "Initialize word net lemmatizer for normword"
    global NORMWORD_WNL
    NORMWORD_WNL = nltk.stem.wordnet.WordNetLemmatizer()
    return NORMWORD_WNL


def normalize_word(word, lowercase=True, lemmatize=True):
    "Normalize word by stripping plural nouns"
    global NORMWORD_CACHE
    global NORMWORD_POS
    if NORMWORD_WNL is None:
        init_normword_wnl()
    if lowercase:
        word = word.lower()
    if word in NORMWORD_CACHE:
        return NORMWORD_CACHE[word]
    if not lemmatize:
        return word
    treebank_tag = nltk.pos_tag([word])[0][1]
    newword = word
    if ( len(newword) > 4 ) and ( treebank_tag == 'NNS' ):
        #  Only lemmatize plural nouns, leave verbs alone
        wnpos = get_wordnet_pos(treebank_tag)
        if wnpos:
            newword = NORMWORD_WNL.lemmatize(newword, wnpos)
        if newword != word:
            LOGGER.debug('Changing %s to %s' % (word, newword))
        NORMWORD_POS[newword] = nltk.pos_tag([newword])[0][1]
    else:
        NORMWORD_POS[word] = treebank_tag
    NORMWORD_CACHE[word] = newword
    return newword


def get_wordnet_pos(treebank_tag):
    "Translate from treebank pos tags to wordnet pos tags."
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



