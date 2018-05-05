"""Semantic analysis utilities"""

import os
import re
import logging
import nltk

LOGGER = logging.getLogger(__name__)

NORMWORD_WNL = None
NORMWORD_POS = {}
NORMWORD_CACHE = {}

MALLET_STOPLISTFILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lib/', "stoplist-mallet-en.txt")
STOPLIST_NONFL = ['yeah', 'um']


## Normalize word

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


def init_normword_wnl():
    "Initialize word net lemmatizer for normword"
    global NORMWORD_WNL
    NORMWORD_WNL = nltk.stem.wordnet.WordNetLemmatizer()
    return NORMWORD_WNL


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


## Stop word processing

# stopwords are upper case: NONE, NLTK, MALLET, NONFL; may also return UNKNOWN-NONE or FILE
# TODO: could add caching for MALLET
def get_stoplist(stopwords, includenonfl=True):
    "Get stopword list for NONE, NLTK, MALLET, NONFL, or filename. Default is NLTK."
    if stopwords == '':  # default
        stopwords = 'NLTK'
    if stopwords == 'NONE':
        return ('NONE', [])
    # Handle lists
    if stopwords == 'NLTK':
        stoplist = nltk.corpus.stopwords.words('english')  # use nltk stop words
    elif stopwords == 'MALLET':
        stoplist = load_stoplist(MALLET_STOPLISTFILE)
    elif stopwords == 'NONFL':
        stoplist = []
    else:  # unknown
        if os.path.isfile(stopwords):
            stoplist = load_stoplist(stopwords)
            return ('FILE', stoplist)
        else:
            return ('UNKNOWN-NONE', [])
    if includenonfl or (stopwords == 'NONFL'):
        stoplist.extend(STOPLIST_NONFL)  # extend nonfl, ntlk, and mallet with nonfl
    return (stopwords, stoplist)


def load_stoplist(fname=MALLET_STOPLISTFILE):
    "Load stoplist file"
    with open(fname) as fp:
        stoplist = fp.read().split()
    return stoplist


## Stand alone utilities

# copied from id_cat_tokens_by_text in fcorpus and simplified
def cleanup_text(text, stoplist=nltk.corpus.stopwords.words('english'), re_token=re.compile(r"[^a-zA-Z0-9 ]+"), lemmatize=True, lowercase=True):
    text = re_token.sub('', text) 
    word_list = text.split()
    tokens = [normalize_word(w, lemmatize=lemmatize, lowercase=lowercase) for w in word_list if w not in stoplist]
    return tokens


