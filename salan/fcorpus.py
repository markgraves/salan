#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os, argparse, sys
import re, string

from gensim import corpora, models, similarities
import gensim.utils
import nltk.corpus
from IPython import embed
import fileinput

from sautil import normalize_word, get_stoplist

LOGGER = logging.getLogger(__name__)

class SACorpus:

    ITER_LINE = 1
    ITER_PARA = 1 #same as line for LNI corpus
    ITER_PARA2 = 3 #requires a blank line; not implemented
    ITER_LNI = 4
    ITER_CHAP = 5 #not implemented
    ITER_ID = 6

    def __init__(self, corpusfname, maxsize=0, itercode=0):
        #10/17/15: changed corpusfname to be a list
        if hasattr(corpusfname, 'lower'):
            #then its a string
            corpusfname = [corpusfname]
        self.corpusfname = corpusfname
        self.maxsize = maxsize
        self.itercode = itercode

    def __iter__(self):
        if self.itercode == self.ITER_LINE:
            logger.debug('Iterating over SACorpus %s by line' % self.corpusfname)
            return self.iter_line()
        elif self.itercode == self.ITER_PARA:
            logger.debug('Iterating over SACorpus %s by paragraph' % self.corpusfname)
            return self.iter_para()
        elif self.itercode == self.ITER_LNI:
            logger.debug('Iterating over SACorpus %s by section' % self.corpusfname)
            return self.iter_lni()
        elif self.itercode == self.ITER_ID:
            logger.debug('Iterating over SACorpus %s by participant' % self.corpusfname)
            return self.iter_participant()
        else:
            logger.warn('Unknown iterator specification for SACorpus %s. Iterating by paragraph.' % self.corpusfname)
            return self.iter_para()
    
    def __len__(self): #Warning: Very inefficient if this is called unnecessarily
        logger.debug('Calculating length for ' + self.name)
        return len(list(self.tokens_by_text()))

    def iter_line(self):
        return self._iter_helper(self.ITER_LINE)

    def iter_para(self): #para is line
        return self._iter_helper(self.ITER_PARA)

    def iter_lni(self): 
        return self._iter_helper(self.ITER_LNI)

    def iter_participant(self): 
        return self._iter_helper(self.ITER_ID)

    def _iter_helper(self, yield_code=ITER_PARA, categories=[]):
        maxsize = self.maxsize
        text_queue = []
        queue_words = 0
        if yield_code == self.ITER_CHAP: #categories, TODO: not implemented
            catdict = {c : [] for c in categories}
        #for corpusfname in self.corpusfname:
        #for loop in iterator causes problems with later zip(*)
        #corpusfname = self.corpusfname[0]
        #with open(corpusfname) as fp:
        #with fileinput.input(files=self.corpusfname) as fp: #with content manager not supported in 2.7
        fp = fileinput.input(files=self.corpusfname)
        try:
            corpusfname = fileinput.filename()
            id = ''
            category = ''
            for line in fp:
                if line.isspace():
                    continue
                if line[0] == '%':
                    if text_queue:
                        if yield_code == self.ITER_LNI: # yield lni section
                            yield id, category, ' '.join(text_queue)
                            text_queue = []
                            queue_words = 0
                    if line[0:3] == '%Q.':
                        #qdict[id].append(category)
                        if line[0:7] == '%Q.LNI.':
                            category = string.strip(line[7:]).lower()
                        elif line[0:6] == '%Q.NP.':
                            category = string.strip(line[6:]).lower()
                        else:
                            category = string.strip(line[3:]).lower()
                        #response, do not continue
                    if line[0:5] == '%DOC.':
                        category = string.strip(line[5:]).lower()
                    elif line[0:6] == '%BEGIN':
                        id_from_header = string.strip(line[7:])
                        id = id_from_header
                    elif line[0:4] == '%END':
                        id_from_header = string.strip(line[4:])
                        if id_from_header != id:
                            logger.error('Inconsistent IDs: %s in %%END statement for %s.', id_from_header, id)
                        if yield_code == self.ITER_ID and text_queue: # yield entire participant text
                            yield id, '', ' '.join(text_queue)
                            text_queue = []
                            queue_words = 0
                    elif line[0:8] == '%COMMENT':
                        continue
                    continue
                if string.find(line, ':') > 0:
                    logger.error("Invalid character ':' in SACorpus file. File may need preprocessing: %s. --Line: %s", self.corpusfname, line)
                    sys.exit(1)
                line = string.strip(line)
                if maxsize > 0:
                    line_split = line.split()
                    line_numwords = len(line_split)
                    if (yield_code == self.ITER_LINE) or (yield_code == self.ITER_PARA): # yield line
                        #output line in <maxsize> segments
                        if line_numwords < maxsize:
                            yield id, category, line
                        else:
                            for start in range(0, line_numwords, maxsize):
                                yield id, category, ' '.join(line_split[start:start+maxsize])
                            remainder = line_split[start+maxsize:]
                            if remainder:
                                yield id, category, ' '.join(line_split[start+maxsize:])
                    else:# (yield_code == self.ITER_LNI or yield_code == self.ITER_ID):
                        #using text_queue
                        if line_numwords + queue_words < maxsize:
                            #continue building queue
                            queue_words += line_numwords
                            text_queue.append(line)
                        else:                         
                            if queue_words > 0:
                                #flush queue
                                if category:
                                    yield id, category, ' '.join(text_queue)
                                else:
                                    yield id, 'maxwords', ' '.join(text_queue)
                                text_queue = []
                                queue_words = 0
                            if line_numwords < maxsize:
                                #start building queue again
                                queue_words += line_numwords
                                text_queue.append(line)
                            else:
                                #line itself too long, split at maxsize
                                for start in range(0, line_numwords, maxsize):
                                    yield id, category, ' '.join(line_split[start:start+maxsize])
                                remainder = line_split[start+maxsize:]
                                if remainder:
                                    queue_words += len(remainder)
                                    text_queue.append(' '.join(remainder))
                else:
                    #do not track size
                    #response line
                    if (yield_code == self.ITER_LINE) or (yield_code == self.ITER_PARA): # yield line
                        yield id, category, line
                    else:
                        text_queue.append(string.strip(line))
        finally:
            fp.close()
            #embed()

    def print_corpus(self, slice=None):
        for id, category, line in self:
            if slice:
                print("%s\t%s\t%s" % (id, category, string.strip(line[slice])))
            else:
                print("%s\t%s\t%s" % (id, category, string.strip(line)))
    
    def tokens_by_text(self, stopwords='', stopwordlist=[]): 
        id, cat, tokens = zip(*self.id_cat_tokens_by_text(stopwords=stopwords, stopwordlist=stopwordlist))
        return tokens

    def id_cat_tokens_by_text(self, stopwords='', stopwordlist=[]): #MAYBE TODO: Could add filters as args here
        (stopwords, stoplist) = get_stoplist(stopwords)
        logger.debug('Filtering %s stopwords from %s.' % (stopwords, self.corpusfname))
        if stopwordlist:
            logger.debug('Filtering additional stopwords from %s: %s.' % (self.corpusfname, stopwordlist))
            stoplist += stopwordlist
        re_token = re.compile(r"[^a-zA-Z0-9 ]+")
        for id, category, text in self:
            text = re_token.sub('', text) 
            word_list = text.split()
            tokens = []
            for w in word_list:
                # Normalization is before stoplist removal in case normalization yields stopword
                nw = normalize_word(w, lemmatize=True)
                if nw not in stoplist:
                    tokens.append(nw)
            yield id, category, tokens

    def print_tokens(self):
        for tokens in self.tokens_by_text():
            print(tokens)



def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Load and Build Corpora.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--corpusfname', dest='corpusfname', nargs='*', default='', help='Corpus file to read.')
    parser.add_argument('--maxwords', dest='maxsize', type=int, default=0, help='Set maximum number of words for each document in the corpus.')
    parser.add_argument('--print', dest='printcorpus', action="store_true", help='Print either the corpus <corpusfname> (without saving) or the corpus <name>')
    parser.add_argument('--width', dest='printwidth', default='', help='Number of characters/tokens of text document to print')
    parser.add_argument('--iterline', dest='itercode', action="store_const", const=SACorpus.ITER_LINE, help='Iterate over lines in reading file, instead of sections.')
    parser.add_argument('--iterpara', dest='itercode', action="store_const", const=SACorpus.ITER_PARA, help='Iterate over paragraphs in reading file, instead of sections.')
    parser.add_argument('--iterid', dest='itercode', action="store_const", const=SACorpus.ITER_ID, help='Iterate over docs/participants/ids in reading file, instead of sections.')
    parser.add_argument('--embed', dest='embed', action="store_true", help='Bring up an IPython prompt after loading the corpus.')
    parser.add_argument('--stop', dest='stopwords', default='', help='Remove stopwords from corpus: none, nltk, mallet, nonfl. Nltk and mallet also include nonfl (yeah, um). Default is nltk.')
    args = parser.parse_args()
    if not args.itercode:
        args.itercode = SACorpus.ITER_LNI
    return args

def main(): 
    #from IPython import embed
    args = init_args()
    if args.printcorpus:
        if args.corpusfname:
            sacorpus = SACorpus(args.corpusfname, maxsize=args.maxsize, itercode=args.itercode)
            if args.printwidth:
                sacorpus.print_corpus(slice(0,int(args.printwidth)))
            else:
                sacorpus.print_corpus(slice(0,50))
        return
    sacorpus = SACorpus(args.corpusfname, maxsize=args.maxsize, itercode=args.itercode)
    if args.embed:
        embed()
        return

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
