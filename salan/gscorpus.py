#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, argparse, logging, sys
import re, string
from gensim import corpora, models, similarities
import gensim.utils
import nltk.corpus
from IPython import embed
import fileinput
from salan.sautil import normalize_word
# import sacorpus
from salan.sacorpus import SACorpus

logger = logging.getLogger('salan')
log_filename = 'log.txt'

#Cache model files for later use, if desired
SALAN_CACHE = os.environ['SALAN_CACHE']
if SALAN_CACHE:
    lsaspacedir = os.path.join(os.environ['SALAN_CACHE'], 'lsa-spaces/')
elif not os.path.exists(lsaspacedir):
    lsaspacedir = 'lsa-spaces/'
    if not os.path.exists(lsaspacedir):
        lsaspacedir = './'        


class GSCorpus():
    def __init__(self, corpus, stopwords='', stopwordlist=[]):
        self.name = ''
        self.corpus = corpus
        self._docnum = 0
        self.doctitle = []
        self.stopwords = stopwords.upper()
        self.stopwordlist = stopwordlist
        self.savetext = False
        self.textdict = {}

    def __iter__(self):
        return self.corpus.__iter__()

    def __len__(self):
        return self.corpus.__len__()

    @classmethod
    def load(cls, name, dictname='', dirpath=lsaspacedir):
        logger.info('Loading GSCorpus %s and dictionary %s' % (name, dictname))
        corpus = corpora.MmCorpus(cls._corp_filename(name, dictname, dirpath=dirpath))
        gscorpus = GSCorpus(corpus)
        gscorpus.name = name
        if dictname:
            gscorpus.dictname = dictname
        else:
            gscorpus.dictname = name
        gscorpus.dictionary = cls.load_dict(gscorpus.dictname, dirpath=dirpath)
        return gscorpus

    def bootstrap(self):
        corpus = self.corpus
        name = self.name
        doctitles = self.load_doctitle(name) #load doctitles instead of generating all gscorpora
        docnum = 0
        for (idx, cat) in doctitles:
            logger.info('Creating bootstrap model %s for %s holding out %s, %s' % (docnum, name, idx, cat))
            gscorpus_bootstrap = self
            gscorpus_bootstrap.corpus_bst = (corpus[0:docnum], corpus[docnum], corpus[docnum+1:], idx, cat) #boot strap tuple
            yield gscorpus_bootstrap
            docnum += 1

    @classmethod
    def build_temp_corpus(cls, name, sacfname, maxsize=0, iterpara=False):
        sacorpus = SACorpus(sacfname, maxsize=maxsize, iterpara=iterpara)
        gscorpus = GSCorpus(sacorpus)
        gscorpus.name = name
        gscorpus.dictname = ['Internal Error: Temporary GSCorpus has no dictionary.']
        return gscorpus

    @classmethod
    def load_dict(cls, name, dirpath=lsaspacedir):
        dictionary = corpora.Dictionary.load(os.path.join(dirpath, name + '.dict'))
        return dictionary

    def save_dict(self, name):
        self.dictionary = corpora.Dictionary(self.corpus.tokens_by_text(stopwords=self.stopwords, stopwordlist=self.stopwordlist))
        self.dictionary.save(os.path.join(lsaspacedir, name + '.dict'))

    @classmethod
    def load_doctitle(cls, name):
        dtfilename = os.path.join(lsaspacedir, name + '-doctitle.pickle')
        logger.info('Reading document titles for %s from %s' % (name, dtfilename))
        doctitle = gensim.utils.unpickle(dtfilename)
        return doctitle

    def save_doctitle(self, name=''):
        if not name:
            name = self.name
        dtfilename = os.path.join(lsaspacedir, name + '-doctitle.pickle')
        logger.info('Writing %s document titles for %s to %s' % (len(self.doctitle), name, dtfilename))
        gensim.utils.pickle(self.doctitle, dtfilename)

    def save_corpus(self, name, type='MM', dictname=''):
        self.save_corpus_woids(name, type, dictname=dictname)

    def save_corpus_woids(self, name, type='MM', dictname=''):
        self.name = name
        if dictname:
            dictionary = self.load_dict(dictname) #call class method
        else:
            dictionary = ''
        corpora.MmCorpus.serialize(self._corp_filename(name,  dictname), [tokens for tokens in self.iter_tokens(dictionary=dictionary)])
        if self.savetext:
            tffilename =  self._text_filename(name)
            logger.info('Writing document text for %s to %s' % (name, tffilename))
            gensim.utils.pickle(self.textdict, tffilename)

    @classmethod
    def _corp_filename(cls, name, dictname="", dirpath=lsaspacedir):
        if dictname:
            return os.path.join(dirpath, name + '-corp-w-dict-' + dictname + '.mm')
        else:
            return os.path.join(dirpath, name + '-corp.mm')

    @classmethod
    def load_doctextdict(cls, name):
        tffilename = cls._text_filename(name)
        logger.info('Reading document text dict for %s from %s' % (name, tffilename))
        textdict = gensim.utils.unpickle(tffilename)
        return textdict

    @classmethod
    def _text_filename(cls, name, dirpath=lsaspacedir):
        return os.path.join(dirpath, name + '-textdict.pickle')

    def iter_tokens(self, dictionary=''):
        if not dictionary:
            dictionary = self.dictionary
        for id, cat, tokens in self.corpus.id_cat_tokens_by_text(stopwords=self.stopwords, stopwordlist=self.stopwordlist):
            self.doctitle.append((id, cat))
            if self.savetext:
                self.textdict[self._docnum] = tokens
            self._docnum += 1
            yield dictionary.doc2bow(tokens)

    def print_corpus(self, width=10):
        docnum = 0
        for (vec, dt) in zip(self.corpus, self.doctitle):
            idx = dt[0]
            cat = dt[1]
            sortedvec = sorted(vec, key=lambda x: x[1], reverse=True)
            if width != 0:
                sortedvec = sortedvec[:width]
            print('%s\t%s\t%s\t%s\n' % (str(docnum), idx, cat, '\t'.join(self.dictionary[wordid]+'('+str(int(count))+')' for wordid, count in sortedvec)))
            docnum += 1

    def print_doctitles(self):
        """Print document titles"""
        docnum = 0
        for dt in self.doctitle:
            idx = dt[0]
            cat = dt[1]
            print('%s\t%s\t%s\n' % (str(docnum), idx, cat))
            docnum += 1

    #Methods called from other modules

    def corpusdict_by_cat_id(self, rollup=0, iddf=[], doctitle={}):
        import pandas as pd
        #from collections import defaultdict
        from collections import OrderedDict #Note: requires python 2.7
        if not doctitle:
            doctitle = self.load_doctitle(self.name)
        #catdict = defaultdict(list)
        #iddict = defaultdict(list)
        #grpdict = defaultdict(list)
        catdict = OrderedDict()
        iddict = OrderedDict()
        grpdict = OrderedDict()
        for (vec, dt) in zip(self.corpus, doctitle):
            idx = dt[0]
            cat = dt[1]            
            if rollup: #everything except None and 0
                parentcats = cat.split('.')
                if rollup > 0:
                    parentcats = parentcats[0-rollup:]
                else:
                    parentcats = parentcats[:0-rollup]
                for pcat in parentcats:
                    if pcat not in catdict:
                        catdict[pcat] = []
                    catdict[pcat].append(vec)
            if rollup >= 0: #skip base categories for roll-downs
                if cat not in catdict:
                    catdict[cat] = []
                catdict[cat].append(vec)
            if rollup == 0:
                #inlcude cat for groups, only when rollup=0
                parentcats = [cat]
            if isinstance(iddf, pd.DataFrame):
                for var in list(iddf.columns):
                    varseries = iddf[var]
                    if idx not in varseries:
                        continue
                    idx_grp = varseries[idx]
                    if idx_grp not in iddict:
                        iddict[idx_grp] = []
                    iddict[idx_grp].append(vec)
                    if rollup != None:
                        for pcat in parentcats:
                            grpkey =  idx_grp + '-' + pcat
                            if grpkey not in grpdict:
                                grpdict[grpkey] = []
                            grpdict[grpkey].append(vec)
        return (iddict, catdict, grpdict)

    #Methods not used

    def corpus_matrix(self):
        self.dictionary = corpora.Dictionary(self.corpus.tokens_by_text())
        for tokens in self.corpus.tokens_by_text():
            yield self.dictionary.doc2bow(tokens)
        

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Load and Build Corpora for Gensim.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--name', dest='name', default="", help='System name of the corpus.')
    parser.add_argument('--dict', dest='dictname', default="", help='System name of the dictionary to use.')
    parser.add_argument('--maxwords', dest='maxsize', type=int, default=0, help='Set maximum number of words for each document in the corpus.')
    parser.add_argument('--print', dest='printcorpus', action="store_true", help='Print either the corpus <corpusfname> (without saving) or the corpus <name>')
    parser.add_argument('--width', dest='printwidth', default='', help='Number of characters/tokens of text document to print')
    parser.add_argument('--printdoctitles', dest='printdoctitles', action="store_true", help='Print document titles of the corpus <name>')
    parser.add_argument('--stop', dest='stopwords', default='', help='Remove stopwords from corpus: none, nltk, mallet, nonfl. Nltk and mallet also include nonfl (yeah, um). Default is nltk.')
    parser.add_argument('--stopwords', dest='stopwordlist', default='', help='Additional stopwords to include, separated by commas with no spaces')
    parser.add_argument('--embed', dest='embed', action="store_true", help='Bring up an IPython prompt after loading the corpus.')
    parser.add_argument('--corpusfname', dest='corpusfname', nargs='*', default='', help='Corpus file to read.')
    parser.add_argument('--iterline', dest='itercode', action="store_const", const=SACorpus.ITER_LINE, help='Iterate over lines in reading file, instead of sections.')
    parser.add_argument('--iterpara', dest='itercode', action="store_const", const=SACorpus.ITER_PARA, help='Iterate over paragraphs in reading file, instead of sections.')
    parser.add_argument('--iterid', dest='itercode', action="store_const", const=SACorpus.ITER_ID, help='Iterate over docs/participants/ids in reading file, instead of sections.')
    parser.add_argument('--savetext', dest='savetext', action="store_true", help='Save the document text of <corpusfname> for later access')
    args = parser.parse_args()
    if not args.itercode:
        args.itercode = SACorpus.ITER_SECT
    return args

def main(): 
    #from IPython import embed
    args = init_args()
    if args.printcorpus:
        if args.name:
            gscorpus = GSCorpus.load(args.name)
            gscorpus.doctitle = GSCorpus.load_doctitle(args.name)
            if args.printwidth:
                gscorpus.print_corpus(int(args.printwidth))
            else:
                gscorpus.print_corpus(10)
        elif args.dictname:
            gscorpus = GSCorpus.load(args.dictname)
            for (k,v) in sorted(gscorpus.dictionary.iteritems()):
                print('%s\t%s' % (k,v))
        elif args.corpusfname:
            sacorpus = SACorpus(args.corpusfname, maxsize=args.maxsize, itercode=args.itercode)
            if args.printwidth:
                sacorpus.print_corpus(slice(0,int(args.printwidth)))
            else:
                sacorpus.print_corpus(slice(0,50))
        return
    if args.printdoctitles and args.name:
        gscorpus = GSCorpus.load(args.name)
        gscorpus.doctitle = GSCorpus.load_doctitle(args.name)
        gscorpus.print_doctitles()
        return
    sacorpus = SACorpus(args.corpusfname, maxsize=args.maxsize, itercode=args.itercode)
    gscorpus = GSCorpus(sacorpus, stopwords=args.stopwords, stopwordlist=args.stopwordlist.split(','))
    if args.embed:
        embed()
        return
    if args.savetext:
        gscorpus.savetext = True
    if args.dictname:
        gscorpus.save_corpus(args.name, dictname=args.dictname)
    else:
        gscorpus.save_dict(args.name)
        gscorpus.save_corpus(args.name)
        gscorpus.save_doctitle(args.name)
    #for doc in gscorpus.corpus_matrix():
    #    print(doc)    
    if args.embed:
        embed()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
