#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, argparse, logging, sys
import re, string
from string import strip, split, replace, find
import load_word_count
from load_word_count import normalize_word, inc_word_count, print_word_count, init_word_count
from segment_sentences import normalize_sent
from compare_score import read_scores
import nltk
# from nltk.corpus import stopwords
from IPython import embed

logger = logging.getLogger('salan')
log_filename = 'log.txt'

defaultcorpus = '../corpus.txt'

stoplistfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lib/', "stoplist-mallet-en.txt")

def load_stoplist(fname=stoplistfile):
    with open(fname) as fp:
        stoplist = fp.read().split()
    return stoplist

class NGram:

    def __init__(self, corpus='', maxn=3):
        if not corpus:
            self.corpus = defaultcorpus
        else:
            self.corpus = corpus
        self.maxn = maxn

    def count_ngrams(self, fname='', maxn=-1, stem_plurals=True):
        sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
        if not fname:
            fname = self.corpus
        if maxn == -1:
            maxn = self.maxn
        init_word_count()
        id_from_file = 'UNKNOWN'
        id_from_header = 'NONE'
        qdict = {}
        category = ''
        with open(fname) as fp:
            for line in fp:
                if line.isspace(): continue
                if line[0] == '%':
                    if line[0:7] == '%Q.LNI.':
                        category = strip(line[7:]).lower()
                        #qdict[id].append(category)
                        #response, do not continue
                    elif line[0:8] == '%Q.CHAP.':
                        category = strip(line[3:]).lower()
                    elif line[0:6] == '%BEGIN':
                        id_from_header = strip(line[7:])
                        id = id_from_header
                    elif line[0:4] == '%END':
                        id_from_header = strip(line[4:])
                        if id_from_header != id:
                            logger.error('Inconsistent IDs: %s in %%END statement for %s.', id_from_header, id)
                    elif line[0:8] == '%COMMENT':
                        continue
                    continue
                prefix = find(line, ':')
                if prefix > 0:
                    if line[0] == 'Q':
                        continue
                    elif line[0:3] == 'ID:': 
                        id_from_transcript = strip(line[4:])
                        if id_from_header != id:
                            logger.warn('Inconsistent IDs: %s in ID statement for %s.', id_from_transcript, id)
                        continue
                    elif line[0:12] == 'PARTICIPANT:': 
                        line = strip(line[12:])
                        #response, do not continue
                    elif line[0:5].upper() == 'DATE:': 
                        continue
                    # interviewer prompt or prologue
                    elif line[0:9] == 'PROLOGUE:': 
                        continue
                    elif (line[0:7] == 'ANDREA:') | (line[0:6] == 'KEVIN:') | (line[0:12].upper() == 'INTERVIEWER:'): 
                        continue
                    else:
                        logger.warn('Unrecognized speaker %s in for id %s.', line[0:prefix], id)
                        continue
                #response line
                sents = sent_tokenizer.tokenize(line)
                if stem_plurals:
                    sents = [normalize_sent(s) for s in sents]
                sents_tokenized = [inc_word_count(s, id, category) for s in sents]
                for i in range(1,maxn):
                    for s in sents_tokenized:
                        ngrams=find_ngrams(s,i+1)
                        inc_word_count('', id, category, word_list=[' '.join(g) for g in ngrams])
                #print(len(sents_tokenized))
        return load_word_count.word_count_dict

    def print_word_count_wids(self, wcdict, total, ids=[], categories=[], outfname='out-wc.txt', fcutoff=0, ccutoff=0, sentfname=''):
        #from IPython import embed
        if sentfname:
            (sentmeasures, sentdict) = read_scores(sentfname) 
        else:
            sentdict = {}
        with open(outfname, 'w') as ofp:
            #write header line
            ofp.write('Word\tWordCount\tFrequency')
            if sentdict:
                ofp.write('\t' + '\t'.join(sentmeasures))
            if ids:
                ofp.write('\t' + '\t'.join(ids))
            if categories:
                ofp.write('\t' + '\t'.join(categories))
            ofp.write('\n')
            #write total lone
            ofp.write('TOTAL\t%s\t1' % total)
            if sentdict:
                ofp.write('\t' + '\t'.join('0' for m in sentmeasures))
            if ids:
                ofp.write('\t' + '\t'.join(str(load_word_count.get_total_word_count_by_id(id)) for id in ids))
            if categories:
                ofp.write('\t' + '\t'.join(str(load_word_count.get_total_word_count_by_cat(cat)) for cat in categories))
            ofp.write('\n')
            #write line for each word
            for w in sorted(wcdict):
                c=wcdict[w]
                f=c/total
                if f >= fcutoff and c>=ccutoff:
                    if sentdict:
                        if w in sentdict:
                            ofp.write('%s\t%s\t%s' % (w, c, c/total)) 
                            ofp.write('\t' + '\t'.join(str(sentdict[w][m]) for m in sentmeasures))
                        else:
                            w_list = w.split(' ')
                            if len(w_list) == 1:
                                continue
                            sent_list = []
                            for tok in w_list:
                                if tok in sentdict:
                                    sent_list.append([float(sentdict[tok][m]) for m in sentmeasures])
                            if sent_list:
                                sent_values = [sum(i) / len(sent_list) for i in zip(*sent_list)]
                                ofp.write('%s\t%s\t%s' % (w, c, c/total)) 
                                ofp.write('\t' + '\t'.join(str(val) for val in sent_values))
                            else:
                                ofp.write('%s\t%s\t%s' % (w, c, c/total))
                                ofp.write('\t' + '\t'.join('0' for m in sentmeasures))
                    else:
                        ofp.write('%s\t%s\t%s' % (w, c, c/total)) 
                    if ids:
                        ofp.write('\t' + '\t'.join(str(load_word_count.get_word_count_dict_with_id(w, id)) for id in ids))
                    if categories:
                        ofp.write('\t' + '\t'.join(str(load_word_count.get_word_count_dict_with_cat(w, cat)) for cat in categories))
                    ofp.write('\n')

        
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Calculate N-Grams from a corpus.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--corpusfname', dest='corpusfname', default='', help='Corpus file to read.')
    parser.add_argument('--maxn', dest='maxn', type=int, default=3, help='Maximum N for n-grams.')
    parser.add_argument('--mino', dest='mino', type=int, default=3, help='Minimum number of occurences for the n-grams.')
    parser.add_argument('--printids', dest='printids', action="store_true", help='Print the ngram counts for each ID')
    parser.add_argument('--printcat', dest='printcat', action="store_true", help='Print the ngram counts for each Category')
    parser.add_argument('--sentiment', dest='sentfname', default='', help='Sentiment file to read.')
    args = parser.parse_args()
    return args

def main(): 
    #from IPython import embed
    args = init_args()
    module = NGram(args.corpusfname, maxn=args.maxn)
    wc_dict = module.count_ngrams(args.corpusfname)
    outfname = 'out-ngrams-%s' % args.maxn
    if args.sentfname:
        outfname += '-sent'
    if args.printids:
        outfname += '-wids'
        ids = sorted(load_word_count.get_word_count_ids())
        outfname += '.txt'
        module.print_word_count_wids(wc_dict, load_word_count.total_word_count, outfname=outfname, ccutoff=args.mino, ids=ids, sentfname=args.sentfname)
    elif args.printcat:
        outfname += '-wcat'
        cats = sorted(load_word_count.get_word_count_cats())
        outfname += '.txt'
        module.print_word_count_wids(wc_dict, load_word_count.total_word_count, outfname=outfname, ccutoff=args.mino, categories=cats, sentfname=args.sentfname)
    else:
        outfname += '.txt'
        print_word_count(wc_dict, load_word_count.total_word_count, outfname=outfname, ccutoff=args.mino, sentfname=args.sentfname)
    print(outfname)
    return wc_dict


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
