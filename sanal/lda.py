#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, argparse, logging, sys
import re, string
from array import array
import numpy as np
from gensim import corpora, models, similarities, matutils
from gensim.models import wrappers #0.12 - needed but not documented
import myldamallet
import gensim.utils
import matplotlib.pyplot as plt
from corpus import SACorpus, GSCorpus, cleanup_text
from IPython import embed
from lsa import SABatch, read_docquery_file
import pandas as pd
logger = logging.getLogger('sa')
log_filename = 'log.txt'

defaultcorpus = '../corpus-responses/All.txt'
lsaspacedir = '/Users/markgraves/Projects/semanticAnalysis-Local/lsa-spaces/'
ldamodeldir = '/Users/markgraves/Projects/semanticAnalysis-Local/lda-topic-models/'

mallet_path = '/Users/markgraves/Projects/semanticAnalysis/topics/mallet/mallet-2.0.7/bin/mallet'
#mallet_path = '/Users/markgraves/Projects/semanticAnalysis/topics/mallet/mallet-2.0.8RC2/bin/mallet' #not supported

stoplistfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lib/', "stoplist-mallet-en.txt")
        
class LDA:
    def __init__(self):
        pass

    def create_model(self, name, numtopics=10, gsname='', save=True, passes=1, iterations=50, chunksize=2000, evalevery=10, cmds='', bootstrap=''):
        self.name = name
        self.numtopics = numtopics
        #set up logging to file; TODO: pull out into function
        logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if save:
            logFileName = self._model_filename()+'-log.txt'
        else:
            logFileName = 'log-build-'+name+'.txt'
        fileHandler = logging.FileHandler(logFileName, mode='w')
        fileHandler.setFormatter(logFormatter)
        logging.getLogger().addHandler(fileHandler) #need default logger to get gensim log messages
        if not gsname:
            logger.error('LDA build needs GSCorpus for building.')
            return 'ERROR-LDA-MODEL' #TODO: throw error
        elif bootstrap:
            gscorpus = GSCorpus.load(gsname) #NOTE: default is lsa-space
            self.bootstrap_base_gscorpus = gscorpus
            gscorpora = GSCorpus.load_bootstrap(gsname) #NOTE: generator
            self.modellist = []
        else:
            gscorpus = GSCorpus.load(gsname) #NOTE: default is lsa-space
            gscorpora = [gscorpus]
        for gscorpus in gscorpora:
            self.gscorpus = gsname
            self.dictionary = gscorpus.dictionary #should be the same for every corpus
            if self.hdp:
                model = models.HdpModel(gscorpus.corpus, id2word=self.dictionary) # do not pass chunksize, as default 256 differs
            elif self.mallet: #lda
                model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=gscorpus.corpus, num_topics=numtopics, id2word=self.dictionary, iterations=iterations)
            elif self.mymallet: #lda
                prefix = self._model_filename()
                logger.info('Using Mallet with cmd string: %s' % cmds)
                if 1:
                    doclabels = [cat for (id, cat) in GSCorpus.load_doctitle(gsname)]
                else:
                    doclabels = []
                model = myldamallet.LdaMallet(mallet_path, corpus=gscorpus.corpus, num_topics=numtopics, id2word=self.dictionary, iterations=iterations, cmds=cmds, doclabels=doclabels)
            else:
                model = models.LdaModel(gscorpus.corpus, num_topics=int(numtopics), id2word=self.dictionary, update_every=1, passes=passes, iterations=iterations, chunksize=chunksize, eval_every=evalevery, alpha='auto')
            if bootstrap:
                self.modellist.append(model)
                #self.model = model #STUB
                #Maybe todo: save
            else:
                self.model = model
                if save:
                    model.save(self._model_filename())
        logging.getLogger().removeHandler(fileHandler)
        if bootstrap:
            return modellist
        else:
            return model

    def load_model(self, name='', numtopics=10, dictname=''):
        if not dictname:
            dictname = name
        if name:
            logger.info('LDA: Loading model %s with %s topics' % (name, numtopics))
            self.name = name
            self.numtopics = numtopics #default
            if self.hdp:
                self.model = gensim.models.hdpmodel.HdpModel.load(self._model_filename())
                self.numtopics = -1
            elif self.mallet:
                self.model = gensim.models.wrappers.ldamallet.LdaMallet.load(self._model_filename())
            elif self.mymallet:
                self.model = myldamallet.LdaMallet.load(self._model_filename())
            else: #LDA
                model = gensim.models.ldamodel.LdaState.load(self._model_filename(), mmap='r')
                ldastate = gensim.models.ldamodel.LdaState.load(self._model_filename()+'.state', mmap='r') #gensim should do this automatically vers 0.12.1; TODO: from gensim log info, looks like explicitly not saved with lda as attribute, along with dispatcher
                self.model = model
                self.model.state = ldastate
        else:
            logger.error('LDA: Model name not specified.')
            return 'ERROR-LDA-MODEL' #TODO: throw error
        self.dictionary = self.model.id2word
        return self.model

    def _model_filename(self, name='', numtopics=''):
        if not name:
            name = self.name
        if self.hdp:
            algtext = 'hdpmodel'
            return os.path.join(ldamodeldir, name + '-' + algtext + '.hdp')
        elif self.mallet:
            if not numtopics:
                numtopics = self.numtopics
            algtext = 'malletmodel'
            return os.path.join(ldamodeldir, name + '-' + algtext + '-' + str(numtopics) + '.gsmallet')
        elif self.mymallet:
            if not numtopics:
                numtopics = self.numtopics
            algtext = 'malletmodel'
            return os.path.join(ldamodeldir, name + '-' + algtext + '-' + str(numtopics) + '-cmds.gsmallet')
        else:
            if not numtopics:
                numtopics = self.numtopics
            algtext = 'ldamodel'
            return os.path.join(ldamodeldir, name + '-' + algtext + '-' + str(numtopics) + '.lda')

    def calculate_perplexity(self, gsname, outfile='', xlswb=None):
        import xlwt
        model = self.model
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
            if not outfile:
                outfile='out-perplexity-%s.xls' % gsname
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws1 = wb.add_sheet('Perplexity')
        ws1.write(0, 0, 'ID', style_reg)
        ws1.write(0, 1, 'Cat', style_reg)
        ws1.write(0, 2, 'Log Word Bound', style_reg)
        ws1.write(0, 3, 'Perplexity', style_reg)
        gscorpus = GSCorpus.load(gsname) #NOTE: default is lsa-space
        doctitles = GSCorpus.load_doctitle(gsname)
        perplexitylist = []
        rownum = 1
        for doc, dt in zip(gscorpus, doctitles):
            perplexitywb = model.log_perplexity([doc])
            logger.debug('Calculated log perplexity per-word bound for document number %s in %s for %s, %s: %s' % (rownum, gsname, dt[0], dt[1], perplexitywb))
            perplexity = np.exp2(-perplexitywb)
            ws1.write(rownum, 0, dt[0], style_reg)
            ws1.write(rownum, 1, dt[1], style_reg)
            ws1.write(rownum, 2, perplexitywb, style_reg)
            ws1.write(rownum, 3, perplexity, style_reg)
            rownum += 1
            perplexitylist.append(perplexitywb)
        if not xlswb:
            wb.save(outfile)
        return perplexitylist

    def bootstrap_model(self, name, numtopics=10, gsname='', save=False, passes=1, iterations=50, chunksize=2000, evalevery=10, cmds='', bootstrap='1', outfile='', xlswb=None):
        import xlwt, itertools
        self.name = name
        self.numtopics = numtopics
        #set up logging to file; TODO: pull out into function
        logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if save:
            logFileName = self._model_filename()+'-log.txt'
        else:
            logFileName = 'log-build-'+name+'.txt'
        fileHandler = logging.FileHandler(logFileName, mode='w')
        fileHandler.setFormatter(logFormatter)
        logging.getLogger().addHandler(fileHandler) #need default logger to get gensim log messages
        if not gsname:
            logger.error('LDA build needs GSCorpus for building.')
            return 'ERROR-LDA-MODEL' #TODO: throw error
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
            if not outfile:
                outfile='out-bootstrap-%s.xls' % gsname
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws1 = wb.add_sheet('Bootstrap')
        ws1.write(0, 0, 'ID', style_reg)
        ws1.write(0, 1, 'Cat', style_reg)
        ws1.write(0, 2, 'Log Word Bound', style_reg)
        ws1.write(0, 3, 'Perplexity', style_reg)
        gscorpus = GSCorpus.load(gsname) #NOTE: default is lsa-space
        #self.bootstrap_base_gscorpus = gscorpus
        gscorpora = gscorpus.bootstrap() #NOTE: generator
        self.modellist = []
        perplexitylist = []
        rownum = 1
        for gscorpus in gscorpora: #NOTE: generator
            self.gscorpus = gsname
            self.dictionary = gscorpus.dictionary #should be the same for every corpus
            corpus_bst = gscorpus.corpus_bst
            model = models.LdaModel(list(itertools.chain(corpus_bst[0], corpus_bst[2])), num_topics=int(numtopics), id2word=self.dictionary, update_every=1, passes=passes, iterations=iterations, chunksize=chunksize, eval_every=evalevery, alpha='auto')
            self.modellist.append(model)
            perplexitywb = model.log_perplexity([corpus_bst[1]])
            logger.info('Calculated bootstrap log perplexity per-word bound for held out document %s, %s: %s' % (corpus_bst[3], corpus_bst[4], perplexitywb))
            perplexity = np.exp2(-perplexitywb)
            ws1.write(rownum, 0, corpus_bst[3], style_reg)
            ws1.write(rownum, 1, corpus_bst[4], style_reg)
            ws1.write(rownum, 2, perplexitywb, style_reg)
            ws1.write(rownum, 3, perplexity, style_reg)
            rownum += 1
            perplexitylist.append(perplexitywb)
            if save: #TODO: add docnum
                model.save(self._model_filename())
        if not xlswb:
            wb.save(outfile)
        logging.getLogger().removeHandler(fileHandler)
        return self.modellist, perplexitylist

    def text2dist(self, text):
        tokens = cleanup_text(text, lemmatize=True) # lemmatize for lda
        bow = self.dictionary.doc2bow(tokens)
        dist_lda = self.model[bow]
        return dist_lda

    def query(self, text):
        dist_lda = self.text2dist(text)
        return dist_lda

    def model2corpus(self, freqmult=100, max_num_words=40):
        #convert topic model to prototype documents
        #TODO: handle alphas
        topics = self.model.show_topics(num_topics=self.numtopics, formatted=False, num_words=max_num_words)
        alphas = self.model.alpha
        dictionary = self.dictionary
        corpus = []
        topicnum = 0
        for t in topics:
            doc = []
            if any(alphas):
                alpha = alphas[topicnum]
            else:
                alpha = 1
            for (prob, word) in t:
                wordid = dictionary.token2id[word]
                wc = int(prob * freqmult)
                if wc > 0:
                    doc.append((wordid, wc))
            corpus.append(doc)
            topicnum += 1
        return corpus

    def write_topics_xls(self, outfile='out-lda-topics.xls', xlswb=None, num_words=100):
        import xlwt
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws1 = wb.add_sheet('Topics')
        tnum = 0
        for topic in self.model.show_topics(num_topics=self.numtopics, formatted=False, num_words=num_words):
            colnum1 = tnum * 2
            ws1.write(0, colnum1, 'Topic '+str(tnum), style_reg)
            ws1.write(0, colnum1 + 1, 'Prob '+str(tnum), style_reg)
            rownum1 = 1
            for (prob, word) in topic:
                ws1.write(rownum1, colnum1, word, style_reg)
                ws1.write(rownum1, colnum1 + 1, float(prob), style_reg)
                rownum1 += 1
            tnum += 1
            if tnum > 127:
                logger.error('Too many variables (>127). Stopping.')
                break
        if not xlswb:
            wb.save(outfile)
        return wb

    def display_lda_matplot(self, num_top_words=10, probcutoff=0.01):
        import matplotlib.pyplot as plt
        num_topics = self.numtopics
        topics = self.model.show_topics(num_topics=num_topics, formatted=False)
        alphas = self.model.alpha
        maxprob = np.max([p for t in topics for (p,w) in t ])
        fontsize_base = 70 / maxprob
        plotmaxgridwidth = 7
        if num_topics % plotmaxgridwidth == 0:
            plotheight = num_topics / plotmaxgridwidth
        else:
            plotheight = num_topics / plotmaxgridwidth + 1
        plotwidth = min(plotmaxgridwidth, num_topics)
        plotnum=0
        for topicnum in range(num_topics):
            if 0: #programmatic flag for subselections
                selected_topics = [2, 8, 11, 12, 20]
                if t not in selected_topics: #include only these topics, numbering starts with 0
                    continue
                plt.subplot(1, len(selected_topics), plotnum + 1)  # plot numbering starts with 1
            else:
                plt.subplot(plotheight, plotwidth, topicnum + 1)  # plot numbering starts with 1
            plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
            plt.xticks([])  # remove x-axis markings ('ticks')
            plt.yticks([]) # remove y-axis markings ('ticks')
            plt.title('Topic %s ' % topicnum, fontsize=12)
            i=0
            for (prob, word) in topics[topicnum][:num_top_words]:
                if prob < probcutoff:
                    break
                plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base*prob)
                i+=1
            plotnum += 1
        #plt.tight_layout(pad=0.3, hpad=0.5)
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9, wspace=0.05, hspace=0.18)
        plt.suptitle('LLDA Topics (top %s topics with word prob cutoff=%s; max prob=%s)' % (num_top_words, probcutoff, maxprob))
        plt.show()

def topicdist_euclidean(topic1, topic2, num_words=25, normconst=1, squared=False):
    #not used (or tested)
    sum = 0    
    tdiff = topicdiff(topic1, topic2, num_words=num_words, sortresult=False)
    for (p,w) in tdiff:
        sum += p**2
    if squared:
        return np.sqrt(sum)
    else:
        return sum

def topicdist(topic1, topic2, num_words=25, normconst=1):
    #TODO: num_words not used
    words1 = [w for (p,w) in topic1]
    words2 = [w for (p,w) in topic2]
    wordset = set().union(words1, words2)
    dict1 = {w:p for (p,w) in topic1}
    dict2 = {w:p for (p,w) in topic2}
    sum = 0
    for word in wordset:
        if word in dict1:
            p1 = dict1[word]
        else:
            p1 = 0
        if word in dict2:
            p2 = dict2[word]
        else:
            p2 = 0
        #sum += (p1 - p2)**2
        sum += p1*p2*(p1 - p2)
    return float(sum / normconst)

def topicdiff(topic1, topic2, num_words=25, sortresult=True):
    #TODO: num_words not used
    words1 = [w for (p,w) in topic1]
    words2 = [w for (p,w) in topic2]
    wordset = set().union(words1, words2)
    dict1 = {w:p for (p,w) in topic1}
    dict2 = {w:p for (p,w) in topic2}
    tdiff = []
    for word in wordset:
        if word in dict1:
            p1 = dict1[word]
        else:
            p1 = 0
        if word in dict2:
            p2 = dict2[word]
        else:
            p2 = 0
        tdiff.append((p1 - p2, word))
    if sortresult:
        tdiff.sort(key=lambda p_w: p_w[0], reverse=True)
    return tdiff

def compare_model_topics (model1, model2, numtopics1=10, numtopics2=10, num_words=25):
    topics1 = model1.show_topics(num_topics=numtopics1, formatted=False, num_words=num_words)
    topics2 = model2.show_topics(num_topics=numtopics2, formatted=False, num_words=num_words)
    result = []
    for t1 in topics1:
        row = []
        for t2 in topics2:
            tdist = topicdist(t1, t2, num_words, normconst=1)
            row.append(tdist)
        result.append(row)
    return result

def models_by_cat_id(gsname, iterations=50, passes=1, usemallet=True, rollup=0, iddf=[]):
    from corpus import GSCorpus
    from collections import OrderedDict #Note: requires python 2.7
    gscorpus = GSCorpus.load(gsname)
    (iddict, catdict, grpdict) = gscorpus.corpusdict_by_cat_id(rollup=rollup, iddf=iddf)
    modeldict = OrderedDict()
    for (ctype, cname, cdict) in zip(['ID', 'CAT', 'GRP'], ['ID Group', 'Category', 'ID-Cat Group'], [iddict, catdict, grpdict]):
        modeldict[ctype] = OrderedDict()
        for (key, corpus) in cdict.iteritems():
            logger.info('Building LDA model for %s %s with %s documents' % (cname, key, len(corpus)))
            if usemallet:
                modeldict[ctype][key] = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=1, id2word=gscorpus.dictionary, iterations=iterations)
            else:
                modeldict[ctype][key] = models.LdaModel(corpus, num_topics=1, id2word=gscorpus.dictionary, update_every=1, passes=passes, iterations=iterations, chunksize=2000, eval_every=1, alpha='auto')
    return modeldict

def write_modeldict_xls(modeldict, outfile='out-lda-models.xls', xlswb=None, num_words=100, fullmatrix=True, rankorder=True):
    import xlwt
    from collections import defaultdict
    from scipy.stats import ks_2samp
    if xlswb:
        wb = xlswb
    else:
        wb = xlwt.Workbook()
    style_reg = xlwt.easyxf("font: height 240")
    style_bold = xlwt.easyxf("font: height 240, bold 1")
    ws1 = wb.add_sheet('Topics')
    ws2 = wb.add_sheet('Topic Dist')
    ws2.write(0,0, "sum(p_i*p_j*(p_i - p_j))", style_reg)
    ws3 = wb.add_sheet('Max Diff')
    colnum1 = 0
    completed = []
    topicnum = 0
    pwdict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    numdict = sum([len(td) for (m, td) in modeldict.items()])
    logger.info('Creating Excel workbook comparing %s models.' % numdict)
    rank_rowoffset = numdict + 5
    for (mtype, typedict) in modeldict.items():
        if not typedict:
            continue
        ws1.write(0,colnum1, mtype, style_bold)
        for (mname, model) in typedict.items():
            ws1.write(1,colnum1, mname, style_reg)
            ws_p = wb.add_sheet((mname + ' Pairwise Diff')[:31]) #31 max chars in XLS sheet name
            pwdict[mname]['SHEET'] = ws_p
            rownum_pw = topicnum + 2
            pwdict[mname]['ROWNUM'] = rownum_pw
            ws_p.write(0,0, mname, style_reg)
            ws_p.write(0,1, 'Euclidean Distance', style_reg)
            ws_p.write(1,0, 'actual prob', style_reg)
            if rankorder:
                ws_p.write(rank_rowoffset ,0, 'Rank Order', style_bold)    
                #ws_p.write(rank_rowoffset ,1, 'KS Z p', style_reg)    
                ws_p.write(rank_rowoffset + 1,0, 'actual rank', style_reg)
            #Assume only one topic per model
            topic = model.show_topics(num_topics=1, formatted=False, num_words=num_words)[0]
            rownum1 = 2
            for (prob, word) in topic:
                colnum_p = rownum1
                ws1.write(rownum1, colnum1, word, style_reg)
                ws1.write(rownum1, colnum1 + 1, float(prob), style_reg)
                ws_p.write(0, colnum_p, word, style_reg)
                ws_p.write(1, colnum_p, float(prob), style_reg)
                ws_p.write(rownum_pw, colnum_p, 0, style_reg)
                pwdict[mname][word]['PROB'] = float(prob)
                pwdict[mname][word]['COLNUM'] = colnum_p
                if rankorder:
                    rank = colnum_p - 1
                    pwdict[mname][word]['RANK'] = rank
                    ws_p.write(rank_rowoffset + 1, colnum_p, rank, style_reg)     
                    ws_p.write(rank_rowoffset + rownum_pw, colnum_p, 0, style_reg)               
                rownum1 += 1
            #Calculate comparisons
            rownum2 = int(colnum1/2)+1
            ws2.write(rownum2, 0, mname, style_reg)
            ws2.write(0, rownum2, mname, style_reg) #symmetric
            ws3.write(rownum2, 0, mname, style_reg)
            ws3.write(0, rownum2, mname, style_reg) #symmetric
            if fullmatrix:
                #write diagonal
                ws2.write(rownum2, rownum2, 0, style_reg)
                ws3.write(rownum2, rownum2, 0, style_reg)
            ws_p.write(rownum_pw,0, mname, style_reg)
            ws_p.write(rownum_pw,1, 0, style_reg) #rms=0 for self
            if rankorder:
                ws_p.write(rank_rowoffset + rownum_pw,0, mname, style_reg)
            rownum_c = pwdict[mname]['ROWNUM']
            colnum2 = 1
            for (mname2, topic2) in completed:
                #write lower pairwise Compare with tdist
                tdist = topicdist(topic, topic2)
                ws2.write(rownum2, colnum2, float(tdist), style_reg)
                if fullmatrix:
                    ws2.write(colnum2, rownum2, float(tdist), style_reg)
                #write MaxDiff and Pairwise sheets
                maxdiff = 0.0
                maxdiff_word = ''
                ws_c = pwdict[mname2]['SHEET']
                ws_c.write(rownum_c, 0, mname, style_reg)
                rownum_pw = pwdict[mname2]['ROWNUM']
                ws_p.write(rownum_pw, 0, mname2, style_reg) 
                if rankorder:
                    ws_c.write(rank_rowoffset + rownum_c, 0, mname, style_reg)
                    ws_p.write(rank_rowoffset + rownum_pw, 0, mname2, style_reg) 
                rms_sum = 0
                rank2 = 1
                ks_list = []
                for (p, w) in topic2:
                    if w not in pwdict[mname]:
                        rms_sum += p**2 # diff from 0
                        continue
                    colnum_pw = pwdict[mname][w]['COLNUM']  
                    #ws_p.write(rownum_pw,colnum_pw, w, style_reg) #DEBUG Statement
                    #ws_p.write(rownum_pw,colnum_pw, float(p), style_reg) #DEBUG Statement
                    p1 = pwdict[mname][w]['PROB']
                    ws_p.write(rownum_pw, colnum_pw, xlwt.Formula("%s - %s" % (float(p), p1)), style_reg)
                    #write converse entry
                    colnum_c = pwdict[mname2][w]['COLNUM']
                    ws_c.write(rownum_c, colnum_c, xlwt.Formula("%s - %s" % (p1, float(p))), style_reg)
                    rms_sum += (p - p1)**2
                    if abs(p1 - p) > maxdiff:
                        maxdiff = p1 - p
                        maxdiff_word = w
                    if rankorder:
                        #ks_list.append((p1, p))
                        rank1 = pwdict[mname][w]['RANK']
                        ws_p.write(rank_rowoffset + rownum_pw, colnum_pw, xlwt.Formula("%s - %s" % (rank2, rank1)), style_reg)    
                        ws_c.write(rank_rowoffset + rownum_c, colnum_c, xlwt.Formula("%s - %s" % (rank1, rank2)), style_reg)     
                        rank2 += 1
                ws3.write(rownum2, colnum2, float(maxdiff), style_reg)
                ws3.write(colnum2, rownum2, maxdiff_word, style_reg)
                rms = float(np.sqrt(rms_sum)) #Note: Euclidean distance, not RMS
                ws_p.write(rownum_pw, 1, rms, style_reg)
                ws_c.write(rownum_c, 1, rms, style_reg)
                if rankorder:
                    #TODO: ks_list only has values that occur in both lists. Not correct.
                    #(ks_z, ks_z_p) = ks_2samp(zip(*ks_list)[0], zip(*ks_list)[1])
                    #ws_p.write(rank_rowoffset + rownum_pw, 1, ks_z_p, style_reg)
                    #ws_c.write(rank_rowoffset + rownum_c, 1, ks_z_p, style_reg)
                    pass
                colnum2 += 1
            topicnum += 1
            completed.append((mname, topic))
            colnum1 += 2
            if colnum1 > 254:
                logger.error('Too many variables (>127). Stopping.')
                break
    if not xlswb:
        wb.save(outfile)
    return wb


def print_docquery_file(fname, yield_code=1, slice=None):
    for sectname, text in read_docquery_file(fname, yield_code):
        if slice:
            print("%s\t%s" % (sectname, string.strip(text[slice])))
        else:
            print("%s\t%s" % (sectname, string.strip(text)))

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Load and Run LDA (from gensim).')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--build', dest='buildcmd',  action="store_true", help='Build a LDA topic model called <model> of the <GSCorpus> with <num> topics.')
    parser.add_argument('--gsname', dest='gsname', default="", help='Name of the GS Corpus.')
    parser.add_argument('--model', dest='modelname', default="", help='Name of the LDA model.')
    parser.add_argument('--numtopics', dest='numtopics', type=int, default=10, help='Number of topics in the LDA model.')
    parser.add_argument('--hdp', dest='hdpcmd',  action="store_true", help='Use hierarchical dirichlet process topic model.')
    parser.add_argument('--mallet', dest='malletcmd',  action="store_true", help='Use mallet topic model.')
    parser.add_argument('--mymallet', dest='mymalletcmd',  default="", help='Use modified mallet topic model, with <arg> as mallet training commands.')
    parser.add_argument('--passes', dest='numpasses', type=int, default=1, help='Number of passes to make when building in the LDA model.')
    parser.add_argument('--iterations', dest='numiterations', type=int, default=50, help='Number of iterations to make when building in the LDA model.')
    parser.add_argument('--chunksize', dest='chunksize', type=int, default=2000, help='Number of documents to consider before updating model.')
    parser.add_argument('--evalevery', dest='evalevery', type=int, default=10, help='Number of updates before estimating log perplexity.')
    parser.add_argument('--printtopics', dest='printtopics',  type=int, default=0, help='Print top <arg> words for LDA topic model <model> with <numtopics> topics.')
    #parser.add_argument('--nearest', dest='nearesttext', default="", help='Nearest using arg as doctext against model.')
    #parser.add_argument('--nearestfile', dest='nearestfile', default="", help='Nearest using arg as filename of doctext against model.')
    parser.add_argument('--query', dest='querytext', default="", help='Query using arg as doctext against GSCorpus <gsname> in the LSA space <space> with <dims> dimensions.')
    parser.add_argument('--queries', dest='queryfile', default="", help='Query using arg as filename of doctext against GSCorpus <gsname> in the LSA space <space> with <dims> dimensions.')
    parser.add_argument('--printdocqueryfile', dest='dqfile', default="", help='Print doc query file.')
    #parser.add_argument('--pairwisecos', dest='pairwisecos', action="store_true", help='Calculate pairwise cosines for queries against every document in the GSCorpus <gsname>')
    parser.add_argument('--xlsfile', dest='xlsfile', default="", help='Create Excel file with topics.')
    parser.add_argument('--outfile', dest='outfile', default="", help='Output file for queries.')
    #parser.add_argument('--modelfile', dest='modelfile', default="", help='Name of the LDA model file.')
    parser.add_argument('--idvars', dest='varfile', default="", help='File of auxillary variables by which to group ids (tab-delimited table keyed off of ids).')
    parser.add_argument('--matplot', dest='matplot', action="store_true", help='Display topics using matplot.')
    parser.add_argument("--probcutoff", dest="probcutoff", type=float, help="Minimum word probability to display on plot", default=0.01)
    parser.add_argument('--compmodel', dest='compmodelname', default="", help='Compare <model> with LDA model name <arg>.')
    parser.add_argument('--numtopics2', dest='numtopics2', type=int, default=10, help='Number of topics in the second LDA model.')
    parser.add_argument('--bycat', dest='bycat', default="", help='Compare topic models by category and include <arg> levels of rolled-up categories (negative numbers roll down from top).')
    parser.add_argument('--bootstrap', dest='bootstrap', default="", help='Bootstrap comparison of individuals in the group.')
    parser.add_argument('--perplexity', dest='perplexity', action="store_true", help='Calculate perplexity of <gsname> against existing model <model> with <numtopics>.') 
    parser.add_argument('--embed', dest='embed', action="store_true", help='Bring up an IPython prompt after loading the model.')
    args = parser.parse_args()
    return args

def main():
    from IPython import embed
    args = init_args()
    if args.dqfile:
        print_docquery_file(args.dqfile, 1)#, slice(0,25))
        return 1
    lda = LDA()
    if args.hdpcmd:
        lda.hdp = True
    else:
        lda.hdp = False
    if args.malletcmd:
        lda.mallet = True
    else:
        lda.mallet = False
    if args.mymalletcmd:
        lda.mymallet = True
    else:
        lda.mymallet = False
    #create or load model
    iddf=''
    if args.varfile:
        #Note: To create topic for every participant, duplicate id column as another variable
        if os.path.isfile(args.varfile):
            iddf = pd.read_table(args.varfile, header=0, index_col=0)
        else:
            logger.error('Auxvar file does not exist: %s' % args.varfile)
    if args.bycat:
        rollup = int(args.bycat) # convert to int here, so "0" gets passed in
        #Note: 0 and 1 have similar outcomes, but not identical, eg, past.ten and future.ten will be merged on rollup 1
    else:
        rollup = None
    if args.varfile or args.bycat:
        modeldict = models_by_cat_id(args.gsname, iterations=args.numiterations, passes=args.numpasses, usemallet=args.malletcmd, rollup=rollup, iddf=iddf)
        if args.xlsfile:
            write_modeldict_xls(modeldict, outfile=args.xlsfile)
        if args.embed:
            embed()
        return
    if args.bootstrap:
        lda.bootstrap_model(name=args.modelname, numtopics=args.numtopics, gsname=args.gsname, passes=args.numpasses, iterations=args.numiterations, chunksize=args.chunksize, evalevery=args.evalevery, cmds=args.mymalletcmd, bootstrap=args.bootstrap)
        if args.embed:
            embed()
        return
    if args.buildcmd:
        if args.perplexity:
            logger.error('Can only calculate perplexity against existing model. Use build and perplexity cmds separately.')
        lda.create_model(name=args.modelname, numtopics=args.numtopics, gsname=args.gsname, passes=args.numpasses, iterations=args.numiterations, chunksize=args.chunksize, evalevery=args.evalevery, cmds=args.mymalletcmd, bootstrap=args.bootstrap)
    else:
        if args.bootstrap: #no longer accessible, but left
            logger.error('Loading bootstrap models not implemented. Use build cmd.')
            return
        lda.load_model(name=args.modelname, numtopics=args.numtopics)
    if args.printtopics > 0:
        if lda.hdp:
            alphas = lda.model.hdp_to_lda()[0]
            print('alpha\tTopic')
            for (a, topic) in zip(alphas, lda.model.show_topics(-1, topn=args.printtopics)):
                if a > 0.0005: #arbitrary cutoff
                    print(str(a)+'\t'+topic)
        else:
            print('TopicNo\talpha\tTopic')
            for i in range(lda.numtopics):
                print(str(i)+'\t'+str(lda.model.alpha[i])+'\t'+lda.model.print_topic(i,topn=args.printtopics))
    if args.perplexity:
        lda.calculate_perplexity(args.gsname)
    if args.xlsfile:
        if lda.hdp:
            logger.error('Excel file writing not implemented for HDP.')
        else:
            lda.write_topics_xls(args.xlsfile)
    if args.matplot:
        lda.display_lda_matplot(num_top_words=args.printtopics, probcutoff=args.probcutoff)
    if args.querytext:
        result = lda.query(args.querytext)
        print('Topic\tProbability')
        for x in result:
            print('\t'.join(str(i) for i in x))
        return 1
    if args.compmodelname:
        lda2 = LDA()
        lda2.hdp = lda.hdp
        lda2.mallet = lda.mallet
        lda2.mymallet = lda.mymallet
        lda2.load_model(name=args.compmodelname, numtopics=args.numtopics2)
        modelcompare = compare_model_topics(lda.model, lda2.model, lda.numtopics, lda2.numtopics)
        print(modelcompare)
    if args.queryfile:
        sabatch = SABatch(lda)
        sabatch.queries_from_file(in_query_file=args.queryfile, in_auxvar_file=args.varfile, outfile=args.outfile)
        return 1
    if args.embed:
        embed()

    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
