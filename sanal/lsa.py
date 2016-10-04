#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, argparse, logging, sys
import re, string
from array import array
import numpy as np
from gensim import corpora, models, similarities
import gensim.utils
import matplotlib.pyplot as plt
from corpus import SACorpus, GSCorpus, cleanup_text
from IPython import embed

logger = logging.getLogger('salan')
log_filename = 'log.txt'


SALAN_CACHE = os.environ['SALAN_CACHE']
if SALAN_CACHE:
    lsaspacedir = os.path.join(os.environ['SALAN_CACHE'], 'lsa-spaces/')
else:
    lsaspacedir = '.'

#Will need to update with mallet stoplist locations
stoplistfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lib/', "stoplist-mallet-en.txt")

defaultcorpus = '../corpus.txt'

stoplistfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lib/', "stoplist-mallet-en.txt")
        
class LSA:
    def __init__(self):
        pass

    def create_model(self, name, numtopics=10, save=True, weighted=True, production=0):
        self.name = name
        self.numtopics = numtopics
        self.production = production
        gscorpus = GSCorpus.load(name)
        gsdict = gscorpus.dictionary
        if weighted:
            if weighted == 'TFIDF':
                weighted_model = models.TfidfModel(gscorpus)
            else:
                weighted = 'LE'
                weighted_model = models.LogEntropyModel(gscorpus)
            self.weighted_name = weighted
            corpus_weighted = weighted_model[gscorpus]
            self.weighted = weighted_model
        else:
            corpus_weighted = gscorpus # no weighting
            self.weighted = None
        if production == 1:
            lsi = models.LsiModel(corpus_weighted, id2word=gsdict, num_topics=numtopics, onepass=False, power_iters=4, extra_samples=300, chunksize=50000)
        else:
            lsi = models.LsiModel(corpus_weighted, id2word=gsdict, num_topics=numtopics)
        self.model = lsi
        #self.print_transformed_corpus(gscorpus, corpus_weighted)
        #embed()
        #corpus_lsi = lsi[corpus_weighted]
        #self.corpus = corpus_lsi
        if save:
            #fname = self._corpus_filename(name, numtopics)
            #corpora.MmCorpus.serialize(fname, corpus_lsi)
            lsi.save(self._model_filename(name, numtopics, weighted, production))
            if weighted:
                weighted_model.save(self._weighted_filename(gscorpus.name, weighted))
        return lsi

    def _model_filename(self, name, dims, weighted='', production=0):
        fname = name + '-lsi-' + str(dims)
        if weighted:
            fname += '-' + weighted
        if production:
            fname += '-prod'
        fname += '.lsi'
        return os.path.join(lsaspacedir, fname)

    def load_model(self, name, numtopics=10, weighted=''):
        logger.info('LSA: Loading model %s with %s dimensions and weight %s' % (name, numtopics, weighted))
        self.name = name
        self.numtopics = numtopics
        self.dictionary = corpora.Dictionary.load(os.path.join(lsaspacedir, name + '.dict'))
        #fname = self._corpus_filename(name, numtopics)
        #if not os.path.isfile(fname):
        #    logger.warn('LSI file does not exist: %s' % fname)
        #    return False
        weighted_fname = ''
        if weighted:
            weighted_fname = self._weighted_filename(name, weighted)
        else:
            #if weight not specified, try TFIDF (because it was specified at some time), then default LE
            #TODO: Update all code to handle 'NONE' rather than occassionally default to ''
            for weighted in ['TFIDF', 'LE', '', 'NONE']:
                weighted_fname = self._weighted_filename(name, weighted)
                if os.path.isfile(weighted_fname):
                    break
        if os.path.isfile(weighted_fname):
            if weighted == 'LE':
                self.weighted_name = weighted
                self.weighted = models.LogEntropyModel.load(weighted_fname)
            elif weighted == 'TFIDF':
                self.weighted_name = weighted
                self.weighted = models.TfidfModel.load(weighted_fname)
            else:
                logger.warn('No weighted model found for %s: %s' % (weighted, weighted_fname))
                self.weighted_name = 'NONE'
                self.weighted = None
        else:
            logger.warn('No weighted model file for %s found.' % weighted)
            self.weighted_name = 'NONE'
            self.weighted = None
        #If production version, load it
        modelfilename = self._model_filename(name, numtopics, self.weighted_name, 1)
        self.production = 1
        if not os.path.isfile(modelfilename):
            modelfilename = self._model_filename(name, numtopics, self.weighted_name, 0)
            self.production = 0
        self.model = models.LsiModel.load(modelfilename)
        return self.model

    def add_corpus(self, gscorpus='', gsname=''):
        """Add a corpus <gscorpus> to the LSA model, of if empty, load from <gsname>."""
        #Not working. Need to change model to Similarity and figure out how to update dictionary.
        #Causes a segmentation fault in add_document
        if not hasattr(self, 'model'):
            try:
                if self.name and self.numtopics:
                    logger.info('Autoload LSA model for %s with %s dims' \
                                % (self.name, self.numtopics))
                    #could be specified in another module to be loaded here
                    self.load_model(self.name, self.numtopics)
            except AttributeError:
                logger.error('Model needs to be loaded or specified for add_corpus.')
                return -1
        if not gscorpus:
            gscorpus = GSCorpus.load(gsname) #may need to use lsa's dictionary
        embed()
        self.model.add_documents(gscorpus.corpus)
        embed()

    #not used
    def _corpus_filename(self, name, numtopics):
        return os.path.join(lsaspacedir, name + '-lsicorp-' + str(numtopics) + '.mm')

    def _index_filename(self, gsname, weighted=''):
        if weighted:
            if weighted == 'TFIDF':
                weightedsuffix = 'TFIDF'
            else:
                weightedsuffix = 'LE'
            weightedsuffix = '-' + weightedsuffix
        else:
            weightedsuffix = ''
        return os.path.join(lsaspacedir, gsname + '-corpindx-' + self.name + '-' + str(self.numtopics) + weightedsuffix + '.index')

    def _weighted_filename(self, gsname, weighted):
        if weighted == 'TFIDF':
            weightedsuffix = 'TFIDF.wghtmodel'
        elif weighted == 'LE':
            weightedsuffix = 'LE.wghtmodel'
        elif not weighted or weighted == 'NONE':
            return ''
        else:
            logger.warn('Unknown weight %s. Using LE for %s.' % (weighted, gsname))
            weightedsuffix = 'LE.wghtmodel'
        weightedsuffix = '-' + weightedsuffix
        return os.path.join(lsaspacedir, gsname + '-corpindx-' + self.name + '-' + str(self.numtopics) + weightedsuffix)

    #not used
    def reload_corpus(self):
        fname = self._corpus_filename(self.name, self.numtopics)
        self.corpus = corpora.MmCorpus(fname)
        return self.corpus

    def index_corpus(self, gscorpus, name='', save=True, ms=True, weighted=True):
        #Note: doctitle is not loaded
        #Note: ms=True is currently the only way to create index that can later be loaded
        if not name:
            name = gscorpus.name
        weightedsuffix = ''
        if self.weighted:
            gscorpus_weighted = self.weighted[gscorpus]
        else:
            gscorpus_weighted = gscorpus # no weighting
        if ms:
            corp = self.model[gscorpus_weighted]
            logger.info('LSA: Indexing %s against %s with Matrix Similarity' % (gscorpus.name, self.name))
            index = similarities.MatrixSimilarity(corp, num_features=int(self.numtopics)) #Note: In memory
        else:
            logger.info('LSA: Indexing %s against %s with Similarity' % (gscorpus.name, self.name))
            index = similarities.Similarity(self._index_filename(name, weighted=weighted), self.model[gscorpus_weighted], num_features=int(self.numtopics)) 
        logger.info('LSA: Completed indexing')
        self.index = index
        if save:
            index.save(self._index_filename(name, weighted=weighted))
        return index

    def load_index(self, gsname, weighted=True, load_doctitle=True):
        #Note: Also loads doctitle
        self.index = similarities.MatrixSimilarity.load(self._index_filename(gsname, weighted=weighted))
        self.gsname = gsname
        self.doctitle = GSCorpus.load_doctitle(gsname)
        return self.index

    def print_transformed_corpus(self, gscorpus, tcorpus, width=10):
        docnum = 0
        if not gscorpus.doctitle:
            gscorpus.doctitle = GSCorpus.load_doctitle(gscorpus.name)
        for (vec, dt) in zip(tcorpus, gscorpus.doctitle):
            idx = dt[0]
            cat = dt[1]
            sortedvec = sorted(vec, key=lambda x: x[1], reverse=True)
            if width != 0:
                sortedvec = sortedvec[:width]
            print('%s\t%s\t%s\t%s\n' % (str(docnum), idx, cat, '\t'.join(gscorpus.dictionary[wordid]+'('+str(float(count))+')' for wordid, count in sortedvec)))
            docnum += 1

    def text2vec(self, text, weighted=True):
        #from corpus import cleanup_text
        dictionary = self.dictionary
        #tokens = text.lower().split()  #4/28/16: Bug fix: was not stripping punctionation properly
        tokens = cleanup_text(text, stoplist=[], lowercase=True, lemmatize=False)
        vec_bow = dictionary.doc2bow(tokens)
        if weighted and self.weighted:
            vec_bow = self.weighted[vec_bow]
        vec_lsi = self.model[vec_bow]
        return vec_lsi

    def query(self, text, sort=True, translate=False, weighted=True):
        vec_lsi = self.text2vec(text, weighted)
        sims = self.index[vec_lsi] # perform a similarity query against the corpus
        if sort:
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
        doctitle = self.doctitle
        if translate:
            return [(docnum, cos, doctitle[docnum][0], doctitle[docnum][1]) for docnum, cos in sims]
        else:
            return enumerate(sims)

    def nearest(self, text, topn=10, weighted=True):
        vec_lsi = self.text2vec(text, weighted)
        vec = np.array([w for d,w in vec_lsi])
        sqrtsigma = np.sqrt(self.model.projection.s)
        dists = cos_mat_vec(self.model.projection.u / sqrtsigma, vec / sqrtsigma)
        best = np.argsort(dists)[::-1]
        result = [(self.dictionary[sim], float(dists[sim])) for sim in best]
        if topn:
            return result[:topn]
        else:
            return result

def cos_mat_vec(mat, vec):
    (size0, size1) = np.shape(mat)
    if len(vec) != size1:
        logger.error('Unequal dimensions in cos_mat_vec %s %s' % (size1, len(vec)))
    res = np.zeros(size0)
    for i in range(size0):
        res[i]=np.dot(mat[i],vec)
    return res

class SABatch:
    def __init__(self, lsa=None):
        self.lsa = lsa

    def query(self, text, sort=True, translate=False, weighted=True):
        self.lsa.query(text, sort=sort, translate=translate, weighted=weighted)

    def query_auxvars(self, text, in_auxvar_file='', name='Augmented Query', sort=True):
        #in_auxvar_file has first column ids, remaning columns demographics or other variables
        import xlwt
        from compare_score import read_scores
        if in_auxvar_file:
            (scores, scoredict) = read_scores(in_auxvar_file)
        #doctitle = self.doctitle
        wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws = wb.add_sheet(name)
        ws.write(0, 0, 'ID', style_reg)
        ws.write(0, 1, 'Section', style_reg)
        ws.write(0, 2, 'Cosine', style_reg)
        colnum = 3
        for var in scores:
            ws.write(0, colnum, var, style_reg)
            colnum += 1
        rownum = 1
        for docnum, cos, id, sect in self.query(text, sort=sort, translate=True):
            ws.write(rownum, 0, id, style_reg)
            ws.write(rownum, 1, sect, style_reg)
            ws.write(rownum, 2, float(cos), style_reg)
            colnum = 3
            sd = scoredict[id]
            for var in scores:
                ws.write(rownum, colnum, sd[var], style_reg)
                colnum += 1
            rownum += 1
        wb.save('out.xls')

    def queries_from_file(self, queries=[], in_query_file='docqueries.txt', in_auxvar_file='', outfile='out.xls', xlswb=None, sheetname='Compare', sort=True, includedoctext=False):
        #in_auxvar_file has first column ids, remaning columns demographics or other variables
        import xlwt
        from compare_score import read_scores
        from collections import defaultdict
        if not queries:
            logger.debug('Reading queries from %s' % in_query_file)
            queries = read_docquery_file(in_query_file)
        doctitle = self.lsa.doctitle
        if in_auxvar_file:
            (scores, scoredict) = read_scores(in_auxvar_file)
        if includedoctext:
            doctextdict = GSCorpus.load_doctextdict(self.lsa.gsname)
            if False: #cache
                self.lsa.doctextdict = doctextdict #not tested
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws = wb.add_sheet(sheetname)
        logger.debug('Creating sheet %s in workbook %s' % (sheetname, wb))
        querynames = []
        cosinedict = defaultdict(lambda: defaultdict(float))
        cosvecdict = defaultdict(list)
        docdict = defaultdict()
        for name, doctext in queries:
            querynames.append(name)
            cosdict = cosinedict[name]
            for docnum, cos in self.lsa.query(doctext, sort=False, translate=False):
                cosdict[docnum] = float(cos)
                if not docnum in docdict: #probably a more efficient way to do this
                    docdict[docnum] = True
        rownum = 1
        for docnum in sorted(docdict.keys()):
            tup = doctitle[docnum]
            ws.write(rownum, 0, tup[0], style_reg)
            ws.write(rownum, 1, tup[1], style_reg)
            colnum = 2
            for queryname in querynames:
                cosdict = cosinedict[queryname]
                ws.write(rownum, colnum, cosdict[docnum], style_reg)
                cosvecdict[queryname].append(cosdict[docnum])
                colnum += 1
            if in_auxvar_file:
                sd = scoredict[tup[0]] #id
                for var in scores:
                    ws.write(rownum, colnum, sd[var], style_reg)
                    colnum += 1
            if includedoctext:
                text_for_docnum = doctextdict[docnum] #is list of tokens
                text_for_docnum = ' '.join(text_for_docnum)
                #text_for_docnum = ''.join([i if ord(i) < 128 else ' ' for i in text_for_docnum]) #needed for nonascii chars
                ws.write(rownum, colnum, text_for_docnum)
                colnum += 1
            rownum += 1
        ws.write(0, 0, 'ID', style_reg)
        ws.write(0, 1, 'Section', style_reg)
        colnum = 2
        for queryname in querynames:
            ws.write(0, colnum, 'Cosine ' + queryname, style_reg)
            colnum += 1
        if in_auxvar_file:
            for var in scores:
                ws.write(0, colnum, var, style_reg)
                colnum += 1
        if includedoctext:
            ws.write(0, colnum, 'Document Text', style_reg)
            colnum += 1
        if not xlswb:
            wb.save(outfile)
        return (cosvecdict, querynames)

    def pairwise_cosines(self, spacename, outfile='out-pwcos-grid.txt', sort=True):
        doctitle = self.lsa.doctitle
        docnum = 0
        with open(outfile, 'w') as ofp:
            ofp.write('ID\t')
            for k in doctitle:
                ofp.write('\t'+k[0])
            ofp.write('\n\tSection')
            for k in doctitle:
                ofp.write('\t'+k[1])
            ofp.write('\n')
            for similarities in self.lsa.index: # perform a similarity query for every doc against itself
                tup = doctitle[docnum]
                ofp.write('\t' + tup[0])
                ofp.write('\t' + tup[1])
                for cos in similarities:
                    ofp.write('\t')
                    ofp.write(str(cos))
                ofp.write('\n')
                docnum += 1
        return outfile

    def pairwise_cosines2(self, spacename, outfile='out-pwcos-tab.txt', sort=True):
        doctitle = self.lsa.doctitle
        docnum = 0
        with open(outfile, 'w') as ofp:
            ofp.write('ID 1\tSection 1\tID 2\tSection 2\tcosine\n')
            for similarities in self.lsa.index: # perform a similarity query for every doc against itself
                tup = doctitle[docnum]
                id = tup[0] 
                section = tup[1]
                docnum2 = 0
                for cos in similarities:
                    ofp.write(id + '\t' + section + '\t')
                    tup = doctitle[docnum2]
                    ofp.write(tup[0] + '\t')
                    ofp.write(tup[1] + '\t')
                    ofp.write(str(cos))
                    ofp.write('\n')
                    docnum2 += 1
                docnum += 1
        return outfile

    def compare_spaces(self, gsname, spaces=[], queries=[], in_query_file='docqueries.txt', outfile='out-cmpspaces.xls', xlswb=None, sort=True):
        import xlwt
        from collections import defaultdict
        import scipy
        #import math
        from scipy import stats
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        cosinedict = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # space, query, docnum
        for (space, dims) in spaces:
            logger.info('Loading model for space %s and dims %s to run queries against %s.' % (space, dims, gsname))
            lsa = LSA()
            self.lsa = lsa
            lsa.load_model(space, dims) #WARNING: TODO: Weight not specified
            lsa.load_index(gsname)
            lsa.gscorpus = GSCorpus.load(gsname, dictname=space)
            (cosinedict[space+'-'+str(dims)], querynames) = self.queries_from_file(queries=queries, in_query_file=in_query_file, in_auxvar_file='', xlswb=wb, sheetname=space+'-'+str(dims), sort=sort)
        for queryname in querynames: #querynames should be the same for every iterations, so just use the last (TODO: could cache and reuse)
            ws = wb.add_sheet('Pearson %s' % queryname)
            logger.debug('Calculating Pearson coefficients for %s' % queryname)
            rownum = 1
            for (space1, dims1) in spaces:
                spacename1 = space1+'-'+str(dims1)
                colnum = 1
                for (space2, dims2) in spaces:
                    spacename2 = space2+'-'+str(dims2)
                    if spacename1 == spacename2:
                        #symmetric, so only triagular matrix
                        break
                    cos1 = cosinedict[spacename1][queryname]
                    cos2 = cosinedict[spacename2][queryname]
                    (corrcoef, pvalue) = scipy.stats.pearsonr(cos1, cos2)
                    corrcoef = corrcoef.item()
                    #pvalue = pvalue.item()
                    #r2 = corrcoef * corrcoef
                    #tscore = (corrcoef * math.sqrt(len(cos1) - 2)) / math.sqrt(1 - r2)
                    ws.write(rownum, colnum, corrcoef, style_reg)
                    colnum += 1
                ws.write(rownum, 0, spacename1, style_reg)
                ws.write(0, rownum, spacename1, style_reg) #write row and col headers through same loop
                rownum += 1
        if not xlswb:
            wb.save(outfile)
        return cosinedict

    def tfidf_calc(self, cmd, spacename, dims):
        #TODO (major): Shouldn't need dims here, maybe remove from some corpus file names
        lsa = self.lsa
        if hasattr(lsa, 'weighted') and lsa.weighted:
            tfidf = lsa.weighted
        else:
            lsa.name = spacename
            lsa.numtopics = dims
            weighted_fname = lsa._weighted_filename(spacename, 'TFIDF')
            if os.path.isfile(weighted_fname):
                tfidf = models.TfidfModel.load(weighted_fname)
            else:
                tfidf = lsa.create_model(name, dims, save=True, weighted='TFIDF')
            lsa.weighted = tfidf
        gscorpus = GSCorpus.load(spacename)
        lsa.doctitle = GSCorpus.load_doctitle(spacename)
        #index = similarities.SparseMatrixSimilarity(tfidf[lsa.gscorpus], num_features=int(self.numtopics)) 
        index = similarities.MatrixSimilarity(tfidf[gscorpus])
        lsa.index = index
        #for (n, t) in gscorpus.dictionary.items():
         #   sims = index[[(n, 1)]]
          #  val = 'NOT_IMPLEMENTED' #TODO
            #print(t+'\t'+str(val))
        of1 = self.pairwise_cosines(spacename, outfile='out-'+spacename+'-tfidf-grid.txt', sort=True) #tfidf similarity scores, not cosines
        of2 = self.pairwise_cosines2(spacename, outfile='out-'+spacename+'-tfidf-tab.txt', sort=True) #  but the code is the same
        return [of1, of2]

    def nearest_from_file(self, queries=[], in_query_file='docqueries.txt', topn=1000, termsonly=False, outfile='out-word2vec-nearest.xls', xlswb=None, sheetname='Nearest Neighbor', sort=True):
        import xlwt
        from compare_score import read_scores
        from collections import defaultdict
        if hasattr(self, 'obj') and self.obj:
            obj = self.obj
        elif hasattr(self, 'lsa') and self.lsa:
            obj = self.lsa
        else:
            logger.error('Unknown Batch object. Internal error.')
            return 'INTERNAL_ERROR_batch'
        if not queries:
            logger.debug('Reading queries from %s' % in_query_file)
            queries = read_docquery_file(in_query_file)
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws = wb.add_sheet(sheetname)
        querynames = []
        colnum = 0
        for name, doctext in queries:
            querynames.append(name)
            if termsonly:
                res = obj.nearest(doctext, topn=False) #loop over entire list
            else:
                res = obj.nearest(doctext, topn=topn) #if phrases, know how many we need
            ws.write(0, colnum, name + '.Term', style_reg)
            ws.write(0, colnum + 1, name + '.Cosine', style_reg)
            rownum = 1
            for x in res:
                if '_' in x[0]:
                    continue
                ws.write(rownum, colnum, x[0], style_reg)                
                ws.write(rownum, colnum + 1, x[1], style_reg)                
                rownum += 1
                if topn and rownum > topn:
                    break
            colnum += 2
        if not xlswb:
            wb.save(outfile)
            return outfile
        else:
            return ws

    def write_space_vec(self, outfile='outfile-spacevec.xls', topwords=5, xlswb=None, sheetname='Space Vec'):
        import xlwt
        if xlswb:
            wb = xlswb
        else:
            wb = xlwt.Workbook()
        style_reg = xlwt.easyxf("font: height 240")
        style_bold = xlwt.easyxf("font: height 240, bold 1")
        ws = wb.add_sheet(sheetname)
        lsa = self.lsa
        space = lsa.name
        dims = int(lsa.numtopics)
        proj = lsa.model.projection.u #[num words, num dims]
        actualdims = np.size(proj,1)
        dims = min(dims, actualdims) #proj can be smaller than stated dims
        numwords = len(proj)
        dictionary = lsa.dictionary
        ws.write(0,0, 'Words', style_reg)
        for i in range(topwords): #write header
            ws.write(0, i+1, i, style_reg)
            ws.write(0, (topwords*2)-i+2, numwords-i-1, style_reg)
        for dim in range(dims): #for each dim
            idxs = np.argsort(proj[:,dim])
            ws.write(dim+1, 0, dim, style_reg)
            for i in range(topwords):
                idx = idxs[i]
                idx2 = idxs[numwords-i-1]
                ws.write(dim+1, i+1, dictionary[idx], style_reg)
                ws.write(dim+1, (topwords*2)-i+2, dictionary[idx2], style_reg)
        #TODO: Might also be useful to write out words close to value 0
        if not xlswb:
            wb.save(outfile)
            return outfile
        else:
            return ws

    def write_token_vec(self, outfile='outfile-tokenvec.txt', dumpformat="TXT"):
        #TODO: dump pickle to lsadir as <space>-tokenvec.pickle
        lsa = self.lsa
        proj = lsa.model.projection.u #[num words, num dims]
        dictionary = lsa.dictionary
        if dumpformat == 'TXT':
            with open(outfile, 'wb') as of:
                for r in range(np.size(proj,0)):
                    of.write(dictionary[r]+'\t'+'\t'.join([str(x) for x in proj[r]])+'\n')
        elif dumpformat == 'PICKLE':
            import pickle
            tokenvecdict = {}
            for r in range(np.size(proj,0)):
                tokenvecdict[dictionary[r]] = proj[r]
            with open(outfile, 'wb') as of:
                pickle.dump(tokenvecdict, of, protocol=pickle.HIGHEST_PROTOCOL)
        return outfile

    def run10(self, name, filter=''):
        from load_questions import participant_cluster_dict, lni_questions_list
        yilnovice = participant_cluster_dict['yilnovice']
        yilexpert = participant_cluster_dict['yilexpert']
        #filter = 'actual'
        if filter != '':
            re_filter = re.compile('([\w.]*?%s[\w.]*?)' % filter.strip())
        else:
            re_filter = re.compile('[\w.]*')
        #
        sections = ['actual', 'temporal', 'social', 'expected', 'exper']
        lsi.print_debug(10)
        for doc, doctitle in zip(corpus_lsi, doctitle): 
            id = doctitle[0]
            cat = doctitle[1]
            sectnum = 0
            for s in sections:
                if s in cat:
                    section = s
                    break
                sectnum += 1
            if not re_filter.match(cat):
                continue 
            yil = ''
            if id in yilexpert:
                color = 'bo'
                yil = 'Expert'
            elif id in yilnovice:
                color = 'ro'
                yil = 'Novice'
            else:
                continue
            for plotnumy in range(0,3):
                for plotnumx in range(0,3):
                    plotnum = plotnumy*3+plotnumx
                    x = doc[plotnum][1]
                    y = doc[plotnum+1][1]
                    plt.plot(x + plotnumx*2, y+plotnumy*2 + 3*sectnum, color)
                    if y < -0.15 or y > 0.15:
                        print('\t'.join([id, yil, section, cat, str(x), str(y)]))
        plt.show()

def read_docquery_file(fname, yield_code=1):
    text_queue = [];
    with open(fname, 'rb') as fp:
        id = ''
        category = ''
        for line in fp:
            if line.isspace():
                continue
            if line[0] == '%':
                if text_queue:
                    textstr = ' '.join(text_queue)
                    if yield_code == 1: # yield section
                        #logger.debug('Processing docquery: ' + sectname)
                        yield sectname, textstr
                    text_queue = []
                if line[0:7] == '%PROBE.':
                    sectname = string.strip(line[7:]).lower()
                if line[0:7] == '%Q.LNI.':
                    category = string.strip(line[7:]).lower()
                    if id:
                        sectname = id + '.' + category
                elif line[0:6] == '%BEGIN':
                    id_from_header = string.strip(line[7:])
                    id = id_from_header
                elif line[0:4] == '%END':
                    id_from_header = string.strip(line[4:])
                    if id_from_header != id:
                        logger.error('Inconsistent IDs: %s in %%END statement for %s.', id_from_header, id)
                    id = ''
                elif line[0:8] == '%COMMENT':
                    continue
                continue
            if string.find(line, ':') > 0:
                logger.error("Invalid character ':' in SACorpus file. File may need preprocessing: %s. --Line: %s", self.lsa.corpusfname, line)
                sys.exit(1)
            text_queue.append(line)
            #response line
            if yield_code == 0: # yield line
                yield sectname, line
        if yield_code==1 and text_queue: #write remaining text queue, if needed
            textstr = ' '.join(text_queue)
            #logger.debug('Processing docquery: ' + sectname)
            yield sectname, textstr

#Functions

def print_docquery_file(fname, yield_code=1, slice=None):
    for sectname, text in read_docquery_file(fname, yield_code):
        if slice:
            print("%s\t%s" % (sectname, string.strip(text[slice])))
        else:
            print("%s\t%s" % (sectname, string.strip(text)))
            

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Load and Run LSA.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--build', dest='buildname', default="", help='Build a LSA space using arg as name of the GSCorpus with dimension <dims>.')
    parser.add_argument('--index', dest='indexname', default="", help='Index a GSCorpus using arg as name against the LSA space <space> with <dims> dimensions.')
    parser.add_argument('--bi', dest='biname', default="", help='Build a LSA space from <space> and index a GSCorpus using arg as name of the GSCorpus with dimension <dims>.')
    parser.add_argument('--query', dest='querytext', default="", help='Query using arg as doctext against GSCorpus <gsname> in the LSA space <space> with <dims> dimensions.')
    parser.add_argument('--queries', dest='queryfile', default="", help='Query using arg as filename of doctext against GSCorpus <gsname> in the LSA space <space> with <dims> dimensions.')
    parser.add_argument('--includedoctext', dest='includedoctext', default="", help='Include doctext in query result.')
    parser.add_argument('--printdocqueryfile', dest='dqfile', default="", help='Print doc query file.')
    parser.add_argument('--cmpspaces', dest='cmpspaces', action="store_true", help='Compare spaces for queries against the GSCorpus <gsname>')
    parser.add_argument('--pairwisecos', dest='pairwisecos', action="store_true", help='Calculate pairwise cosines for queries against every document in the GSCorpus <gsname>')
    parser.add_argument('--xlsfile', dest='xlsfile', default="", help='Create Excel file with matrix of IDs and cosines.')
    parser.add_argument('--outfile', dest='outfile', default="", help='Output file for queries.')
    parser.add_argument('--gsname', dest='gsname', default="", help='Name of the GS Corpus.')
    parser.add_argument('--gsfile', dest='gsfile', default="", help='Name of the files to use for GS Corpus instead of GS Corpus. [NOT IMPLEMENTED]')
    parser.add_argument('--space', dest='lsaspacename', default="", help='Name of the LSA space.')
    parser.add_argument('--dims', dest='lsadims', default=10, help='Dimensions of the LSA space.')
    parser.add_argument('--weighted', dest='weighted', default="LE", help='Weighting to use for corpus. Options are: LE (default), TFIDF, or None')
    parser.add_argument('--spaceoptions', '-s', dest='spaceoptions', default="", help='Short name of configuration options for LSA space (dims, corpus stopwords and iterator, weighting.')
    parser.add_argument('--tfidf', dest='tfidf_calc', default="", help='TFIDF weighting for corpus.')
    parser.add_argument('--auxvars', dest='varfile', default="", help='File of auxillary variables to include in report (tab-delimited table keyed off of ids).')
    #parser.add_argument('--gscorpus', dest='gscorpus', default='temp', help='GENSIM corpus name to use.')
    parser.add_argument('--nearest', dest='nearesttext', default="", help='Nearest using arg as doctext against model.')
    parser.add_argument('--nearestfile', dest='nearestfile', default="", help='Nearest using arg as filename of doctext against model.')
    parser.add_argument('--spacevec', dest='spacevec', action="store_true", help='Describe space vector <space> with <dims> dims')
    parser.add_argument('--tokenvec', dest='tokenvec', default="", help='Dump space vector <space> with <dims> dims in specified format (txt, pickle).')
    parser.add_argument('--prod', dest='production', action="store_const", const=1, help='Set LSA space creation (of model) to use production values (takes more time).')
    parser.add_argument('--verbosity', '-v', dest='verbosity', default=1, type=int, help='Verbosity levels. 0:Warnings, 1:Info, 2:Debug [default 1]')
    parser.add_argument('--embed', dest='embed', action="store_true", help='Bring up an IPython prompt after loading the space and running commands.')
    args = parser.parse_args()
    if args.weighted:
        args.weighted = args.weighted.upper()
    if args.spaceoptions:
        if args.spaceoptions in lsa_space_shortnames:
            soptions = lsa_space_shortnames[args.spaceoptions]
            logger.info('Setting space options for %s to space %s with %s dims and %s weighting.' % (
                args.spaceoptions, soptions[0], soptions[1], soptions[2]))
            (args.lsaspacename, args.lsadims, args.weighted) = soptions
        else:
            logger.warn('Unknown space options for ' + args.spaceoptions)
    return args

lsa_space_shortnames = {
    'TASA' : ('TASA_sect_stopmallet', 300, 'LE'),
    'TASA300' : ('TASA_sect_stopmallet', 300, 'LE'),
    'TASA300nltk' : ('TASA_sect_stopnltk', 300, 'LE'),
}

def main():
    from IPython import embed
    args = init_args()
    if args.dqfile:
        print_docquery_file(args.dqfile, 1)#, slice(0,25))
        return 1
    if args.cmpspaces:
        sabatch = SABatch()
        spaces = [('TASA', 300), ('TASA', 450)]
        sabatch.compare_spaces(args.gsname, spaces)
        return 1
    if args.tfidf_calc:
        sabatch = SABatch(LSA())
        val = sabatch.tfidf_calc(args.tfidf_calc, args.lsaspacename, args.lsadims)
        print(val)
        return 1
    lsa = LSA()
    #Create or load model
    if args.buildname:
        lsaspacename = args.buildname
        lsa.create_model(lsaspacename, args.lsadims, weighted=args.weighted, production=args.production)
        return 1
    elif args.biname:
        lsa.create_model(args.lsaspacename, args.lsadims, weighted=args.weighted, save=False)
    else:
        lsa.load_model(args.lsaspacename, args.lsadims, weighted=args.weighted)
    if args.nearesttext:
        result = lsa.nearest(args.nearesttext, topn=10)
        print('Word\tCosine')
        for x in result:
            print('\t'.join(str(i) for i in x))
        return 1
    if args.nearestfile:
        batch = SABatch(lsa)
        outfile = batch.nearest_from_file(in_query_file=args.nearestfile, topn=100, outfile='out-lsa-nearest-'+args.lsaspacename+'-'+str(args.lsadims)+'.xls')
        print(outfile)
        return 1
    if args.spacevec:
        batch = SABatch(lsa)
        outfile = batch.write_space_vec(outfile='out-lsa-spacevec-'+args.lsaspacename+'-'+str(args.lsadims)+'.xls')
        print(outfile)
        if args.embed:
            embed()
        return 1
    if args.tokenvec:
        batch = SABatch(lsa)
        tvoutfilename = 'out-lsa-tokenvec-'+args.lsaspacename+'-'+str(args.lsadims)+'.'+string.lower(args.tokenvec)
        outfile = batch.write_token_vec(outfile=tvoutfilename, dumpformat=string.upper(args.tokenvec))
        print(outfile)
        if args.embed:
            embed()
        return 1
    #Create or load index
    if args.indexname:
        gscorpus = GSCorpus.load(args.indexname, dictname=args.lsaspacename)
        lsa.index_corpus(gscorpus, weighted=args.weighted)        
        return 1
    elif args.biname:
        gscorpus = GSCorpus.load(args.biname, dictname=args.lsaspacename)
        lsa.index_corpus(gscorpus)        
        return 1
    elif args.gsfile: #Create temporary corpus and index
        lsa.gscorpus = GSCorpus.build_temp_corpus('adhoc', args.gsfile) 
        lsa.index_corpus(lsa.gscorpus, save=False)
    else:
        lsa.load_index(args.gsname)
    #Load query corpus and run query
    if not args.gsfile:
        lsa.gscorpus = GSCorpus.load(args.gsname, dictname=args.lsaspacename)
    if args.querytext:
        result = lsa.query(args.querytext, translate=True)
        print('DocNum\tCosine\tID\tCategory')
        for x in result:
            print('\t'.join(str(i) for i in x))
        return 1
    if args.queryfile:
        sabatch = SABatch(lsa)
        sabatch.queries_from_file(in_query_file=args.queryfile, in_auxvar_file=args.varfile, outfile=args.outfile, includedoctext=args.includedoctext)
        return 1
    if args.pairwisecos:
        sabatch = SABatch(lsa)
        print('Warning: These files may be large.')
        sabatch.pairwise_cosines(args.lsaspacename)
        sabatch.pairwise_cosines2(args.lsaspacename)
        return 
    if args.embed:
        embed()
    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
