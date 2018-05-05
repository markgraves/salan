#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import logging

import nltk

import salan

LOGGER = logging.getLogger('salan')

# constants for iterators and flags
ITER_TOKEN = 1
ITER_SENT = 2
ITER_LINE = 3  # SENT and LINE are independent; LINE can comprise several sentences or SENT can split across lines
ITER_PARA = 4  # requires a blank line or command to separate, so requires precise formatting
ITER_SECT = 6  # requires a section command (or larger division) to separate
ITER_DOC = 8  # BEGIN/END pairs
ITER_FILE = 9

FLAG_NORMAL = 0
FLAG_EOF = 1
FLAG_CMD = 2


class FCorpus:

    # constants for state output
    STATE_FILENAME = 'FILE'
    STATE_DOCID = 'DOC'
    STATE_SECTION = 'SECT'
    STATE_ORDERNUM = 'ORD'  # Section order number
    STATE_TUPLES = 'TUP'

    # document format variables
    CMD_CHAR = '%'  # commands have this character at the beginning of the line
    CMD_BEGIN = 'BEGIN'  # name of the command to start a new doc
    CMD_END = 'END'  # name of the command to end a doc
    CMD_COMMENT = 'COMMENT'
    CMD_SECT_LIST = ['Q.', 'Q.LNI.', 'Q.NP.', 'DOC.', 'CHAP.', 'BOOK.', 'SECT.', 'DIV.']

    def __init__(self, itercode=None):
        if itercode:
            default_itercode = itercode

    ## Main parsing functions

    def read_fcorp_file(self, filename, itercode=ITER_FILE):
        "Open and iterate through fcorp file."
        with open(filename, 'r') as fp:
            newstate = dict()
            newstate[self.STATE_FILENAME] = 'augustine-smallcorpus-fcorp.txt'
            newstate[self.STATE_DOCID] = None
            newstate[self.STATE_SECTION] = None
            newstate[self.STATE_ORDERNUM] = None
            newstate[self.STATE_TUPLES] = list()
            return self.read_fcorp_block_file(fp, newstate, itercode)

    def read_fcorp_block_file(self, stream, current_state, itercode):
        result = list()
        LOGGER.debug('Reading ' + current_state[self.STATE_FILENAME])
        while True:
            lines, flag = self.read_fcorp_block_sect(stream)
            if current_state[self.STATE_TUPLES]:
                current_state[self.STATE_TUPLES].extend(lines)
            else:
                current_state[self.STATE_TUPLES] = lines
            if flag == FLAG_NORMAL:
                result =  self.handle_fcorp_iterate_emit(current_state, result, itercode)
                newstate = current_state.copy()
                newstate[self.STATE_TUPLES] = list()
                current_state = newstate
            elif flag == FLAG_EOF:
                result =  self.handle_fcorp_iterate_emit(current_state, result, itercode, flush=True)
                return result
            elif flag == FLAG_CMD:
                newstate = self.handle_fcorp_cmd(stream, current_state)
                if newstate:
                    result =  self.handle_fcorp_iterate_emit(current_state, result, itercode)
                    current_state = newstate

    ## Readers from file
       
    def read_fcorp_block_para(self, stream, ignore_blank_line=False):
        "Returns next paragraph of contiguous data lines from stream (as stripped list)."
        lines = list()
        while True:
            oldpos = stream.tell()
            line = stream.readline()
            if not line:  # end of file
                return lines, FLAG_EOF
            if line[0] == self.CMD_CHAR:
                stream.seek(oldpos)
                return lines, FLAG_CMD
            line = line.strip()
            if not line:  # blank line
                if ignore_blank_line:
                    continue
                else:
                    return lines, FLAG_NORMAL
            else:
                lines.append(line)

    def read_fcorp_block_sect(self, stream):
        "Returns next section of data lines from stream (as stripped list)."
        return self.read_fcorp_block_para(stream, True)

    def read_fcorp_block_line(self, stream):
        "Returns next data line from stream (unstripped)."
        oldpos = stream.tell()
        line = stream.readline()
        if not line:  # end of file
            return '', FLAG_EOF
        if line[0] == self.CMD_CHAR:
            stream.seek(oldpos)
            return None, FLAG_CMD
        else:
            return line, FLAG_NORMAL

    ## Handle parsing events

    def handle_fcorp_cmd(self, stream, current_state):
        "Returns a new reader state based upon the next immediate cmd in stream."
        line = stream.readline()
        if line[0] != self.CMD_CHAR:
            msg = 'Internal inconsistency with fcorp command at: ' + line
            LOGGER.error(msg)
            raise RuntimeError(msg) from error
        if line.startswith(self.CMD_COMMENT, 1):
            pass
        elif line.startswith(self.CMD_BEGIN, 1):
            if current_state[self.STATE_DOCID]:
                LOGGER.warning('Document %s was not properly ended.' % current_state[self.STATE_DOCID])
            newstate = current_state.copy()
            newstate[self.STATE_DOCID] = line[len(self.CMD_BEGIN)+2:].strip()  #+1 for CMD_CHAR and +1 for trailing space
            newstate[self.STATE_SECTION] = None
            newstate[self.STATE_ORDERNUM] = None
            newstate[self.STATE_TUPLES] = list()
            return newstate
        elif line.startswith(self.CMD_END, 1):
            cmdid = line[len(self.CMD_END)+2:].strip()
            if current_state[self.STATE_DOCID] != cmdid:
                LOGGER.warning('Inconsistent IDs: %s in %%END statement for %s.', cmdid, current_state[self.STATE_DOCID])
            newstate = current_state.copy()
            newstate[self.STATE_DOCID] = None
            newstate[self.STATE_SECTION] = None
            newstate[self.STATE_ORDERNUM] = None
            newstate[self.STATE_TUPLES] = list()
            return newstate
        elif any(line.startswith(cmd, 1) for cmd in self.CMD_SECT_LIST):
            newsect = line[1:].strip()  #+1 to skip CMD_CHAR
            newstate = current_state.copy()
            if current_state[self.STATE_SECTION] == newsect:
                newstate[self.STATE_ORDERNUM] += 1
            else:
                newstate[self.STATE_SECTION] = newsect  #+1 to skip CMD_CHAR
                newstate[self.STATE_ORDERNUM] = 0
                newstate[self.STATE_TUPLES] = list()
            return newstate
        else:
            LOGGER.warning('Ignoring unknown command: ' + line)
            return None

    def handle_fcorp_iterate_emit(self, current_state, result, itercode, flush=False):
        # Flush forces all states to be emitted
        if flush or itercode in [ITER_TOKEN, ITER_SENT, ITER_LINE, ITER_PARA]:
            if flush:
                LOGGER.debug('Flushing ' + current_state[self.STATE_FILENAME])
            if result:
                result.append(current_state)
                self.emit_states(result, itercode)
            else:
                self.emit_states([current_state], itercode)
            return list()
        elif (itercode == ITER_FILE):
            result.append(current_state)
            return result
        elif itercode == ITER_DOC or itercode == ITER_SECT:
            if not result:  # nothing to compare or emit
                return [current_state]
            else:
                if ((itercode == ITER_DOC and result[-1][self.STATE_DOCID] == current_state[self.STATE_DOCID]) or
                    (itercode == ITER_SECT and result[-1][self.STATE_SECTION] == current_state[self.STATE_SECTION])):
                    # continue collecting states
                    result.append(current_state)
                    return result
                else:
                    # significant state change
                    self.emit_states(result, itercode)
                    return [current_state]
        else:
            msg = 'Internal error. Unknown iterator %s' % itercode
            LOGGER.error(msg)
            raise RuntimeError(msg) from error

    ## Emit states

    def emit_states(self, statelist, itercode):
        if itercode in [ITER_TOKEN, ITER_SENT, ITER_LINE, ITER_PARA]:
            for state in statelist:
                self.emit_state(state, itercode)
        elif itercode == ITER_SECT:
            grouped_statelist = [list(g) for k, g in itertools.groupby(statelist, lambda s: s[self.STATE_SECTION])]
            if len(grouped_statelist) > 1:
                LOGGER.debug('Emitting %s sections beginning with %s' 
                             % (len(grouped_statelist), grouped_statelist[0][0][self.STATE_SECTION]))
            for group in grouped_statelist:
                key_state = group[0]
                if len(group) > 1:
                    key_state[self.STATE_TUPLES] = itertools.chain.from_iterable(state[self.STATE_TUPLES] for state in group)
                self.emit_state(key_state, itercode)
        elif itercode == ITER_DOC:
            grouped_statelist = [list(g) for k, g in itertools.groupby(statelist, lambda s: s[self.STATE_DOCID])]
            for group in grouped_statelist:
                key_state = group[0]
                key_state[self.STATE_SECTION] = None
                if len(group) > 1:
                    key_state[self.STATE_TUPLES] = itertools.chain.from_iterable(state[self.STATE_TUPLES] for state in group)
                self.emit_state(key_state, itercode)
        elif itercode == ITER_FILE:
            grouped_statelist = [list(g) for k, g in itertools.groupby(statelist, lambda s: s[self.STATE_FILENAME])]
            if len(grouped_statelist) > 1:
                LOGGER.debug('Emitting %s files beginning with %s'
                             % (len(grouped_statelist), grouped_statelist[0][0][self.STATE_FILENAME]))
            for group in grouped_statelist:
                key_state = group[0]
                key_state[self.STATE_DOCID] = None
                key_state[self.STATE_SECTION] = None
                if len(group) > 1:
                    key_state[self.STATE_TUPLES] = itertools.chain.from_iterable(state[self.STATE_TUPLES] for state in group)
                self.emit_state(key_state, itercode)
        else:
            msg = 'Internal error. Unknown iterator %s' % itercode
            LOGGER.error(msg)
            raise RuntimeError(msg) from error

    def emit_state(self, state, itercode):
        # yield state
        print(state)

    # end of FCorpus

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Load and Build Corpora.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--corpusfname', dest='corpusfname', nargs='*', default='', help='Corpus file to read.')
    parser.add_argument('--print', dest='printcorpus', action="store_true", help='Print the corpus <corpusfname>')
    parser.add_argument('--stopwords', dest='stopwordlist', default='', help='Stopwords to remove, separated by commas with no spaces')
    args = parser.parse_args()
    if not args.itercode:
        args.itercode = SACorpus.ITER_SECT
    return args


def main(): 
    args = init_args()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
    main()
