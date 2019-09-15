import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import base64
import csv


def load_nav_graphs(scan, path=None):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    if path is not None:
        DATA_DIR = path
    else:
        DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    with open(os.path.join(DATA_DIR, 'connectivity/%s_connectivity.json' % scan)) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item['included']:
                for j, conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],
                            weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')
        return G

def load_region_map(scan):
    DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    region_map = {}
    with open(os.path.join(DATA_DIR, 'view_to_region/%s.panorama_to_region.txt' % scan)) as f:
        for line in f:
            fields = line.rstrip().split()
            view = fields[1]
            region = fields[-1]
            region_map[view] = region
    return region_map

def load_datasets(splits, path, prefix=''):
    data = []
    for split in splits:
        with open(os.path.join(path, prefix + '_%s.json' % split)) as f:
            data += json.load(f)
    return data


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab, encoding_length):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        return sentence.split()

    def encode_sentence(self, sentence, encoding_length=None, reverse=False,
            eos=True, to_numpy=True):

        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = []

        # Split sentence
        if isinstance(sentence, list):
            sent_split = sentence
        else:
            sent_split = self.split_sentence(sentence)

        # Cut sentence
        if encoding_length is None:
            encoding_length = self.encoding_length

        sent_split = sent_split[:encoding_length]

        # Reverse sentence
        if reverse:
            sent_split = sent_split[::-1]

        # Encode sentence
        for word in sent_split: # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        # Append EOS and PAD.
        if len(encoding) < encoding_length and eos:
            encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) < encoding_length:
            encoding.extend([self.word_to_index['<PAD>']] * (encoding_length-len(encoding)))

        assert len(encoding) == encoding_length

        if to_numpy:
            encoding = np.array(encoding)

        return encoding

def read_vocab(paths):
    vocab = []
    added = set()
    for path in paths:
        with open(path) as f:
            words = [word.strip() for word in f.readlines()]
            for w in words:
                if w not in added:
                    added.add(w)
                    vocab.append(w)
    print('Read vocab of size', len(vocab))

    return vocab

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def decode_base64(string):
    if sys.version_info[0] == 2:
        return base64.decodestring(bytearray(string))
    elif sys.version_info[0] == 3:
        return base64.decodebytes(bytearray(string, 'utf-8'))
    else:
        raise ValueError("decode_base64 can't handle python version {}".format(sys.version_info[0]))


