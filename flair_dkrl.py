#!/usr/bin/env python
# coding: utf-8

# Script to run baseline:
#
# ```
#
# python train.py link_prediction with \
# dataset='FB15k-237' \
# inductive=True \
# model='bert-dkrl' \
# rel_model='transe' \
# loss_fn='margin' \
# regularizer=1e-2 \
# max_len=32 \
# num_negatives=64 \
# lr=1e-4 \
# use_scheduler=False \
# batch_size=64 \
# emb_batch_size=512 \
# eval_batch_size=128 \
# max_epochs=5 \
# checkpoint=None \
# use_cached_text=False
#
# ```

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os.path as osp
import networkx as nx
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import concurrent.futures
from time import time
from torchcoder.autoencoders import LSTM_AE
import pandas as pd


#from data import CATEGORY_IDS
#from data import GraphDataset, TextGraphDataset, GloVeTokenizer
#import models
#import utils


# In[2]:


import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# In[3]:


from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings
from flair.data import Sentence
import flair
flair.device = torch.device('cpu')


# In[4]:


def transe_score(heads, tails, rels):
    return -torch.norm(heads + rels - tails, dim=-1, p=1)


# In[5]:


def margin_loss(pos_scores, neg_scores):
    loss = 1 - pos_scores + neg_scores
    loss[loss < 0] = 0
    return loss.mean()

def nll_loss(pos_scores, neg_scores):
    return (F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()) / 2


def l2_regularization(heads, tails, rels):
    reg_loss = 0.0
    for tensor in (heads, tails, rels):
        reg_loss += torch.mean(tensor ** 2)

    return reg_loss / 3.0


# In[6]:


class LinkPrediction(nn.Module):
    """A general link prediction model with a lookup table for relation
    embeddings."""
    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer, batch_size):
        super().__init__()
        self.dim = dim
        self.normalize_embs = False
        self.regularizer = regularizer
        self.batch_size = batch_size

        if rel_model == 'transe':
            self.score_fn = transe_score
            self.normalize_embs = True
        else:
            raise ValueError(f'Unknown relational model {rel_model}.')

        self.rel_emb = nn.Embedding(num_relations, self.dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

        if loss_fn == 'margin':
            self.loss_fn = margin_loss
        elif loss_fn == 'nll':
            self.loss_fn = nll_loss
        else:
            raise ValueError(f'Unkown loss function {loss_fn}')

    def encode(self, *args, **kwargs):
        ent_emb = self._encode_entity(*args, **kwargs)
        if self.normalize_embs:
            ent_emb = F.normalize(ent_emb, dim=-1)

        return ent_emb

    def _encode_entity(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, ent_embs, rels, neg_idx):
        batch_size = ent_embs.shape[0]

        # Scores for positive samples
        rels = self.rel_emb(rels)
        heads, tails = torch.chunk(ent_embs, chunks=2, dim=1)
        pos_scores = self.score_fn(heads, tails, rels)

        if self.regularizer > 0:
            reg_loss = self.regularizer * l2_regularization(heads, tails, rels)
        else:
            reg_loss = 0

        # Scores for negative samples
        neg_embs = ent_embs.view(batch_size * 2, -1)[neg_idx]
        heads, tails = torch.chunk(neg_embs, chunks=2, dim=2)
        neg_scores = self.score_fn(heads.squeeze(), tails.squeeze(), rels)

        model_loss = self.loss_fn(pos_scores, neg_scores)
        return model_loss + reg_loss


# In[7]:


class InductiveLinkPrediction(LinkPrediction):
    """Description-based Link Prediction (DLP)."""
    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError

    def forward(self, text, rels=None, neg_idx=None):
        
        # Encode text into an entity representation from its description
        ent_embs = self.encode(text)

        if rels is None and neg_idx is None:
            # Forward is being used to compute entity embeddings only
            out = ent_embs
        else:
            # Forward is being used to compute link prediction loss
            ent_embs = ent_embs.view(self.batch_size, 2, -1)
            out = self.compute_loss(ent_embs, rels, neg_idx)

        return out


# In[8]:


class WordEmbeddingsLP(InductiveLinkPrediction):
    """Description encoder with pretrained embeddings, obtained from BERT or a
    specified tensor file.
    """
    def __init__(self, rel_model, loss_fn, num_relations, regularizer, batch_size,
                 dim=None, encoder_name=None, embeddings=None):
        if not encoder_name and not embeddings:
            raise ValueError('Must provided one of encoder_name or embeddings')

        if encoder_name is not "flair":
            encoder = TransformerWordEmbeddings(encoder_name)
        elif encoder_name=="flair":
            encoder = FlairEmbeddings("en-forward-fast")
        else:
            #then it is GLOVE in this case
            encoder = WordEmbeddings('glove')

        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer, batch_size)

        self.embeddings = encoder

    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError


# In[9]:


class DKRL(WordEmbeddingsLP):
    """Description-Embodied Knowledge Representation Learning (DKRL) with CNN
    encoder, after
    Zuo, Yukun, et al. "Representation learning of knowledge graphs with
    entity attributes and multimedia descriptions."
    """

    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer, batch_size,
                 encoder_name=None, embeddings=None):
        super().__init__(rel_model, loss_fn, num_relations, regularizer, batch_size,
                         dim, encoder_name, embeddings)

        self.emb_dim = self.embeddings.embedding_length
        self.conv1 = nn.Conv1d(self.emb_dim, self.dim, kernel_size=2)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=2)

    def _encode_entity(self, text):
        # Extract word embeddings and mask padding
        max_len = 32
        all_emb = []
        for j,ent_text in enumerate(text):
            # For training loop where 2 entities are present in each ent_text
            text_mask = torch.ones((self.batch_size*2, max_len), dtype=torch.float, device=device)
            if isinstance(ent_text, list):
                emb = torch.zeros(len(ent_text), max_len, self.emb_dim)
                for i, ent in enumerate(ent_text):
                    #print(ent)
                    self.embeddings.embed(ent)
                    toks = []
                    for token in ent:
                        toks.append(token.embedding.cpu().numpy())
                    toks = torch.tensor(toks)
                    if max_len>len(toks):
                        emb[i,:len(toks)] = toks
                        text_mask[(2*j)+i, len(toks):] = 0
                    else:
                        emb[i, :max_len] = toks[:max_len]
                    #print(emb)
            else:
                #For the evaluation loop
                text_mask = torch.ones((len(text), max_len), dtype=torch.float, device=device)
                emb = torch.zeros(max_len, self.emb_dim)
                self.embeddings.embed(ent_text)
                toks = []
                for token in ent_text:
                    toks.append(token.embedding.cpu().numpy())
                toks = torch.tensor(toks)
                if max_len>len(toks):
                    emb[:len(toks)] = toks
                    text_mask[j, len(toks):] = 0
                else:
                    emb[:max_len] = toks[:max_len]
                #print(emb)
            all_emb.append(emb.numpy())
        #print(all_emb)
        all_emb = np.array(all_emb)
        embs = torch.tensor(all_emb)
        #print(embs.shape)
        embs = embs.to(device)

        #First reshape the batch*2 number of entities - this is done in blp in indcutive forward only
        embs = embs.view(-1, max_len, self.emb_dim)
        text_mask = text_mask.unsqueeze(1)
        # Reshape to (N, C, L)
        embs = embs.transpose(1, 2)
        #text_mask = text_mask.unsqueeze(1)

        # Pass through CNN, adding padding for valid convolutions
        # and masking outputs due to padding
        embs = F.pad(embs, [0, 1])
        embs = self.conv1(embs)
        embs = embs * text_mask
        if embs.shape[2] >= 4:
            kernel_size = 4
        elif embs.shape[2] == 1:
            kernel_size = 1
        else:
            kernel_size = 2
        embs = F.max_pool1d(embs, kernel_size=kernel_size)
        text_mask = F.max_pool1d(text_mask, kernel_size=kernel_size)
        embs = torch.tanh(embs)
        embs = F.pad(embs, [0, 1])
        embs = self.conv2(embs)
        lengths = torch.sum(text_mask, dim=-1)

        embs = torch.sum(embs * text_mask, dim=-1) / lengths
        embs = torch.tanh(embs)

        return embs


class EntAutoEnc(WordEmbeddingsLP):
    def __init__(self, dim, rel_model, loss_fn, num_entities, num_relations, regularizer, batch_size,
                 encoder_name=None, embeddings=None, aec=None):
        super().__init__(rel_model, loss_fn, num_relations, regularizer, batch_size,
                         dim, encoder_name, embeddings)
        if aec is None:
            print("ValueError: Train the autoencoder and pass as param to EntAutoEnc")
            raise ValueError
        self.aec = aec
        self.emb_dim = self.embeddings.embedding_length

        self.ent_emb = nn.Embedding(num_entities, self.dim)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)

        self.W1 = nn.Linear(self.dim, self.dim)
        self.W2 = nn.Linear(self.dim, self.dim)

    def get_encoding(self, ent_text):
        emb = torch.zeros(self.max_len, self.emb_dim)
        self.embeddings.embed(ent_text)
        toks = []
        for token in ent_text:
            toks.append(token.embedding.cpu().numpy())
        toks = torch.tensor(toks)
        if self.max_len>len(toks):
            emb[:len(toks)] = toks
        else:
            emb[:self.max_len] = toks[:self.max_len]
        return emb.numpy()

    def train_get_encoding(self, ent_text):
        emb = torch.zeros(len(ent_text), self.max_len, self.emb_dim)
        for i, ent in enumerate(ent_text):
            #print(ent)
            self.embeddings.embed(ent)
            toks = []
            for token in ent:
                toks.append(token.embedding.cpu().numpy())
            toks = torch.tensor(toks)
            if self.max_len>len(toks):
                emb[i,:len(toks)] = toks
            else:
                emb[i, :self.max_len] = toks[:self.max_len]
        return emb.numpy()

    def _encode_entity(self, entities, text):
        # Extract word embeddings and mask padding
        self.max_len = 32
        all_emb = []
        if isinstance(text[0], list):
            train = True
        else:
            train = False
        
        if train:
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                all_emb = [self.train_get_encoding(ent_text) for ent_text in text]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                all_emb = [self.get_encoding(ent_text) for ent_text in text]
        all_emb = np.array(all_emb)
        embs = torch.tensor(all_emb)
        #print(embs.shape)
        embs = embs.to(device)

        #First reshape the batch*2 number of entities - this is done in blp in indcutive forward only
        embs = embs.view(-1, self.max_len, self.emb_dim)
        input_embs, _, _ = prepare_dataset(embs)
        
        # Now adding encoding using TorchCoder
        enc_embs = self.aec.encode(input_embs)

        #encoding of the embeddings for the entity
        ent_emb = self.ent_emb(entities)
        ent_emb = ent_emb.view(-1, self.dim)

        embs = self.W1(enc_embs) + self.W2(ent_emb)
        embs = torch.tanh(embs)

        return embs

    def forward(self, pos_pair, text, rels=None, neg_idx=None):
        
        # Encode text into an entity representation from its description
        ent_embs = self.encode(pos_pair, text)

        if rels is None and neg_idx is None:
            # Forward is being used to compute entity embeddings only
            out = ent_embs
        else:
            # Forward is being used to compute link prediction loss
            ent_embs = ent_embs.view(self.batch_size, 2, -1)
            out = self.compute_loss(ent_embs, rels, neg_idx)

        return out

# data file functions

# In[10]:


from torch.utils.data import Dataset
import string
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
import logging


# In[11]:


UNK = '[UNK]'
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = stopwords.words('english')
DROPPED = STOP_WORDS + list(string.punctuation)
CATEGORY_IDS = {'1-to-1': 0, '1-to-many': 1, 'many-to-1': 2, 'many-to-many': 3}


# In[12]:


def file_to_ids(file_path):
    """Read one line per file and assign it an ID.

    Args:
        file_path: str, path of file to read

    Returns: dict, mapping str to ID (int)
    """
    str2id = dict()
    with open(file_path) as file:
        for i, line in enumerate(file):
            str2id[line.strip()] = i

    return str2id


def get_negative_sampling_indices(batch_size, num_negatives, repeats=1):
    """"Obtain indices for negative sampling within a batch of entity pairs.
    Indices are sampled from a reshaped array of indices. For example,
    if there are 4 pairs (batch_size=4), the array of indices is
        [[0, 1],
         [2, 3],
         [4, 5],
         [6, 7]]
    From this array, we corrupt either the first or second element of each row.
    This yields one negative sample.
    For example, if the positions with a dash are selected,
        [[0, -],
         [-, 3],
         [4, -],
         [-, 7]]
    they are then replaced with a random index from a row other than the row
    to which they belong:
        [[0, 3],
         [5, 3],
         [4, 6],
         [1, 7]]
    The returned array has shape (batch_size, num_negatives, 2).
    """
    num_ents = batch_size * 2
    idx = torch.arange(num_ents).reshape(batch_size, 2)

    # For each row, sample entities, assigning 0 probability to entities
    # of the same row
    zeros = torch.zeros(batch_size, 2)
    head_weights = torch.ones(batch_size, num_ents, dtype=torch.float)
    head_weights.scatter_(1, idx, zeros)
    random_idx = head_weights.multinomial(num_negatives * repeats,
                                          replacement=True)
    random_idx = random_idx.t().flatten()

    # Select randomly the first or the second column
    row_selector = torch.arange(batch_size * num_negatives * repeats)
    col_selector = torch.randint(0, 2, [batch_size * num_negatives * repeats])

    # Fill the array of negative samples with the sampled random entities
    # at the right positions
    neg_idx = idx.repeat((num_negatives * repeats, 1))
    neg_idx[row_selector, col_selector] = random_idx
    neg_idx = neg_idx.reshape(-1, batch_size * repeats, 2)
    neg_idx.transpose_(0, 1)

    return neg_idx


# In[13]:


class GraphDataset(Dataset):
    """A Dataset storing the triples of a Knowledge Graph.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        write_maps_file: bool, if set to True, dictionaries mapping
            entities and relations to IDs are saved to disk (for reuse with
            other datasets).
    """
    def __init__(self, triples_file, neg_samples=None, write_maps_file=False,
                 num_devices=1):
        directory = osp.dirname(triples_file)
        maps_path = osp.join(directory, 'maps.pt')

        # Create or load maps from entity and relation strings to unique IDs
        if not write_maps_file:
            if not osp.exists(maps_path):
                raise ValueError('Maps file not found.')

            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']
        else:
            ents_file = osp.join(directory, 'entities.txt')
            rels_file = osp.join(directory, 'relations.txt')
            ent_ids = file_to_ids(ents_file)
            rel_ids = file_to_ids(rels_file)

        entities = set()
        relations = set()

        # Read triples and store as ints in tensor
        file = open(triples_file)
        triples = []
        for i, line in enumerate(file):
            values = line.split()
            # FB13 and WN11 have duplicate triples for classification,
            # here we keep the correct triple
            if len(values) > 3 and values[3] == '-1':
                continue
            head, rel, tail = line.split()[:3]
            entities.update([head, tail])
            relations.add(rel)
            triples.append([ent_ids[head], ent_ids[tail], rel_ids[rel]])

        self.triples = torch.tensor(triples, dtype=torch.long)

        self.rel_categories = torch.zeros(len(rel_ids), dtype=torch.long)
        rel_categories_file = osp.join(directory, 'relations-cat.txt')
        self.has_rel_categories = False
        if osp.exists(rel_categories_file):
            with open(rel_categories_file) as f:
                for line in f:
                    rel, cat = line.strip().split()
                    self.rel_categories[rel_ids[rel]] = CATEGORY_IDS[cat]
            self.has_rel_categories = True

        # Save maps for reuse
        torch.save({'ent_ids': ent_ids, 'rel_ids': rel_ids}, maps_path)

        self.num_ents = len(entities)
        self.num_rels = len(relations)
        self.entities = torch.tensor([ent_ids[ent] for ent in entities])
        self.num_triples = self.triples.shape[0]
        self.directory = directory
        self.maps_path = maps_path
        self.neg_samples = neg_samples
        self.num_devices = num_devices

    def __getitem__(self, index):
        return self.triples[index]

    def __len__(self):
        return self.num_triples

    def collate_fn(self, data_list):
        """Given a batch of triples, return it together with a batch of
        corrupted triples where either the subject or object are replaced
        by a random entity. Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list)
        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples)
        return pos_pairs, rels, neg_idx


class TextGraphDataset(GraphDataset):
    """A dataset storing a graph, and textual descriptions of its entities.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        max_len: int, maximum number of tokens to read per description.
        neg_samples: int, number of negative samples to get per triple
        tokenizer: transformers.PreTrainedTokenizer or GloVeTokenizer, used
            to tokenize the text.
        drop_stopwords: bool, if set to True, punctuation and stopwords are
            dropped from entity descriptions.
        write_maps_file: bool, if set to True, dictionaries mapping
            entities and relations to IDs are saved to disk (for reuse with
            other datasets).
        drop_stopwords: bool
    """

    def __init__(self, triples_file, neg_samples,
                 drop_stopwords, write_maps_file=False,
                 num_devices=1):
        super().__init__(triples_file, neg_samples, write_maps_file,
                         num_devices)

        maps = torch.load(self.maps_path)
        ent_ids = maps['ent_ids']
        max_length = 32


        #self.text_data = torch.zeros((len(ent_ids), max_len + 1),
        #                             dtype=torch.long)
        self.text_data = dict()
        read_entities = set()
        progress = tqdm(desc='Reading entity descriptions',
                        total=len(ent_ids), mininterval=5)
        for text_file in ('entity2textlong.txt', 'entity2text.txt'):
            file_path = osp.join(self.directory, text_file)
            if not osp.exists(file_path):
                continue

            with open(file_path) as f:
                for line in f:
                    values = line.strip().split('\t')
                    entity = values[0]
                    text = ' '.join(values[1:])
                    if entity not in ent_ids:
                        continue
                    if entity in read_entities:
                        continue

                    read_entities.add(entity)
                    ent_id = ent_ids[entity]

                    if drop_stopwords:
                        tokens = nltk.word_tokenize(text)
                        text = ' '.join([t for t in tokens[:max_length] if
                                         t.lower() not in DROPPED])

                    #text_tokens = tokenizer.encode(text,
                    #                               max_length=max_len,
                    #                               return_tensors='pt')
                    text_sent = Sentence(text)
                    #text_len = text_tokens.shape[1]

                    # Starting slice of row contains token IDs
                    #print(ent_id)
                    self.text_data[ent_id] = text_sent

                    progress.update()

        progress.close()

        if len(read_entities) != len(ent_ids):
            raise ValueError(f'Read {len(read_entities):,} descriptions,'
                             f' but {len(ent_ids):,} were expected.')

        #torch.save(self.text_data,
        #           osp.join(self.directory, 'text_data.pt'))

    def get_entity_description(self, ent_ids):
        """Get entity descriptions for a tensor of entity IDs."""
        #print(ent_ids.shape)
        # Let us do parallel lookup for the different elements of ent_ids
        lookup = lambda ent: self.text_data[ent.item()]
        def arraylookup(t): 
            if t.shape!= torch.Size([]) and t.shape[0]>1:
                out = []
                for i in t:
                    out.append(lookup(i))
            else:
                out = lookup(t)
            return out
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            all_text = [arraylookup(ent_id) for ent_id in ent_ids]
        return all_text

    def collate_fn(self, data_list):
        """Given a batch of triples, return it in the form of
        entity descriptions, and the relation types between them.
        Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list) // self.num_devices
        if batch_size <= 1:
            raise ValueError('collate_text can only work with batch sizes'
                             ' larger than 1.')

        #print(data_list)
        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        #print(pos_pairs)
        text = self.get_entity_description(pos_pairs)

        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples,
                                                repeats=self.num_devices)

        return text, rels, neg_idx

    def collate_fn_with_ent(self, data_list):
        """
        Given a batch of triples, return it in form of entity descriptions, the entity type
        and the relation type between them. Use as a collate_fn for DataLoader when using autoencoder models.
        """
        batch_size = len(data_list) // self.num_devices
        if batch_size <= 1:
            raise ValueError('collate_text can only work with batch sizes'
                             ' larger than 1.')

        #print(data_list)
        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        #print(pos_pairs)
        text = self.get_entity_description(pos_pairs)

        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples,
                                                repeats=self.num_devices)

        return pos_pairs, text, rels, neg_idx

# utils

# In[14]:


import logging


# In[15]:


def get_model(model, dim, rel_model, loss_fn, num_entities, num_relations,
              encoder_name, regularizer, batch_size, aec):
    if model == 'bert-dkrl':
        return DKRL(dim, rel_model, loss_fn, num_relations, regularizer, batch_size,
                           encoder_name=encoder_name)
    elif model == 'glove-dkrl':
        return DKRL(dim, rel_model, loss_fn, num_relations, regularizer, batch_size,
                           embeddings='data/glove/glove.6B.300d.pt')
    elif model == 'flair-torchcoder':
        return EntAutoEnc(dim, rel_model, loss_fn, num_entities, num_relations, regularizer, batch_size, 
                    encoder_name=encoder_name, aec=aec)
    else:
        raise ValueError(f'Unkown model {model}')


def make_ent2idx(entities, max_ent_id):
    """Given a tensor with entity IDs, return a tensor indexed with
    an entity ID, containing the position of the entity.
    Empty positions are filled with -1.

    Example:
    > make_ent2idx(torch.tensor([4, 5, 0]))
    tensor([ 2, -1, -1, -1,  0,  1])
    """
    idx = torch.arange(entities.shape[0])
    ent2idx = torch.empty(max_ent_id + 1, dtype=torch.long).fill_(-1)
    ent2idx.scatter_(0, entities, idx)
    return ent2idx


def get_triple_filters(triples, graph, num_ents, ent2idx):
    """Given a set of triples, filter candidate entities that are valid
    substitutes of an entity in the triple at a given position (head or tail).
    For a particular triple, this allows to compute rankings for an entity of
    interest, against other entities in the graph that would actually be wrong
    substitutes.
    Results are returned as a mask array with a value of 1.0 for filtered
    entities, and 0.0 otherwise.

    Args:
        triples: Bx3 tensor of type torch.long, where B is the batch size,
            and each row contains a triple of the form (head, tail, rel)
        graph: nx.MultiDiGraph containing all edges used to filter candidates
        num_ents: int, number of candidate entities
        ent2idx: tensor, contains at index ent_id the index of the column for
            that entity in the output mask array
    """
    num_triples = triples.shape[0]
    heads_filter = torch.zeros((num_triples, num_ents), dtype=torch.bool)
    tails_filter = torch.zeros_like(heads_filter)

    triples = triples.tolist()
    for i, (head, tail, rel) in enumerate(triples):
        head_edges = graph.out_edges(head, data='weight')
        for (h, t, r) in head_edges:
            if r == rel and t != tail:
                ent_idx = ent2idx[t]
                if ent_idx != -1:
                    tails_filter[i, ent_idx] = True

        tail_edges = graph.in_edges(tail, data='weight')
        for (h, t, r) in tail_edges:
            if r == rel and h != head:
                ent_idx = ent2idx[h]
                if ent_idx != -1:
                    heads_filter[i, ent_idx] = True

    return heads_filter, tails_filter


def hit_at_k(predictions, ground_truth_idx, hit_positions):
    """Calculates mean number of hits@k. Higher values are ranked first.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class
        hit_positions: list, containing number of top K results to be
            considered as hits.

    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.
    """
    max_position = max(hit_positions)
    _, indices = predictions.topk(k=max_position)
    hits = []

    for position in hit_positions:
        idx_at_k = indices[:, :position]
        hits_at_k = (idx_at_k == ground_truth_idx).sum(dim=1).float().mean()
        hits.append(hits_at_k.item())

    return hits


def mrr(predictions, ground_truth_idx):
    """Calculates mean reciprocal rank (MRR) for given predictions
    and ground truth values. Higher values are ranked first.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class

    Returns: float, Mean reciprocal rank score
    """
    indices = predictions.argsort(descending=True)
    rankings = (indices == ground_truth_idx).nonzero()[:, 1].float() + 1.0
    return rankings.reciprocal()


def split_by_new_position(triples, mrr_values, new_entities):
    """Split MRR results by the position of new entity. Use to break down
    results for a triple where a new entity is at the head and the tail,
    at the head only, or the tail only.
    Since MRR is calculated by corrupting the head first, and then the head,
    the size of mrr_values should be twice the size of triples. The calculated
    MRR is then the average of the two cases.
    Args:
        triples: Bx3 tensor containing (head, tail, rel).
        mrr_values: 2B tensor, with first half containing MRR for corrupted
            triples at the head position, and second half at the tail position.
        new_entities: set, entities to be considered as new.
    Returns:
        mrr_by_position: tensor of 3 elements breaking down MRR by new entities
            at both positions, at head, and tail.
        mrr_pos_counts: tensor of 3 elements containing counts for each case.
    """
    mrr_by_position = torch.zeros(3, device=mrr_values.device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)
    num_triples = triples.shape[0]

    for i, (h, t, r) in enumerate(triples):
        head, tail = h.item(), t.item()
        mrr_val = (mrr_values[i] + mrr_values[i + num_triples]).item() / 2.0
        if head in new_entities and tail in new_entities:
            mrr_by_position[0] += mrr_val
            mrr_pos_counts[0] += 1.0
        elif head in new_entities:
            mrr_by_position[1] += mrr_val
            mrr_pos_counts[1] += 1.0
        elif tail in new_entities:
            mrr_by_position[2] += mrr_val
            mrr_pos_counts[2] += 1.0

    return mrr_by_position, mrr_pos_counts


def split_by_category(triples, mrr_values, rel_categories):
    mrr_by_category = torch.zeros([2, 4], device=mrr_values.device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float,
                                device=mrr_by_category.device)
    num_triples = triples.shape[0]

    for i, (h, t, r) in enumerate(triples):
        rel_category = rel_categories[r]

        mrr_val_head_pred = mrr_values[i]
        mrr_by_category[0, rel_category] += mrr_val_head_pred

        mrr_val_tail_pred = mrr_values[i + num_triples]
        mrr_by_category[1, rel_category] += mrr_val_tail_pred

        mrr_cat_count[0, rel_category] += 1

    return mrr_by_category, mrr_cat_count


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger("")
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger


# In[16]:

OUT_PATH = 'output/'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# In[17]:

# function for torchcoder autoencoder training
def prepare_dataset(sequential_data) :
    if type(sequential_data) == pd.DataFrame:
        data_in_numpy = np.array(sequential_data)
        data_in_tensor = torch.tensor(data_in_numpy, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == np.array:
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    else:
        unsqueezed_data = torch.tensor(sequential_data, dtype = torch.float)
        #unsqueezed_data = data_in_tensor.unsqueeze(2)
        
    seq_len = unsqueezed_data.shape[1]
    no_features = unsqueezed_data.shape[2] 
    # shape[0] is the number of batches
    
    return unsqueezed_data, seq_len, no_features

# function for flair embedding of the entity texts for training by the autoencoder.
def embed_text(text):
    encoder = FlairEmbeddings("en-forward-fast")
    emb_dim = encoder.embedding_length
    max_len = 32
    all_emb = []
    for j,ent_text in enumerate(text):
        emb = torch.zeros(max_len, emb_dim)
        encoder.embed(ent_text)
        toks = []
        for token in ent_text:
            toks.append(token.embedding.cpu().numpy())
        toks = torch.tensor(toks)
        if max_len>len(toks):
            emb[:len(toks)] = toks
        else:
            emb[:max_len] = toks[:max_len]
        #print(emb)
        all_emb.append(emb.numpy())
    #print(all_emb)
    all_emb = np.array(all_emb)
    embs = torch.tensor(all_emb)
    #print(embs.shape)
    embs = embs.to(device)

    embs = embs.view(-1, max_len, emb_dim).cpu()
    return embs

def eval_link_prediction(model, triples_loader, text_dataset, entities,
                         epoch, emb_batch_size,
                         prefix='', max_num_batches=None,
                         filtering_graph=None, new_entities=None,
                         return_embeddings=False):
    compute_filtered = filtering_graph is not None
    mrr_by_position = torch.zeros(3, dtype=torch.float).to(device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)

    rel_categories = triples_loader.dataset.rel_categories.to(device)
    mrr_by_category = torch.zeros([2, 4], dtype=torch.float).to(device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float).to(device)

    hit_positions = [1, 3, 10]
    hits_at_k = {pos: 0.0 for pos in hit_positions}
    mrr_value = 0.0
    mrr_filt = 0.0
    hits_at_k_filt = {pos: 0.0 for pos in hit_positions}

    if device != torch.device('cpu'):
        model = model.module

    if isinstance(model, InductiveLinkPrediction):
        num_entities = entities.shape[0]
        if compute_filtered:
            max_ent_id = max(filtering_graph.nodes)
        else:
            max_ent_id = entities.max()
        ent2idx = make_ent2idx(entities, max_ent_id)
    else:
        print("Error in model type not InductiveLink Pred")
        return -1

    # Create embedding lookup table for evaluation
    ent_emb = torch.zeros((num_entities, model.dim), dtype=torch.float,
                          device=device)
    idx = 0
    num_iters = np.ceil(num_entities / emb_batch_size)
    iters_count = 0
    while idx < num_entities:
        # Get a batch of entity IDs and encode them
        batch_ents = entities[idx:idx + emb_batch_size]
        #print(batch_ents)
        if isinstance(model, InductiveLinkPrediction):
            # Encode with entity descriptions
            data = text_dataset.get_entity_description(batch_ents)
            batch_emb = model(data)
        else:
            # Encode from lookup table
            batch_emb = model(batch_ents)

        ent_emb[idx:idx + batch_ents.shape[0]] = batch_emb

        iters_count += 1
        if iters_count % np.ceil(0.2 * num_iters) == 0:
            print(f'[{idx + batch_ents.shape[0]:,}/{num_entities:,}]')

        idx += emb_batch_size

    ent_emb = ent_emb.unsqueeze(0)

    batch_count = 0
    print('Computing metrics on set of triples')
    total = len(triples_loader) if max_num_batches is None else max_num_batches
    for i, triples in enumerate(triples_loader):
        if max_num_batches is not None and i == max_num_batches:
            break

        heads, tails, rels = torch.chunk(triples, chunks=3, dim=1)
        # Map entity IDs to positions in ent_emb
        heads = ent2idx[heads].to(device)
        tails = ent2idx[tails].to(device)

        assert heads.min() >= 0
        assert tails.min() >= 0

        # Embed triple
        head_embs = ent_emb.squeeze()[heads]
        tail_embs = ent_emb.squeeze()[tails]
        rel_embs = model.rel_emb(rels.to(device))

        # Score all possible heads and tails
        heads_predictions = model.score_fn(ent_emb, tail_embs, rel_embs)
        tails_predictions = model.score_fn(head_embs, ent_emb, rel_embs)

        pred_ents = torch.cat((heads_predictions, tails_predictions))
        true_ents = torch.cat((heads, tails))

        hits = hit_at_k(pred_ents, true_ents, hit_positions)
        for j, h in enumerate(hits):
            hits_at_k[hit_positions[j]] += h
        mrr_value += mrr(pred_ents, true_ents).mean().item()

        if compute_filtered:
            filters = get_triple_filters(triples, filtering_graph,
                                               num_entities, ent2idx)
            heads_filter, tails_filter = filters
            # Filter entities by assigning them the lowest score in the batch
            filter_mask = torch.cat((heads_filter, tails_filter)).to(device)
            pred_ents[filter_mask] = pred_ents.min() - 1.0
            hits_filt = hit_at_k(pred_ents, true_ents, hit_positions)
            for j, h in enumerate(hits_filt):
                hits_at_k_filt[hit_positions[j]] += h
            mrr_filt_per_triple = mrr(pred_ents, true_ents)
            mrr_filt += mrr_filt_per_triple.mean().item()

            if new_entities is not None:
                by_position = split_by_new_position(triples,
                                                          mrr_filt_per_triple,
                                                          new_entities)
                batch_mrr_by_position, batch_mrr_pos_counts = by_position
                mrr_by_position += batch_mrr_by_position
                mrr_pos_counts += batch_mrr_pos_counts

            if triples_loader.dataset.has_rel_categories:
                by_category = split_by_category(triples,
                                                      mrr_filt_per_triple,
                                                      rel_categories)
                batch_mrr_by_cat, batch_mrr_cat_count = by_category
                mrr_by_category += batch_mrr_by_cat
                mrr_cat_count += batch_mrr_cat_count

        batch_count += 1
        if (i + 1) % int(0.2 * total) == 0:
            print(f'[{i + 1:,}/{total:,}]')

    for hits_dict in (hits_at_k, hits_at_k_filt):
        for k in hits_dict:
            hits_dict[k] /= batch_count

    mrr_value = mrr_value / batch_count
    mrr_filt = mrr_filt / batch_count

    log_str = f'{prefix} mrr: {mrr_value:.4f}  '
    print(f'{prefix}_mrr', mrr_value, epoch)
    for k, value in hits_at_k.items():
        log_str += f'hits@{k}: {value:.4f}  '
        print(f'{prefix}_hits@{k}', value, epoch)

    if compute_filtered:
        log_str += f'mrr_filt: {mrr_filt:.4f}  '
        print(f'{prefix}_mrr_filt', mrr_filt, epoch)
        for k, value in hits_at_k_filt.items():
            log_str += f'hits@{k}_filt: {value:.4f}  '
            print(f'{prefix}_hits@{k}_filt', value, epoch)

    print(log_str)

    if new_entities is not None and compute_filtered:
        mrr_pos_counts[mrr_pos_counts < 1.0] = 1.0
        mrr_by_position = mrr_by_position / mrr_pos_counts
        log_str = ''
        for i, t in enumerate((f'{prefix}_mrr_filt_both_new',
                               f'{prefix}_mrr_filt_head_new',
                               f'{prefix}_mrr_filt_tail_new')):
            value = mrr_by_position[i].item()
            log_str += f'{t}: {value:.4f}  '
            print(t, value, epoch)
        print(log_str)

    if compute_filtered and triples_loader.dataset.has_rel_categories:
        mrr_cat_count[mrr_cat_count < 1.0] = 1.0
        mrr_by_category = mrr_by_category / mrr_cat_count

        for i, case in enumerate(['pred_head', 'pred_tail']):
            log_str = f'{case} '
            for cat, cat_id in CATEGORY_IDS.items():
                log_str += f'{cat}_mrr: {mrr_by_category[i, cat_id]:.4f}  '
            print(log_str)

    if return_embeddings:
        out = (mrr_value, ent_emb)
    else:
        out = (mrr_value, None)

    return out


# In[18]:


def link_prediction(dataset, inductive, dim, model, rel_model, loss_fn,
                    encoder_name, regularizer, max_len, num_negatives, lr,
                    use_scheduler, batch_size, emb_batch_size, eval_batch_size,
                    max_epochs, checkpoint):
    drop_stopwords = model in {'bert-dkrl', 'glove-dkrl'}

    prefix = 'ind-' if inductive and model != 'transductive' else ''
    triples_file = f'data/{dataset}/{prefix}train.tsv'

    if device != torch.device('cpu'):
        num_devices = torch.cuda.device_count()
        if batch_size % num_devices != 0:
            raise ValueError(f'Batch size ({batch_size}) must be a multiple of'
                             f' the number of CUDA devices ({num_devices})')
        print(f'CUDA devices used: {num_devices}')
        #raise ValueError('Training on GPU')
    else:
        num_devices = 1
        print('Training on CPU')


    train_data = TextGraphDataset(triples_file, num_negatives,
                                      drop_stopwords,
                                      write_maps_file=True,
                                      num_devices=num_devices)

    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=0, drop_last=True)

    train_eval_loader = DataLoader(train_data, eval_batch_size)

    valid_data = GraphDataset(f'data/{dataset}/{prefix}dev.tsv')
    valid_loader = DataLoader(valid_data, eval_batch_size)

    test_data = GraphDataset(f'data/{dataset}/{prefix}test.tsv')
    test_loader = DataLoader(test_data, eval_batch_size)

    # Build graph with all triples to compute filtered metrics

    graph = nx.MultiDiGraph()
    all_triples = torch.cat((train_data.triples,
                                valid_data.triples,
                                test_data.triples))
    graph.add_weighted_edges_from(all_triples.tolist())

    train_ent = set(train_data.entities.tolist())
    train_val_ent = set(valid_data.entities.tolist()).union(train_ent)
    train_val_test_ent = set(test_data.entities.tolist()).union(train_val_ent)
    val_new_ents = train_val_ent.difference(train_ent)
    test_new_ents = train_val_test_ent.difference(train_val_ent)
    
    print('num_train_entities', len(train_ent))

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    if model=='flair-torchcoder':
        ### TODO: Add autoencoder here which will take all entity texts - train the lstmaec 
        ### and pass to model. probably add unftcion in textgraphdata to get all texts
        input_data = train_data.get_entity_description(train_ent)
        input_data = embed_text(input_data)
        refined_input_data, seq_len, no_features = prepare_dataset(input_data)
        aec = LSTM_AE(seq_len, no_features, embedding_dim=128, learning_rate=1e-3, every_epoch_print=100, epochs=2, patience=20, max_grad_norm=0.005)
        aec = aec.to(device)
        final_loss = aec.fit(refined_input_data) 
        print("Autoencoder training loss: ", final_loss)
    else:
        aec = None

    model = get_model(model, dim, rel_model, loss_fn,
                            len(train_val_test_ent), train_data.num_rels,
                            encoder_name, regularizer, batch_size, aec)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    if device != torch.device('cpu'):
        model = torch.nn.DataParallel(model).to(device)
    
    if device== torch.device('cpu'):
        model = model.cpu()

    optimizer = Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * max_epochs
    if use_scheduler:
        warmup = int(0.2 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup,
                                                    num_training_steps=total_steps)
    best_valid_mrr = 0.0
    checkpoint_file = osp.join(OUT_PATH, f'model-flair.pt')
    for epoch in range(0, max_epochs):
        start = time()
        train_loss = 0
        for step, data in enumerate(train_loader):
            loss = model(*data).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            train_loss += loss.item()

            if step % int(0.05 * len(train_loader)) == 0:
                print(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                print('batch_loss', loss.item())

            #if step==0:
            #    break

        stop = time()
        print('train_loss', train_loss / len(train_loader), epoch)
        print('train time for epoch:', stop-start)

        #if dataset != 'Wikidata5M':
        #    print('Evaluating on sample of training set')
        #    eval_link_prediction(model, train_eval_loader, train_data, train_ent,
        #                         epoch, emb_batch_size, prefix='train',
        #                         max_num_batches=len(valid_loader))

        #print('Evaluating on validation set')
        #val_mrr, _ = eval_link_prediction(model, valid_loader, train_data,
        #                                  train_val_ent, epoch,
        #                                  emb_batch_size, prefix='valid')

        # Keep checkpoint of best performing model (based on raw MRR)
        #if val_mrr > best_valid_mrr:
        #    best_valid_mrr = val_mrr
        # SUPARNA - original code has the next line indented to save for best performing model
        if epoch%5==0:
            torch.save(model.state_dict(), checkpoint_file)
        


    # Evaluate with best performing checkpoint
    #if max_epochs > 0:
    #print(model)
    model.load_state_dict(torch.load(checkpoint_file))

    start = time()
    #print('Evaluating on validation set (with filtering)')
    #eval_link_prediction(model, valid_loader, train_data, train_val_ent,
    #                     max_epochs + 1, emb_batch_size, prefix='valid',
    #                     filtering_graph=graph,
    #                     new_entities=val_new_ents)

    print('Evaluating on test set')
    _, ent_emb = eval_link_prediction(model, test_loader, train_data,
                                      train_val_test_ent, max_epochs + 1,
                                      emb_batch_size, prefix='test',
                                      filtering_graph=graph,
                                      new_entities=test_new_ents,
                                      return_embeddings=True)

    stop = time()

    print("Eval run times:", stop-start)
    # Save final entity embeddings obtained with trained encoder
    #torch.save(ent_emb, osp.join(OUT_PATH, f'ent_emb-base.pt'))
    #torch.save(train_val_test_ent, osp.join(OUT_PATH, f'ents-base.pt'))


# In[19]:


link_prediction(dataset='FB15k-237', inductive=True, dim=128, model='bert-dkrl', rel_model='transe', loss_fn='margin', encoder_name='flair', regularizer=1e-2, max_len=32, num_negatives=64, lr=1e-4, use_scheduler=False, batch_size=64, emb_batch_size=512, eval_batch_size=128, max_epochs=30, checkpoint=None)

