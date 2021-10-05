import langdetect
import pandas as pd
import numpy as np
import nltk
import json
import flask
from flask import request
import faiss
import torch
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple
import string
import os
from tqdm import tqdm
import time
import torch.nn.functional as F

FAISS_INIT = False
LOADED_INIT = False

def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return float(y_value)
    elif gain_scheme == 'exp2':
        return float((2 ** y_value) - 1)


def hadle_punctuation(inp_str: str) -> str:
    handled = inp_str
    for i in range(len(string.punctuation)):
        handled = handled.replace(string.punctuation[i], ' ')
    return handled


def simple_preproc(inp_str: str) -> List[str]:
    if inp_str == None:
        return None
    handled_str = hadle_punctuation(inp_str).lower()
    tokens = nltk.word_tokenize(handled_str)
    return tokens


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [20, 10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.load(EMB_PATH_KNRM)['weight'],
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(torch.load(MLP_PATH))

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                    self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class Solution:
    def __init__(self,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.1,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [100, 100, 100],
                 ):

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp

        self.docs = {}
        self.idx_to_texts = {}
        self.indexed_texts = {}

        self.faiss_init = False
        self.loaded_state = False
        self.vec_length = 0
        self.model, self.vocab = self.build_knrm_model()
        self.loaded_state = True


    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int]]:

        torch.manual_seed(self.random_seed)

        knrm = KNRM(freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)

        with open(VOCAB_PATH) as json_file:
            vocab = json.load(json_file)

        return knrm, vocab

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        vocab = {}
        f = open(file_path, encoding="utf8")
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            vocab[word] = coefs
        f.close()
        return vocab

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        if tokenized_text == None:
            return None
        result = []
        for token in tokenized_text:
            result.append(self.vocab[token])
        return result


EMB_PATH_KNRM = os.environ.get('EMB_PATH_KNRM')
VOCAB_PATH = os.environ.get('VOCAB_PATH')
MLP_PATH = os.environ.get('MLP_PATH')
EMB_PATH_GLOVE = os.environ.get('EMB_PATH_GLOVE')


LOADED_INIT = True


app = flask.Flask(__name__)


solution = Solution()


def vec_to_compare(query_vec):
    if query_vec == None:
        return None
    comparison_vec = np.zeros(solution.vec_length)
    selected_len = min(solution.vec_length, len(query_vec))
    comparison_vec[:selected_len] = query_vec[:selected_len]
    return comparison_vec.astype(np.int32)


def vec_neighbours(compare_vec, num):
    # num = min(100, len(solution.docs))
    _, indices = solution.index.search(np.asarray([compare_vec]).astype('float32'), num)
    return indices[0]


def vec_true_indices(vec_neigh):
    return [solution.idx_to_texts[str(idx)] for idx in vec_neigh]


def vec_true_text(true_idx):
    return [solution.docs[idx] for idx in true_idx]


def vec_knrm_preds(df):
    num = min(200, len(solution.docs))
    indices_for_preds = [solution.indexed_texts[idx] for idx in df['true_idx']]
    tensor_for_preds = torch.tensor(indices_for_preds, dtype=torch.int64)
    tensor_for_query = torch.tensor([df['comparison_vecs']], dtype=torch.int64).repeat(num, 1)

    inputs = {
        'query': tensor_for_query,
        'document': tensor_for_preds
    }

    preds = solution.model.predict(inputs)
    return preds


def vec_suggested_by_knrm(df):
    return [(df['true_idx'][i], df['true_text'][i]) for i in torch.argsort((df['preds'].T[0]) * -1, dim=0)]


def np_knrm_preds(true_idx, comparison_vec, num):
    # num = min(15, len(solution.docs))
    indices_for_preds = [solution.indexed_texts[idx] for idx in true_idx]
    tensor_for_preds = torch.tensor(indices_for_preds, dtype=torch.int64)
    tensor_for_query = torch.tensor([comparison_vec], dtype=torch.int64).repeat(num, 1)

    inputs = {
        'query': tensor_for_query,
        'document': tensor_for_preds
    }

    preds = solution.model.predict(inputs)
    return preds


def np_vec_suggested_by_knrm(true_idx, true_text, preds):
    return [(true_idx[i], true_text[i]) for i in torch.argsort((preds.T[0]) * -1, dim=0)]


@app.route('/ping')
def ping():
    if LOADED_INIT:
        result = {'status': 'ok'}
        return result


@app.route('/query', methods=['POST'])
def query():
    if not(solution.faiss_init):
        result = {'status': 'FAISS is not initialized!'}
        return result

    else:

        data = json.loads(request.json)
        # data = request.json
        lang_check = []
        suggestions = []
        num = min(200, len(solution.docs))

        simple_time = time.time()
        for query_faiss in tqdm(data['queries']):
            if langdetect.detect(query_faiss) == 'en':

                lang_check.append(True)
                query_vec = solution._tokenized_text_to_index(simple_preproc(query_faiss))
                comparison_vec = np.zeros(solution.vec_length )
                selected_len = min(solution.vec_length, len(query_vec))
                comparison_vec[:selected_len] = query_vec[:selected_len]
                faiss_vec = torch.mean(solution.model.embeddings(torch.LongTensor(np.asarray(query_vec[:selected_len]))), dim=0)
                distances, indices = solution.index.search(np.asarray([np.asarray(faiss_vec)]), num)
                suggest = []
                indices_for_preds = []
                for i in range(len(indices[0])):
                    idx = str(indices[0, i])
                    if idx != '-1':
                        true_idx = solution.idx_to_texts[idx]
                        true_text = solution.docs[true_idx]
                        suggest.append((true_idx, true_text))

                        true_token_indices = solution.indexed_texts[true_idx]
                        indices_for_preds.append(true_token_indices)

                tensor_for_preds = torch.tensor(indices_for_preds, dtype=torch.int64)
                tensor_for_query = torch.tensor([comparison_vec], dtype=torch.int64).repeat(len(tensor_for_preds), 1)

                inputs = {
                    'query': tensor_for_query,
                    'document': tensor_for_preds
                }
                preds = solution.model.predict(inputs)
                sorted_suggest = [suggest[i] for i in torch.argsort((preds.T[0]) * -1, dim=0)]
                suggestions.append(sorted_suggest[:10])
            else:
                lang_check.append(False)
                suggestions.append(None)
        result = {
                    'lang_check': lang_check,
                    'suggestions': suggestions
                  }
        print('simple')
        print(time.time() - simple_time)
        return result


@app.route('/update_index', methods=['POST'])
def update_index():
    data = json.loads(request.json)
    solution.docs = data['documents']

    listed_docs = []
    max_length = 0
    for key in tqdm(solution.docs.keys()):
        tokens = simple_preproc(solution.docs[key])
        max_length = max(len(tokens), max_length)
        solution.idx_to_texts[str(len(listed_docs))] = key
        indexed_text = solution._tokenized_text_to_index(tokens)
        listed_docs.append(indexed_text)

    fetched_length = int(np.mean(list(map(len, listed_docs))) + np.std(list(map(len, listed_docs))))

    vector = np.zeros((len(solution.docs), fetched_length))
    vector_emb = np.zeros((len(solution.docs), 50))

    for i in range(len(listed_docs)):
        vector[i, :len(listed_docs[i])] = listed_docs[i][:fetched_length]
        vector_emb[i] = torch.mean(solution.model.embeddings(torch.LongTensor(listed_docs[i][:fetched_length])), dim=0)
        solution.indexed_texts[solution.idx_to_texts[str(i)]] = vector[i]

    solution.index = faiss.index_factory(50, "Flat", faiss.METRIC_L2)
    train_vector = vector_emb.astype('float32')
    solution.index.add(train_vector)
    solution.vec_length = fetched_length

    result = {
        'status': 'ok',
        'index_size': len(solution.docs)
              }

    solution.faiss_init = True

    return result

