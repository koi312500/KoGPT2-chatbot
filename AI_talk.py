# -*- coding: utf-8 -*-
import argparse
import logging
import math

import gluonnlp as nlp
import mxnet as mx
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from kogpt2.utils import get_tokenizer
from mxnet import gluon, nd
from mxnet.gluon import nn


logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'


class ChatDataset(gluon.data.Dataset):
    def __init__(self, chats, tok_path, vocab, max_len=32):
        self._data = chats
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token])

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [<mask>, <mask>, ...., <mask>, ..., A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        return (self.padder(self.vocab[q_toked + a_toked]), nd.array(mask),
                self.padder(self.vocab[labels]))


class KoGPT2Chat(nn.HybridBlock):
    def __init__(self, kogpt2, prefix=None, params=None):
        super(KoGPT2Chat, self).__init__(prefix=prefix, params=params)
        self.kogpt2 = kogpt2

    def hybrid_forward(self, F, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output


if mx.context.num_gpus() > 0:
    ctx = mx.gpu()
else:
    ctx = mx.cpu()

def Load_Model():
    global vocab_global
    global sent_tokens_global
    global kogptqa_global
    global tok_global
    tok_path = get_tokenizer()
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    kogptqa = KoGPT2Chat(model)
    kogptqa.load_parameters("KoGPT2-chatbot\kogpt2_chat.params", ctx=ctx)
    sent_tokens = tok("0")
    vocab_global = vocab
    sent_tokens_global = sent_tokens
    kogptqa_global = kogptqa
    tok_global = tok

def Chat_To_AI(word_from_user):    
    word_from_user = word_from_user.strip()
    word_from_user_tok = tok_global(word_from_user)
    a = ''
    a_tok = []
    while 1:
        input_ids = mx.nd.array([vocab_global[U_TKN]] + vocab_global[word_from_user_tok] +
                                vocab_global[EOS, SENT] + vocab_global[sent_tokens_global] +
                                vocab_global[EOS, S_TKN] +
                                vocab_global[a_tok]).expand_dims(axis=0)
        pred = kogptqa_global(input_ids.as_in_context(ctx))
        gen = vocab_global.to_tokens(
            mx.nd.argmax(
                pred,
                axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace('▁', ' ')
        a_tok = tok_global(a)
    return a



if __name__ == "__main__":
    Load_Model()
    print(Chat_To_AI("안녕!"))
    
