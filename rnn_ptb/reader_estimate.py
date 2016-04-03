# -*- coding: utf-8 -*-

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile


"""
# args: file path
# returns: array list which has separated words
"""
def _read_words(filename):
  with gfile.GFile(filename, "r") as f:
    # Convert the new line to the End of String.
    return f.read().replace("\n", "<eos>").split()

"""
# args: テキストファイル(訓練文書)
# returns: 引数に与えた訓練文書内に存在する各単語を, そのファイル内での出現回数の多い順にソートし(この時文末の<EOS>も単語の一つとしてカウント), その出現頻度順位(0から始まる)を単語に付与したハッシュマップ. 要はText Frequency.
"""
def _build_vocab(filename):
  data = _read_words(filename)

  # count a word. and keep as hash map. ---(1)
  counter = collections.Counter(data)
  # sort (1) by counter. ---(2)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])

  # do tuple of (2), and exclude counter. ---(3)
  words, _ = list(zip(*count_pairs))
  # (3)に対して, (3)の各単語の並び順を各単語に数値として付与し, ハッシュマップとして保持する. ちなみに数値は0から始まる. つまり, 数が小さいほど出現回数が高い.
  word_to_id = dict(zip(words, range(len(words))))

  # print ("\n", "word_to_id...")
  # print (word_to_id, "\n")

  return word_to_id

"""
# Word to Vector
# args: テキストファイル(評価文書), TextFrequencyを持つハッシュマップ.
# returns: 引数に与えた評価文書内の単語(*)の並び順と, 訓練文書内での単語(*)の出現頻度順位を持つ, 数字配列(配列の並びが単語(*)の並び順で, 各要素が各単語(*)が持つ出現頻度順位である).
# なので, 評価文書にあるが, 訓練文書に存在しない単語があった場合には, エラーとなる
"""
def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  # 入力した文書の各単語に対してその文書における出現回数を付与する. 返り値は, 語の並び順と出現回数の情報を持つ.
  return [word_to_id[word] for word in data]

##########
vocab = {}
inv_vocab = {}
##########
def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  # valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  ##########
  def load_data(filename):
    global vocab
    global inv_vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
      if word not in vocab:
        vocab[word] = len(vocab)
        # 単語逆引き用辞書
        inv_vocab[len(vocab)-1]=word
      dataset[i] = vocab[word]

    print('#vocab =', len(vocab))
    print (inv_vocab)

    return dataset

  load_data(test_path)
  load_data(train_path)

  # def load_data_train(f):
  #   ff={}
  #   data = _read_words(f)
  #   for i, w in enumerate(data):
  #     ff[str(i)] = str(w)
  #   print (ff)
  #
  # load_data_train(train_path)
  ##########

  # 訓練文書内の単語の出現頻度順に並び替えて, その順位を付与した辞書
  word_to_id = _build_vocab(train_path)
  # word_to_id = _build_vocab(test_path)

  # 訓練文書を基に, 評価文書内の単語の訓練文書における出現頻度順位を持った数字配列
  # train_data = _file_to_word_ids(train_path, word_to_id)
  # valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)

  # 訓練文書の総単語数
  vocabulary = len(word_to_id)

  # return train_data, valid_data, test_data, vocabulary
  return test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
