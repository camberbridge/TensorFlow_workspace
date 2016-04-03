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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

To compile on CPU:
  bazel build -c opt tensorflow/models/rnn/ptb:ptb_word_lm
To compile on GPU:
  bazel build -c opt tensorflow --config=cuda \
    tensorflow/models/rnn/ptb:ptb_word_lm
To run:
  ./bazel-bin/.../ptb_word_lm --data_path=/tmp/simple-examples/data/

教師データの入力テキストには行頭に半角スペースを入れて, 行末には半角スペースを入れるルールで.
実際に学習する時は, 行頭の半角スペースは無視して, 行末の半角スペースは<EOS>に置換して1単語扱いにする.
でも, 単語の予測をする際には文末の<EOS>は無視(∵単語総数-1)する（文中の<EOS>は予測単語の対象とする). ゆえに, 評価文書内の単語の数だけrun_epochを繰り返す.
vocab_sizeは, train_dataの単語種類数にする.
test_dataの各単語の次に出現する単語は, train_dataから選ばれる.

実際にモデルで試す際に(学習も同様(trainとvalid間の)), 未知語(訓練データにあるが評価データに出現しない単語)は<UNK>に変換する必要がある
訓練データ, 評価データともに, 重複単語は1つにする. したがって, vocab_sizeの10000は訓練データ内の単語の種類数.

"""
from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function


import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
# from tensorflow.models.rnn.ptb import reader
import reader_estimate
import time
import os

import tensorflow.python.platform
import datetime
dateTime = datetime.datetime.today()

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("save_model", "model-" + str(dateTime.month) + str(dateTime.day) + str(dateTime.hour) + str(dateTime.minute) + ".ckpt", "File name of a model data.")

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    states = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
        states.append(state)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    # This logits has a (y = Wx + b). Because tf.nn.xw_plus_b is convenience wrapper to perform the Wx + b matrix multiplication.
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))

    ##########
    # logits = tf.matmul(output, [size, vocab_size]) + [vocab_size]
    probabilities = tf.nn.softmax(logits)
    print ("Prpbability...\n", probabilities, "\n")
    ##########

    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = states[-1]

    ##########
    self._proba = tf.nn.softmax(logits)
    ##########

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  ##########
  @property
  def proba(self):
    return self._proba
  ##########

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data.
  data: 入力テキストの単語の語順(配列の並び)と出現頻度順位(要素の値) ex.[0, 3, 1, 0, 5]
  """
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader_estimate.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    # cost, state, _ = session.run([m.cost, m.final_state, eval_op],
    #                              {m.input_data: x,
    #                               m.targets: y,
    #                               m.initial_state: state})
    cost, state, proba,  _ = session.run([m.cost, m.final_state, m.proba, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})

    # 評価文書の文末の<EOS>を除いた単語の数だけ回す
    print ("==========")
    print ("The length of proba[0]: ", len(proba[0]))
    print ("The best probability is ...", np.argmax(proba))

    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


##########
vocab = {}
inv_vocab = {}


def load_data(filename):
  global vocab
  global inv_vocab
  words = open(filename).read().replace('\n', '<eos>').strip().split()
  dataset = np.ndarray((len(words),), dtype=np.int32)
  for i, word in enumerate(words):
    if word not in vocab:
      vocab[word] = len(vocab)
      # 単語逆引き用辞書
      inv_vocab[len(vocab) - 1] = word
  dataset[i] = vocab[word]

  print('#vocab =', len(vocab))
  print(inv_vocab)

  return dataset
##########


def main(unused_args):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader_estimate.ptb_raw_data(FLAGS.data_path)
  # raw_data: 訓練文書を基に, 評価文書内の単語の訓練文書における出現頻度順位を持った数字配列, 訓練文書の総単語数
  # train_data, valid_data, test_data, _ = raw_data
  # raw_data: 訓練文書を基に, 評価文書内の単語の訓練文書における出現頻度順位を持った数字配列, 訓練文書の総単語数
  test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  train_path = os.path.join(FLAGS.data_path, "ptb.train.txt")
  valid_path = os.path.join(FLAGS.data_path, "ptb.valid.txt")
  test_path = os.path.join(FLAGS.data_path, "ptb.test.txt")

  ##########
  load_data(train_path)
  load_data(valid_path)
  load_data(test_path)
  ##########

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    # with tf.variable_scope("model", reuse=None, initializer=initializer):
    #   m = PTBModel(is_training=True, config=config)
    # with tf.variable_scope("model", reuse=True, initializer=initializer):
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      # mvalid = PTBModel(is_training=False, config=config)
      mtest = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()

    """
    # Load a model which use as the classifier
    """
    saver = tf.train.Saver()
    saver.restore(session, "./SAVE/model-351636.ckpt")

    # for i in range(config.max_max_epoch):
    #   lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
    #   m.assign_lr(session, config.learning_rate * lr_decay)
    #
    #   print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
    #   train_perplexity = run_epoch(session, m, train_data, m.train_op,
    #                                verbose=True)
    #   print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    #   valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
    #   print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    """
    # Here. Give test_data a optional word vector
    """
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("\nTest Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()

  # save_path = saver.save(session, FLAGS.save_model)
