from __future__ import print_function, division
from rnnlm_ops import RnnlmOp, run_epoch
from dataset import SentenceSet
from config import Config 
import tensorflow as tf
import os

class LogProbs(RnnlmOp):
  def __init__(self, model_dir, batch_size = 1, model="default"):

    import flags
    flags.FLAGS.model_dir = model_dir
    flags.FLAGS.model = model
    self.config, self.params = flags.get_config(), flags.FLAGS

    self.config.batch_size = batch_size
    self.config.num_steps = 0 # seen as independent sentence.

    self.sentence_set = None

    self.graph = tf.Graph()
    with self.graph.as_default():
      super(LogProbs, self).__init__(self.config, self.params)
    
      tf_config = tf.ConfigProto()
      tf_config.allow_soft_placement = True
      self.sess = tf.Session(config=tf_config, graph = self.graph)

      self.build_graph()
      self.io.restore_session(self.sess)

    print ('init done.')

  def _load_data(self,raw):
    if self.sentence_set == None:
      self.sentence_set = SentenceSet(raw, self.config.batch_size, shuffle=False)
    else:
      self.sentence_set.sentences = self.sentence_set._raw_to_sentences(raw)

  def _build_graph(self):
    config = self.config
    eval_config = Config(clone=config)

    initializer = self.model_initializer
    with tf.name_scope("LogProbs"):
      with tf.variable_scope("Model", reuse=False, initializer=initializer):
        self.test_model = self.Model(config=eval_config, is_training=False)

  def __call__(self, raw_data = None):
    return self._run(raw_data)

  def _run(self, raw_data = None):
    assert self.sentence_set != None or raw_data != None, "Do not load a sentence set!"
    self._load_data(raw_data)
    with self.graph.as_default():
      ans = run_epoch(self.sess, self.test_model, self.sentence_set, outputs = ['costs'], sentences_mode=True)
    return ans

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  L = LogProbs('small_model', batch_size = 2)
  res = L([[4, 2], [4, 0, 3], [4, 3], [4, 4, 3, 4]])
  # res = L([[4, 2, 3]])
  print (res)
