"""Allow for making call to endpoint for sentiment analysis."""

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import nltk
import util.dataprocessor
import models.sentiment
import util.vocabmapping
import ConfigParser
from flask import Flask, request

app = Flask(__name__)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/',
                    'Directory to store/restore checkpoints')
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')
flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')

model = None


def prepare_text(text, max_seq_length, vocab_mapping):
    """Input: text_list. Returns: inputs, seq_lengths, targets."""
    data = np.array([i for i in range(max_seq_length)])
    targets = []
    seq_lengths = []
    tokens = tokenize(text)
    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]

    indices = [vocab_mapping.getIndex(j) for j in tokens]
    if len(indices) < max_seq_length:
        indices = indices + \
            [vocab_mapping.getIndex("<PAD>")
             for i in range(max_seq_length - len(indices))]
    else:
        indices = indices[0:max_seq_length]
    seq_lengths.append(len(tokens))

    data = np.vstack((data, indices))
    targets.append(1)

    onehot = np.zeros((len(targets), 2))
    onehot[np.arange(len(targets)), targets] = 1
    return data[1::], np.array(seq_lengths), onehot


def load_model(session, vocab_size):
    """Load trained model."""
    hyper_params = read_config_file()
    model = models.sentiment.SentimentModel(vocab_size,
                                            hyper_params["hidden_size"],
                                            1.0,
                                            hyper_params["num_layers"],
                                            hyper_params["grad_clip"],
                                            hyper_params["max_seq_length"],
                                            hyper_params["learning_rate"],
                                            hyper_params["lr_decay_factor"],
                                            1)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from {0}".format(
            ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Double check you got the checkpoint_dir right..."
        print "Model not found..."
        model = None
    return model


def read_config_file():
    """Read config file, returns dictionary of network params."""
    config = ConfigParser.ConfigParser()
    config.read(FLAGS.config_file)
    dic = {}
    sentiment_section = "sentiment_network_params"
    general_section = "general"
    dic["num_layers"] = config.getint(sentiment_section, "num_layers")
    dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
    dic["dropout"] = config.getfloat(sentiment_section, "dropout")
    dic["batch_size"] = config.getint(sentiment_section, "batch_size")
    dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
    dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
    dic["lr_decay_factor"] = config.getfloat(
        sentiment_section, "lr_decay_factor")
    dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
    dic["use_config_file_if_checkpoint_exists"] = config.getboolean(
        general_section, "use_config_file_if_checkpoint_exists")
    dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
    dic["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
    dic["max_seq_length"] = config.getint(general_section,
                                          "max_seq_length")
    dic["steps_per_checkpoint"] = config.getint(general_section,
                                                "steps_per_checkpoint")
    return dic


def tokenize(text):
    """Tokenize the text using nltk."""
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)

vocab_mapping = util.vocabmapping.VocabMapping()

sess = tf.Session()
model = load_model(sess, vocab_mapping.getSize())
print 'model loaded'


@app.route('/')
def hello_world():
    """Simple hello world function."""
    return 'MAIN FUNCTION CALLED READY FOR STUFF'


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    """Analyze text and extract sentiment."""
    content = request.get_json()
    if model is None:
        return
    max_seq_length = model.max_seq_length
    test_data = content['text']
    data, seq_lengths, targets = prepare_text(
        test_data, max_seq_length, vocab_mapping)
    input_feed = {}
    input_feed[model.seq_input.name] = data
    input_feed[model.target.name] = targets
    input_feed[model.seq_lengths.name] = seq_lengths
    output_feed = [model.y]
    outputs = sess.run(output_feed, input_feed)
    score = np.argmax(outputs[0])
    probability = outputs[0].max(axis=1)[0]
    message = 'Value of sentiment: '
    if score > 0:
        message = message + 'positive'
    else:
        message = message + 'negative'
    message = message + ' with probability: ' + str(probability)
    return message

app.run(host='0.0.0.0')
