from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from qa_bidaf_model import BidafEncoder, BidafQASystem, BidafDecoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 20.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 300, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("question_max_length", 60, "Maximum length of an input question.")
tf.app.flags.DEFINE_integer("context_paragraph_max_length", 300, "Maximum length of a context paragraph.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train), it will use the model name as a second directory.")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.") # want to print every 1000 examples (so approx 100 times per epoch)
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_boolean("debug", False, "Are we debugging?")
tf.app.flags.DEFINE_integer("debug_training_size", 100, "A smaller training size for debugging, so that epochs are quick, and we can test logging etc")
tf.app.flags.DEFINE_boolean("log_score", True, "If we want to log f1 and em score in a txt file, alongside the model params in the pa4/train/<model_name> directory")
tf.app.flags.DEFINE_string("model_name", "BiDAF", "The model to use, pick from: 'baseline', 'embedding_backprop', 'deep_encoder_2layer', 'deep_encoder_3layer', 'deep_decoder_2layer', 'deep_decoder_3layer', 'QRNNs', 'BiDAF'")
tf.app.flags.DEFINE_string("model_version", "_3", "Make this '' for initial model, if we ever want to retrain a model, then we can use this (with '_i') to not overwrite the original data")
tf.app.flags.DEFINE_boolean("clip_norms", True, "Do we wish to clip norms?")
tf.app.flags.DEFINE_string("train_prefix", "train", "Prefix of all the training data files")
tf.app.flags.DEFINE_string("val_prefix", "val", "Prefix of all the validation data files")
tf.app.flags.DEFINE_integer("epoch_base", 10, "The first epoch, so that we are saving the correct model and outputting the correct numbers if we restarted")


FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir
            

def main(_):

    # Work out flags that decide the architecture we're doing to use (defaults for the baseline)
    backprop_word_embeddings = False
    encoder_layers = 1
    decoder_layers = 1

    if FLAGS.model_name == "embedding_backprop":
        backprop_word_embeddings = True
    elif FLAGS.model_name == "deep_encoder_2layer":
        backprop_word_embeddings = True # false if that did better
        encoder_layers = 2
    elif FLAGS.model_name == "deep_encoder_3layer":
        backprop_word_embeddings = True # false if that did better
        encoder_layers = 3
    elif FLAGS.model_name == "deep_decoder_2layer":
        backprop_word_embeddings = True # false if that did better
        encoder_layers = 2 # 1, 2 if one of them did better 
        decoder_layer = 2
    elif FLAGS.model_name == "deep_decoder_3layer":
        backprop_word_embeddings = True # false if that did better
        encoder_layers = 3 # 1, 2 if one of them did better
        decoder_layers = 3 
    elif FLAGS.model_name == "BiDAF":
        # do nothing
        pass
    elif not (FLAGS.model_name == "baseline"): 
        raise Exception("Invalid model name selected")

    # Do what you need to load datasets from FLAGS.data_dir
    #train_dataset = load_dataset(FLAGS.data_dir, "train")
    #val_dataset = load_dataset(FLAGS.data_dir, "val")
    train_dataset_address = FLAGS.data_dir + "/" + FLAGS.train_prefix 
    val_dataset_address = FLAGS.data_dir + "/" + FLAGS.val_prefix 
    
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path) # vocab = {words : indices}, rev_vocab = [words] (where each word is at index as dictated by vocab)
    
    # load in the embeddings
    embeddings = np.load(embed_path)['glove']

    if FLAGS.model_name == "BiDAF":
        encoder = BidafEncoder(FLAGS.state_size, FLAGS.embedding_size)
        decoder = BidafDecoder(FLAGS.state_size)
        qa = BidafQASystem(encoder, decoder, embeddings, backprop_word_embeddings)
    else:
        encoder = Encoder(FLAGS.state_size, FLAGS.embedding_size, FLAGS.dropout, encoder_layers, FLAGS.model_name)
        decoder = Decoder(FLAGS.state_size, FLAGS.dropout, decoder_layers)
        qa = QASystem(encoder, decoder, embeddings, backprop_word_embeddings)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        load_train_dir += "/" + FLAGS.model_name + FLAGS.model_version # each model gets its own subdirectory
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        save_train_dir += "/" + FLAGS.model_name + FLAGS.model_version # each model gets its own subdirectory
        if not os.path.exists(save_train_dir):
            os.makedirs(save_train_dir)
        create_train_dir = (save_train_dir)
        qa.train(sess, train_dataset_address, val_dataset_address, save_train_dir)


if __name__ == "__main__":
    tf.app.run()
