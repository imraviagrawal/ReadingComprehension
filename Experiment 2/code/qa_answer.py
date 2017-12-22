from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from qa_bidaf_model import BidafEncoder, BidafQASystem, BidafDecoder

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
# tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
# tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
# tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
# tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
# tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
# tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
# tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
# tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
# tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 24, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 50, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 300, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("question_max_length", 100, "Maximum length of an input question.")
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
tf.app.flags.DEFINE_string("model_version", "", "Make this '' for initial model, if we ever want to retrain a model, then we can use this (with '_i') to not overwrite the original data")
tf.app.flags.DEFINE_boolean("clip_norms", True, "Do we wish to clip norms?")
tf.app.flags.DEFINE_string("train_prefix", "train.short", "Prefix of all the training data files")
tf.app.flags.DEFINE_string("val_prefix", "val", "Prefix of all the validation data files")
tf.app.flags.DEFINE_integer("epoch_base", 0, "The first epoch, so that we are saving the correct model and outputting the correct numbers if we restarted")

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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [vocab.get(w, qa_data.UNK_ID) for w in context_tokens]
                qustion_ids = [vocab.get(w, qa_data.UNK_ID) for w in question_tokens]

                context_data.append(context_ids)
                query_data.append(qustion_ids)
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, qa, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """

    context_data, question_data, question_uuid_data = dataset
    answers = {}

    for i in range(len(context_data)):
        paragraph = [context_data[i]]
        question = [question_data[i]]
        uuid = question_uuid_data[i]
        predictions = qa.answer(sess, paragraph, question)
	prediction_str_list = []
	for j in range(predictions[0][0], predictions[0][1]+1):
	    #print("j ", j)
	    #print("p ", paragraph[0][j])
	    #print(rev_vocab[paragraph[0][j]])
	    prediction_str_list.append(rev_vocab[paragraph[0][j]])
        #prediction_str_list = [rev_vocab[paragraph[j]] for j in range(predictions[0][0], predictions[0][1]+1)]
        prediction_string = ' '.join(prediction_str_list)
        answers[uuid] = prediction_string  

    return answers


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
    
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path) # vocab = {words : indices}, rev_vocab = [words] (where each word is at index as dictated by vocab)
    
    # load in the embeddings
    embeddings = np.load(embed_path)['glove']

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    if FLAGS.model_name == "BiDAF":
        encoder = BidafEncoder(FLAGS.state_size, FLAGS.embedding_size)
        decoder = BidafDecoder(FLAGS.state_size)
        qa = BidafQASystem(encoder, decoder, embeddings, backprop_word_embeddings)
    else:
        encoder = Encoder(FLAGS.state_size, FLAGS.embedding_size, FLAGS.dropout, encoder_layers, FLAGS.model_name)
        decoder = Decoder(FLAGS.state_size, FLAGS.dropout, decoder_layers)
        qa = QASystem(encoder, decoder, embeddings, backprop_word_embeddings)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        load_train_dir += "/" + FLAGS.model_name + FLAGS.model_version # each model gets its own subdirectory
        initialize_model(sess, qa, load_train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
