from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.ops import variable_scope as vs
from util import get_sample, get_minibatches, unlistify, flatten, reconstruct, softmax, softsel
from evaluate import exact_match_score, f1_score
import math

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    elif opt == "adadelta":
        optfn = tf.train.AdadeltaOptimizer
    else:
        assert (False)
    return optfn


# the session is created in train.py but it is passed into all functions



def build_birnn(cell, inpt, mask, scope="default_scope"):
    """
    Helper function to build a rolled out rnn. We need the mask to tell tf how far to roll out the network
        
    :param cell: The rnn cell to use for the forward network
    :param cell: The rnn cell to use for the backward network
    :param inpt: input to the rnn, should be a tf.placeholder/tf.variable
        Dims: [batch_size, input_sequence_length, embedding_size], input_sequence_length is the question/context_paragraph length
    :param mask: a tf.variable for the mask of the input, if there is a 0 in this, then the corresponding word is a <pad>
    :param scope: the variable scope to use
    :return: (fw_final_state, bw_final_state), concat(fw_all_states, bw_all_states)), 
            final_state is the final states of (RELEVANT part) of the states
            dim = [batch_size, cell_size] (cell_size = hidden state size of the lstm)
            so dim of concatinated state is [batch_size, 2*cell_size]
            all_states are the tf.variable's for all of the hidden states
            dim = [batch_size, input_sequence_length, cell_size]
            dim of the concatinated states is [batch_size, input_sequence_length, 2*cell_size]
    """

    # build the dynamic_rnn, compute lengths of the sentences (using mask) so tf knows how far it needs to roll out rnn
    # fw_outputs, bw_outputs is of dim [batch_size, input_sequence_length, embedding_size], n.b. "input_lengths" below is smaller than "input_sequence_length"
    # we think second output is the ("true") final state, but TF docs are ambiguous AF, so I don't really know. There may be problems here...
    with tf.variable_scope(scope):
        input_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inpt, sequence_length=input_length, dtype=tf.float32, time_major=False) 
        fw_final_state = fw_outputs[:,-1,:]
        bw_final_state = bw_outputs[:,0,:]
        state_history = tf.concat(2, [fw_outputs, bw_outputs])
        return ((fw_final_state, bw_final_state), state_history)



def build_deep_birnn(cell, inpt, mask, layers, scope="default_scope"):
    """
    Use build_rnn to build a rolled out RNN, using the cell of type self.cell
    We will make this 'self.layers' layers deep
    See deep_rnn for description of the inputs
    We feed the output history (concatination of the forward and backward history) into the next layer each time
    Returns the the history of states for the TOPMOST layer of BiLSTM's and all of the final states concatinated together as a single vector representation
    So if the forward final states are f1, ..., fn and backward are b1, ..., bn if there are n layers, then the single vector representation is
    [f1; ...; fn; b1; ...; bn]
    """
    with tf.variable_scope(scope):
        fw_final_states = []
        bw_final_states = []
        states = inpt
        for i in range(layers):
            scope = "birnn_layer_" + str(i)
            (fw_final_state_i, bw_final_state_i), states = build_birnn(cell, inpt, mask, scope=scope)
            fw_final_states.append(fw_final_state_i)
            bw_final_states.append(bw_final_state_i)
        final_representation = tf.concat(1, fw_final_states + bw_final_states)
        return final_representation, states




class BidafEncoder(object):
    def __init__(self, state_size, embedding_dim):
        self.state_size = state_size
        self.embedding_dim = embedding_dim # the dimension of the word embeddings
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)
        self.FLAGS = tf.app.flags.FLAGS # Get a link to tf.app.flags

    def encode(self, question, question_mask, context_paragraph, context_mask, dropout_placeholder):
        """
        Encoder function. Encodes the question and context paragraphs into some hidden representation.
        This function assumes that the question has been padded already to be of length self.FLAGS.question_max_length 
        and that the context paragraph has been padded to the length of self.FLAGS.context_paragraph_max_length

        :param question: A tf.placeholder for words in the question. 
                Dims = [batch_size, question_length, embedding_dimension]
        :param question_mask: A tf.placholder for a mask over the question. 
                Dims = [batch_size, question_length, 1]
                0 in mask indicates if the word is padding (<pad>), 1 otherwise
        :param context_paragraph: A tf.placeholder for words in the context paragraph 
                Dims = [batch_size, context_paragraph_length, embedding_dimension]
        :param context_mask: A tf.placholder for a mask over the context paragraph. 
                Dims = [batch_size, context_paragraph_length, 1]
                0 in mask indicates if the word is padding (<pad>), 1 otherwise
        :return: context_aware_question_states (G from the paper) 
        """

        # Pass question and context sentence (representations) through a BiLSTM layer (seperate)
        _, question_states = build_deep_birnn(self.cell, question, question_mask, layers=1, scope="question_BiLSTM")
        _, context_states = build_deep_birnn(self.cell, context_paragraph, context_mask, layers=1, scope="context_BiLSTM")

        # Generate the similarity matrix
        # Part one of attention
        similarity = self.compute_similarity(context_states, question_states, dropout_placeholder)

        # Compute the soft select vectors
        # Part two of attention
        context_to_query = self.compute_context_to_query(similarity, question_states)
        query_to_context = self.compute_query_to_context(similarity, context_states)

        # Use the attnetions (above) to learn a context aware querstion representation
        # This is G in the paper
        question_aware_context_states = self.compute_final_attention(context_states, context_to_query, query_to_context)

        # Return
        return question_aware_context_states



    def compute_similarity(self, context_states, question_states, dropout_placeholder): 
        """
        :param context_states: H from paper, dims = [batch_size, T, 2d] 
        :param question_states: U from paper, dims = [batch_size, J, 2d]
        :return: Similarity matrix. S from paper. dims = [batch_size, T, J]
        """
        with tf.variable_scope("similarity_matrix"):
            # Create a matrix hu with dims [batch_size, T, J, 6d], 
            # [H_t; U_j; H_t o U_j] from the paper
            T = tf.shape(context_states)[1]
            J = tf.shape(question_states)[1] 
            context_states_tiled = tf.expand_dims(context_states, 2) # dim = [batch_size, T, 1, 2d]
            context_states_tiled = tf.tile(context_states_tiled, [1, 1, J, 1]) # tile so dim = [batch_size, T, J, 2d]
            question_states_tiled = tf.expand_dims(question_states, 1) # dim = [batch_size, 1, J, 2d]
            question_states_tiled = tf.tile(question_states_tiled, [1, T, 1, 1]) # dim = [batch_size, T, J, 2d]
            hu = tf.concat(3, [context_states_tiled, question_states_tiled, context_states_tiled * question_states_tiled]) # dim = [batch_size, T, J, 6d]

            # Create the weights (dims=[1,1,6d]) to learn for the similarity function + tile to make it of dimension [T, J, 6d]
            weights_size = self.state_size * 6 # 6d
            weights = tf.get_variable("weights", shape=(1,1,weights_size), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # dims = [1,1,6d]
            weights = tf.nn.dropout(weights, 1.0 - dropout_placeholder)
            weights = tf.tile(weights, [T, J, 1]) # dims = [T, J, 6d]

            # Compute the final similarity matrix using:
            # 1. Broadcast a pointwise product of the weights over the hu tensor
            # [batch_size, T, J, 6d] * [T, J, 6d] -> [T, J, 6d] 
            # 2. Reduce sum over the 4th dimension (so step 1 + step 2 = a dot product at every [batch_size, T, J, :]
            # [batch_size, T, J, 6d] -> [batch_size, T, J]
            # 3. squeeze it into the correct shape
            # [batch_size, T, J, 1] -> [batch_size, T, J]
            unsummed_dots = tf.multiply(hu, weights)
            similarity = tf.reduce_sum(unsummed_dots, axis=3)

            return similarity


    
    def compute_context_to_query(self, similarity, question_states):
        """
        Compute context to query attention. U_tilde in the paper.
        :param similarity: S in the paper. dims = [batch_size, T, J]
        :param question_states: U in the paper. dims = [batch_size, J, 2d]
        :output: U_tilde in the paper. [batch_size, T, 2d]
        """
        with tf.variable_scope("context_to_query"):
            # Compute attentions. This is softmax over similarity (S) in the 2nd dimension
            # dims = [batch_size, T, J]
            attentions = tf.nn.softmax(similarity) #, dim=2) # can't specify dim=<last_dim>, because who wants code readability???

            # Tile querstions over T so that we can compute a soft select per t
            T = tf.shape(similarity)[1]
            question_states_tiled = tf.expand_dims(question_states, 1) # dim = [batch_size, 1, J, 2d]
            question_states_tiled = tf.tile(question_states_tiled, [1, T, 1, 1]) # dim = [batch_size, T, J, 2d]

            # Use the attention to compute soft selection
            # To do this, tile attentions to make it dim = [batch_size, T, J, 2d]
            # Then pointwise multiply with question_states (effectively, computing a weighted vecrsion of question states)
            # Call this aU (as aU[:, t, j, :] = a_tj U_j from the paper (which we need to sum over))
            twod = tf.shape(question_states)[2] 
            attentions_tiled = tf.expand_dims(attentions, 3) # dim = [batch_size, T, J, 1]
            attentions_tiled = tf.tile(attentions_tiled, [1, 1, 1, twod]) # dim = [batch_size, T, J, 2d]
            aU = tf.multiply(attentions_tiled, question_states_tiled) # dim = [batch_size, T, J, 2d]

            # to get the final soft selected vector (U_tilde) sum over the J values, in dim 2
            U_tilde = tf.reduce_sum(aU, axis=2) # dim = [batch_size, T, 2d]

            return U_tilde



    def compute_query_to_context(self, similarity, context_states):
        """
        Compute query to context attention. H_tilde from the paper. (N.B. H_tilde = tiled version of h_tilde)
        :param similarity: S in the paper. dims = [batch_size, T, J]
        :param context_states: U in the paper. dims = [batch_size, T, 2d]
        :return: H_tilde in the paper. dims = [batch_size, T, 2d]
        """
        with tf.variable_scope("query_to_context"):
            # Take max per column (actually column, as we DO have a TxJ matrix) 
            # Then run it through a softmax to get our softmax scores
            similarity_col_max = tf.reduce_max(similarity, axis=2) # dim = [batch_size, T].
            attention = tf.nn.softmax(similarity_col_max) #, dim=1) # dim = [batch_size, T]. N.B. cannot specify the default behaviour, because making you're code clear isn't something you should try to do....

            # Now compute the soft selection over the context state
            # Use tiling, pointwise multiplication and reduce sum to compute this, similar to context_to_query attention
            # bH[:,t,:] is b_t H_t from the paper
            twod = tf.shape(context_states)[2]
            attention_tiled = tf.expand_dims(attention, 2) # dim = [batch_size, T, 1]
            attention_tiled = tf.tile(attention_tiled, [1, 1, twod]) # dim = [batch_size, T, 2d]
            bH = tf.multiply(attention_tiled, context_states)

            # Get the soft selected vector by summing over dimension 1 (the T context words)
            # Return a tiled version of this vector
            T = tf.shape(context_states)[1]
            h_tilde = tf.reduce_sum(bH, axis=1) # dim = [batch_size, 2d]
            H_tilde = tf.expand_dims(h_tilde, 1) # dim = [batch_size, 1, 2d]
            H_tilde = tf.tile(H_tilde, [1, T, 1]) # dim = [batch_size, T, 2d]

            return H_tilde



    def compute_final_attention(self, context_states, context_to_query, query_to_context):
        """
        Compute the output of the attention + encoding. Paper suggests to just concatinate these
        :param context_states: H from the paper. dim = [batch_size, T, 2d]
        :param context_to_query: U_tilde from the paper. dim = [batch_size, T, 2d]
        :param query_to_context: H_tilde from the paper. dim = [batch_size, T, 2d]
        :return: [H; U_tilde; H o U_tilde; H o H_tilde] from the paper. dim = [batch_size, T, 8d]
        """
        with tf.variable_scope("final_attention"):
            HU_tilde = tf.multiply(context_states, context_to_query)
            HH_tilde = tf.multiply(context_states, query_to_context)
            question_aware_context_states = tf.concat(2, [context_states, context_to_query, HU_tilde, HH_tilde])
            return question_aware_context_states



class BidafDecoder(object):
    def __init__(self, state_size):
        self.state_size = state_size
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)
        self.FLAGS = tf.app.flags.FLAGS # Get a link to tf.app.flags  

    def decode(self, question_aware_context_states, mask, dropout_placeholder):
        """
        From the paper, takes the output from the encoder (G) and computes a probability distribution over the beginning and end index
        :param question_aware_context_states: G from the paper. dims = [batch_size, T, 8d]
        :param mask: For use in the LSTM networks, and is the context paragraph mask
        :return: A probability distribution over the context paragraph for where the beginning and end of the answer is
        """

        with tf.variable_scope("decode"):
            # M^1 from the paper        
            _, M1 = build_deep_birnn(self.cell, question_aware_context_states, mask, layers=2, scope="Model_Layer") # dim = [batch_size, T, 2d]

            # M^2 from paper
            _, M2 = build_deep_birnn(self.cell, M1, mask, layers=1, scope="Model_Layer_2") # dim = [batch_size, T, 2d]

            # Compute GM^1 from paper and GM^2 from paper
            GM1 = tf.concat(2, [question_aware_context_states, M1]) # dim = [batch_size, T, 10d]
            GM2 = tf.concat(2, [question_aware_context_states, M2]) # dim = [batch_size, T, 10d]

            # Define weights for to get logits (scores) from GM1 and GM2
            T = tf.shape(question_aware_context_states)[1] 
            tend = 10 * self.state_size # got lazy, not calculating it from the inputs
            weights_beg = tf.get_variable("weights_beg", shape=(1,tend), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # dims = [1,10d]
            weights_beg = tf.nn.dropout(weights_beg, 1.0 - dropout_placeholder)
            weights_beg = tf.tile(weights_beg, [T, 1]) # dim = [T,10d]
            weights_end = tf.get_variable("weights_end", shape=(1,tend), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # dims = [1,10d]
            weights_end = tf.tile(weights_beg, [T, 1]) # dim = [T,10d]
            weights_end = tf.nn.dropout(weights_end, 1.0 - dropout_placeholder)

            # Compute logits (using broadcasting to multiply)
            to_sum_beg = tf.multiply(weights_beg, GM1) # dim = [batch_size, T, 10d]
            to_sum_end = tf.multiply(weights_beg, GM2) # dim = [batch_size, T, 10d]
            predict_beg = tf.reduce_sum(to_sum_beg, axis=2) # dim = [batch_size, T]
            predict_end = tf.reduce_sum(to_sum_end, axis=2) # dim = [batch_size, T]

            # We would take softmax, BUT, cross entropy takes the logits (not the softmax)
            return predict_beg, predict_end


   
class BidafQASystem(object):
    # i added embeddings down here, make sure you'll change it in train.py
    def __init__(self, encoder, decoder, embeddings, backpropogate_embeddings):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param backpropogate_embeddings: boolean for if we want to backpropogate through the word embeddings
        """

        # Get a link to tf.app.flags
        self.FLAGS = tf.app.flags.FLAGS

        # ==== set up placeholder tokens ========
        # is this the way to do it????
        # do we need to do reshape like we did in q2_rnn
        if backpropogate_embeddings:
            self.embeddings = tf.Variable(tf.convert_to_tensor(embeddings, dtype = tf.float32, name = 'embedding'))
        else:
            self.embeddings = tf.convert_to_tensor(embeddings, dtype = tf.float32, name = 'embedding')

        self.encoder = encoder
        self.decoder = decoder

        # start answer and end answer
        # we probably need to change output_size to max_context_length or something similar
        self.answer_start = tf.placeholder(tf.int32, shape = (None,), name="answer_end")
        self.answer_end = tf.placeholder(tf.int32, shape = (None,), name="answer_end")

        # question, paragraph (context), answer, and dropout placeholders
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.question_word_ids_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.question_max_length), name="question_word_ids_placeholder")
        self.context_word_ids_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.context_paragraph_max_length), name="context_word_ids_placeholder")
        self.question_mask = tf.cast(tf.sign(self.question_word_ids_placeholder, name="question_mask"), tf.bool)
        self.context_mask = tf.cast(tf.sign(self.context_word_ids_placeholder, name ="context_mask"), tf.bool)
        
        self.answer_placeholder = tf.placeholder(tf.int32, (None, 2), name="answer_placeholder")
        
        # ==== assemble pieces ====
        #with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        self.setup_embeddings()
        self.setup_system()
        self.setup_loss()

        # ==== set up training/updating procedure ====
        params = tf.trainable_variables()
        self.setup_optimizer()

        # ==== Give the system a saver (also used for loading) ====
        self.saver = tf.train.Saver()

        

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question_aware_context_states = self.encoder.encode(self.question_var, self.question_mask, self.context_var, self.context_mask, self.dropout_placeholder)
        self.predict_beg, self.predict_end = self.decoder.decode(question_aware_context_states, self.context_mask, self.dropout_placeholder)
        self.predict_beg_probs = tf.nn.softmax(self.predict_beg)
        self.predict_end_probs = tf.nn.softmax(self.predict_end)
        

        
    def create_feed_dict(self, question_batch, context_batch, answer_start_batch = None, answer_end_batch = None, dropout = 0.0):
        feed_dict = {}
        list_data = type(context_batch) is list and (type(context_batch[0]) is list or type(context_batch[0]) is np.ndarray) 
        if not list_data:
            context_batch = [context_batch]
            question_batch = [question_batch]
        
        for i in range(len(question_batch)):
            if len(question_batch[i]) >= self.FLAGS.question_max_length:
                question_batch[i] = question_batch[i][:self.FLAGS.question_max_length]
            else:
                padding_length = self.FLAGS.question_max_length - len(question_batch[i])
                padding = [0] * padding_length
                question_batch[i].extend(padding)

        feed_dict[self.question_word_ids_placeholder] = question_batch

        for i in range(len(context_batch)):
            if len(context_batch[i]) >= self.FLAGS.context_paragraph_max_length:
                context_batch[i] = context_batch[i][:self.FLAGS.context_paragraph_max_length]
            else:
                padding_length = self.FLAGS.context_paragraph_max_length - len(context_batch[i])
                padding = [0] * padding_length
                context_batch[i].extend(padding)

        feed_dict[self.context_word_ids_placeholder] = context_batch
            
        if answer_start_batch is not None:
            feed_dict[self.answer_start] = answer_start_batch

        if answer_end_batch is not None:
            feed_dict[self.answer_end] = answer_end_batch

        feed_dict[self.dropout_placeholder] = dropout

        return feed_dict                
            


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with tf.variable_scope("loss"):
            predict_beg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.predict_beg, self.answer_start)
            predict_end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.predict_end, self.answer_end)
            self.loss = tf.add(predict_beg_loss, predict_end_loss)



    def setup_optimizer(self):
        optimizer_option = self.FLAGS.optimizer
        optimizer = get_optimizer(optimizer_option)(self.FLAGS.learning_rate)

        grads = optimizer.compute_gradients(self.loss)
        only_grads = [grad[0] for grad in grads]
        only_vars = [grad[1] for grad in grads]

        if not self.FLAGS.clip_norms:
            self.updates = optimizer.minimize(self.loss)
            self.global_grad_norm = tf.global_norm(only_grads)
            return

        max_grad_norm = tf.constant(self.FLAGS.max_gradient_norm)
        only_grads, _ = tf.clip_by_global_norm(only_grads, max_grad_norm)
        grads = zip(only_grads, only_vars)
        self.updates = optimizer.apply_gradients(grads)

        # Get global norm so we can print it in training
        self.global_grad_norm = tf.global_norm(only_grads)


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            #glove_matrix = np.load()['glove']
            #params = tf.constant(glove_matrix) # if you wanna train the embeddings too, put it in a variable (inside the init function)
            self.question_var = tf.nn.embedding_lookup(self.embeddings, self.question_word_ids_placeholder)
            self.context_var = tf.nn.embedding_lookup(self.embeddings, self.context_word_ids_placeholder)

    # this function calls answer bellow and is called by train at the bottom of the page
    # returns the f1 and em scores of an epoch


    # check whether the for loop is necessary. feel like should work with the full batch
    def evaluate_answer(self, session, dataset_address, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        ######### the f1_score and exact_match_score functions defined work only with strings
        ######### need to write new ones that work with lists like below 
        
        f1 = 0.
        em = 0.
        dataset, num_samples = get_sample(dataset_address, self.FLAGS.context_paragraph_max_length, sample) # gets 100 samples
        test_questions, test_paragraphs, test_start_answers, test_end_answers = dataset

        # need to split the samples testing into a reasonable workload, so that we can fit each prediction batch into memory
        predictions = []
	num_minibatches = int(math.ceil(num_samples / self.FLAGS.batch_size)) # integer div, rounding up
	for i in range(num_minibatches): # would be nice to have this as "for question, paragraph, start, end in dataset" (TODO: when clean code up)
            paragraphs = []
            questions = []
            beg_pos = i * self.FLAGS.batch_size
            end_pos = beg_pos + self.FLAGS.batch_size
            if i == num_minibatches - 1:
                paragraphs = test_paragraphs[beg_pos:]
                questions = test_questions[beg_pos:]
            else: 
                paragraphs = test_paragraphs[beg_pos:end_pos]
                questions = test_questions[beg_pos:end_pos]
            predictions += self.answer(session, paragraphs, questions)

        for i in range(num_samples): # num_samples = number of samples in dataset, it's not necessarily 100, as we cut out things we couldn't predict 
            answer_beg = test_start_answers[i][0] # this is a list of length 1
            answer_end = test_end_answers[i][0] # same
            answer_str_list = [str(test_paragraphs[i][j]) for j in range(answer_beg, answer_end+1)]
            true_answer = ' '.join(answer_str_list)
            prediction_str_list = [str(test_paragraphs[i][j]) for j in range(predictions[i][0], predictions[i][1]+1)]
            prediction_string = ' '.join(prediction_str_list)
            f1 += f1_score(prediction_string, true_answer)
            em += exact_match_score(prediction_string, true_answer)
        f1 = 1.0 * f1 / sample
        em = 1.0 * em / sample
        
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em
    

    # this function is only called by evaluate_answer above and calls decode below.
    # returns indices of the most probable start and end words
        
    def answer(self, session, paragraphs, questions):
        beg, end = self.predict(session, paragraphs, questions)

        # take the argmax of (beg)_i(end)_j where j >= i (and beg_pr, end_pr are the beginning and end probabilities (actually logits, but it doesn't matter))
        shape = np.shape(beg)[1:] # shape of the predictions

        beg = np.argmax(beg, axis = 1)
        end = np.argmax(end, axis = 1)
        predictions = [(beg[i], end[i]) for i in range(len(beg))]
        return predictions
    
    # this function is only called by answer above. returns probabilities for the start and end words
    
    def predict(self, session, test_paragraphs, test_questions):
        #with tf.variable_scope("qa"):
        input_feed = self.create_feed_dict(test_questions, test_paragraphs)
        output_feed = [self.predict_beg_probs, self.predict_end_probs]
        outputs = session.run(output_feed, feed_dict = input_feed)
        return outputs    


    def optimize(self, session, dataset_address):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        counter = -1
        #loss = 0.0
        if self.FLAGS.debug:
            dataset, _ = get_sample(dataset_address, self.FLAGS.context_paragraph_max_length, self.FLAGS.batch_size)
            dataset = [dataset] # put in a list, becuase get_sample returns one minibatch and we want a list of minibatches
        else:
            dataset = get_minibatches(dataset_address, self.FLAGS.context_paragraph_max_length, self.FLAGS.batch_size)

        #with tf.variable_scope("qa"):
        #tf.get_variable_scope().reuse_variables()
        epoch_loss = 0.0
        for question_batch, context_batch, answer_start_batch, answer_end_batch in dataset:
            answer_start_batch = unlistify(answer_start_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
            answer_end_batch = unlistify(answer_end_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
            input_feed = self.create_feed_dict(question_batch, context_batch, answer_start_batch, answer_end_batch, self.FLAGS.dropout)
            output_feed = [self.updates, self.loss, self.global_grad_norm]
            outputs = session.run(output_feed, feed_dict = input_feed)
            epoch_loss += np.sum(outputs[1])
            global_grad_norm = outputs[2]
            counter = (counter + 1) % self.FLAGS.print_every
            if counter == 0:
                logging.info("Global grad norm for update: {}".format(global_grad_norm))
        return epoch_loss



    def train(self, session, train_dataset_address, val_dataset_address, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # Set reuse of variables up
        tf.get_variable_scope().reuse_variables()

        # Make files to append the scores as we get them
        training_scores_filename = train_dir + "/training_scores.txt"
        validation_scores_filename = train_dir + "/validation_scores.txt"
        self.append_file_line(training_scores_filename, "Epoch", "F1_score", "EM_score", "Epoch_time")
        self.append_file_line(validation_scores_filename, "Epoch", "F1_score", "EM_score", "Epoch_time")

        #q_val, p_val, a_val_s, a_val_e = val_dataset
        #q, p, a_s, a_e = train_dataset 
        for e in range(self.FLAGS.epochs):
            e_proper = e + self.FLAGS.epoch_base
            tic = time.time()
            train_loss = self.optimize(session, train_dataset_address)
            toc = time.time()
            epoch_time = toc - tic
            # save your model here
            self.saver.save(session, train_dir + "/model_params", global_step=e_proper)
            val_loss = self.validate(session, val_dataset_address)
            logging.info("Training error in epoch {}: {}".format(str(e), str(train_loss)))
            logging.info("Validation error in epoch {}: {}".format(str(e), str(val_loss)))
            f1_train, em_train = self.evaluate_answer(session, train_dataset_address, 100, True) # doing this cuz we wanna make sure it at least works well for the stuff it's already seen
            f1_val, em_val = self.evaluate_answer(session, val_dataset_address, 100, True)
            
            # Log scores
            self.append_file_line(training_scores_filename, e_proper, f1_train, em_train, epoch_time)
            self.append_file_line(validation_scores_filename, e_proper, f1_val, em_val, epoch_time)
            

    
    # A function to write out epoch times and corresponding 
    def append_file_line(self, filename, epoch, f1_score, em_score, epoch_time):
        with open(filename, "a") as my_file:
            my_file.write(" ".join([str(epoch), str(f1_score), str(em_score), str(epoch_time)]) + "\n")
                        



    # the following function is called from train above and only calls the function test below

    def validate(self, sess, valid_dataset_address): # only used for unseen examples, ie when you wanna check your model
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0.
        if self.FLAGS.debug:
            dataset, _ = get_sample(valid_dataset_address, self.FLAGS.context_paragraph_max_length, self.FLAGS.batch_size)
            dataset = [dataset] # expecting a list of minibatches, but get sample returns a single minibatch
        else:
            dataset = get_minibatches(valid_dataset_address, self.FLAGS.context_paragraph_max_length, self.FLAGS.batch_size)
        for question_batch, context_batch, answer_start_batch, answer_end_batch in dataset:
            valid_cost += self.test(sess, question_batch, context_batch, answer_start_batch, answer_end_batch)

        return valid_cost

    
    # the following function is called from validate above
    
    def test(self, session, question_batch, context_batch, answer_start_batch, answer_end_batch):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        output_feed = [self.loss]     
        answer_start_batch = unlistify(answer_start_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
        answer_end_batch = unlistify(answer_end_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
        input_feed = self.create_feed_dict(question_batch, context_batch, answer_start_batch, answer_end_batch)
        loss = np.sum(session.run(output_feed, input_feed)) # sessions always return real things
            
        return loss



        
