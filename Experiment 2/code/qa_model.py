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
import pdb

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


# the session is created in train.py but it is passed into all functions


class Encoder(object):
    def __init__(self, state_size, embedding_dim, dropout_prob, layers, model_name):
        self.state_size = state_size
        self.embedding_dim = embedding_dim # the dimension of the word embeddings
        self.dropout_prob = tf.constant(dropout_prob)
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)
        self.layers = layers # number of layers for a deep BiLSTM encoder, for both
        self.model_name = model_name
        self.FLAGS = tf.app.flags.FLAGS # Get a link to tf.app.flags

    def encode(self, question, question_mask, context_paragraph, context_mask):
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
        :param var_scope: The tf variable scope to use
        :param reuse: boolean whether to reuse variables in this scope
        :return: An (tf op of) encoded representation of the input of the form (attention_vector, context_vectors)
                'attention_vector' is the attention vector for the sequence
                dims = [batch_size, state_size]
                'context_vectors' is the states of the (rolled out) rnn for the context paragraph
                dims = [batch_size, context_paragraph_length, state_size]
        """

        # Build a BiLSTM layer for the question (we only want the concatinated end vectors here)
        question_vector, _ = self.build_deep_rnn(question, question_mask, scope="question_BiLSTM", reuse=True)

        # Concatanate the question vector to every word in the context paragraph, by tiling the question vector and concatinating
        question_vector_tiled = tf.expand_dims(question_vector, 1)
        question_vector_tiled = tf.tile(question_vector_tiled, tf.pack([1, tf.shape(context_paragraph)[1], 1]))
        context_input = tf.concat(2, [context_paragraph, question_vector_tiled])

        # Build BiLSTM layer for the context (want all output states here)
        _, context_states = self.build_deep_rnn(context_input, context_mask, scope="context_BiLSTM", reuse=True)

        # Create a 'context' vector from attention
        attention = self.create_attention_context_vector(context_states, question_vector, scope="AttentionVector", reuse=True)

        # Retuuuurn
        return (attention, context_states)



    def build_rnn(self, inpt, mask, scope="default_scope", reuse=False):
        """
        Helper function to build a rolled out rnn. We need the mask to tell tf how far to roll out the network
        
        :param inpt: input to the rnn, should be a tf.placeholder/tf.variable
            Dims: [batch_size, input_sequence_length, embedding_size], input_sequence_length is the question/context_paragraph length
        :param mask: a tf.variable for the mask of the input, if there is a 0 in this, then the corresponding word is a <pad>
            Dims: [batch_size, input_sequence_length]
        :param scope: the variable scope to use
        :param reuse: boolean if we should reuse parameters in each cell in the rolled out network (i.e. is it theoretically the "same cell" or different?)
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
        with vs.variable_scope(scope, reuse):
            input_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1) # dim = [batch_size]
            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inpt, sequence_length=input_length, dtype=tf.float32, time_major=False) 
            fw_final_state = fw_outputs[:,-1,:]
            bw_final_state = bw_outputs[:,0,:]
            state_history = tf.concat(2, [fw_outputs, bw_outputs])
            return ((fw_final_state, bw_final_state), state_history)


    
    def build_deep_rnn(self, inpt, mask, scope="default_scope", reuse=False):
        """
        Use build_rnn to build a rolled out RNN, using the cell of type self.cell
        We will make this 'self.layers' layers deep
        See deep_rnn for description of the inputs
        We feed the output history (concatination of the forward and backward history) into the next layer each time
        Returns the the history of states for the TOPMOST layer of BiLSTM's and all of the final states concatinated together as a single vector representation
        So if the forward final states are f1, ..., fn and backward are b1, ..., bn if there are n layers, then the single vector representation is
        [f1; ...; fn; b1; ...; bn]
        """
        with vs.variable_scope(scope, reuse):
            fw_final_states = []
            bw_final_states = []
            states = inpt
            for i in range(self.layers):
                scope = "rnn_layer_" + str(i)
                (fw_final_state_i, bw_final_state_i), states = self.build_rnn(states, mask, scope=scope, reuse=True)
                fw_final_states.append(fw_final_state_i)
                bw_final_states.append(bw_final_state_i)
            final_representation = tf.concat(1, fw_final_states + bw_final_states)
            return final_representation, states



    def create_attention_context_vector(self, rnn_states, cur_state, scope="default scope", reuse=False):
        """
        Helper function to create an attention vector. rnn_states and cur_state are vectors with dimension state_size
        rnn_states encorporates inputes of length 'seq_len'

        :param rnn_state: the states which an rnn went through, and we want to learn which are relevant
            dim = [batch_size, seq_len, state_size]
        :param cur_state: the current state which we want to attend to
            dim = [batch_size, state_size]       
        :param scope: the variable scope to use
        :param reuse: boolean if we should reuse parameters in each cell in the rolled out network (i.e. is it theoretically the "same cell" or different?)
        :return: an attention vector, which is a weighted combination of the rnn_states, encorporating relevant information from rnn_states
        """

        # Compute scores for each rnn state
        final_lstm_layer_state_size = self.state_size * 2 # concatination of two vectors
        question_rep_state_size = self.state_size * 2 * self.layers # the question vector concatinates self.layers * 2 vectors 
        batch_size = tf.shape(rnn_states)[0]
        seq_len = tf.shape(rnn_states)[1]
        with vs.variable_scope(scope, reuse):
            # Setup variables to be able to make the matrix product
            inner_product_shape = (final_lstm_layer_state_size, question_rep_state_size)
            inner_product_matrix = tf.get_variable("inner_produce_matrix", shape=inner_product_shape, initializer=tf.contrib.layers.xavier_initializer()) # dim = [statesize, statesize]
            inner_product_matrix_tiled = tf.expand_dims(tf.expand_dims(inner_product_matrix, 0), 0) # dim = [1, 1, statesize, statesize]
            inner_product_matrix_tiled = tf.tile(inner_product_matrix_tiled, tf.pack([batch_size, seq_len, 1, 1])) # dim = [batch_size, seq_len, statesize, statesize]
            cur_state_tiled = tf.expand_dims(cur_state, 1) # dim = [batch_size, 1, statesize]
            cur_state_tiled = tf.tile(cur_state_tiled, tf.pack([1, seq_len, 1])) # dim = [batch_size, seq_len, statesize]
            cur_state_tiled = tf.expand_dims(cur_state_tiled, 3) # dim = [batch_size, seq_len, state_size, 1]
            rnn_state_expanded = tf.expand_dims(rnn_states, 2) # dim = [batch_size, seq_len, 1, state_size]

            # Apply dropout to the inner product
            cur_state_tiled = tf.nn.dropout(cur_state_tiled, self.dropout_prob)
            rnn_state_expanded = tf.nn.dropout(rnn_state_expanded, self.dropout_prob)
            
            # Matrix product. Each input is a rank 4 tensor. For each index in batch_size, seq_len, we comupute an quadratic form. [1, state_size] * [state_size, state_size] * [state_size, 1]
            attention_scores = tf.matmul(tf.matmul(rnn_state_expanded, inner_product_matrix_tiled), cur_state_tiled) # dim = [batch_size, seq_len, 1, 1]
            attention_scores = tf.reduce_max(tf.reduce_max(attention_scores, axis=3), axis=2) # dim = [batch_size, seq_len], just used to reduce rank of tensor

            # Attention vector is attention scores run through softmax
            attention = tf.nn.softmax(attention_scores) # dim = [batch_size, seq_len]
            attention = tf.expand_dims(attention, 2) # dim = [batch_size, seq_len, 1]

            # Take a weighted sum over the vectors in the rnn, the multiply broadcasts appropriately (1 over state_size)
            attention_context_vector = tf.reduce_sum(tf.multiply(rnn_states, attention), axis = 1) # before reduce sum dim = [batch_size, seq_len, state_size], after dim = [batch_size, state_size]
            return attention_context_vector



class Decoder(object):
    def __init__(self, state_size, dropout_prob, layers):
        self.state_size = state_size
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)
        self.dropout_prob = tf.constant(dropout_prob)
        self.layers = layers # number of layers for a deep LSTM decoder
        self.FLAGS = tf.app.flags.FLAGS # Get a link to tf.app.flags  

    def decode(self, attention, context_par_vectors, context_mask):
        """
        Takes the encoder output, which is the whole replac of states for the context BiLSTM and 
        an attention vector, computed from the question representation and uses this to feed a LSTM for decoding
        
        The decoder concatinates the attention vector to the end of each of the states from the encoder, and feeds 
        this into an LSTM. THe outputs from the LSTM are used as scores (proportional to a probability distribution) 
        for the beginning of the answer in the context paragraph.

        THere is a second LSTM for the output.

        The number of layers is how many LSTM's are used / how deep the network is 

        :param attention: the attention context vector, computed using the question representation over the output from the context states (in the encoder)
        :param context_par_vectors: the states from the BiLSTM in the encoder from the context paragraph
        :return: A probability distribution over the context paragraph for where the beginning and end of the answer is
        """
        attention = tf.expand_dims(attention, 1)
        attention_ctx_vector_tiled = tf.tile(attention, tf.pack([1, tf.shape(context_par_vectors)[1], 1]))
        rnn_input = tf.concat(2, [context_par_vectors, attention_ctx_vector_tiled])
        
        with vs.variable_scope("answer_start"):
            rnn_output = self.build_deep_rnn(rnn_input)
            start_scores = self.linear(rnn_output, reuse=True) 

        with vs.variable_scope("answer_end"):
            rnn_output = self.build_deep_rnn(rnn_input)
            end_scores = self.linear(rnn_output, reuse=True) 

        return start_scores, end_scores
    


    def build_deep_rnn(self, rnn_input):
        """ 
        Builds a deep rnn using dynamic_rnn layers
        Returns the states from the FINAL layer
        """
        states = rnn_input
        for i in range(self.layers):
            scope = "rnn_layer_" + str(i)
            with vs.variable_scope(scope, True):
                states, _ = tf.nn.dynamic_rnn(self.cell, states, dtype=tf.float32)
        return states           

    def build_rnn(self, inpt, mask, scope="default_scope", reuse=False):
        """
        Helper function to build a rolled out rnn. We need the mask to tell tf how far to roll out the network
        
        :param inpt: input to the rnn, should be a tf.placeholder/tf.variable
            Dims: [batch_size, input_sequence_length, embedding_size], input_sequence_length is the question/context_paragraph length
        :param mask: a tf.variable for the mask of the input, if there is a 0 in this, then the corresponding word is a <pad>
            Dims: [batch_size, input_sequence_length]
        :param scope: the variable scope to use
        :param reuse: boolean if we should reuse parameters in each cell in the rolled out network (i.e. is it theoretically the "same cell" or different?)
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
        with vs.variable_scope(scope, reuse):
            input_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1) # dim = [batch_size]
            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inpt, sequence_length=input_length, dtype=tf.float32, time_major=False) 
            fw_final_state = fw_outputs[:,-1,:]
            bw_final_state = bw_outputs[:,0,:]
            state_history = tf.concat(2, [fw_outputs, bw_outputs])
            return ((fw_final_state, bw_final_state), state_history)

   

    def linear(self, rnn_output, scope="default_linear", reuse=False):
        with vs.variable_scope(scope, reuse):
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_prob) # apply dropout to the beginning of the linear layer
            weights = tf.get_variable("weights", shape=(self.FLAGS.state_size, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            weights_tiled = tf.expand_dims(weights, axis=0)
            weights_tiled = tf.tile(weights_tiled, tf.pack([tf.shape(rnn_output)[0], 1, 1]))
            bias = tf.get_variable("bias", shape=(self.FLAGS.context_paragraph_max_length,), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            matmul = tf.matmul(rnn_output, weights_tiled) 
            matmul = tf.reduce_max(matmul, axis=2) # dim = [batch_size, context_len, 1] -> [batch_size, context_len]
            result = tf.add(matmul, bias)
        return result

        """
        # h_q, h_p: 2-d TF variable
        with vs.scope("answer_start"):
            a_s = rnn_cell._linear([h_q, h_p], output_size= self.output_size)
        with vs.scope("answer_end"):
            a_e = rnn_cell._linear([h_q, h_p], output_size= self.output_size)
        # with vs.scope("linear_default"):
        #   tf.get_variable("W", shape = [])
        #   tf.get_variable("b", shape = [])
        #   h_q = W + b
        #   tf.get_variable("W", shape = [])
        #   tf.get_variable("b", shape = [])
        #   h_p = W + b
        # creates h_qW + b_q + h_pW + b_p
        return
        """
    
class QASystem(object):
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
        self.question_word_ids_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.question_max_length), name="question_word_ids_placeholder")
        self.context_word_ids_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.context_paragraph_max_length), name="context_word_ids_placeholder")
        self.question_mask = tf.cast(tf.sign(self.question_word_ids_placeholder, name="question_mask"), tf.bool)
        self.context_mask = tf.cast(tf.sign(self.context_word_ids_placeholder, name ="context_mask"), tf.bool)
        
        self.answer_placeholder = tf.placeholder(tf.int32, (None, 2), name="answer_placeholder")
        
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
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
        attentionVector, contextVectors = self.encoder.encode(self.question_var, self.question_mask, self.context_var, self.context_mask)
        self.a_s, self.a_e = self.decoder.decode(attentionVector, contextVectors, self.context_mask)
        
        # q_o, q_h = encoder.encode(self.question_Var)
        # p_o, p_h= encoder.encode(self.paragraph_Var, init_state = q_h, reuse = True)
        # self.a_s, self.a_e = decoder.decode(q_h, p_h)

        
    def create_feed_dict(self, question_batch, context_batch, answer_start_batch = None, answer_end_batch = None):
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

        return feed_dict                
            

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
		    l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.a_s, labels=self.answer_start)
		    l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.a_e, labels=self.answer_end)
		    self.loss = l1 + l2



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
        with vs.variable_scope("embeddings"):
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
        dataset, num_samples = get_sample(dataset_address, self.FLAGS.context_paragraph_max_length, sample)
        test_questions, test_paragraphs, test_start_answers, test_end_answers = dataset
        predictions = self.answer(session, test_paragraphs, test_questions)
        for i in range(num_samples):
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
        
    def answer(self, session, paragraph, question):
        yp1, yp2 = self.decode(session, paragraph, question)

        a_s = np.argmax(yp1, axis = 1)
        a_e = np.argmax(yp2, axis = 1)
        predictions = [[a_s[i], a_e[i]] for i in range(len(a_e))]
        return predictions
    
    # this function is only called by answer above. returns probabilities for the start and end words
    
    def decode(self, session, test_paragraph, test_question):

        input_feed = {}
        """
        could we just have the following?
        input_feed = create_feed_dict(test_paragraph, test_question)
        """
	#input_feed[self.context_var] = test_paragraph
        #input_feed[self.question_var] = test_question
        input_feed = self.create_feed_dict(test_question, test_paragraph)
        output_feed = [self.a_s, self.a_e]
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
        for question_batch, context_batch, answer_start_batch, answer_end_batch in dataset:
            answer_start_batch = unlistify(answer_start_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
            answer_end_batch = unlistify(answer_end_batch) # batch returns dim=[batch_size,1] need dim=[batch_size,]
            input_feed = self.create_feed_dict(question_batch, context_batch, answer_start_batch, answer_end_batch)
            output_feed = [self.updates, self.loss, self.global_grad_norm]
            outputs = session.run(output_feed, feed_dict = input_feed)
            #loss += outputs[1]
            global_grad_norm = outputs[2]
            counter = (counter + 1) % self.FLAGS.print_every
            if counter == 0:
                logging.info("Global grad norm for update: {}".format(global_grad_norm))




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
            self.optimize(session, train_dataset_address)
            toc = time.time()
            epoch_time = toc - tic
            # save your model here
            self.saver.save(session, train_dir + "/model_params", global_step=e_proper)
            val_loss = self.validate(session, val_dataset_address)

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



        
