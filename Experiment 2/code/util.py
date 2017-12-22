import numpy as np
import tensorflow as tf
import random
from functools import reduce
from operator import mul

"""
def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)
"""


# flattens tensor of shape [..., d] into [x, d] where x is the product of ...
def flatten(tensor):
    fixed_shape = tensor.get_shape().as_list()
    tensor_rank = len(fixed_shape) 
    flatten_dim = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(1,tensor_rank)])
    minibatch_size = fixed_shape[0] or tf.shape(tensor)[0]
    out_shape = [minibatch_size, flatten_dim]
    #out_shape = [left] + [fixed_shape[i] or tf.shape(tenor)[i] for i in range(start, len(fixed_shape))]    
    flat = tf.reshape(tensor, out_shape)
    return flat
#def flatten(tensor):
#    return tf.reshape(tensor, [-1, tensor.get_shape().as_list()[-1]])


# reshapes tensor into ref's shape
def reconstruct(tensor, ref):
    shape = ref.get_shape().as_list() # will be none in the minibatch dim
    shape[0] = -1 # want -1 rather than none, so it fills correctly
    reshaped = tf.reshape(tensor, shape)
    return reshaped
    """
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - 1
    tensor_start = len(tensor_shape) - 1
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    if tensor.get_shape().as_list()[-1] == 1:
        out = tf.squeeze(out, [len(ref.get_shape().as_list())-1])
    return out
#def reconstruct(tensor, ref):
#    return tf.reshape(tensor, ref.get_shape().as_list())
    """


# first flattens logits, takes softmax, and reshapes it back
def softmax(logits):
    with tf.name_scope("Softmax"):
        flat_logits = flatten(logits)
        #logits_shape = logits.get_shape().as_list()
        #flat_dim = np.prod(logits_shape[1:])
        #flat_logits = tf.reshape(logits, [-1, flat_dim])
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits)
        #logits_shape[0] = -1 # would be None when read shape, BUT, need as -1 when inputting into reshape
        #out = tf.reshape(flat_logits, logits_shape)
        return out



# creates U^tilde and h^tilde in c2q and q2c attention (and later in decode)
def softsel(target, logits):
    with tf.name_scope("Softsel"):
        a = softmax(logits)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def unlistify(llist):
    return [item for sublist in llist for item in sublist]



def get_minibatches(dataset_address, max_context_length, minibatch_size = 100):
    data_size = sum(1 for line in open(dataset_address + ".span"))
    question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')
    file_eof = False
    
    while not file_eof:
        
        questions = []
        contexts = []
        start_answers = []
        end_answers = []

        minibatch_counter = 0
        while minibatch_counter < minibatch_size:
            question_line = question_id_file.readline()
            context_line = context_id_file.readline()
            answer_line = answer_file.readline()

            if (question_line == ""): # EOF
                file_eof = True
                break

            context = map(int, context_line.split(" "))
            question = map(int, question_line.split(" "))
            answer = map(int, answer_line.split(" "))
            
            if answer[0] >= max_context_length or answer[1] >= max_context_length:
                continue;
            
            contexts.append(context)            
            questions.append(question)            
            start_answers.append([answer[0]])
            end_answers.append([answer[1]])
            minibatch_counter += 1

        minibatch = [questions, contexts, start_answers, end_answers]
        yield minibatch # yield, return as a generator
    
    question_id_file.close()
    context_id_file.close()
    answer_file.close()



def get_sample(dataset_address, max_context_length, sample_size): 
    
    data_size = sum(1 for line in open(dataset_address + ".span"))

    #question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')    

    indices = set(random.sample(range(0, data_size), sample_size))
    max_index = max(indices)
    
    contexts = []
    questions = []
    start_answers = []
    end_answers = []
    num_samples = sample_size
    line_number = 0
    for question_line in open(dataset_address + ".ids.question", 'r'):
        if line_number > max_index:
            break
        context_line = context_id_file.readline()
        answer_line = answer_file.readline()
        if line_number in indices:
            # This is one of the selected rows, add it to the minibatch
            question = map(int, question_line.split(" "))
            context = map(int, context_line.split(" "))
            answer = map(int, answer_line.split(" "))
            if answer[0] >= max_context_length or answer[1] >= max_context_length:
                num_samples -= 1
                line_number += 1
                continue
            questions.append(question)
            contexts.append(context)
            start_answers.append([answer[0]])
            end_answers.append([answer[1]])
        line_number += 1

    context_id_file.close()
    answer_file.close()
    return [[questions, contexts, start_answers, end_answers], num_samples]
