# Bidirectional Attention Flow Model
import os, re, sys, time, json, codecs
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from string import punctuation
from sklearn import metrics


from keras import utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional
import h5py

from keras.layers import Dense, Input, Concatenate, TimeDistributed
from deep_qa.layers import ComplexConcat
from deep_qa.layers.attention import MatrixAttention, MaskedSoftmax, WeightedSum
from deep_qa.layers.backend import Max, RepeatLike, Repeat
from deep_qa.training import TextTrainer
# from deep_qa.data.instances.reading_comprehension import McQuestionPassageInstance
from deep_qa.layers import L1Normalize
from deep_qa.layers import OptionAttentionSum
from deep_qa.layers.attention import Attention

# from deep_qa.training.models import DeepQaModel
# from deep_qa.common.params import Params

# import matplotlib.pyplot as plt
# from tensorflow.python.lib.io import file_io
import os,re,sys,argparse
# from StringIO import StringIO

# pd.set_option('display.max_columns', None)
os.chdir("/Users/rxia/Desktop/QA/data/100000_reidx")

from help import *
# analyze the model for further improvement

# from keras.models import load_model
# attention = load_model('biDAF.h5',custom_objects={'MatrixAttention':MatrixAttention,'Max':Max,'MaskedSoftmax':MaskedSoftmax,'WeightedSum':WeightedSum,'RepeatLike':RepeatLike,'OptionAttentionSum':OptionAttentionSum,'ComplexConcat':ComplexConcat,'L1Normalize':L1Normalize})
# weights= attention.load_weights('biDAF.h5')

# file_path = 'gs://si630rita_bucket/data'
Q_train = np.load('Q_train100000_reidx.npy')
# Q_test = np.load('Q_test10000.npy')[:2976]
N_train = np.load('N_train100000_reidx.npy')
# N_test = np.load('N_train10000.npy')[:2976]
Y_train = np.load('y_train_multi100000_reidx.npy')
# Y_test = np.load('y_test_multi10000.npy')[:2976] # check 'UNK'
train_option_input = np.load('O_train100000_reidx.npy') #shape of 8704
# option_input = np.load('option_output_320.npy')[:320]
Q_val = np.load('Q_val100000_reidx.npy')
N_val=np.load('N_val100000_reidx.npy')
Y_val = np.load('y_val_multi100000_reidx.npy')
val_option_input = np.load('O_val100000_reidx.npy')

# parameters of model
embeddings = np.load('embedding_input_matrix100000_reidx.npy')
em_len, emb_dim = embeddings.shape
# em_len = 16320 #unique tokens
# emb_dim= 325
max_len_P = 300
max_len_Q = 46
max_num_options = 102 #include UNK
# max_option_length = 224 # check for all_option lengths in every news
batch_size = 32
# N_train = np.load('N_train10000.npy')[:8704]
#
# with open('word_index10000.json','r') as file:
#     word_index = json.loads(file.read())
#
# with open('entity_index10000.json') as file:
#     entity_index = json.loads(file.read())
#
# word_index_reverse = {code:i for i,code in word_index.items()}
#
# entity_index_word = [code for i,code in word_index.items() if i.startswith('@entity') or i=='<UNK_ENTITY>']
#
# option_input = np.zeros((8704,371))
# count=0
# for record in N_train:
#     for code in record:
#         if code in entity_index_word:
#             word = word_index_reverse[code]
#             entity = entity_index[word]
#             option_input[count][entity] = code
#     count+=1
#     if count%1000 ==0:
#         print('load'+str(count))


#TODO: Extract all options' word index in every news as input
def run_biDAF():
    # Create embedding for both Question and News ON both word level and char level
    question_input = Input(shape=(max_len_Q,),
                           dtype='int32', name="question_input")
    passage_input = Input(shape=(max_len_P,),
                          dtype='int32', name="passage_input")
    # Load num of options input
    options_input = Input(shape=(max_num_options,),
                          dtype='int32', name="options_input")  # in order to map only options output
    embedding_layer_P = Embedding(em_len,
                                  emb_dim,
                                  weights=[embeddings],
                                  input_length=max_len_P,
                                  batch_input_shape=(batch_size, max_len_P),
                                  trainable=False)
    embedding_layer_Q = Embedding(em_len,
                                  emb_dim,
                                  weights=[embeddings],
                                  input_length=max_len_Q,
                                  batch_input_shape=(batch_size, max_len_Q),
                                  trainable=False)

    passage_embedding = embedding_layer_P(passage_input)
    question_embedding = embedding_layer_Q(question_input)



    bi_lstm_Q = Bidirectional(LSTM(256, return_sequences=True), batch_input_shape=(batch_size, max_len_Q, emb_dim))(
        question_embedding)
    bi_lstm_Q1 = Bidirectional(LSTM(256), batch_input_shape=(batch_size, max_len_Q, emb_dim))(question_embedding)
    bi_lstm_P = Bidirectional(LSTM(256, return_sequences=True), batch_input_shape=(batch_size, max_len_P, emb_dim))(
        passage_embedding)
    ##### Create Attention Layer

    similarity_function_params = {'type': 'linear', 'combination': 'x,y,x*y'}
    matrix_attention_layer = MatrixAttention(similarity_function=similarity_function_params,name='matrix_attention_layer')
    # Shape: (batch_size, num_passage_words, num_question_words)
    passage_question_similarity = matrix_attention_layer([bi_lstm_P, bi_lstm_Q])

    # Shape: (batch_size, num_passage_words, num_question_words), normalized over question words for each passage word.
    passage_question_attention = MaskedSoftmax()(passage_question_similarity)

    weighted_sum_layer = WeightedSum(name="passage_question_vectors",
                                     use_masking=False)  # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    passage_question_vectors = weighted_sum_layer([bi_lstm_Q, passage_question_attention])  # sum at(U~:t)=1
    ## Query - Passage 2d * max_len_Q
    # find most important context words by max() passage_question_similarity

    question_passage_similarity = Max(axis=-1)(passage_question_similarity)  # Shape: (batch_size, num_passage_words)
    # use softmax for b (max softmax value for similarity matrix column wise)
    question_passage_attention = MaskedSoftmax()(question_passage_similarity)  # Shape: (batch_size, num_passage_words)

    weighted_sum_layer = WeightedSum(name="question_passage_vector",
                                     use_masking=False)  # h~ = sum(weighted_bt * H:t) 2*embed_dim
    question_passage_vector = weighted_sum_layer([bi_lstm_P, question_passage_attention])  # sum bt(H~:t)=1

    repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    tiled_question_passage_vector = repeat_layer([question_passage_vector, bi_lstm_P])

    # Shape: (batch_size, num_passage_words, embedding_dim * 8)
    complex_concat_layer = ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
    final_merged_passage = complex_concat_layer([bi_lstm_P,
                                                 passage_question_vectors,
                                                 tiled_question_passage_vector])  # Denote G
    # Modelling layer. Take input of (?,?,emb*8) and apply bi-directional LSTM each with d dimensions, finally get 2d * Max_len_[]
    bi_model_passage = Bidirectional(LSTM(256, return_sequences=True),
                                     batch_input_shape=(batch_size, max_len_P, emb_dim))(final_merged_passage)
    # denote M

    # span begin output is calculated by Attention weight & LSTM softmax(Wp1 * [G;M])
    span_begin_input = Concatenate()([final_merged_passage, bi_model_passage])
    span_begin_weights = TimeDistributed(Dense(units=1))(span_begin_input)  # Wp1
    # Shape: (batch_size, num_passage_words)
    span_begin_probabilities = MaskedSoftmax(name="span_begin_softmax")(span_begin_weights)  # (700,)

    # as Minjoon's bidaf indicated, after obtain p1, span_start_prob, he sum all probability values of the entity instances
    # by mask out all non-entity value. and the loss function apply withoutp2
    multiword_option_mode = 'mean'
    options_sum_layer_minj = OptionAttentionSum(multiword_option_mode, name="options_probability_sum_minj")
    options_probabilities_minj = options_sum_layer_minj([passage_input, span_begin_probabilities, options_input])
    l1_norm_layer = L1Normalize()
    option_normalized_probabilities_cnn = l1_norm_layer(options_probabilities_minj)
    # dense = Dense(377, activation='sigmoid')(option_normalized_probabilities_cnn)

    biDAF = Model(inputs=[question_input, passage_input, options_input],
                      outputs=option_normalized_probabilities_cnn)
    biDAF.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return biDAF

# loss: 1.6481 - acc: 0.4578 - val_loss: 1.2756 - val_acc: 0.5968
biDAF = run_biDAF()

# Input Data:
# Passage, Question word index in word occur sequences ==> find embedding value later ;
# Option Input: entity index in word_index ==> feed into embedding layer later
# Output: 1 out of 371 entity

# Load data
# later: train and test split add batch size split batch_size=32
# job_dir = "gs://si630rita_bucket"
#
# logs_path = job_dir + '/logs/' + datetime.now().isoformat()
# print('Using logs_path located at {}'.format(logs_path))




# test the model, before run on whole dataset


# fit model
# epoch=12 can reach 92%
# introduce val data, add call backs
check_point = ModelCheckpoint("biDAF.h5",monitor='val_acc',save_best_only=True,save_weights_only=False)
early_stop = EarlyStopping(monitor='val_acc',patience=10)
# biDAF.fit([Q_train,N_train,option_input],Y_train,validation_data=([Q_val,N_val,val_option_input],Y_val),batch_size=32,epochs=10)
biDAF.fit([Q_train,N_train,train_option_input],Y_train,validation_data=([Q_val,N_val,val_option_input],Y_val),callbacks = [check_point, early_stop],batch_size=32,epochs=10)

biDAF.save('biDAF_reidx.h5')
biDAF.save_weights('biDAF_weights_reindx.h5')




# with file_io.FileIO('biDAF.h5', mode='r') as input_f:
#     with file_io.FileIO(job_dir + '/biDAF.h5', mode='w+') as output_f:
#         output_f.write(input_f.read())
#
# print('Model Saved.')

# max_option = 224
#
#
# option_list = np.lmmhhoad('entity_code10000.npy')  # all options in each news
#
# option_matrix = []
#
# for i in range(len(option_list)):
#     encode = utils.to_categorical(option_list[i], num_classes=386).T  # count correct
#     option_matrix.append(encode)
#
# pad_matrix =[]
# for i in rnage(len(option_matrix)):
#     option_sequence = pad_sequences(option_matrix[i], maxlen=max_option)
#     pad_matrix.append(list(option_matrix))