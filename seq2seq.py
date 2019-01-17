import tensorflow as tf
import sys
import numpy as np
import math
import collections
import time
import os.path
import string
import glob

WINDOW_SZ = 13
BATCH_SZ = 20
EMBED_SZ = 30
LEARN_RATE = 1e-3
VOCAB_SZ_F = 10000
VOCAB_SZ_E = 10000

SAVE_FREQ = 500
STATE_SZ = 64

NUM_EPOCHS = 1


TRAIN_FLAG =False
#TRAIN_FLAG = True

MODEL_DIR = os.path.join(os.getcwd(), "model_files")
MODEL_PATH = os.path.join(MODEL_DIR, "model")

MODEL_DIR_2 = os.path.join(os.getcwd(), "model_files_2")
MODEL_PATH_2 = os.path.join(MODEL_DIR_2, "model_2")

TRAIN_F_PATH = '/course/cs1470/asgn/seq2seq_mt/french_train.txt'
TRAIN_E_PATH = '/course/cs1470/asgn/seq2seq_mt/english_train.txt'
DEV_F_PATH = '/course/cs1470/asgn/seq2seq_mt/french_test.txt'
DEV_E_PATH = '/course/cs1470/asgn/seq2seq_mt/english_test.txt'

RNN_MODEL='0'

start_time = time.time()


class LSTMRNN:
    def __init__(self):
        # define session and grpah
        self.sess = tf.Session()
        self.defineGraph()
        self.saver = tf.train.Saver()

        # check if the model exists already
        checkPoint = tf.train.get_checkpoint_state(MODEL_DIR)
        modelExists = checkPoint and checkPoint.model_checkpoint_path
        if modelExists:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
            print 'restored'
        else:
            self.sess.run(tf.global_variables_initializer())
            print 'initialized'

    def defineGraph(self):
        self.encIn = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.decIn = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.ans = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.keepP = tf.placeholder(tf.float32)

        self.enc_len = tf.placeholder(tf.int32, [BATCH_SZ])
        self.dec_len = tf.placeholder(tf.int32, [BATCH_SZ])

        with tf.variable_scope('enc'):
            F = tf.Variable(tf.random_normal([VOCAB_SZ_F, EMBED_SZ], stddev=.1))
            embs = tf.nn.embedding_lookup(F, self.encIn)
            embs = tf.nn.dropout(embs, self.keepP)
            cell = tf.contrib.rnn.GRUCell(STATE_SZ)
            initState = cell.zero_state(BATCH_SZ, tf.float32)
            encOut, encState = tf.nn.dynamic_rnn(cell, embs, sequence_length=self.enc_len, initial_state=initState)

        with tf.variable_scope('dec'):
            E = tf.Variable(tf.random_normal([VOCAB_SZ_E, EMBED_SZ], stddev=.1))
            embs = tf.nn.embedding_lookup(E, self.decIn)
            embs = tf.nn.dropout(embs, self.keepP)
            cell = tf.contrib.rnn.GRUCell(STATE_SZ)

            decOut, _ = tf.nn.dynamic_rnn(cell, embs, sequence_length=self.dec_len, initial_state=encState)

        W = tf.Variable(tf.random_normal([STATE_SZ, VOCAB_SZ_E], stddev=.1))
        b = tf.Variable(tf.random_normal([VOCAB_SZ_E], stddev=.1))
        logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b
        #probs = tf.nn.softmax(logits)  # shape = bs x ws x vs
        predict_label = tf.argmax(logits, 2, output_type=tf.int32)  # shape = bs x ws
        weights = tf.sequence_mask(self.dec_len, WINDOW_SZ, dtype=tf.float32)

        self.accu =tf.contrib.metrics.accuracy(predict_label, self.ans, weights)

        # self.accu=0.0
        # for i in range(BATCH_SZ):
        #     counter=0.0
        #     correct=0.0
        #     for j in range(WINDOW_SZ):
        #         label=self.ans[i][j]
        #         if label == 0 or j==12:
        #             break
        #         counter+=1
        #         if label==predict_label[i][j]:
        #             correct+=1
        #
        #     self.accu+=correct/counter
        # self.accu=self.accu/BATCH_SZ

        #self.ans=tf.cast(self.ans, tf.float32)
        #print self.ans

        #num_correct = tf.equal(predict_label, self.ans)
        #self.accu = tf.reduce_mean(tf.cast(num_correct, tf.float32))

        self.loss = tf.contrib.seq2seq.sequence_loss(logits, self.ans,
                                                     weights)
        self.train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)


    def save(self):
        self.saver.save(self.sess, MODEL_PATH)

class AttentionRNN():
    def __init__(self):
        # define session and grpah
        self.sess = tf.Session()
        self.defineGraph()
        self.saver = tf.train.Saver()

        # check if the model exists already
        checkPoint = tf.train.get_checkpoint_state(MODEL_DIR_2)
        modelExists = checkPoint and checkPoint.model_checkpoint_path
        if modelExists:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
            print 'restored'
        else:
            self.sess.run(tf.global_variables_initializer())
            print 'initialized'

        # print self.sess.run(self.attWt)
        # print self.sess.run(self.attWt_sum)
        # print self.sess.run(self.attWt_prob)

    def defineGraph(self):
        self.encIn = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.decIn = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.ans = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
        self.attWt = tf.Variable(tf.ones([WINDOW_SZ,WINDOW_SZ], tf.float32))

        self.attWt_sum = tf.reduce_sum(self.attWt, 0)
        self.attWt.assign(tf.div(self.attWt, self.attWt_sum))
        #self.attWt_prob=tf.div(self.attWt, self.attWt_sum)

        self.enc_len = tf.placeholder(tf.int32, [BATCH_SZ])
        self.dec_len = tf.placeholder(tf.int32, [BATCH_SZ])

        self.keepP = tf.placeholder(tf.float32)

        with tf.variable_scope('enc'):
            F = tf.Variable(tf.random_normal([VOCAB_SZ_F, EMBED_SZ], stddev=.1))
            embs = tf.nn.embedding_lookup(F, self.encIn)
            embs = tf.nn.dropout(embs, self.keepP)
            cell = tf.contrib.rnn.GRUCell(STATE_SZ)
            initState = cell.zero_state(BATCH_SZ, tf.float32)
            encOut, encState = tf.nn.dynamic_rnn(cell, embs, sequence_length=self.enc_len, initial_state=initState)
            encOT=tf.transpose(encOut, [0,2,1])
            decIT=tf.tensordot(encOT, self.attWt, [[2], [0]])
            decI=tf.transpose(decIT, [0,2,1])


        with tf.variable_scope('dec'):
            E = tf.Variable(tf.random_normal([VOCAB_SZ_E, EMBED_SZ], stddev=.1))
            embs = tf.nn.embedding_lookup(E, self.decIn)
            embs = tf.nn.dropout(embs, self.keepP)
            both=tf.concat([embs, decI], 2)
            cell = tf.contrib.rnn.GRUCell(STATE_SZ)
            decOut, _ = tf.nn.dynamic_rnn(cell, both, initial_state=encState)

        W = tf.Variable(tf.random_normal([STATE_SZ, VOCAB_SZ_E], stddev=.1))
        b = tf.Variable(tf.random_normal([VOCAB_SZ_E], stddev=.1))
        logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b
        probs = tf.nn.softmax(logits)  # shape = bs x ws x vs
        predict_label = tf.argmax(probs, 2, output_type=tf.int32)  # shape = bs x ws
        #self.ans = tf.cast(self.ans, tf.float32)
        num_correct= tf.equal(predict_label, self.ans)
        self.accu=tf.reduce_mean(tf.cast(num_correct, tf.float32))
        self.loss = tf.contrib.seq2seq.sequence_loss(logits, self.ans,
                                                     tf.ones([BATCH_SZ, WINDOW_SZ]))
        self.train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)


    def save(self):
        self.saver.save(self.sess, MODEL_PATH_2)
#################################################################
def preprocess(thepath):
    # read in corpus as words then process to number IDs.
    corpusWords = []  # turn corpus into a list of tokened sentence
    allowedWords = {'STOP'}  # the dictionary

    with open(thepath, "r") as corpusFile:
        lines = corpusFile.readlines()

    for line in lines:
        token_line = line.split()
        allowedWords.update(token_line)
        corpusWords.append(token_line)

    # wordCounts = collections.Counter(corpusWords)

    # allowedWords = set(list(commonWords) + ["*UNK*"])

    windowData = []  # corpus of numbers representing words
    windowData_len=[]
    wordToIDs = {}
    wordToIDs['STOP'] = 0
    nextIDToUse = 1

    for line in corpusWords:  # line is a list of tokened words
        padded_line = [0] * WINDOW_SZ
        windowData_len.append(len(line))

        for i in xrange(len(line)):  # i is the word index
            if line[i] not in wordToIDs:  # if word not in our dictionary, buy it valid, give it ID
                wordToIDs[line[i]] = nextIDToUse
                nextIDToUse += 1

            padded_line[i] = wordToIDs[line[i]]
        windowData.append(padded_line)

    return (windowData, allowedWords, windowData_len)


if __name__ == "__main__":
    # read in data into batches then train...

    if len(sys.argv) > 1:
        TRAIN_F_PATH = sys.argv[1]
        TRAIN_E_PATH = sys.argv[2]
        DEV_F_PATH = sys.argv[3]
        DEV_E_PATH = sys.argv[4]
        RNN_MODEL=sys.argv[5]



    train_f_windowData, train_f_voc, train_f_len = preprocess(TRAIN_F_PATH)
    VOCAB_SZ_F = len(train_f_voc)

    train_e_windowData_label, train_e_voc, train_e_len = preprocess(TRAIN_E_PATH)
    VOCAB_SZ_E = len(train_e_voc)
    #train_e_len=[x+1 for x in train_e_len]

    train_e_windowData = []
    for line in train_e_windowData_label:
        train_e_windowData.append(line[-1:] + line[:-1])

    ###################################train

    if RNN_MODEL == '0':
        rnnModel = LSTMRNN()
    else:
        rnnModel = AttentionRNN()
    #raw_input('Enter your input:')

    if TRAIN_FLAG:
        train_f_numBatches = len(
            train_f_windowData) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).

        train_e_numBatches = len(
            train_e_windowData) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).

        train_e_numBatches_label = len(
            train_e_windowData_label) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).

        numBatches = min(train_f_numBatches, train_e_numBatches, train_e_numBatches_label)
        loss_sum = 0.0

        for i in xrange(0, numBatches):
            slicer=slice(i * BATCH_SZ, BATCH_SZ+i * BATCH_SZ)
            train_f_batchData = train_f_windowData[slicer]
            train_f_batchLen= train_f_len[slicer]
            #x=np.array(train_f_batchData)
            #print x.shape
            #print i
            train_e_batchData = train_e_windowData[slicer]
            train_e_batchLabel = train_e_windowData_label[slicer]
            train_e_batchLen = train_e_len[slicer]




            feedDict = {rnnModel.encIn: train_f_batchData, rnnModel.decIn: train_e_batchData,
                        rnnModel.ans: train_e_batchLabel, rnnModel.enc_len: train_f_batchLen,
                        rnnModel.dec_len: train_e_batchLen, rnnModel.keepP: 0.5}
            sessArgs = [rnnModel.accu, rnnModel.train_step]

            loss, _ = rnnModel.sess.run(sessArgs, feed_dict=feedDict)

            loss_sum += loss

            if i % SAVE_FREQ == 0:
                rnnModel.save()
                end_time = time.time()
                print "saved at batches : ", i
                print 'the accu for training is ', loss
                print "Time elapsed: %f s" % (end_time - start_time)
        rnnModel.save()

        print 'the avg loss for training is ', loss_sum / numBatches
        print 'time elapsed: ', time.time() - start_time

    ##########################test
    test_f_windowData, _, test_f_len= preprocess(DEV_F_PATH)
    test_e_windowData_label, _, test_e_len = preprocess(DEV_E_PATH)
    #test_e_len = [x + 1 for x in test_e_len]

    test_e_windowData = []
    for line in test_e_windowData_label:
        test_e_windowData.append(line[-1:] + line[:-1])

    test_f_numBatches = len(
        test_f_windowData) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).
    test_e_numBatches = len(
        test_e_windowData) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).
    test_e_numBatches_label = len(
        test_e_windowData_label) / BATCH_SZ  # number of batches that will fit. also length of one sequence (in windows).

    numBatches = min(test_f_numBatches, test_e_numBatches, test_e_numBatches_label)

    accu_sum = 0.0
    loss_total=0.0
    for i in xrange(0, numBatches):
        slicer = slice(i * BATCH_SZ, BATCH_SZ + i * BATCH_SZ)
        test_f_batchData = test_f_windowData[slicer]
        test_f_batchLen = test_f_len[slicer]

        test_e_batchData = test_e_windowData[slicer]
        test_e_batchLabel = test_e_windowData_label[slicer]
        test_e_batchLen = test_e_len[slicer]

        feedDict = {rnnModel.encIn: test_f_batchData, rnnModel.decIn: test_e_batchData,
                    rnnModel.ans: test_e_batchLabel, rnnModel.enc_len: test_f_batchLen,
                        rnnModel.dec_len: test_e_batchLen, rnnModel.keepP: 1.0}
        sessArgs = [rnnModel.loss, rnnModel.accu]
        loss, accu = rnnModel.sess.run(sessArgs, feed_dict=feedDict)
        accu_sum += accu
        loss_total +=loss

    print 'the avg accuracy for test is ', accu_sum / numBatches
    print 'the avg loss for test is ', loss_total / numBatches
    print 'time elapsed: ', time.time() - start_time
