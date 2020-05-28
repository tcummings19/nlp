import os
import datetime

import tensorflow as tf

from reanlp import nlp_utils


class RNN:

    def __init__(self, models_dir, model_name, vocab_size, num_classes, batch_size=1, num_steps=200, lstm_size=256,
                 num_layers=1, learning_rate=0.01, keep_prob=0.5, grad_clip=5):

        self.models_dir = models_dir
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.g = tf.Graph()

        with self.g.as_default():

            tf.set_random_seed(123)

            self.build()

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

            # fix for tensorflow multiprocess thread lock issue
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

    def add_placeholders(self):

        self.tf_x = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='tf_x')
        self.tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
        self.tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')
        self.tf_batch_size = tf.placeholder(tf.int32, [], name='tf_batch_size')

    def add_embeddings(self):

        # create the word embeddings layer.
        # The word embedding layer will be initialized randomly and over the training duration, the system
        # will learn which words tend to be associated with others.
        embedding_size = 150

        init_embeds = tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0)
        embeddings = tf.Variable(init_embeds)
        self.embed = tf.nn.embedding_lookup(embeddings, self.tf_x, name='embed')

    def add_cells(self):

        self.cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                output_keep_prob=self.tf_keepprob)
                for _ in range(self.num_layers)])

        # Define the initial state
        self.initial_state = self.cells.zero_state(self.tf_batch_size, tf.float32)

    def add_logits(self):

        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.cells, self.embed,
                                                                initial_state=self.initial_state)

        print(' << lstm_outputs >>', self.lstm_outputs)
        print(' << final_state >>', self.final_state)

        self.logits = tf.layers.dense(inputs=self.final_state[-1][1],  units=self.num_classes,
                                 activation=None, name='logits')

        print(' << logits >> ', self.logits)

    def add_proba(self):

        self.proba = tf.nn.softmax(self.logits, name='probabilities')

        print(' << proba >> ', self.proba)

    def add_preds(self):

        self.pred_labels = tf.argmax(self.logits, axis=1, name='labels')

        print(' << Label Predictions >> ', self.pred_labels)

    def add_loss(self):

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_y, name='xentropy')

        cost = tf.reduce_mean(xentropy, name='cost')

        # Gradient clipping to avoid  "exploding gradients"
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_op')

    def build(self):

        self.add_placeholders()

        self.add_embeddings()

        self.add_cells()

        self.add_logits()

        self.add_proba()

        self.add_preds()

        self.add_loss()

    def train(self, x_train, y_train, num_epochs):
        # Create the checkpoint directory if it does not exist

        model_dir_name = self.models_dir + self.model_name

        # create model directory if it doesnt exist
        nlp_utils.create_dir(model_dir_name)

        with tf.Session(graph=self.g) as sess:
            print('Initializing Tensor Variables')
            sess.run(self.init_op)

            print('Beginning Training')

            iteration = 1
            for epoch in range(num_epochs):

                # Train network
                state = sess.run(self.initial_state)
                loss = 0

                for batch_x, batch_y in nlp_utils.create_batch_generator(x_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keepprob:0': self.keep_prob,
                            self.initial_state: state}

                    loss, _, state = sess.run(['cost:0', 'train_op', self.final_state], feed_dict=feed)

                    if iteration % 10 == 0:
                        print('Epoch %d/%d Iteration %d | Training loss: %.4f' % (epoch + 1, num_epochs,
                                                                                  iteration, loss))
                    iteration += 1

                ## Save the trained model
                if (epoch + 1) % 10 == 0:
                    self.saver.save(sess, os.path.join(model_dir_name, self.model_name))

            self.saver.save(sess, os.path.join(model_dir_name, self.model_name + '_final'))

        return None


class CLFBiRNN(RNN):

    # modify logits to have a forward and backward RNN layers

    def add_cells(self):

        self.cells_fw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                output_keep_prob=self.tf_keepprob)
                for _ in range(self.num_layers)])

        self.cells_bw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                output_keep_prob=self.tf_keepprob)
                for _ in range(self.num_layers)])

        # Define the initial state
        self.initial_state_fw = self.cells_fw.zero_state(self.tf_batch_size, tf.float32)
        self.initial_state_bw = self.cells_bw.zero_state(self.tf_batch_size, tf.float32)

    def add_logits(self):

        bi_lstm_outputs, self.final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cells_fw,
            cell_bw=self.cells_bw,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            inputs=self.embed,
            dtype=tf.float32)

        self.final_states_fw, self.final_states_bw = self.final_states

        print(' << lstm_outputs >>', bi_lstm_outputs)
        print(' << final_state_fw >>', self.final_states_fw)
        print(' << final_state_bw >>', self.final_states_bw)

        final_states_concat = tf.concat([self.final_states_fw[-1][1], self.final_states_bw[-1][1]], 1)
        print(' << final_state_concat >>', final_states_concat)

        self.logits = tf.layers.dense(inputs=final_states_concat, units=self.num_classes,
                                      activation=None, name='logits')

        print('\n << logits >> ', self.logits)

    #revise train and predict to reflect the forward and backward RNN layers

    def train(self, x_train, y_train, num_epochs):
        ## Create the checkpoint directory if it does not exist

        model_dir_name = self.models_dir + self.model_name

        # create model directory if it doesnt exist
        nlp_utils.create_dir(model_dir_name)

        with tf.Session(graph=self.g) as sess:
            print('Initializing Tensor Variables')
            sess.run(self.init_op)

            print('Beginning Training')

            iteration = 1
            for epoch in range(num_epochs):

                # Train network
                states = sess.run([self.initial_state_fw, self.initial_state_bw], {'tf_batch_size:0': self.batch_size})
                loss = 0

                for batch_x, batch_y in nlp_utils.create_batch_generator(x_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keepprob:0': self.keep_prob,
                            self.initial_state_fw: states[0],
                            self.initial_state_bw: states[1]}

                    pred, loss, _, states = sess.run(['labels:0', 'cost:0', 'train_op', self.final_states], feed_dict=feed)

                    #if iteration % 10 == 0:
                    print('Epoch %d/%d Iteration %d | Training loss: %.4f' % (epoch + 1, num_epochs,
                                                                                  iteration, loss))
                    iteration += 1

                ## Save the trained model
                if (epoch + 1) % 10 == 0:
                    self.saver.save(sess, os.path.join(model_dir_name, self.model_name))

            self.saver.save(sess, os.path.join(model_dir_name, self.model_name + '_final'))

            # exporting model for serving
            export_path = model_dir_name + '_v_' + datetime.datetime.now().strftime("%m-%d-%y")

            print('Exporting Trained Model to: ' + export_path)

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # Build the signature_def_map

            classification_inputs = tf.saved_model.utils.build_tensor_info(self.tf_x)
            classification_output_classes = tf.saved_model.utils.build_tensor_info(self.pred_labels)
            classification_output_scores = tf.saved_model.utils.build_tensor_info(self.proba)

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                      tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
                    },
                    outputs={
                      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_output_classes,
                      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_output_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

            tensor_info_x = tf.saved_model.utils.build_tensor_info(self.tf_x)
            tensor_info_keepprob = tf.saved_model.utils.build_tensor_info(self.tf_keepprob)
            tensor_info_batch_size = tf.saved_model.utils.build_tensor_info(self.tf_batch_size)
            tensor_info_x = tf.saved_model.utils.build_tensor_info(self.tf_x)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(self.pred_labels)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'words': tensor_info_x, 'keepprob': tensor_info_keepprob, 'batch_size': tensor_info_batch_size},
                    outputs={'labels': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={
                                                     'predict_words':
                                                         prediction_signature,
                                                     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                         classification_signature,
                                                 },
                                                 legacy_init_op=legacy_init_op)
            builder.save()

            print('Done Exporting!')

        return None


class NERBiRNN(CLFBiRNN):

    # revised placeholders to reflect tf_y to have a shape of self.batch_size, self.num_steps
    def add_placeholders(self):

        self.tf_x = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='tf_x')
        self.tf_y = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='tf_y')
        self.tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')
        self.tf_batch_size = tf.placeholder(tf.int32, [], name='tf_batch_size')

    # one hot the y values for soft max
    def add_one_hot(self):

        self.tf_y_onehot = tf.one_hot(self.tf_y, depth=self.num_classes, name='tf_y_onehot')

    # override method to get output for each input sequence (rather than just one output for the entire sequence)
    def add_logits(self):

        # Run each sequence step through the RNN
        (bi_lstm_output_fw, bi_lstm_output_bw), self.final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cells_fw,
            cell_bw=self.cells_bw,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            inputs=self.embed,
            dtype=tf.float32)

        self.final_states_fw, self.final_states_bw = self.final_states

        print(' << lstm_outputs_fw >>', bi_lstm_output_fw)
        print(' << lstm_outputs_bw >>', bi_lstm_output_bw)
        print(' << final_state_fw >>', self.final_states_fw)
        print(' << final_state_bw >>', self.final_states_bw)

        outputs_concat = tf.concat([bi_lstm_output_fw, bi_lstm_output_bw], axis=-1)

        print(' << outputs_concat >>', outputs_concat)

        outputs_reshaped = tf.reshape(outputs_concat, [-1, 2*self.lstm_size])

        print(' << outputs_reshaped >>', outputs_reshaped)

        self.logits = tf.layers.dense(inputs=outputs_reshaped,  units=self.num_classes, activation=None, name='logits')

    # override method to leverage softmax cross entropy with y_onehot labels
    def add_loss(self):

        # new version of softmax, previous version from tensorflow is deprecated
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.tf_y_onehot, name='xentropy')
        cost = tf.reduce_mean(xentropy, name='cost')

        # Gradient clipping to avoid  "exploding gradients"
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_op')

        return None

    # override build to call one hot method
    def build(self):

        self.add_placeholders()

        self.add_embeddings()

        self.add_one_hot()

        self.add_cells()

        self.add_logits()

        self.add_proba()

        self.add_preds()

        self.add_loss()