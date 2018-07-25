import collections
import tensorflow as tf
import re
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

try:
    import cPickle as pickle
except ImportError:
    import pickle


class MLTModel(object):
    def __init__(self, config):
        self.config = config
        self.tasks = config["tasks"].strip().split(":")
        self.UNK = "<unk>"
        self.CUNK = "<cunk>"

        self.word2id = None
        self.char2id = None
        self.singletons = None

    def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
        data_source = []
        vocab_data = [data_train]
        if self.config["vocab_include_devtest"]:
            vocab_data.extend([data_dev, data_test])
        for split in vocab_data:
            for _, dataset in split.items():
                data_source += dataset
        char_counter = collections.Counter()
        for sentence, sent_label in data_source:
            for word in sentence:
                char_counter.update(word[0])
        self.char2id = collections.OrderedDict([(self.CUNK, 0)])
        for char, count in char_counter.most_common():
            if char not in self.char2id:
                self.char2id[char] = len(self.char2id)
        # self.sentlabel2id = collections.OrderedDict([(self.UNK, 0)])

        word_counter = collections.Counter()

        for sentence, sent_label in data_source:
            # if sent_label not in self.sentlabel2id:
            #     self.sentlabel2id[sent_label] = len(self.sentlabel2id)
            for word in sentence:
                w = word[0]
                if self.config["lowercase"]:
                    w = w.lower()
                if self.config["replace_digits"]:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1

        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

        self.singletons = set([word for word in word_counter if word_counter[word] == 1])

        if embedding_path is not None and self.config["vocab_only_embedded"]:
            self.embedding_vocab = {self.UNK}
            with open(embedding_path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <= 2:
                        continue
                    w = line_parts[0]
                    if self.config["lowercase"] == True:
                        w = w.lower()
                    if self.config["replace_digits"] == True:
                        w = re.sub(r'\d', '0', w)
                    self.embedding_vocab.add(w)
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in self.embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            self.word2id = word2id_revised

        print("n_words: " + str(len(self.word2id)))
        print("n_chars: " + str(len(self.char2id)))
        print("n_singletons: " + str(len(self.singletons)))

    def construct_network(self):
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
        self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
        self.sentence_labels = tf.placeholder(tf.float32, [None,], name="sentence_labels")
        self.word_labels = tf.placeholder(tf.float32, [None,None], name="word_labels")
        self.word_objective_weights = tf.placeholder(tf.float32, [None,None], name="word_objective_weights")
        self.sentence_objective_weights = tf.placeholder(tf.float32, [None], name="sentence_objective_weights")
        self.learningrate = tf.placeholder(tf.float32, name="learningrate")
        self.is_training = tf.placeholder(tf.int32, name="is_training")

        self.loss = 0.0
        input_tensor = None
        input_vector_size = 0

        self.initializer = None
        if self.config["initializer"] == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        elif self.config["initializer"] == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config["initializer"] == "xavier":
            self.initializer = tf.glorot_normal_initializer()

        zeros_initializer = tf.zeros_initializer()

        self.word_embeddings = tf.get_variable("word_embeddings", 
            shape=[len(self.word2id), self.config["word_embedding_size"]], 
            initializer=(zeros_initializer if self.config["emb_initial_zero"] == True else self.initializer), 
            trainable=(True if self.config["train_embeddings"] == True else False))
        input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
        input_vector_size = self.config["word_embedding_size"]

        if self.config["char_embedding_size"] > 0 and self.config["char_recurrent_size"] > 0:
            with tf.variable_scope("chars"), tf.control_dependencies([tf.assert_equal(tf.shape(self.char_ids)[2], tf.reduce_max(self.word_lengths), message="Char dimensions don't match")]):
                self.char_embeddings = tf.get_variable("char_embeddings", 
                    shape=[len(self.char2id), self.config["char_embedding_size"]], 
                    initializer=self.initializer, 
                    trainable=True)
                char_input_tensor = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)

                s = tf.shape(char_input_tensor)
                char_input_tensor = tf.reshape(char_input_tensor, shape=[s[0]*s[1], s[2], self.config["char_embedding_size"]])
                _word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                char_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"], 
                    use_peepholes=self.config["lstm_use_peepholes"], 
                    state_is_tuple=True, 
                    initializer=self.initializer,
                    reuse=False)
                char_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"], 
                    use_peepholes=self.config["lstm_use_peepholes"], 
                    state_is_tuple=True, 
                    initializer=self.initializer,
                    reuse=False)

                char_lstm_outputs = tf.nn.bidirectional_dynamic_rnn(char_lstm_cell_fw, char_lstm_cell_bw, char_input_tensor, sequence_length=_word_lengths, dtype=tf.float32, time_major=False)
                _, ((_, char_output_fw), (_, char_output_bw)) = char_lstm_outputs
                char_output_tensor = tf.concat([char_output_fw, char_output_bw], axis=-1)
                char_output_tensor = tf.reshape(char_output_tensor, shape=[s[0], s[1], 2 * self.config["char_recurrent_size"]])
                char_output_vector_size = 2 * self.config["char_recurrent_size"]

                if self.config["lmcost_char_gamma"] > 0.0:
                    self.loss += self.config["lmcost_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "separate", "lmcost_char_separate")
                if self.config["lmcost_joint_char_gamma"] > 0.0:
                    self.loss += self.config["lmcost_joint_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "joint", "lmcost_char_joint")

                if self.config["char_hidden_layer_size"] > 0:
                    char_output_tensor = tf.layers.dense(char_output_tensor, self.config["char_hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
                    char_output_vector_size = self.config["char_hidden_layer_size"]

                if self.config["char_integration_method"] == "concat":
                    input_tensor = tf.concat([input_tensor, char_output_tensor], axis=-1)
                    input_vector_size += char_output_vector_size
                elif self.config["char_integration_method"] == "none":
                    input_tensor = input_tensor
                else:
                    raise ValueError("Unknown char integration method")

        self.word_representations = input_tensor

        dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
        input_tensor =  tf.nn.dropout(input_tensor, dropout_input, name="dropout_word")

        word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
        word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)

        with tf.control_dependencies([tf.assert_equal(tf.shape(self.word_ids)[1], tf.reduce_max(self.sentence_lengths), message="Sentence dimensions don't match")]):
            (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(word_lstm_cell_fw, word_lstm_cell_bw, input_tensor, sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)

        dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
        lstm_outputs_fw =  tf.nn.dropout(lstm_outputs_fw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
        lstm_outputs_bw =  tf.nn.dropout(lstm_outputs_bw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
        lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)

        if self.config["whidden_layer_size"] > 0:
            lstm_outputs = tf.layers.dense(lstm_outputs, self.config["whidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)

        self.lstm_outputs = lstm_outputs

        lstm_output = tf.concat([lstm_output_fw, lstm_output_bw], -1)
        lstm_output = tf.nn.dropout(lstm_output, dropout_word_lstm)

        if self.config["sentence_composition"] == "last":
            processed_tensor = lstm_output
            self.attention_weights_unnormalised = tf.zeros_like(self.word_ids, dtype=tf.float32)
        elif self.config["sentence_composition"] == "attention":
            with tf.variable_scope("attention"):
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.tanh, kernel_initializer=self.initializer)

                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids))

                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                word_objective_loss = tf.square(attention_weights - self.word_labels)
                word_objective_loss = tf.where(tf.sequence_mask(self.sentence_lengths), word_objective_loss, tf.zeros_like(word_objective_loss))
                # self.loss += self.config["word_objective_weight"] * tf.reduce_sum(self.word_objective_weights * word_objective_loss)
                self.loss += tf.reduce_sum(
                    self.word_objective_weights * word_objective_loss)

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.sentence_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                processed_tensor = tf.reduce_sum(lstm_outputs * attention_weights[:,:,np.newaxis], 1)
                self.attention_weights = attention_weights

        if self.config["hidden_layer_size"] > 0:
            processed_tensor = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)

        self.sentence_scores = tf.layers.dense(processed_tensor, 1, activation=tf.sigmoid, kernel_initializer=self.initializer, name="output_ff")
        self.sentence_scores = tf.reshape(self.sentence_scores, shape=[tf.shape(processed_tensor)[0]])

        # self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(self.sentence_objective_weights * tf.square(self.sentence_scores - self.sentence_labels))
        self.loss += tf.reduce_sum(self.sentence_objective_weights *
                                   tf.square(self.sentence_scores -
                                             self.sentence_labels))

        if self.config["attention_objective_weight"] > 0.0:
            self.loss += self.config["attention_objective_weight"] * \
                (tf.reduce_sum(
                    self.sentence_objective_weights *tf.square(
                        tf.reduce_max(
                            tf.where(
                                tf.sequence_mask(self.sentence_lengths), 
                                self.attention_weights_unnormalised, 
                                tf.zeros_like(self.attention_weights_unnormalised) - 1e6), 
                            axis=-1) - self.sentence_labels))
                + 
                tf.reduce_sum(
                    self.sentence_objective_weights * tf.square(
                        tf.reduce_min(
                            tf.where(
                                tf.sequence_mask(self.sentence_lengths), 
                                self.attention_weights_unnormalised, 
                                tf.zeros_like(self.attention_weights_unnormalised) + 1e6), 
                            axis=-1) - 0.0)))

        self.token_scores = [tf.where(tf.sequence_mask(self.sentence_lengths), self.attention_weights_unnormalised, tf.zeros_like(self.attention_weights_unnormalised) - 1e6)]


        if self.config["lmcost_lstm_gamma"] > 0.0:
            self.loss += self.config["lmcost_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "separate", "lmcost_lstm_separate")
        if self.config["lmcost_joint_lstm_gamma"] > 0.0:
            self.loss += self.config["lmcost_joint_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "joint", "lmcost_lstm_joint")

        self.train_op = self.construct_optimizer(self.config["opt_strategy"], self.loss, self.learningrate, self.config["clip"])

    def construct_lmcost(self, input_tensor_fw, input_tensor_bw, sentence_lengths, target_ids, lmcost_type, name):
        with tf.variable_scope(name):
            lmcost_max_vocab_size = min(len(self.word2id), self.config["lmcost_max_vocab_size"])
            target_ids = tf.where(tf.greater_equal(target_ids, lmcost_max_vocab_size-1), x=(lmcost_max_vocab_size-1)+tf.zeros_like(target_ids), y=target_ids)
            cost = 0.0
            if lmcost_type == "separate":
                lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:]
                lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,:-1]
                lmcost_fw = self._construct_lmcost(input_tensor_fw[:,:-1,:], lmcost_max_vocab_size, lmcost_fw_mask, target_ids[:,1:], name=name+"_fw")
                lmcost_bw = self._construct_lmcost(input_tensor_bw[:,1:,:], lmcost_max_vocab_size, lmcost_bw_mask, target_ids[:,:-1], name=name+"_bw")
                cost += lmcost_fw + lmcost_bw
            elif lmcost_type == "joint":
                joint_input_tensor = tf.concat([input_tensor_fw[:,:-2,:], input_tensor_bw[:,2:,:]], axis=-1)
                lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:-1]
                cost += self._construct_lmcost(joint_input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids[:,1:-1], name=name+"_joint")
            else:
                raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
            return cost

    def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
        with tf.variable_scope(name):
            lmcost_hidden_layer = tf.layers.dense(input_tensor, self.config["lmcost_hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
            lmcost_output = tf.layers.dense(lmcost_hidden_layer, lmcost_max_vocab_size, activation=None, kernel_initializer=self.initializer)
            lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lmcost_output, labels=target_ids)
            lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
            return tf.reduce_sum(lmcost_loss)

    def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
        optimizer = None
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
        else:
            raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

        if clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)
        return train_op

    def preload_word_embeddings(self, embedding_path):
        loaded_embeddings = set()
        embedding_matrix = self.session.run(self.word_embeddings)
        with open(embedding_path, 'r') as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                w = line_parts[0]
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                if w in self.word2id and w not in loaded_embeddings:
                    word_id = self.word2id[w]
                    embedding = np.array(line_parts[1:])
                    embedding_matrix[word_id] = embedding
                    loaded_embeddings.add(w)
        self.session.run(self.word_embeddings.assign(embedding_matrix))
        print("n_preloaded_embeddings: " + str(len(loaded_embeddings)))

    def translate2id(self, token, token2id, unk_token, lowercase=False, replace_digits=False, singletons=None, singletons_prob=0.0):
        if lowercase == True:
            token = token.lower()
        if replace_digits == True:
            token = re.sub(r'\d', '0', token)

        token_id = None
        if singletons != None and token in singletons and token in token2id and unk_token != None and np.random.uniform() < singletons_prob:
            token_id = token2id[unk_token]
        elif token in token2id:
            token_id = token2id[token]
        elif unk_token != None:
            token_id = token2id[unk_token]
        else:
            raise ValueError("Unable to handle value, no UNK token: " + str(token))
        return token_id

    # TODO dynamic change of weights depending on task
    def create_input_dictionary_for_batch(self, task, batch, is_training, learningrate):
        sentence_lengths = np.array([len(sentence) for sentence, sent_label in batch])
        max_sentence_length = sentence_lengths.max()
        max_word_length = np.array([np.array([len(word[0]) for word in sentence]).max() for sentence, sent_label in batch]).max()
        if self.config["allowed_word_length"] > 0 and self.config["allowed_word_length"] < max_word_length:
            max_word_length = min(max_word_length, self.config["allowed_word_length"])

        word_ids = np.zeros((len(batch), max_sentence_length), dtype=np.int32)
        char_ids = np.zeros((len(batch), max_sentence_length, max_word_length), dtype=np.int32)
        word_lengths = np.zeros((len(batch), max_sentence_length), dtype=np.int32)
        word_labels = np.zeros((len(batch), max_sentence_length), dtype=np.float32)
        sentence_labels = np.zeros((len(batch)), dtype=np.float32)
        word_objective_weights = np.zeros((len(batch), max_sentence_length), dtype=np.float32)
        sentence_objective_weights = np.zeros((len(batch)), dtype=np.float32)

        singletons = self.singletons if is_training == True else None
        singletons_prob = self.config["singletons_prob"] if is_training == True else 0.0
        for i in range(len(batch)):
            sent, sent_label = batch[i]
            # TODO sentence labels need to be one-hot for multi-class classification
            if sent_label == "" or sent_label == self.config["ignore_label"]:
                sentence_objective_weights[i] = 0.0
            else:
                sentence_objective_weights[i] = 1.0
                sentence_labels[i] = 0.0 if \
                    sent_label == self.config["default_label"] else 1.0

            if self.config["sentence_objective_persistent"]:
                sentence_objective_weights[i] = 1.0

            for j in range(len(sent)):
                word_ids[i][j] = self.translate2id(sent[j][0], self.word2id, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"], singletons=singletons, singletons_prob=singletons_prob)
                # print(sent[j], self.config["default_label"], sent[j][-1] == self.config["default_label"])
                word_labels[i][j] = 0.0 if sent[j][-1] in [
                    self.config["default_label"], self.config["ignore_label"]] \
                    else float(sent[j][-1])
                word_lengths[i][j] = len(sent[j][0])
                for k in range(min(len(sent[j][0]), max_word_length)):
                    char_ids[i][j][k] = self.translate2id(sent[j][0][k], self.char2id, self.CUNK)
                
                # if len(sent[j]) == 2 or (len(sent[j]) >= 3 and sent[j][1] == "T"):
                #     word_objective_weights[i][j] = 1.0

                if not sent[j][-1] == self.config["ignore_label"]:
                    word_objective_weights[i][j] = 1.0


            # if len(sent[j]) == 2 or (len(sent[j]) >= 3 and sent[0][1] == "S")
            # or self.config["sentence_objective_persistent"] == True:

        input_dictionary = {self.word_ids: word_ids, self.char_ids: char_ids, self.sentence_lengths: sentence_lengths, self.word_lengths: word_lengths, self.word_labels: word_labels, self.word_objective_weights: word_objective_weights, self.sentence_labels: sentence_labels, self.sentence_objective_weights: sentence_objective_weights, self.learningrate: learningrate, self.is_training: is_training}
        # print(sentence_objective_weights[-1])
        # print(batch[-1])
        # print("weights", word_objective_weights[-1])
        # print("labels", word_labels[-1])
        return input_dictionary

    def process_batch(self, task, batch, is_training, learningrate):
        feed_dict = self.create_input_dictionary_for_batch(task,
                                                           batch,
                                                           is_training,
                                                           learningrate)
        cost, sentence_scores, token_scores, attention_scores = \
            self.session.run([self.loss, self.sentence_scores,
                              self.token_scores, self.attention_weights] +
                             ([self.train_op] if is_training == True else []),
                             feed_dict=feed_dict)[:4]  # don't return train_op
        return cost, sentence_scores, token_scores, attention_scores

    def initialize_session(self):
        tf.set_random_seed(self.config["random_seed"])
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
        session_config.gpu_options.per_process_gpu_memory_fraction = self.config["tf_per_process_gpu_memory_fraction"]
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def get_parameter_count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def get_parameter_count_without_word_embeddings(self):
        shape = self.word_embeddings.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        return self.get_parameter_count() - variable_parameters

    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["UNK"] = self.UNK
        dump["CUNK"] = self.CUNK
        dump["word2id"] = self.word2id
        dump["char2id"] = self.char2id
        dump["singletons"] = self.singletons

        dump["params"] = {}
        for variable in tf.global_variables():
            assert(variable.name not in dump["params"]), "Error: variable with this name already exists" + str(variable.name)
            dump["params"][variable.name] = self.session.run(variable)
        f = open(filename, 'wb')
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


    @staticmethod
    def load(filename, new_config=None):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            # for safety, so we don't overwrite old models
            dump["config"]["save"] = None

            # we use the saved config, except for values that are present in the new config
            if new_config != None:
                for key in new_config:
                    dump["config"][key] = new_config[key]

            labeler = MLTModel(dump["config"])
            labeler.UNK = dump["UNK"]
            labeler.CUNK = dump["CUNK"]
            labeler.word2id = dump["word2id"]
            labeler.char2id = dump["char2id"]
            labeler.singletons = dump["singletons"]

            labeler.construct_network()
            labeler.initialize_session()

            labeler.load_params(filename)

            return labeler


    def load_params(self, filename):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            for variable in tf.global_variables():
                assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
                assert(variable.shape == dump["params"][variable.name].shape), "Variable shape not as expected: " + str(variable.name) + " " + str(variable.shape) + " " + str(dump["params"][variable.name].shape)
                value = np.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))

