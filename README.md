Multi-Level Tagger
==============================

Run experiment with 

    python experiment.py config_file.conf data_config_file.conf

The first of these config files defines tasks and hyperparameters, while the
second lists paths for the datasets for each task.
Examples of the config files can be found in `conf`.



Data format
-------------------------

The training and test data is expected in standard CoNLL-type tab-separated format. One word per line, separate column for token and label, empty line between sentences.

For error detection, this would be something like:

    I       -
    saws    +
    the     -
    show    -
    
Sentence-level labels are optionally marked on an extra line preceding the first word:

    -
    I       -
    saws    +
    the     -
    show    -

    +
    Did     -
    you     +
    see     -
    it      -
    ?       -


Any word with `default_label` gets label 0, any word with other labels gets assigned 1.
Sentences with annotations only at the sentence-level mark each word with the `ignore_label`.

    X
    I       _
    saws    _
    the     _
    show    _


Configuration
-------------------------

Edit the values in config.conf as needed:

* **default_label** - The most common (negative) label in the dataset. For example, the correct label in error detection or neutral label in sentiment detection.
* **ignore_label** - Null-label for un-annotated tokens, e.g. in partially annotated data.
* **model_selector** - What is measured on the dev set for model selection. For example, "dev_sent_f:high" means we're looking for the highest sentence-level F score on the development set.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **emb_initial_zero** - Whether word embeddings should be initialized with zeros. Otherwise, they are initialized randomly. If 'preload_vectors' is set, the initialization will be overwritten either way for words that have pretrained embeddings.
* **train_embeddings** - Whether word embeddings are updated during training.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **hidden_layer_size** - Final hidden layer size, right before word-level predictions.
* **char_hidden_layer_size** - Char-level representation size, right before it gets combined with the word embeddings.
* **lowercase** - Whether words should be lowercased.
* **replace_digits** - Whether all digits should be replaced by zeros.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **singletons_prob** - The probability with which words that occur only once are replaced with OOV during training.
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
* **max_train_sent_length** - Discard sentences in the training set that are longer than this.
* **vocab_include_devtest** - Whether the loaded vocabulary includes words also from the dev and test set. Since the word embeddings for these words are not updated during training, this is equivalent to preloading embeddings at test time as needed. This seems common practice for many sequence labeling toolkits, so I've included it as well. 
* **vocab_only_embedded** - Whether to only include words in the vocabulary if they have pre-trained embeddings.
* **initializer** - Method for random initialization
* **opt_strategy** - Optimization methods, e.g. adam, adadelta, sgd.
* **learningrate** - Learning rate
* **clip** - Gradient clip limit
* **batch_equal_size** - NB: DEPRECATED! No longer implemented! Whether to construct batches from sentences of equal length.
* **max_batch_size** - Maximum batch size.
* **aux_training_probability** - Decides sampling probability for aux task during training. Will be normalized such that main task and aux task sum to 1 (main is already set to 1, so setting aux to 1 as well will weigh them equally). Constant if set to float between 0 and 1. If set to 'decrease', aux task batches will be used for training with p=(1/epoch)
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Stop if there has been no improvement for this many epochs.
* **learningrate_decay** - Learning rate decay when performance hasn't improved.
* **dropout_input** - Apply dropout to word representations.
* **dropout_word_lstm** - Apply dropout after the LSTMs.
* **tf_per_process_gpu_memory_fraction** - Set 'tf_per_process_gpu_memory_fraction' for TensorFlow.
* **tf_allow_growth** - Set 'allow_growth' for TensorFlow
* **lmcost_max_vocab_size** - Maximum vocabulary size for the language modeling objective.
* **lmcost_hidden_layer_size** - Hidden layer size for LMCost.
* **lmcost_lstm_gamma** - LMCost weight
* **lmcost_joint_lstm_gamma** - Joint LMCost weight
* **lmcost_char_gamma** - Char-level LMCost weight
* **lmcost_joint_char_gamma** - Joint char-level LMCost weight
* **char_integration_method** - Method for combining character-based representations with word embeddings.
* **save** - Path for saving the model.
* **garbage_collection** - Whether to force garbage collection.
* **lstm_use_peepholes** - Whether LSTMs use the peephole architecture.
* **whidden_layer_size** - Hidden layer size after the word-level LSTMs.
* **attention_evidence_size** - Layer size for predicting attention weights.
* **attention_activation** - Type of activation to apply for attention weights.
* **attention_objective_weight** - The weight for pushing the attention weights to a binary classification range.
* **sentence_objective_weight** - Sentence-level objective weight.
* **sentence_objective_persistent** - Whether the sentence-level objective should always be given to the network.
* **word_objective_weight** - Word-level classification objective weight.
* **sentence_composition** - The method for sentence composition.
* **random_seed** - Random seed.
