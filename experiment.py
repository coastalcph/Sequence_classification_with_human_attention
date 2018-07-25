import sys
import collections
import numpy
import random
import math
import os
import gc
from tqdm import tqdm
import numpy as np
import numbers

try:
    import ConfigParser as configparser
except ImportError:
    import configparser

from model import MLTModel
from evaluator import MLTEvaluator


def read_input_files(file_paths, max_sentence_length=-1):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    """
    # JB: I've changed this so a sentence-level label can be read at the
    # beginning of the sentence (single label on extra line preceding sentence)
    # output is a list of (sentence, label) tuples, where sentence is a list of
    # (token, token_label) tuples
    sentences = []
    line_length = None
    sent_label = ""
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) == 1:
                    sent_label = line
                elif len(line) > 1:
                    line_parts = line.split()
                    assert(len(line_parts) >= 2), line
                    assert(len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append((sentence, sent_label))
                    sentence = []
                    sent_label = ""
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append((sentence, sent_label))
    return sentences


def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def parse_data_config(data_config_path):
    cfg = configparser.ConfigParser()
    cfg.read(data_config_path)
    return cfg


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def weighted_choice(dist):
    dist = dist / dist.sum()
    dart = random.uniform(0, 1)
    for i in range(len(dist)):
        if dart < dist[:i+1].sum():
            return i


def create_batches_of_sentence_ids(data, config, is_training=False, epoch=0):
    """
    Groups together sentences into batches
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    # print(data)
    max_batch_size = config["max_batch_size"]
    tasks = config["tasks"].strip().split(':')
    task_sent_id_batches = []
    if is_training:
        task_sampling_distribution = np.ones(len(tasks))
        if epoch and config['aux_training_probability'] == 'decreasing':
            for i in range(1, len(tasks)):
                task_sampling_distribution[i] = 1 / epoch
        elif isinstance(config['aux_training_probability'], numbers.Number):
            aux_prob = float(config['aux_training_probability'])
            if 0 <= aux_prob <= 1:
                for i in range(1, len(tasks)):
                    task_sampling_distribution[i] = \
                        config['aux_training_probability']
            else:
                print("Parameter config['aux_training_probability'] needs"
                      "to be 'decreasing' or float between 0 and 1.")

        for _batch in range(config['batches_in_epoch']):
            task_id = weighted_choice(task_sampling_distribution)
            task = tasks[task_id]
            task_sents = data[task]
            batch_size = min(max_batch_size, len(task_sents))
            current_batch = random.sample(range(len(task_sents)), batch_size)
            task_sent_id_batches.append((task, current_batch))
    else:
        for task, sentences in data.items():
            current_batch = []
            max_sentence_length = 0
            for i in range(len(sentences)):
                current_batch.append(i)
                if len(sentences[i]) > max_sentence_length:
                    max_sentence_length = len(sentences[i])
                if (0 < max_batch_size <= len(current_batch)) or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
                    task_sent_id_batches.append((task, current_batch))
                    current_batch = []
                    max_sentence_length = 0
            if len(current_batch) > 0:
                task_sent_id_batches.append((task, current_batch))
    return task_sent_id_batches


def process_sentences(data, model, is_training, learningrate, config, name,
                      epoch=0, write_out=""):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluators = {task: MLTEvaluator(config) for task in data.keys()}
    task_sent_id_batches = create_batches_of_sentence_ids(
        data, config, is_training, epoch)
    if is_training:
        for task, sentence_ids_in_batch in task_sent_id_batches:
            random.shuffle(sentence_ids_in_batch)

    for task, sentence_ids_in_batch in tqdm(task_sent_id_batches):
        batch = [data[task][i] for i in sentence_ids_in_batch]
        cost, sentence_scores, token_scores_list, attention_scores_list = \
            model.process_batch(task, batch, is_training, learningrate)

        evaluators[task].append_data(cost, batch, sentence_scores, token_scores_list, attention_scores_list)

        while config["garbage_collection"] and gc.collect() > 0:
            pass

    results = {}
    for task in data.keys():
        print("\n=== TASK: {} ===".format(task))
        results[task] = evaluators[task].get_results(name)
        for key, res in results[task].items():
            print(task + " " + key + ": " + str(res))

        if write_out:
            write_out_task = write_out+"_"+task
            if not os.path.exists(write_out_task):
                os.mkdir(write_out_task)
            evaluators[task].write_predictions(write_out_task)

    return results


def run(config, data_config):

    # temp_model_base = config.get("save") if config.get("save") else config.get("load")
    temp_model_base = "."
    temp_model_path = temp_model_base+"/tmp.model"
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    tasks = config["tasks"].strip().split(":")
    main_task = tasks[0]
    data_train, data_dev, data_test = {}, {}, {}
    for task in tasks:
        data_train[task] = read_input_files(data_config[task]['tr'],
                                            config["max_train_sent_length"])
        if 'dv' in data_config[task]:
            data_dev[task] = read_input_files(data_config[task]['dv'])
        if 'te' in data_config[task]:
            data_test[task] = read_input_files(data_config[task]['te'])

    model = None
    if config.get("load"):
        print("Loading model from "+config["load"])
        model = MLTModel.load(config["load"])

    else:
        model = MLTModel(config)
        model.build_vocabs(data_train, data_dev, data_test,
                           config["preload_vectors"])


        model.construct_network()
        model.initialize_session()
        if config["preload_vectors"]:
            model.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(model.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " +
          str(model.get_parameter_count_without_word_embeddings()))

    if data_train and config.get("do_train"):
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        for epoch in range(1, config["epochs"]+1):
            # random.shuffle(data_train)  # TODO make curriculum here
            print("EPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))

            process_sentences(data_train, model, is_training=True, epoch=epoch,
                              learningrate=learningrate, config=config,
                              name="train")

            if data_dev:
                results_dev_main_task = process_sentences(
                    data_dev, model, is_training=False, learningrate=0.0,
                    config=config, name="dev")[main_task]

                if math.isnan(results_dev_main_task["dev_cost_sum"]) or \
                        math.isinf(results_dev_main_task["dev_cost_sum"]):
                    raise ValueError("Cost is NaN or Inf. Exiting.")

                print(results_dev_main_task[model_selector], best_selector_value)

                if (epoch == 1 or (model_selector_type == "high" and results_dev_main_task[model_selector] > best_selector_value)
                               or (model_selector_type == "low" and results_dev_main_task[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev_main_task[model_selector]
                    model.saver.save(model.session, temp_model_path,
                                     latest_filename=os.path.basename(
                                         temp_model_path)+".checkpoint")
                print("best_epoch: " + str(best_epoch))

                if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
                    break

                if (epoch - best_epoch) > 3:
                    learningrate *= config["learningrate_decay"]

            while config["garbage_collection"] == True and gc.collect() > 0:
                pass

        if data_dev and best_epoch >= 0:
            # loading the best model so far
            model.saver.restore(model.session, temp_model_path)
            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")
            predictions_dev = process_sentences(
                data_dev, model, is_training=False, learningrate=0.0,
                config=config, name="dev",
                write_out=config["save"] + "_predictions_dv")

    if config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"]+".model")

    if data_dev:
        predictions_dev = process_sentences(
            data_dev, model, is_training=False, learningrate=0.0,
            config=config, name="dev",
            write_out=config["save"] + "_predictions_dv")

    if data_test:
        predictions_test = process_sentences(
            data_test, model, is_training=False, learningrate=0.0,
            config=config, name="test",
            write_out=config["save"]+"_predictions_te")


if __name__ == "__main__":
    config = parse_config("config", sys.argv[1])
    data_config = parse_data_config(sys.argv[2])
    run(config, data_config)

