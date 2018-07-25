import time
import collections
import numpy
import math

class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.cost_sum = 0.0
        self.sentence_count = 0.0
        # self.sentence_correct_binary = 0.0
        self.sentence_predicted = 0.0
        self.sentence_correct = 0.0
        self.sentence_total = 0.0

        # self.token_ap_sum = []
        # self.token_predicted = []
        # self.token_gold = []
        # self.token_total = []

        self.start_time = time.time()
        self.sentences = []
        self.true_labels_sent = []
        self.true_values_tokens = []
        self.predictions_tokens = []
        self.scores_tokens = []
        self.sent2token_predictions = []
        self.sent2token_true = []
        self.predictions_sents = []
        self.attention_scores = []


    def calculate_ap(self, true_labels, predicted_scores):
        assert(len(true_labels) == len(predicted_scores))
        indices = numpy.argsort(numpy.array(predicted_scores))[::-1]
        summed, correct, total = 0.0, 0.0, 0.0
        for index in indices:
            total += 1.0
            if true_labels[index] >= 0.5:
                correct += 1.0
                summed += correct / total
        return (summed / correct) if correct > 0.0 else 0.0

    # def append_token_data_for_sentence(self, index, true_values, token_scores):
    #     print(index, true_values, token_scores)
    #     if len(self.token_ap_sum) <= index:
    #         self.token_ap_sum.append(0.0)
    #         self.token_predicted.append(0.0)
    #         self.token_gold.append(0.0)
    #         self.token_total.append(0)
    #
    #     for i in range(len(true_values)):
    #         self.token_total[index] += 1
    #         self.token_predicted[index] = token_scores[i]
    #         self.token_gold[index] += true_values[i]

    def append_data(self, cost, batch, sentence_scores, token_scores_list,
                    attention_scores):
        # print(attention_scores_list.shape, token_scores_list[0].shape)
        self.cost_sum += cost

        token_scores_list = token_scores_list[0]

        for i in range(len(batch)):
            sent, sent_label = batch[i]
            sent_label = int(sent_label not in [self.config["default_label"],
                                                self.config["ignore_label"]])

            self.sentence_count += 1.0
            # print(sent, sent_label)

            # print(true_labels, [sent[j][-1] for j in range(len(sent))])
            # count_interesting_labels = numpy.array(true_labels_tokens).sum()

            prediction_sent = int(sentence_scores[i] >= 0.5)
            # if (count_interesting_labels == 0.0 and prediction_sent == 0) or \
            #         (count_interesting_labels > 0.0 and prediction_sent == 1):
            #     self.sentence_correct_binary += 1.0
            if prediction_sent == 1:
                self.sentence_predicted += 1.0
            if sent_label == 1:
                self.sentence_total += 1.0
            if sent_label == 1 and prediction_sent:
                self.sentence_correct += 1.0

            true_values_tokens = []
            predictions_tokens = []
            # attention_scores_filtered = []
            for t, token in enumerate(sent):
                if token[-1] not in [self.config["default_label"],
                                     self.config["ignore_label"]]:
                    true_values_tokens.append(float(token[-1]))
                    predictions_tokens.append(token_scores_list[i][t])

                # attention_scores_filtered.append(attention_scores[i][t])


            # true_values_tokens = [
            #     float(sent[j][-1]) if sent[j][-1] not in [
            #         self.config["default_label"], self.config["ignore_label"]]
            #     else 0.0 for j in range(len(sent))]
            #
            # predictions_tokens = [token_scores_list[k][i]
            #                       for k in range(len(token_scores_list))][0]

            # predictions_tokens = [int(token_score >= 0.5) for token_score in
            #                       scores_tokens]

            self.sentences.append(sent)
            self.true_labels_sent.append(sent_label)
            self.predictions_sents.append(prediction_sent)
            self.sent2token_predictions.append(token_scores_list[i][:len(sent)])
            self.sent2token_true.append(true_values_tokens)
            self.true_values_tokens.extend(true_values_tokens)
            self.predictions_tokens.extend(predictions_tokens)

            self.attention_scores.append(attention_scores[i][:len(sent)])

    # def write_predictions(self, path):
    #     assert len(self.sentences) == \
    #            len(self.true_labels_sent) == \
    #            len(self.predictions_sents) == \
    #            len(self.predictions_tokens)
    #     outfile = open(path+"/output.txt", "w")
    #     for sent, sent_label, sent_pred, token_preds in \
    #             zip(self.sentences, self.true_labels_sent,
    #                 self.predictions_sents, self.predictions_tokens):
    #         outfile.write("{}\t{}\n".format(sent_label, sent_pred))
    #         for tok, tok_pred in zip(sent, token_preds):
    #             outfile.write("{}\t{}\t{}\n".format(
    #                 tok[0], tok[1], tok_pred))
    #         outfile.write("\n")
    #     outfile.close()


    # wihtout writing token predictions
    def write_predictions(self, path):
        assert len(self.sentences) == \
               len(self.true_labels_sent) == \
               len(self.predictions_sents) == \
               len(self.sent2token_predictions) == \
               len(self.attention_scores), print(len(self.sentences), len(self.true_labels_sent), len(self.predictions_sents), len(self.sent2token_predictions), len(self.attention_scores))

        outfile = open(path + "/output.txt", "w")
        for sent, sent_label, sent_pred, token_pred, att_scores in \
                zip(self.sentences, self.true_labels_sent,
                    self.predictions_sents, self.sent2token_predictions,
                    self.attention_scores):
            outfile.write("{}\t{}\n".format(sent_label, sent_pred))
            for tok, token_pred, att_score in zip(sent, token_pred, att_scores):
                outfile.write("{}\t{}\t{}\t{}\n".format(
                    tok[0], tok[1], token_pred, att_score))
            outfile.write("\n")
        outfile.close()

    def get_results(self, name):
        p = (float(self.sentence_correct) / float(self.sentence_predicted)) if (self.sentence_predicted > 0.0) else 0.0
        r = (float(self.sentence_correct) / float(self.sentence_total)) if (self.sentence_total > 0.0) else 0.0
        f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
        f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (((0.5*0.5 * p) + r) > 0.0) else 0.0

        results = collections.OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        if self.sentence_count > 0:
            results[name + "_cost_avg"] = self.cost_sum / float(self.sentence_count)
            results[name + "_sent_count"] = self.sentence_count
            results[name + "_sent_predicted"] = self.sentence_predicted
            results[name + "_sent_correct"] = self.sentence_correct
            results[name + "_sent_total"] = self.sentence_total
            results[name + "_sent_p"] = p
            results[name + "_sent_r"] = r
            results[name + "_sent_f"] = f
            results[name + "_sent_f05"] = f05
            # results[name + "_sent_correct_binary"] = self.sentence_correct_binary
            # results[name + "_sent_accuracy_binary"] = \
            #     self.sentence_correct_binary / float(self.sentence_count)
        else:
            results[name + "_cost_avg"] = 'n/a'
            results[name + "_sent_count"] = 0
            results[name + "_sent_predicted"] = 0
            results[name + "_sent_correct"] = 'n/a'
            results[name + "_sent_total"] = 'n/a'
            results[name + "_sent_p"] = 'n/a'
            results[name + "_sent_r"] = 'n/a'
            results[name + "_sent_f"] = 'n/a'
            results[name + "_sent_f05"] = 'n/a'
            results[name + "_sent_correct_binary"] = 'n/a'
            results[name + "_sent_accuracy_binary"] = 'n/a'

        # print(self.token_predicted, self.token_total)
        squared_errors = []
        # TODO check assertion below, sth wrong here...
        # print(len(self.predictions_tokens), len(self.true_values_tokens))
        assert len(self.predictions_tokens) == len(self.true_values_tokens), print(len(self.predictions_tokens), len(self.true_values_tokens))
        for t in range(len(self.true_values_tokens)):
            squared_errors.append((self.predictions_tokens[t] -
                                   self.true_values_tokens[t])
                                  **2)
        if len(squared_errors) > 0:
            # print(squared_errors)
            # print(self.true_values_tokens)
            # print(self.predictions_tokens[k])
            token_rmse = math.sqrt(sum(squared_errors)/len(squared_errors))
            results[name + "_tok_rmse"] = token_rmse

        results[name + "_time"] = float(time.time()) - float(self.start_time)

        return results



