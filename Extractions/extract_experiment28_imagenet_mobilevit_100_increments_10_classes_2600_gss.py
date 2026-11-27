"""
This module allows extracting data from the experiments run by
`experiment28_imagenet_mobilevit_100_increments_10_classes_2600_gss.py` and
`experiment28_imagenet_mobilevit_100_increments_10_classes_2600_gss_2.py`
"""

from typing import Dict, List, Tuple, Union, Any
import dill
import pickle
import math
import torch
import numpy as np
from scipy.special import softmax

EXP_ID = 28
"""
Experiment ID
"""

N_EXPS = 3
"""
Number of experiences actually performed in the experiments.
"""

N_EXPS_TOTAL = 100
"""
Number of experiences that could be run theoretically
"""

N_CLASSES_PER_EXP = 10
"""
Number of classes per experience in the experiments
"""

ACTUAL_N_CLASS_ORDERS = 2
"""
Actual number of class orders in the experiments 
"""

N_SEEDS = 1
"""
Number of seeds in the experiments per class order
"""

TRAIN_EPOCHS = 100
"""
Number of training epochs in the experiments
"""

N_TRAIN_SAMPLES_PER_CLASS = 1300
"""
Number of training samples per class
"""

N_EXEMPLARS = int(0.2 * N_TRAIN_SAMPLES_PER_CLASS * N_CLASSES_PER_EXP)
"""
Number of exemplars in the rehearsal buffer after the first experience
"""

FIRST_CHECKPOINT_PATH = (f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}" +
                   "_classes_{index_class_order}_{n_exemplars}_experience_0.pth")
"""
Path of the checkpoints of the experiments for the first experience
"""

CHECKPOINT_PATH = (f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}" +
                   "_classes_{index_class_order}_{n_exemplars}_gss_experience_{index_exp}.pth")
"""
Path of the checkpoints of the experiments from the second experience onwards
"""

SAVE_PATH = f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}_classes_{N_EXEMPLARS}_gss_data.pkl"
"""
Path where to save all the extracted data
"""

INDEX_INTERVENTION_SCHEDULER = 3
"""
Index of the Last-Layer Gradient Interventions Scheduler Plugin
"""

INDEX_EVAL_PLUGIN = 4
"""
Index of the evaluation plugin
"""

accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [(np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1))),
                                                                 np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1))))
                                                                for j in range(N_EXPS)]
                                                            for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
of the current experience for the ith seed. 
"""

forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [(np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)),
                                                                   np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)))
                                                                  for j in range(1, N_EXPS)]
                                                              for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
training forgetting of class j at the end of the current experience relative to its original training accuracy
for the ith seed. The ith, jth element in the second numpy array is the test forgetting of class j
at the end of the current experience relative to its original test accuracy for the ith seed. 
"""

normalized_forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [(np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)),
                                                                              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)))
                                                                             for j in range(1, N_EXPS)]
                                                                         for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
normalized training forgetting of class j at the end of the current experience relative to its original training
accuracy for the ith seed. The ith, jth element in the second numpy array is the normalized test forgetting of class j
at the end of the current experience relative to its original test accuracy for the ith seed. 
"""

conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = \
    {i: {j: (
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64)
            )
         for j in range(N_EXPS)}
     for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective 4-tuples as values. All elements in the 4-tuples are 3D numpy
arrays of size NxCxC, where N is equal to the number of seeds and C is equal to the number of unique classes encountered
up to the current experience. Specifically, all elements in the 4-tuples contain N confusion matrices, one for each
seed. While the confusion matrices in the first and second element of the 4-tuples are computed prior to perform
training on the given experience, the confusion matrices in the third and fourth element of the 4-tuples are computed
after performing training on the given experience. The confusion matrices in the first and third elements are training
confusion matrices. The confusion matrices in the second and fourth elements are test confusion matrices.
"""

softmax_prob: Dict[int, Dict[int, np.ndarray]] = \
    {i: {j: np.zeros((N_SEEDS, TRAIN_EPOCHS, N_CLASSES_PER_EXP*(j+1)))
         for j in range(1, N_EXPS)}
     for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective 3D numpy arrays as values.
The first experience, the experience with index 0, is skipped. Each 3D numpy array is of size NxExC,
where N is equal to the number of seeds, E is equal to the number of training epochs and C is equal to the number of
unique classes encountered up to the current experience. The ith, eth, jth element in each array is the
average softmax probability for class j computed across all training samples in the current experience
(only the new classes) for the ith seed and prior to start training if eth is 0, after the eth training epoch otherwise.
The last training epoch is skipped.  
"""

logits: Dict[int, Dict[int, np.ndarray]] = \
    {i: {j: np.zeros((N_SEEDS, TRAIN_EPOCHS, N_CLASSES_PER_EXP*(j+1)))
         for j in range(1, N_EXPS)}
     for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
index 0, is skipped. Each array is a 3D numpy array of size NxExC, where N is equal to the number of seeds, E is equal
to the number of training epochs, and C is equal to the number of unique classes encountered up to the current
experience. The ith, eth, jth element in each array is the average logit for class j computed across all training
samples in the current experience (only the new classes) for the ith seed and prior to start training if eth is 0,
after the eth training epoch otherwise. The last training epoch is skipped.  
"""

collected_data: Dict[int, Dict[int, List[Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]]]] = \
    {i: {j: []
         for j in range(1, N_EXPS)}
     for i in range(ACTUAL_N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective lists as values. The first experience, the experience with
index 0, is skipped.  Each list contains as many elements as the number of training epochs used for that experience.
Each element is a dictionary collected by the class `LLIFCDataCollection` (have a look at that class to see the content
of each dictionary) (the `logits` key and its corresponding value are removed from each dictionary to save space)
at the end of each epoch during that experience and prior to performing training on that experience.
The last epoch is skipped.
"""


def insert_accuracy(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    train_acc, test_acc = accuracy[index_class_order][index_exp]
    for t in range(train_acc.shape[1]):
        exp = int(math.floor(t/N_CLASSES_PER_EXP))
        train_str = f'Class_Accuracy_Incremental_Classifier_Exp/eval_phase/train_stream/Task000/Exp00{exp}/{t}'
        test_str = f'Class_Accuracy_Incremental_Classifier_Exp/eval_phase/test_stream/Task000/Exp00{exp}/{t}'
        # insert the train accuracy
        train_acc[index_seed, t] = metric_results[train_str][1][-1]
        # insert the test accuracy
        test_acc[index_seed, t] = metric_results[test_str][1][-1]


def fill_forgetting_normalized_forgetting_from_accuracy():
    for index_class_order, accuracies in accuracy.items():
        for n0, (train_acc_0, test_acc_0) in enumerate(accuracies):
            start_index = N_CLASSES_PER_EXP * n0
            end_index = N_CLASSES_PER_EXP * (n0 + 1)
            for n1, (train_acc_1, test_acc_1) in enumerate(accuracies[n0+1:]):
                n1 = n1 + n0 + 1
                train_forgetting, test_forgetting = forgetting[index_class_order][n1 - 1]
                norm_train_forgetting, norm_test_forgetting = normalized_forgetting[index_class_order][n1 - 1]
                train_diff = train_acc_0[:, start_index: end_index] - train_acc_1[:, start_index: end_index]
                test_diff = test_acc_0[:, start_index: end_index] - test_acc_1[:, start_index: end_index]
                train_forgetting[:, start_index: end_index] = train_diff
                test_forgetting[:, start_index: end_index] = test_diff
                norm_train_forgetting[:, start_index: end_index] = train_diff /  train_acc_0[:, start_index: end_index]
                norm_test_forgetting[:, start_index: end_index] = test_diff / test_acc_0[:, start_index: end_index]


def insert_conf_matrix(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    train_conf_matrix_prior, test_conf_matrix_prior, train_conf_matrix_after, test_conf_matrix_after = (
        conf_matrix)[index_class_order][index_exp]
    train_conf_matrix_prior, test_conf_matrix_prior, train_conf_matrix_after, test_conf_matrix_after = (
        train_conf_matrix_prior[index_seed], test_conf_matrix_prior[index_seed], train_conf_matrix_after[index_seed],
        test_conf_matrix_after[index_seed])
    for i in range(index_exp + 1):
        start_index = N_CLASSES_PER_EXP * i
        end_index = N_CLASSES_PER_EXP * (i+1)
        train_str = f'Confusion_Matrix_Incremental_Classifier_Exp/eval_phase/train_stream/Task000/Exp00{i}'
        test_str = f'Confusion_Matrix_Incremental_Classifier_Exp/eval_phase/test_stream/Task000/Exp00{i}'
        train_conf_matrix_prior[start_index: end_index] = [e for e in metric_results[train_str][1]
                                                           if len(e) == N_CLASSES_PER_EXP * (index_exp + 1)][0][start_index: end_index]
        test_conf_matrix_prior[start_index: end_index] = [e for e in metric_results[test_str][1]
                                                          if len(e) == N_CLASSES_PER_EXP * (index_exp + 1)][0][start_index: end_index]
        train_conf_matrix_after[start_index: end_index] = metric_results[train_str][1][-1][start_index: end_index]
        test_conf_matrix_after[start_index: end_index] = metric_results[test_str][1][-1][start_index: end_index]


def insert_logits_softmax_prob(all_collected_data: List[Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]],
                               index_class_order: int, index_seed: int, index_exp: int):
    if index_exp > 0:
        train_logits = logits[index_class_order][index_exp]
        train_softmax_prob = softmax_prob[index_class_order][index_exp]
        d_list = [d for d in all_collected_data
                  if d["logits"].shape[1] == N_CLASSES_PER_EXP * (index_exp + 1)]
        if len(d_list) != TRAIN_EPOCHS:
            raise RuntimeError("The number of elements does not match the number of training epochs")
        for n, d in enumerate(d_list):
            new_classes_mask = d["orig_targets"] >= N_CLASSES_PER_EXP * index_exp
            new_classes_logits = d["logits"][new_classes_mask]
            train_logits[index_seed, n] = np.mean(new_classes_logits, axis=0)
            train_softmax_prob[index_seed, n] = np.mean(softmax(new_classes_logits, axis=1), axis=0)


def insert_collected_data(all_collected_data: List[Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]],
                          index_class_order: int, index_exp: int):
    if index_exp > 0:
        d_list = [{k: v for k, v in d.items() if k != "logits"}
                  for d in all_collected_data
                  if d["logits"].shape[1] == N_CLASSES_PER_EXP * (index_exp + 1)]
        if len(d_list) != TRAIN_EPOCHS:
            raise RuntimeError("The number of elements does not match the number of training epochs")

        collected_data[index_class_order][index_exp].extend(d_list)

if __name__ == "__main__":
    for index_class_order in range(ACTUAL_N_CLASS_ORDERS):
        for index_seed in range(N_SEEDS):
            for index_exp in range(N_EXPS):
                # load checkpoint
                if index_exp == 0:
                    ckp = torch.load(FIRST_CHECKPOINT_PATH.format(index_class_order=index_class_order,
                                                                  n_exemplars=N_EXEMPLARS), pickle_module=dill,
                                     map_location="cpu")
                else:
                    ckp = torch.load(CHECKPOINT_PATH.format(index_class_order=index_class_order, n_exemplars=N_EXEMPLARS,
                                                            index_exp=index_exp), pickle_module=dill, map_location="cpu")
                # extract strategy
                strategy = ckp["strategy"]

                # get the dictionary containing all the metric results
                all_metric_results = strategy.plugins[INDEX_EVAL_PLUGIN].all_metric_results
                # get all the data collected by the last-layer gradient interventions scheduler plugin
                all_collected_data = strategy.plugins[INDEX_INTERVENTION_SCHEDULER].data

                # insert the accuracies
                insert_accuracy(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the confusion matrices
                insert_conf_matrix(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the logits and softmax probabilities
                insert_logits_softmax_prob(all_collected_data, index_class_order, index_seed, index_exp)
                # insert the collected data
                insert_collected_data(all_collected_data, index_class_order, index_exp)


    # fill `forgetting` and `normalized_forgetting` using the accuracies in `accuracy`
    fill_forgetting_normalized_forgetting_from_accuracy()
    # create dictionary encapsulating all the extracted data
    data = dict(accuracy=accuracy, forgetting=forgetting,
                normalized_forgetting=normalized_forgetting, conf_matrix=conf_matrix, softmax_prob=softmax_prob,
                logits=logits, collected_data=collected_data)
    # save dictionary encapsulating all the extracted data
    with open(SAVE_PATH, 'wb') as file:
        pickle.dump(data, file)