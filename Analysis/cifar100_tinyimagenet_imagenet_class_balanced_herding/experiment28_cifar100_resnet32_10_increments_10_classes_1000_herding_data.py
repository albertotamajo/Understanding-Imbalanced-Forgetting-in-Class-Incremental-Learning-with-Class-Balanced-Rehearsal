from typing import Dict, List, Tuple, Union
import pickle
import numpy as np

EXP_ID = 28
"""
Experiments ID
"""

DATASET = "cifar100"
"""
Name of the dataset
"""

NETWORK = "resnet32"
"""
Name of the network architecture
"""

N_EXPS = 3
"""
Number of experiences actually performed in the experiments.
"""

N_EXPS_TOTAL = 10
"""
Number of experiences that could be run theoretically
"""

N_CLASSES_PER_EXP = 10
"""
Number of classes per experience
"""

N_TRAIN_SAMPLES_PER_CLASS = 500
"""
Number of training samples per class
"""

N_REHEARSAL = 1000
"""
Total number of samples in the rehearsal buffer after the first incremental step
"""

N_CLASS_ORDERS = 40
"""
Number of class orders in the experiments
"""

ACTUAL_N_CLASS_ORDERS = 5
"""
The actual number of class orders
"""

N_SEEDS = 1
"""
Number of seeds
"""

N_EPOCHS = 170
"""
Number of epochs
"""

MB_SIZE_SECOND_EXP = 128
"""
Mini-batch size after the first experience
"""

DATA_PATH = f"experiment{EXP_ID}_{DATASET}_{NETWORK}_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}_classes_{N_REHEARSAL}_herding_data.pkl"
"""
Path to the extracted data
"""

with open(DATA_PATH, "rb") as file:
    data = pickle.load(file)


accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = data["accuracy"]
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
of the current experience for the ith seed. 
"""

forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = data["forgetting"]
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
training forgetting of class j at the end of the current experience relative to its original training accuracy
for the ith seed. The ith, jth element in the second numpy array is the test forgetting of class j
at the end of the current experience relative to its original test accuracy for the ith seed. 
"""

normalized_forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = data["normalized_forgetting"]
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
normalized training forgetting of class j at the end of the current experience relative to its original training
accuracy for the ith seed. The ith, jth element in the second numpy array is the normalized test forgetting of class j
at the end of the current experience relative to its original test accuracy for the ith seed. 
"""

conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = data["conf_matrix"]
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

softmax_prob: Dict[int, Dict[int, np.ndarray]] = data["softmax_prob"]

logits: Dict[int, Dict[int, np.ndarray]] = data["logits"]

collected_data: Dict[int, Dict[int, List[Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]]]] = data["collected_data"]
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective lists as values. The first experience, the experience with
index 0, is skipped.  Each list contains as many elements as the number of training epochs used for that experience.
Each element is a dictionary collected by the class `LLIFCDataCollection` (have a look at that class to see the content
of each dictionary) (the `logits` key and its corresponding value are removed from each dictionary to save space)
at the end of each epoch during that experience and prior to performing training on that experience.
The last epoch is skipped.
"""