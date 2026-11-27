import pickle
from typing import Dict, Tuple, Protocol, List, Sequence, Literal, Union, Iterable, Any, Optional
from numbers import Number
from itertools import chain
import importlib
import os

from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from statsmodels.formula.api import ols
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


##################################Stylize seaborn############################
# Set seaborn-white style
sns.set_style("white")

# NeurIPS style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7
})

# Colorblind palette
colors = sns.color_palette("colorblind")
##################################Stylize seaborn############################


##################################Lists of keys for modules############################
N_REHEARSAL_PER_CLASS = [40, 100, 200]
"""
Number of rehearsal samples per class stored at the end of each incremental step
"""

CIFAR100_10_CLS_1 = [("cifar100", "resnet32", 10, 10, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with cifar100, resnet32, 10 increments
of 10 classes each.
"""

CIFAR100_10_CLS_2 = [("cifar100", "resnet32", 10, 10, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with cifar100, resnet32, 10 increments
of 10 classes each.
"""

CIFAR100_5_CLS_1 = [("cifar100", "resnet32", 5, 20, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with cifar100, resnet32, 5 increments
of 20 classes each.
"""

CIFAR100_5_CLS_2 = [("cifar100", "resnet32", 5, 20, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with cifar100, resnet32, 5 increments
of 20 classes each.
"""

CIFAR100_2_CLS_1 = [("cifar100", "resnet32", 2, 50, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with cifar100, resnet32, 2 increments
of 50 classes each.
"""

TINYIN_10_CLS_1 = [("tinyimagenet", "resnet18", 10, 20, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet, resnet18,
10 increments of 20 classes
each.
"""

TINYIN_10_CLS_2 = [("tinyimagenet", "resnet18", 10, 20, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet, resnet18,
10 increments of 20 classes
each.
"""

TINYIN_5_CLS_1 = [("tinyimagenet", "resnet18", 5, 40, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet, resnet18,
5 increments of 40 classes each.
"""

TINYIN_5_CLS_2 = [("tinyimagenet", "resnet18", 5, 40, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet, resnet18,
5 increments of 40 classes each.
"""

TINYIN_2_CLS_1 = [("tinyimagenet", "resnet18", 2, 100, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet, resnet18,
2 increments of 100 classes each.
"""

ALL_CIFAR_1 = CIFAR100_10_CLS_1 + CIFAR100_5_CLS_1 + CIFAR100_2_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with cifar100.
"""

ALL_CIFAR_10_5_1 = CIFAR100_10_CLS_1 + CIFAR100_5_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with cifar100 but only for
those with 10 increments of 10 classes each or 5 increments of 20 classes each.
"""

ALL_CIFAR_2 = CIFAR100_10_CLS_2 + CIFAR100_5_CLS_2
"""
All the keys for the modules containing the third experience of experiments run with cifar100.
"""

ALL_TINYIN_1 = TINYIN_10_CLS_1 + TINYIN_5_CLS_1 + TINYIN_2_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet.
"""

ALL_TINYIN_10_5_1 = TINYIN_10_CLS_1 + TINYIN_5_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet but only for
those with 10 increments of 20 classes each or 5 increments of 40 classes each.
"""

ALL_TINYIN_2 = TINYIN_10_CLS_2 + TINYIN_5_CLS_2
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet.
"""

BENCHMARKS = [("ALL_CIFAR_1", ALL_CIFAR_1), ("ALL_TINYIN_1", ALL_TINYIN_1),
              ("ALL_CIFAR_2", ALL_CIFAR_2), ("ALL_TINYIN_2", ALL_TINYIN_2),
              ("ALL_CIFAR_10_5_1", ALL_CIFAR_10_5_1), ("ALL_TINYIN_10_5_1", ALL_TINYIN_10_5_1)]
"""
"""
##################################Lists of keys for modules############################


##################################Classes############################
class ExperimentModule(Protocol):
    """
    A protocol for a module containing data about experiments.
    """

    EXP_ID: int
    """
    Experiments ID
    """

    DATASET: str
    """
    Name of the dataset
    """

    NETWORK: str
    """
    Name of the network architecture
    """

    N_EXPS: int
    """
    Number of experiences actually performed in the experiments.
    """

    N_EXPS_TOTAL: int
    """
    Number of experiences that could be run theoretically
    """

    N_CLASSES_PER_EXP: int
    """
    Number of classes per experience
    """

    N_REHEARSAL: int
    """
    Total number of samples in the rehearsal buffer after the first experience
    """

    N_CLASS_ORDERS: int
    """
    Number of class orders in the experiments
    """

    N_SEEDS: int
    """
    Number of seeds
    """

    N_EPOCHS: int
    """
    Number of epochs
    """

    MB_SIZE_SECOND_EXP: int
    """
    Mini-batch size during the second experience
    """

    N_TRAINING_EXAMPLES_PER_CLASS: int
    """
    Number of training examples per class
    """

    N_THEORY_MEASURES: int
    """
    Number of measures derived from the theoretical analysis
    """

    DATA_PATH: str
    """
    Path to the extracted data
    """

    accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
    where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
    experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
    experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
    of the current experience for the ith seed. 
    """

    forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
    where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
    number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
    training forgetting of class j at the end of the current experience relative to its training accuracy at the end of the
    previous experience for the ith seed. The ith, jth element in the second numpy array is the test forgetting of class j
    at the end of the current experience relative to its test accuracy at the end of the previous experience for the ith
    seed. 
    """

    exemplar_indices: Dict[int, List[np.ndarray]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences. Each element is a 2D numpy array of size NxC, where N is equal to the number of
    seeds and C is equal to the number of exemplars in the rehearsal buffer after each experience. The ith row in each numpy
    array contains the indices of all the samples selected for rehearsal at the end of the current experience for the ith
    seed.
    """

    conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
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

    softmax_prob: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective 2-tuples as values. The first experience, the experience with
    index 0, is skipped. Both elements in the 2-tuples are 2D numpy arrays of size NxC, where N is equal to the number of
    seeds and C is equal to the number of unique classes encountered up to the current experience.
    The ith, jth element in the first array is the average softmax probability for class j computed across all training
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    The ith, jth element in the second array is the average softmax probability for class j computed across all test
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    """

    logits: Dict[int, Dict[int, np.ndarray]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
    index 0, is skipped. Each array is a 2D numpy array of size NxC, where N is equal to the number of seeds and C is equal
    to the number of unique classes encountered up to the current experience.
    The ith, jth element in each array is the average softmax probability for class j computed across all training
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    """

    theory_measures: Dict[int, np.ndarray]
    """
    a dictionary containing the class order indexes as keys and respective arrays as values. Each array is a 4D numpy array
    of size TxExNxC, where T is equal to the number of theoretical measures derived from the theoretical analysis (11),
    E is equal to the number of training epochs for the second/third experience +1---this is because it
    contains one element computed prior to perform training on the second/third experience---N is equal to the number
    of seeds, C is equal to the number of old classes during the second/third experience.
        - The first theoretical measure is (f_c @ b_c) / ||f_c||
        - The second theoretical measure is (f_c @ b_c^{others}) / ||f_c||
        - The third theoretical measure is (f_c @ f^{bias}) / ||f_c||
        - The fourth theoretical measure is (f_c @ f^{bias-except-self}) / ||f_c||
        - The fifth theoretical measure is (f_c @ f^{no-bias}) / ||f_c||
        - The sixth theoretical measure is (f_c(c-row) @ b_c^{others}(c-row)) / ||f_c(c-row)||
        - The seventh theoretical measure is (f_c(c-row) @ f^{bias}(c-row)) / ||f_c(c-row)||
        - The eighth theoretical measure is (f_c(c-row) @ f^{bias-except-self}(c-row)) / ||f_c(c-row)||
        - The ninth theoretical measure is (f_c(c-row) @ f^{no-bias}(c-row)) / ||f_c(c-row)||
        - The tenth theoretical measure is ||f_c||
        - The eleventh theoretical measure is ||f_c(c-row)||

    where
        - f_c is the average gradient of class c computed using its full training dataset
        - b_c = r_c - f_c, where r_c is the average gradient of class c computed using its rehearsal set
        - b_c^{others} is the sum of the bias components of the other old classes (all old classes except for c)
        - f^{bias} is the sum of the average gradients of all old classes computed using their full training dataset (it 
          includes class c)
        - f^{bias-except-self} is the sum of the average gradients of the other old classes (all old classes except for c)
          computed using their full training dataset
        - f^{no-bias} is the sum of the average gradients of the new classes computes using their full training dataset
        - f_c(c-row) is the average gradient of class c computed using its full training dataset wrt only its class-specific
          parameters (final-layer row of class c + its bias)
        - b_c^{others}(c-row) is the sum of the bias components of the other old classes (all old classes except for c) wrt
          only the class-specific parameters of c (final-layer row of class c + its bias)
        - f^{bias}(c-row) is the sum of the average gradients of all old classes computed using their full training dataset
          (it includes class c) wrt only the class-specific parameters of c (final-layer row of class c + its bias)
        - f^{bias-except-self}(c-row) is the sum of the average gradients of the other old classes (all old classes except
          for c) computed using their full training dataset wrt only the class-specific parameters of c (final-layer row of
          class c + its bias)
        - f^{no-bias}(c-row) is the sum of the average gradients of the new classes computes using their full training
          dataset wrt only the class-specific parameters of c (final-layer row of class c + its bias)
    """
##################################Classes############################

##################################Collect the experiment modules############################


modules_dict: Dict[Tuple[str, str, int, int, int, int], ExperimentModule] = {}
"""
A dictionary containing 6-tuples as keys and respective modules as values.
The first element in each  6-tuple is the name of the dataset used by the respective module;
the second element is the name of the network architecture;
the third element is the number of total experiences;
the fourth element is the number of classes per experience;
the fifth element is the total number of rehearsal samples stored at the end of each incremental step per class
the sixth element is the experience index
"""
# fill `modules_dict` with the experiment modules present in this folder
for name in os.listdir():
    if name.endswith('.py'):
        name = name.removesuffix(".py")
        splits = name.split('_')
        if len(splits) in [9, 10]:
            key = (splits[1], splits[2], int(splits[3]), int(splits[5]), int(splits[7]) // int(splits[5]))
            if len(splits) == 9:
                key = key + (1,)
            else:
                key = key + (2,)
            module: ExperimentModule = importlib.import_module(name)
            modules_dict[key] = module
##################################Collect the experiment modules############################


##################################Utilities############################
def collect_measures(label_measures: Sequence[str],
                     module: ExperimentModule,
                     n_epochs: Optional[int] = None,
                     apply_only: Optional[Sequence[str]] = None,
                     single_epoch: bool = False,
                     start_epoch: int = 0) -> np.ndarray:
    """
    Collect the given measures from the given module.
    :param label_measures: a sequence of measures that need to be collected
    :param module: a module that contains the given measures
    :param n_epochs: number of epochs to include for computation of the measures. It does not apply to `CF`, `UCF` and
        `LOG`. When None, all epochs are used. Default is None.
    :param: apply_only: if `n_epochs` is not None, it is only applied to the measures included here. If None, `n_epochs`
        is applied to all measures. Default is None.
    :return: a 3D numpy array of size NxCxM, where N is the number of training runs used in the module, C is the number
        of classes used in the module and M is the number of measures collected. Note that N=OxS, where O is the number
        of class orders used in the module and S is the number of seeds used per class order in the module.
    """
    if len(label_measures) == 0:
        raise ValueError("`label_measures cannot be empty`")
    if len(set(label_measures)) != len(label_measures):
        raise ValueError("`label_measures cannot contain duplicates`")

    if apply_only is None:
        apply_only = [x for x in label_measures]

    measures_list: List[np.ndarray] = []
    # proportion1
    prop1 = -(((module.N_REHEARSAL / module.N_CLASSES_PER_EXP) / module.N_REHEARSAL) / 2)
    if module.N_EXPS == 3:  # if it is a module for the third experience
        prop1 = prop1 / 2
    # proportion2
    prop2 = -((module.N_TRAINING_EXAMPLES_PER_CLASS / (module.N_TRAINING_EXAMPLES_PER_CLASS * module.N_CLASSES_PER_EXP))/2)
    # labels, with same order, for the 11 theoretical measures
    theory_measures_names = ["SBI", "CBI", "PCI", "PCI_EXS", "NDI", "CBI-S", "PCI-S", "PCI_EXS-S", "NDI-S", "OM"]
    for class_order in range(module.N_CLASS_ORDERS):
        for seed in range(module.N_SEEDS):
            measures = []
            # append the theoretical measures in label_measures
            for l in label_measures:
                if l == "CF":
                    class_forgetting = (module.forgetting[class_order][-1][1][seed]).copy()
                    accuracy_first_exp = module.accuracy[class_order][0][1][seed]
                    class_forgetting[:module.N_CLASSES_PER_EXP] = (class_forgetting[:module.N_CLASSES_PER_EXP] /
                                                                   accuracy_first_exp)
                    if module.N_EXPS == 3:
                        accuracy_second_exp = module.accuracy[class_order][1][1][seed]
                        class_forgetting[module.N_CLASSES_PER_EXP:] = (class_forgetting[module.N_CLASSES_PER_EXP:] /
                                                                       accuracy_second_exp[module.N_CLASSES_PER_EXP:])
                    measure = class_forgetting
                elif l == "UCF":
                    measure = module.forgetting[class_order][-1][1][seed]
                elif l == "LOG":
                    exp_id = module.N_EXPS
                    logits = (module.logits[class_order][exp_id -1][seed]).copy()
                    measure = logits[:module.N_CLASSES_PER_EXP*(exp_id -1)]
                elif l == "R-SBI":
                    if n_epochs is None:
                        measure_SBI = module.theory_measures[class_order][0, :, seed, :].sum(axis=0)
                        measure_grad_norm = module.theory_measures[class_order][9, :, seed, :].sum(axis=0)
                    else:
                        if l in apply_only:
                            measure_SBI = (module.theory_measures[class_order][0, n_epochs: n_epochs+1, seed, :]
                                           if single_epoch else module.theory_measures[class_order][0, start_epoch:n_epochs, seed, :]).sum(axis=0)
                            measure_grad_norm = (module.theory_measures[class_order][9, n_epochs: n_epochs+1, seed, :]
                                                 if single_epoch else module.theory_measures[class_order][9, start_epoch:n_epochs, seed, :]).sum(axis=0)
                        else:
                            measure_SBI = module.theory_measures[class_order][0, :, seed, :].sum(axis=0)
                            measure_grad_norm = module.theory_measures[class_order][9, :, seed, :].sum(axis=0)

                    measure = (measure_SBI / measure_grad_norm) * prop1
                elif l == "RP":
                    measure_SBI = module.theory_measures[class_order][0, :, seed, :]
                    measure_grad_norm = module.theory_measures[class_order][9, :, seed, :]
                    if n_epochs is not None and l in apply_only:
                        measure_SBI = measure_SBI[n_epochs: n_epochs+1, :] if single_epoch else measure_SBI[start_epoch:n_epochs, :]
                        measure_grad_norm = measure_grad_norm[n_epochs: n_epochs+1, :] if single_epoch else measure_grad_norm[start_epoch:n_epochs, :]

                    measure = (measure_SBI + measure_grad_norm).sum(axis=0)
                elif l in theory_measures_names:
                    index = theory_measures_names.index(l)
                    if n_epochs is None:
                        measure = module.theory_measures[class_order][index, :, seed, :].sum(axis=0)
                    else:
                        if l in apply_only:
                            measure = (module.theory_measures[class_order][index, n_epochs: n_epochs+1, seed, :]
                                       if single_epoch else module.theory_measures[class_order][index, start_epoch:n_epochs, seed, :]).sum(axis=0)
                        else:
                            measure = module.theory_measures[class_order][index, :, seed, :].sum(axis=0)

                    if l in ["NDI", "NDI-S"]:
                        measure = measure * prop2
                    else:
                        measure = measure * prop1
                else:
                    raise ValueError("One of the provided label measures is not one of the theoretical measures")
                measures.append(measure)

            measures = np.column_stack(measures)
            measures_list.append(measures)

    measures_list = np.stack(measures_list, axis=0)
    return measures_list

def collect_raw_measures(label_measure: str,
                         module: ExperimentModule) -> np.ndarray:
    """

    :param label_measures:
    :param module:
    :return: a 3D numpy array of size NxExC, where N is the number of training runs used in the module, E is the number
        of classes used in the module and M is the number of measures collected. Note that N=OxS, where O is the number
        of class orders used in the module and S is the number of seeds used per class order in the module.
    """
    theory_measures_names = ["SBI", "CBI", "PCI", "PCI_EXS", "NDI", "CBI-S", "PCI-S", "PCI_EXS-S", "NDI-S"]
    if label_measure not in theory_measures_names:
        raise ValueError("The provided label measure is not one of the theoretical measures")
    index = theory_measures_names.index(label_measure)

    measures = []
    for class_order in range(module.N_CLASS_ORDERS):
        for seed in range(module.N_SEEDS):
            measures.append(module.theory_measures[class_order][index, :, seed, :])
    return np.stack(measures, axis=0)


def swarm_plot_forgetting(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                          n_exps: int,
                          name_fig: str,
                          palette: str = "tab10",
                          show: bool = True,
                          title: Optional[str] = None,
                          col_titles: Optional[Sequence[str]] = None,
                          row_titles: Optional[Sequence[str]] = None,
                          **kwargs):
    """
    Draw swarm plots for visualizing the distribution of forgetting across past classes in the experiments in the
    given modules.
    :param module_keys: a sequence of module keys
    :param n_exps: number of experiments to select for each module
    :param name_fig: name to use for saving the figure
    :param palette: the palette to be used for the plots
    :param show: whether to show the plot
    :param title: title for the overall figure
    :param col_titles: titles for the columns of the figure. If not None, the length of this sequence must match the
        number of columns in the plot
    :param row_titles: titles for the rows of the figure. If not None, the length of this sequence must match the number
        of rows in the plot
    :param kwargs: some keyword arguments to be passed to `plt.subplots`
    """
    forg_list = [collect_measures(["CF"], modules_dict[key]).squeeze() for key in module_keys]
    fig, axs = plt.subplots(**kwargs)
    n_cols = axs.shape[1]
    if title is not None:
        # Add an overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")
        # Add horizontal lines (in figure coordinates: 0 → left, 1 → right)
        line_top = Line2D([0.1, 0.9], [0.985, 0.985], transform=fig.transFigure, color='black', linewidth=1)
        line_bottom = Line2D([0.1, 0.9], [0.943, 0.943], transform=fig.transFigure, color='black', linewidth=1)
        fig.add_artist(line_top)
        fig.add_artist(line_bottom)
    axs = axs.flatten()
    if col_titles is not None:
        # Add column headers
        for i, col_label in enumerate(col_titles):
            # Get the center x position of each column
            col_center = axs[i].get_position().x0 + axs[i].get_position().width / 2
            fig.text(col_center, 0.90, col_label, ha='center', va='bottom', fontsize=14, fontweight='bold')

    if row_titles is not None:
        # Add row labels
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i * len(col_titles)].get_position().y0 + axs[i * len(col_titles)].get_position().height / 2
            fig.text(0.06, row_center, row_label, ha='right', va='center', rotation=90, fontsize=14, fontweight='bold')
        # Add transparent row labels on the right side to center plots
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i * len(col_titles)].get_position().y0 + axs[i * len(col_titles)].get_position().height / 2
            fig.text(0.99, row_center, "R", ha='right', va='center', rotation=90, fontsize=14, fontweight='bold',
                     alpha=0)

    for n, forg in enumerate(forg_list):
        forg = forg[:n_exps, :].T
        sns.swarmplot(data=forg, palette=palette, ax=axs[n])
        axs[n].set_xlim(-0.5, forg.shape[1] - 0.5)  # perfectly tight to first/last strip
        axs[n].set_ylim(-0.15, 1.05)
        axs[n].set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        axs[n].set_xticks([])
        if n % n_cols == 0:
            axs[n].set_ylabel("FG", fontsize=12, labelpad=-0.1)
        axs[n].grid(axis='y', color='lightgray', linestyle='--', linewidth=0.3, alpha=0.9)
        # Shade alternating background areas
        for i in range(forg.shape[1]):
            if i % 2 != 0:  # shade every other category
                axs[n].axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.2, zorder=0)
        for spine in axs[n].spines.values():
            spine.set_color('gray')
    plt.savefig(f"swarm_plot_forgetting_{name_fig}", bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()


def line_plots(dfs: Sequence[pd.DataFrame],
               x: str,
               y: str,
               estimator: str,
               errorbar: str,
               name_fig: str,
               maxtick: int,
               xlabel: str,
               hue: Optional[str] = None,
               show: bool = True,
               title: Optional[str] = None,
               col_titles: Optional[Sequence[str]] = None,
               row_titles: Optional[Sequence[str]] = None,
               y_ticks: Sequence[Number] = (-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
               y_min: Number = -0.35,
               err_style = "band",
               err_kws = None,
               **kwargs):
    """
    Draw swarm plots for visualizing the distribution of forgetting across past classes in the experiments in the
    given modules.
    :param module_keys: a sequence of module keys
    :param n_exps: number of experiments to select for each module
    :param name_fig: name to use for saving the figure
    :param palette: the palette to be used for the plots
    :param show: whether to show the plot
    :param title: title for the overall figure
    :param col_titles: titles for the columns of the figure. If not None, the length of this sequence must match the
        number of columns in the plot
    :param row_titles: titles for the rows of the figure. If not None, the length of this sequence must match the number
        of rows in the plot
    :param kwargs: some keyword arguments to be passed to `plt.subplots`
    """
    fig, axs = plt.subplots(**kwargs)
    fig.supxlabel(xlabel, fontsize=12, y=0.05)
    if title is not None:
        # Add an overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")
        # Add horizontal lines (in figure coordinates: 0 → left, 1 → right)
        line_top = Line2D([0.1, 0.9], [0.985, 0.985], transform=fig.transFigure, color='black', linewidth=1)
        line_bottom = Line2D([0.1, 0.9], [0.943, 0.943], transform=fig.transFigure, color='black', linewidth=1)
        fig.add_artist(line_top)
        fig.add_artist(line_bottom)
    axs = axs.flatten()
    if col_titles is not None:
        # Add column headers
        for i, col_label in enumerate(col_titles):
            # Get the center x position of each column
            col_center = axs[i].get_position().x0 + axs[i].get_position().width / 2
            fig.text(col_center, 0.90, col_label, ha='center', va='bottom', fontsize=14, fontweight='bold')

    if row_titles is not None:
        # Add row labels
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i * len(col_titles)].get_position().y0 + axs[i * len(col_titles)].get_position().height / 2
            fig.text(0.06, row_center, row_label, ha='right', va='center', rotation=90, fontsize=14)
        # Add transparent row labels on the right side to center plots
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i * len(col_titles)].get_position().y0 + axs[i * len(col_titles)].get_position().height / 2
            fig.text(0.99, row_center, "R", ha='right', va='center', rotation=90, fontsize=14, alpha=0)

    for n, df in enumerate(dfs):
        sns.lineplot(data=df, x=x, y=y, estimator=estimator, errorbar=errorbar, err_style=err_style,
                     err_kws = {} if err_kws is None else err_kws,
                     hue=hue, ax=axs[n])
        axs[n].set_ylim(y_min, 1.)
        axs[n].set_yticks(y_ticks)
        axs[n].set_ylabel("")  # removes y label
        axs[n].set_xlabel("")  # removes x label
        axs[n].xaxis.set_major_formatter(mtick.PercentFormatter(maxtick))
        axs[n].set_xticks(np.linspace(0, maxtick, 5))  # exactly 5 ticks
        if n != 0:
            axs[n].legend_.remove()
        else:
            axs[n].legend(title=None)  # removes legend title
        axs[n].grid(axis='y', color='lightgray', linestyle='--', linewidth=0.3, alpha=0.9)
        for spine in axs[n].spines.values():
            spine.set_color('gray')
    plt.savefig(f"line_plot_{name_fig}", bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()


def forgetting_metric(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                      metric_name: Literal["r", "hsg"],
                      method: str = "BCa",
                      confidence: float = 0.95,
                      n_resamples: int = 1_000_000,
                      random_seed: int = 123,
                      forgetting_name: Literal["CF", "UCF"] = "CF",
                      **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute the mean and std of the given metric across all the experiments in the given modules. The CIs are
    computed as well for both the mean and std. The CI for the mean is computed using the t-distribution, whereas the CI
    for the std is computed using bootstrap. The metrics supported are the forgetting range (`r`) and the half-split
    forgetting gap (`hsg`), which can either be computed  using `CF` or `UCF`.
    """
    forg_list = [collect_measures([forgetting_name], modules_dict[key]).squeeze() for key in module_keys]
    #forg_list = [np.maximum(forg, 0) for forg in forg_list]
    if metric_name == "r":
        metric = np.concatenate([forg.max(axis=1) - forg.min(axis=1) for forg in forg_list])
    elif metric_name == "hsg":
        forg_list = [np.sort(forg, axis=1) for forg in forg_list]
        bottom50 = np.concatenate([forg[:, :(forg.shape[1] // 2)].mean(axis=1) for forg in forg_list])
        top50 = np.concatenate([forg[:, (forg.shape[1] // 2):].mean(axis=1) for forg in forg_list])
        metric = top50 - bottom50
    else:
        raise ValueError("`metric_name` can only be `r` or `hsg`")

    mean = metric.mean()
    std = metric.std()
    sem = stats.sem(metric)  # Standard error for the mean
    # confidence% CI using t-distribution for the mean
    ci_mean = stats.t.interval(confidence, df=len(metric) - 1, loc=mean, scale=sem)
    ci_std = stats.bootstrap(data=(metric,), statistic=np.std, method=method, confidence_level=confidence,
                             n_resamples=n_resamples, random_state=random_seed, **kwargs)
    ci_std = (ci_std.confidence_interval.low, ci_std.confidence_interval.high)
    return pd.DataFrame({'mean': [f"{mean:.3f} ({ci_mean[0]:.3f}, {ci_mean[1]:.3f})"],
                         'std': [f"{std:.3f} ({ci_std[0]:.3f}, {ci_std[1]:.3f})"]}), metric


def correlation(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                label_measures: Sequence[str],
                corr_op: Union[Literal["pearson"], Literal["spearman"]],
                corr_type: Union[Literal["raw"], Literal["partial"]],
                non_included_covars: Optional[Dict[str, Iterable[str]]] = None,
                method: str = "BCa",
                confidence: float = 0.95,
                n_resamples: int = 1_000_000,
                random_seed: int = 123,
                n_epochs: Optional[int] = None,
                apply_only: Optional[Sequence[str]] = None,
                single_epoch: bool = False,
                start_epoch: int = 0,
                **kwargs
                ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute the mean and std of the correlation between each pair of measures across all experiments in the given
    modules. The CIs are computed as well for both the mean and std. The CIs for the means are computed using the
    t-distribution, whereas the CIs for the stds are computed using bootstrap.
    :param module_keys: a sequence of module keys
    :param label_measures: a sequence of measures that need to be considered
    :param corr_op: the operator for the correlation. Either `pearson` or `spearman`
    :param corr_type: the type of the correlation. Either `raw` or `partial`
    :param non_included_covars: a dictionary containing measures as keys and respective iterables of measures as
        values. When the partial correlation between a measure X and a measure Y needs to be computed, if X or Y or both
        are keys in this dictionary, the partial correlation will not include as covariates the respective measures
        included in their iterables. It is used only when `corr_type` is `partial`. When `corr_type` is not `partial`,
        a :class:`ValueError` will be raised. Default is None.
    :param method: the bootstrapping method for computing the confidence intervals of the std
    :param confidence: confidence level for bootstrapping
    :param n_resamples: number of resamples used by bootstrap
    :param random_seed: seed used for bootstrapping
    :param n_epochs: number of epochs to include for computation of the measures. It does not apply to `CF`. When None,
        all epochs are used. Default is None.
    :param: apply_only: if `n_epochs` is not None, it is only applied to the measures included here. If None, `n_epochs`
        is applied to all measures. Default is None.
    :param kwargs: some keyword arguments to be passed to `stats.bootstrap`
    :return: a tuple with two elements. The first element in the tuple is a Dataframe of size MxM, where M is equal to
        the number of measures considered. Each cell contains the mean correlation along with its CI and the std of the
        correlation along with its CI. The second element in the tuple is a 3d array of size NxMxM, where N is the total
        number of experiments across all the modules and M is the number of measures considered. In practice, the second
        element contains N correlation matrices, one for each experiment.
    """
    corr_ops = ["pearson", "spearman"]
    corr_types = ["raw", "partial"]
    if corr_op not in corr_ops:
        raise ValueError("`corr_op` must be either `pearson` or `spearman`")
    if corr_type not in corr_types:
        raise ValueError("`corr_type` must be either `raw` or `partial`")
    if non_included_covars is not None and corr_type == corr_types[0]:
        raise ValueError("`non_included_covars` must be None when `corr_type` is `raw`")

    # convert label measures into a list
    label_measures = list(label_measures)
    if non_included_covars is not None:
        # convert the dictionary so that it has indexes as keys and values rather than the measure labels
        non_included_covars = {label_measures.index(key): [label_measures.index(val) for val in values]
                               for key, values in non_included_covars.items()}

    def compute_raw_corr(arr, corr_op):
        if corr_op == corr_ops[0]:
            return np.corrcoef(arr, rowvar=False)
        else:
            return spearmanr(arr)[0]

    def compute_partial_corr(arr, corr_op):
        C, M = arr.shape
        # where C is the n. of classes and M is the n. of measures
        labels = [f"X{m}" for m in range(M)]
        arr = pd.DataFrame(arr, columns=labels)
        corr_matrix = np.ones((M, M), dtype=np.float64)
        for x in range(M):
            for y in range(M):
                if x != y:
                    label_x = f"X{x}"
                    label_y = f"X{y}"
                    covar = list(set(labels) - set([label_x, label_y]))
                    if non_included_covars is not None:
                        init_set = set()
                        if x in non_included_covars.keys():
                            init_set.update(set([f"X{c}" for c in non_included_covars[x]]))
                        if y in non_included_covars.keys():
                            init_set.update(set([f"X{c}" for c in non_included_covars[y]]))
                        covar = list(set(covar) - init_set)

                    corr_matrix[x, y] = pg.partial_corr(data=arr, x=label_x, y=label_y, covar=covar,
                                                        method=corr_op)["r"].to_numpy()
        return corr_matrix

    measures_list = [collect_measures(label_measures, modules_dict[key], n_epochs=n_epochs, apply_only=apply_only,
                                      single_epoch=single_epoch, start_epoch=start_epoch)
                     for key in module_keys]
    corr_matrices = []
    for measures in measures_list:
        size = len(measures)
        for i in range(size):
            if corr_type == corr_types[0]:
                corr_matrices.append(compute_raw_corr(measures[i], corr_op))
            else:
                corr_matrices.append(compute_partial_corr(measures[i], corr_op))

    corr_matrices = np.stack(corr_matrices, axis=0)
    # transform the correlations into fisher-z values
    fisher_z_corr_matrices = fisher_z_transform(corr_matrices)
    mean_fisher_z_corr_matrix = fisher_z_corr_matrices.mean(axis=0)
    std_corr_matrix = corr_matrices.std(axis=0)
    sem = stats.sem(fisher_z_corr_matrices, axis=0)  # Standard error for the means

    # confidence% CI using t-distribution for the means
    ci_mean = stats.t.interval(confidence, df=len(fisher_z_corr_matrices) - 1, loc=mean_fisher_z_corr_matrix,
                               scale=sem)
    ci_mean_low = inverse_fisher_z_transform(ci_mean[0])
    ci_mean_high = inverse_fisher_z_transform(ci_mean[1])
    mean_corr_matrix = inverse_fisher_z_transform(mean_fisher_z_corr_matrix)

    ci_std_low = np.zeros((len(label_measures), len(label_measures)), dtype=np.float64)
    ci_std_high = np.zeros((len(label_measures), len(label_measures)), dtype=np.float64)

    for i in range(len(label_measures)):
        for j in range(len(label_measures)):
            ci_std = stats.bootstrap(data=(corr_matrices[:, i, j],), statistic=np.std, method=method,
                                     confidence_level=confidence, n_resamples=n_resamples, random_state=random_seed,
                                     **kwargs)
            ci_std = (ci_std.confidence_interval.low, ci_std.confidence_interval.high)
            ci_std_low[i, j] = ci_std[0]
            ci_std_high[i, j] = ci_std[1]

    output = np.empty((len(label_measures), len(label_measures)), dtype=object)
    for i in range(len(label_measures)):
        for j in range(len(label_measures)):
            output[i, j] = (f"{mean_corr_matrix[i, j]:.3f} ({ci_mean_low[i, j]:.3f}, {ci_mean_high[i, j]:.3f}) +- "
                            f"{std_corr_matrix[i, j]:.3f} ({ci_std_low[i, j]:.3f} {ci_std_high[i, j]:.3f})")

    return pd.DataFrame(output, columns=label_measures, index=label_measures), corr_matrices


def grid_plot(data: np.ndarray,
              axes_level_sns_func,
              save_name_fig: Optional[str] = None,
              title: Optional[str] = None,
              col_titles: Optional[Sequence[str]] = None,
              row_titles: Optional[Sequence[str]] = None,
              show: bool = True,
              xlim: Optional[Tuple[float, float]] = None,
              ylim: Optional[Tuple[float, float]] = None,
              xticks: Optional[Sequence[float]] = None,
              yticks: Optional[Sequence[float]] = None,
              ylabel: Optional[str] = None,
              index_y_label: Optional[Tuple[int, int]] = None,
              subplots_kwargs: Optional[Dict[str, Any]] = None,
              sns_kwargs: Optional[Dict[str, Any]] = None
              ):
    """
    Draw a grid of plots.
    :param data: a 3D array of size NxMxV, where N is the number of columns in the grid of plots, M is the number of
        rows in the grid of plots and V is just the number of values.
    :param axes_level_sns_func: an axes-level seaborn function that can draw on a matplotlib axes.
    :param save_name_fig: name to use for saving the figure.
    :param title: overall title of the figure.
    :param col_titles: titles for the columns of the figure. If not None, the length of this sequence must match the
        number of columns in the plot.
    :param row_titles: titles for the rows of the figure. If not None, the length of this sequence must match the
        number of rows in the plot.
    :param show: whether to show the plot.
    :param xlim: limits for the x-axis of the plots.
    :param ylim: limits for the y-axis of the plots.
    :param xticks: ticks for the x-axis of the plots.
    :param yticks: ticks for the y-axis of the plots.
    :param ylabel: label for the y-axis.
    :param index_y_label: tuple of indexes for the single subplot to be used for displaying the y-axis label.
    :param subplots_kwargs: some keyword arguments to pass to `plt.subplots`
    :param sns_kwargs: some keyword arguments to pass to the axes-level seaborn function
    """
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D numpy array")
    if subplots_kwargs is None:
        subplots_kwargs = {}
    if sns_kwargs is None:
        sns_kwargs = {}
    N, M, _ = data.shape
    if col_titles is not None and len(col_titles) != N:
        raise ValueError("`col_titles` must match the number of columns in the plot")
    if row_titles is not None and len(row_titles) != M:
        raise ValueError("`row_titles` must match the number of rows in the plot")

    fig, axs = plt.subplots(nrows=M, ncols=N, **subplots_kwargs)
    if title is not None:
        # Add an overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")

    if col_titles is not None:
        # Add column headers
        for i, col_label in enumerate(col_titles):
            # Get the center x position of each column
            col_center = axs[0, i].get_position().x0 + axs[0, i].get_position().width / 2
            fig.text(col_center, 0.90, col_label, ha='center', va='bottom', fontsize=14, fontweight='bold')

    if row_titles is not None:
        # Add row labels
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i, 0].get_position().y0 + axs[i, 0].get_position().height / 2
            fig.text(0.06, row_center, row_label, ha='right', va='center', rotation=90, fontsize=14, fontweight='bold')
        # Add transparent row labels on the right side to center plots
        for i, row_label in enumerate(row_titles):
            # Get the center y position of each row
            row_center = axs[i, 0].get_position().y0 + axs[i, 0].get_position().height / 2
            fig.text(0.99, row_center, "R", ha='right', va='center', rotation=90, fontsize=14, fontweight='bold',
                     alpha=0)

    for n in range(N):
        for m in range(M):
            axes_level_sns_func(data[n, m], ax=axs[m, n], **sns_kwargs)
            if xlim is not None:
                axs[m, n].set_xlim(*xlim)
            if ylim is not None:
                axs[m, n].set_ylim(*ylim)
            if xticks is not None:
                axs[m, n].set_xticks(xticks)
            if yticks is not None:
                axs[m, n].set_yticks(yticks)
            if index_y_label is None:
                axs[m, n].set_ylabel(ylabel, fontsize=12)
            else:
                if m == index_y_label[0] and n == index_y_label[1]:
                    axs[m, n].set_ylabel(ylabel, fontsize=12)
                else:
                    axs[m, n].set_ylabel(None)
            for spine in axs[m, n].spines.values():
                spine.set_color('gray')

    if save_name_fig is not None:
        plt.savefig(save_name_fig, bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()


def standardized_linear_regression(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                   label_measures: Sequence[str],
                                   y: str,
                                   method: str = "BCa",
                                   confidence: float = 0.95,
                                   n_resamples: int = 1_000_000,
                                   random_seed: int = 123,
                                   **kwargs
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the mean and std of the pearson correlation between predicted values and true values, the adjusted R^2
    and the coefficients resulting from performing standardized linear
    regression across all experiments in the given modules. The CIs are computed as well for both the mean and std.
    The CIs for the means are computed using the t-distribution, whereas the CIs for the stds are computed using
    bootstrap.
    :param module_keys: a sequence of module keys
    :param label_measures: a sequence of measures that need to be considered. It must include both the dependent
        variable and the independent ones.
    :param y: the dependent variable
    :param method: the bootstrapping method for computing the confidence intervals of the std
    :param confidence: confidence level for bootstrapping
    :param n_resamples: number of resamples used by bootstrap
    :param random_seed: seed used for bootstrapping
    :param kwargs: some keyword arguments to be passed to `stats.bootstrap`
    :return: a tuple with six elements. The first element is a singleton Dataframe containing the mean and std
        (along with CIs) of the pearson correlation between predicted values and true values. The second element is a
        singleton Dataframe containing the mean and std (along with CIs) of the adjusted R^2. The third element is a
        1xC Dataframe, where C is the number of independent variables. Each cell contains the mean and std
        (along with CIs) of the respective coefficient. The fourth element is a 1D array of size N including the pearson
        correlation between predicted values and true values, where N is the number of experiments across all the
        modules. The fifth element is a 1D array of size N including the adjusted R^2 for each experiment, where N is
        the number of experiments across all the modules. The sixth element contains a 2D array of size NxC, where N is
        the number of experiments across all the modules and the C is the number of independent variables. Each cell
        contains the coefficient estimated for the respective independent variable for the given experiment.
    """
    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key]) for key in module_keys]
    p_corr = []
    adjusted_r2 = []
    coefficients = []
    for measures in measures_list:
        size = len(measures)
        for i in range(size):
            df = pd.DataFrame(measures[i], columns=label_measures)
            df[independent_vars] = StandardScaler().fit_transform(df[independent_vars])
            independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
            model = smf.ols(f"Q('{y}') ~ {independent_vars_form}", data=df).fit()
            p_corr.append(spearmanr(model.fittedvalues, df[y])[0])
            adjusted_r2.append(model.rsquared_adj)
            coefficients.append(model.params.to_numpy()[1:])

    p_corr = np.asarray(p_corr)
    mean_p_corr = fisher_z_transform(p_corr).mean()
    std_p_corr = p_corr.std()
    sem_p_corr = stats.sem(fisher_z_transform(p_corr))  # Standard error for the mean pearson correlation
    # confidence% CI using t-distribution for the mean of adjsuted r2
    ci_mean_p_corr = stats.t.interval(confidence, df=len(p_corr) - 1, loc=mean_p_corr,
                                      scale=sem_p_corr)
    ci_std_p_corr = stats.bootstrap(data=(p_corr,), statistic=np.std, method=method,
                                    confidence_level=confidence, n_resamples=n_resamples,
                                    random_state=random_seed, **kwargs)
    ci_std_p_corr = (ci_std_p_corr.confidence_interval.low, ci_std_p_corr.confidence_interval.high)
    mean_p_corr = inverse_fisher_z_transform(mean_p_corr)
    ci_mean_p_corr = (inverse_fisher_z_transform(ci_mean_p_corr[0]), inverse_fisher_z_transform(ci_mean_p_corr[1]))

    df_p_corr = pd.DataFrame({"P. Corr.": [f"{mean_p_corr:.3f} ({ci_mean_p_corr[0]:.3f}, "
                                          f"{ci_mean_p_corr[1]:.3f}) +- {std_p_corr:.3f} "
                                          f"({ci_std_p_corr[0]:.3f}, {ci_std_p_corr[1]:.3f})"]})

    adjusted_r2 = np.asarray(adjusted_r2)
    mean_adjusted_r2 = adjusted_r2.mean()
    std_adjusted_r2 = adjusted_r2.std()
    sem_adjusted_r2 = stats.sem(adjusted_r2)  # Standard error for the mean adjusted r2
    # confidence% CI using t-distribution for the mean of adjsuted r2
    ci_mean_adjusted_r2 = stats.t.interval(confidence, df=len(adjusted_r2) - 1, loc=mean_adjusted_r2,
                                           scale=sem_adjusted_r2)
    ci_std_adjusted_r2 = stats.bootstrap(data=(adjusted_r2,), statistic=np.std, method=method,
                                         confidence_level=confidence, n_resamples=n_resamples,
                                         random_state=random_seed, **kwargs)
    ci_std_adjusted_r2 = (ci_std_adjusted_r2.confidence_interval.low, ci_std_adjusted_r2.confidence_interval.high)

    df_adjusted_r2 = pd.DataFrame({"Adj. R2": [f"{mean_adjusted_r2:.3f} ({ci_mean_adjusted_r2[0]:.3f}, "
                                               f"{ci_mean_adjusted_r2[1]:.3f}) +- {std_adjusted_r2:.3f} "
                                               f"({ci_std_adjusted_r2[0]:.3f}, {ci_std_adjusted_r2[1]:.3f})"]})

    coefficients = np.stack(coefficients, axis=0)
    mean_coefficients = coefficients.mean(axis=0)
    std_coefficients = coefficients.std(axis=0)
    sem_coefficients = stats.sem(coefficients, axis=0)
    # confidence% CI using t-distribution for the mean of coefficients
    ci_mean_coefficients = stats.t.interval(confidence, df=len(coefficients) - 1, loc=mean_coefficients,
                                            scale=sem_coefficients)
    ci_std_coefficients_low = np.zeros((len(independent_vars),), dtype=np.float64)
    ci_std_coefficients_high = np.zeros((len(independent_vars),), dtype=np.float64)

    for i in range(len(independent_vars)):
        ci_std_coefficients = stats.bootstrap(data=(coefficients[:, i],), statistic=np.std, method=method,
                                              confidence_level=confidence, n_resamples=n_resamples,
                                              random_state=random_seed, **kwargs)
        ci_std_coefficients = (ci_std_coefficients.confidence_interval.low,
                               ci_std_coefficients.confidence_interval.high)
        ci_std_coefficients_low[i] = ci_std_coefficients[0]
        ci_std_coefficients_high[i] = ci_std_coefficients[1]

    coefficients_output = np.empty((1, len(independent_vars)), dtype=object)
    for i in range(len(independent_vars)):
        coefficients_output[0, i] = (f"{mean_coefficients[i]:.3f} ({ci_mean_coefficients[0][i]:.3f}, "
                                     f"{ci_mean_coefficients[1][i]:.3f}) +- {std_coefficients[i]:.3f} "
                                     f"({ci_std_coefficients_low[i]:.3f}, {ci_std_coefficients_high[i]:.3f})")

    df_coefficients = pd.DataFrame(coefficients_output, columns=independent_vars)

    ps = np.sum(coefficients > 0, axis=0)
    psc_output = np.empty((1, len(independent_vars)), dtype=object)
    for i in range(len(independent_vars)):
        lower, upper = proportion_confint(
            count=ps[i],
            nobs=coefficients.shape[0],
            alpha=1 - confidence,
            method='wilson'
        )
        prop = ps[i] / coefficients.shape[0]
        psc_output[0, i] = f"{prop:.3f} ({lower:.3f}, {upper:.3f})"

    df_psc = pd.DataFrame(psc_output, columns=independent_vars)

    return df_p_corr, df_adjusted_r2, df_coefficients, df_psc, p_corr, adjusted_r2, coefficients


def standardized_linear_regression_leave_one_out_hold(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                                      label_measures: Sequence[str],
                                                      y: str,
                                                      method: str = "BCa",
                                                      confidence: float = 0.95,
                                                      n_resamples: int = 1_000_000,
                                                      random_seed: int = 123,
                                                      **kwargs
                                                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the mean and std of the pearson correlation between predicted values and true values, the adjusted R^2
    and the coefficients resulting from performing standardized linear
    regression across all experiments in the given modules. The CIs are computed as well for both the mean and std.
    The CIs for the means are computed using the t-distribution, whereas the CIs for the stds are computed using
    bootstrap.
    :param module_keys: a sequence of module keys
    :param label_measures: a sequence of measures that need to be considered. It must include both the dependent
        variable and the independent ones.
    :param y: the dependent variable
    :param method: the bootstrapping method for computing the confidence intervals of the std
    :param confidence: confidence level for bootstrapping
    :param n_resamples: number of resamples used by bootstrap
    :param random_seed: seed used for bootstrapping
    :param kwargs: some keyword arguments to be passed to `stats.bootstrap`
    :return: a tuple with six elements. The first element is a singleton Dataframe containing the mean and std
        (along with CIs) of the pearson correlation between predicted values and true values. The second element is a
        singleton Dataframe containing the mean and std (along with CIs) of the adjusted R^2. The third element is a
        1xC Dataframe, where C is the number of independent variables. Each cell contains the mean and std
        (along with CIs) of the respective coefficient. The fourth element is a 1D array of size N including the pearson
        correlation between predicted values and true values, where N is the number of experiments across all the
        modules. The fifth element is a 1D array of size N including the adjusted R^2 for each experiment, where N is
        the number of experiments across all the modules. The sixth element contains a 2D array of size NxC, where N is
        the number of experiments across all the modules and the C is the number of independent variables. Each cell
        contains the coefficient estimated for the respective independent variable for the given experiment.
    """
    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key]) for key in module_keys]
    measures = [l[i] for l in measures_list for i in range(len(l))]
    s_corr = []
    r2 = []
    for i in range(len(measures)):
        params = []
        for j, measure in enumerate(measures):
            if j != i:
                std_measures = StandardScaler().fit_transform(measure)
                df = pd.DataFrame(std_measures, columns=label_measures)
                independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
                model = smf.ols(f"Q('{y}') ~ {independent_vars_form}", data=df).fit()
                params.append(model.params.to_numpy())
        avg_params = np.vstack(params).mean(axis=0)
        std_measures_infer = StandardScaler().fit_transform(measures[i])
        df_infer = pd.DataFrame(std_measures_infer, columns=label_measures)
        df_dropped = df_infer.drop(columns=[y])
        vals = df_dropped.to_numpy()
        preds = (vals @ avg_params[1:]) + avg_params[0]
        s_corr.append(spearmanr(preds, df_infer[y])[0])
        r2.append(r2_score(preds, df_infer[y]))

    return



def standardized_linear_regression_leave_one_out(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                                 label_measures: Sequence[str],
                                                 y: str,
                                                 method: str = "BCa",
                                                 confidence: float = 0.95,
                                                 n_resamples: int = 1_000_000,
                                                 random_seed: int = 123,
                                                 **kwargs
                                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key]) for key in module_keys]
    s_corr = []
    mae = []
    for measures in measures_list:
        for i in range(len(measures)):
            train = np.concatenate((measures[:i], measures[i+1:]), axis=0)
            train = np.concatenate(train, axis=0)
            df_train = pd.DataFrame(train, columns=label_measures)
            df_train[y] = np.arctan(df_train[y])
            independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
            model = smf.ols(f"Q('{y}') ~ {independent_vars_form}", data=df_train).fit()

            df_test = pd.DataFrame(measures[i], columns=label_measures)
            preds = np.tanh(model.predict(df_test))
            s_corr.append(spearmanr(preds, df_test[y])[0])
            mae.append(mean_absolute_error(preds, df_test[y]))

    s_corr = np.asarray(s_corr)
    mean_s_corr = fisher_z_transform(s_corr).mean()
    std_s_corr = s_corr.std()
    sem_s_corr = stats.sem(fisher_z_transform(s_corr))  # Standard error for the mean spearman correlation
    # confidence% CI using t-distribution for the mean of s_corr
    ci_mean_s_corr = stats.t.interval(confidence, df=len(s_corr) - 1, loc=mean_s_corr,
                                      scale=sem_s_corr)
    ci_std_s_corr = stats.bootstrap(data=(s_corr,), statistic=np.std, method=method,
                                    confidence_level=confidence, n_resamples=n_resamples,
                                    random_state=random_seed, **kwargs)
    ci_std_s_corr = (ci_std_s_corr.confidence_interval.low, ci_std_s_corr.confidence_interval.high)
    mean_s_corr = inverse_fisher_z_transform(mean_s_corr)
    ci_mean_s_corr = (inverse_fisher_z_transform(ci_mean_s_corr[0]), inverse_fisher_z_transform(ci_mean_s_corr[1]))

    df_s_corr = pd.DataFrame({"S. Corr.": [f"{mean_s_corr:.3f} ({ci_mean_s_corr[0]:.3f}, "
                                           f"{ci_mean_s_corr[1]:.3f}) +- {std_s_corr:.3f} "
                                           f"({ci_std_s_corr[0]:.3f}, {ci_std_s_corr[1]:.3f})"]})

    mae = np.asarray(mae)
    mean_mae_corr = fisher_z_transform(mae).mean()
    std_mae_corr = mae.std()
    sem_mae_corr = stats.sem(fisher_z_transform(mae))  # Standard error for the mean spearman correlation
    # confidence% CI using t-distribution for the mean of s_corr
    ci_mean_mae_corr = stats.t.interval(confidence, df=len(mae) - 1, loc=mean_mae_corr,
                                      scale=sem_mae_corr)
    ci_std_mae_corr = stats.bootstrap(data=(mae,), statistic=np.std, method=method,
                                    confidence_level=confidence, n_resamples=n_resamples,
                                    random_state=random_seed, **kwargs)
    ci_std_mae_corr = (ci_std_mae_corr.confidence_interval.low, ci_std_mae_corr.confidence_interval.high)
    mean_mae_corr = inverse_fisher_z_transform(mean_mae_corr)
    ci_mean_mae_corr = (inverse_fisher_z_transform(ci_mean_mae_corr[0]), inverse_fisher_z_transform(ci_mean_mae_corr[1]))

    df_mae_corr = pd.DataFrame({"MAE.": [f"{mean_mae_corr:.3f} ({ci_mean_mae_corr[0]:.3f}, "
                                           f"{ci_mean_mae_corr[1]:.3f}) +- {std_mae_corr:.3f} "
                                           f"({ci_std_mae_corr[0]:.3f}, {ci_std_mae_corr[1]:.3f})"]})

    return df_s_corr, df_mae_corr


def mixed_effects_linear_regression_leave_one_out(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                                  label_measures: Sequence[str],
                                                  y: str,
                                                  n_epochs: Optional[int] = None,
                                                  ) -> Tuple[pd.DataFrame, np.asarray, np.asarray]:
    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key], n_epochs=n_epochs) for key in module_keys]
    s_corr = []
    mae = []
    for measures in measures_list:
        for i in range(len(measures)):
            train = np.concatenate((measures[:i], measures[i+1:]), axis=0)
            N,M, _ = train.shape
            d = {}
            for j, label in enumerate(label_measures):
                d[label] = train[:, :, j].flatten()
            d["ID"] = np.asarray([[t]*M for t in range(N)]).flatten()
            train_df = pd.DataFrame(d)
            scaler = StandardScaler().fit(train_df[independent_vars])
            train_df[independent_vars] = scaler.transform(train_df[independent_vars])
            independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
            model = smf.mixedlm(
                f"Q('{y}') ~ {independent_vars_form}",
                train_df,
                groups=train_df["ID"],
                re_formula=f"~{independent_vars_form}"
            ).fit()

            test_df = pd.DataFrame(measures[i], columns=label_measures)
            test_df[independent_vars] = scaler.transform(test_df[independent_vars])
            y_pred = model.predict(test_df)
            s_corr.append(spearmanr(y_pred, test_df[y])[0])
            mae.append(mean_absolute_error(y_pred, test_df[y]))

    mean_s_corr = inverse_fisher_z_transform(fisher_z_transform(s_corr).mean())
    std_s_corr = np.std(s_corr)
    mean_mae = np.mean(mae)
    std_mae = np.std(mae)
    return pd.DataFrame({"Spearman": [f"{mean_s_corr:.3f} +- {std_s_corr:.3f}"],
                         "MAE": [f"{mean_mae:.3f} +- {std_mae:.3f}"]}), np.asarray(s_corr), np.asarray(mae)

def mixed_effects_linear_regression_summary(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                            label_measures: Sequence[str],
                                            y: str,
                                            n_epochs: Optional[int] = None,
                                            ) -> Dict[Tuple[str, str, int, int, int, int], Any]:
    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [(key, collect_measures(label_measures, modules_dict[key], n_epochs=n_epochs))
                     for key in module_keys]
    d_summary = {}
    for key, measures in measures_list:
        train = measures
        N,M, _ = train.shape
        d = {}
        for j, label in enumerate(label_measures):
            d[label] = train[:, :, j].flatten()
        d["ID"] = np.asarray([[t]*M for t in range(N)]).flatten()
        train_df = pd.DataFrame(d)
        scaler = StandardScaler().fit(train_df[label_measures])
        train_df[label_measures] = scaler.transform(train_df[label_measures])
        independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
        model = smf.mixedlm(
            f"Q('{y}') ~ {independent_vars_form}",
            train_df,
            groups=train_df["ID"],
            re_formula=f"~{independent_vars_form}"
        ).fit()

        d_summary[key] = model.summary()

    return d_summary



def standardized_linear_regression_leave_one_out_3(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                                   label_measures: Sequence[str],
                                                   y: str,
                                                   method: str = "BCa",
                                                   confidence: float = 0.95,
                                                   n_resamples: int = 1_000_000,
                                                   random_seed: int = 123,
                                                   **kwargs
                                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key]) for key in module_keys]
    s_corr = []
    mae = []

    s_corr2 = []
    mae2 = []
    for measures in measures_list:
        train = measures
        N,M, _ = train.shape
        d = {}
        for i, label in enumerate(label_measures):
            d[label] = train[:, :, i].flatten()
        d["ID"] = np.asarray([[i]*M for i in range(N)]).flatten()
        train_df = pd.DataFrame(d)
        scaler = StandardScaler().fit(train_df[independent_vars])
        train_df[independent_vars] = scaler.transform(train_df[independent_vars])
        independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
        model = smf.mixedlm(
            f"Q('{y}') ~ {independent_vars_form}",
            train_df,
            groups=train_df["ID"],
            re_formula=f"~{independent_vars_form}"
        ).fit()

        for i in range(N):
            train_exp_df = train_df[train_df["ID"] == i]
            y_pred = model.predict(train_exp_df)
            s_corr.append(spearmanr(y_pred, train_exp_df[y])[0])
            mae.append(mean_absolute_error(y_pred, train_exp_df[y]))

            s_corr2.append(spearmanr(model.fittedvalues[train_df["ID"] == i], train_exp_df[y])[0])
            mae2.append(r2_score(model.fittedvalues[train_df["ID"] == i], train_exp_df[y]))

    return

def standardized_linear_regression_leave_one_out_4(module_keys: Sequence[Tuple[str, str, int, int, int, int]],
                                                   label_measures: Sequence[str],
                                                   y: str,
                                                   method: str = "BCa",
                                                   confidence: float = 0.95,
                                                   n_resamples: int = 1_000_000,
                                                   random_seed: int = 123,
                                                   **kwargs
                                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")
    independent_vars = [x for x in label_measures if x != y]

    measures_list = [collect_measures(label_measures, modules_dict[key]) for key in module_keys]
    s_corr = []
    mae = []

    s_corr2 = []
    mae2 = []

    d = {}

    for i, label in enumerate(label_measures):
        d[label] = np.concatenate([measures[:, :, i].flatten() for measures in measures_list])
    tot_n_exps = sum([len(measures) for measures in measures_list])
    id = 0
    ids = []
    for measures in measures_list:
        for m in measures:
            ids.append([id]*len(m))
            id += 1
    d["ID"] = np.concatenate(ids)

    train_df = pd.DataFrame(d)
    scaler = StandardScaler().fit(train_df[independent_vars])
    train_df[independent_vars] = scaler.transform(train_df[independent_vars])
    independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
    model = smf.mixedlm(
        f"Q('{y}') ~ {independent_vars_form}",
        train_df,
        groups=train_df["ID"],
        re_formula=f"~{independent_vars_form}"
    ).fit()

    for i in range(tot_n_exps):
        train_exp_df = train_df[train_df["ID"] == i]
        y_pred = model.predict(train_exp_df)
        s_corr.append(spearmanr(y_pred, train_exp_df[y])[0])
        mae.append(mean_absolute_error(y_pred, train_exp_df[y]))

        s_corr2.append(spearmanr(model.fittedvalues[train_df["ID"] == i], train_exp_df[y])[0])
        mae2.append(r2_score(model.fittedvalues[train_df["ID"] == i], train_exp_df[y]))


    #         train = np.concatenate(train, axis=0)
    #         df_train = pd.DataFrame(train, columns=label_measures)
    #         df_train[y] = np.arctan(df_train[y])
    #         independent_vars_form = "+".join([f"Q('{x}')" for x in independent_vars])
    #         model = smf.ols(f"Q('{y}') ~ {independent_vars_form}", data=df_train).fit()
    #
    #         df_test = pd.DataFrame(measures[i], columns=label_measures)
    #         df_test[independent_vars] = scaler.transform(df_test[independent_vars])
    #         preds = np.tanh(model.predict(df_test))
    #         s_corr.append(spearmanr(preds, df_test[y])[0])
    #         mae.append(mean_absolute_error(preds, df_test[y]))
    #
    # s_corr = np.asarray(s_corr)
    # mean_s_corr = fisher_z_transform(s_corr).mean()
    # std_s_corr = s_corr.std()
    # sem_s_corr = stats.sem(fisher_z_transform(s_corr))  # Standard error for the mean spearman correlation
    # # confidence% CI using t-distribution for the mean of s_corr
    # ci_mean_s_corr = stats.t.interval(confidence, df=len(s_corr) - 1, loc=mean_s_corr,
    #                                   scale=sem_s_corr)
    # ci_std_s_corr = stats.bootstrap(data=(s_corr,), statistic=np.std, method=method,
    #                                 confidence_level=confidence, n_resamples=n_resamples,
    #                                 random_state=random_seed, **kwargs)
    # ci_std_s_corr = (ci_std_s_corr.confidence_interval.low, ci_std_s_corr.confidence_interval.high)
    # mean_s_corr = inverse_fisher_z_transform(mean_s_corr)
    # ci_mean_s_corr = (inverse_fisher_z_transform(ci_mean_s_corr[0]), inverse_fisher_z_transform(ci_mean_s_corr[1]))
    #
    # df_s_corr = pd.DataFrame({"S. Corr.": [f"{mean_s_corr:.3f} ({ci_mean_s_corr[0]:.3f}, "
    #                                        f"{ci_mean_s_corr[1]:.3f}) +- {std_s_corr:.3f} "
    #                                        f"({ci_std_s_corr[0]:.3f}, {ci_std_s_corr[1]:.3f})"]})
    #
    # mae = np.asarray(mae)
    # mean_mae_corr = fisher_z_transform(mae).mean()
    # std_mae_corr = mae.std()
    # sem_mae_corr = stats.sem(fisher_z_transform(mae))  # Standard error for the mean spearman correlation
    # # confidence% CI using t-distribution for the mean of s_corr
    # ci_mean_mae_corr = stats.t.interval(confidence, df=len(mae) - 1, loc=mean_mae_corr,
    #                                   scale=sem_mae_corr)
    # ci_std_mae_corr = stats.bootstrap(data=(mae,), statistic=np.std, method=method,
    #                                 confidence_level=confidence, n_resamples=n_resamples,
    #                                 random_state=random_seed, **kwargs)
    # ci_std_mae_corr = (ci_std_mae_corr.confidence_interval.low, ci_std_mae_corr.confidence_interval.high)
    # mean_mae_corr = inverse_fisher_z_transform(mean_mae_corr)
    # ci_mean_mae_corr = (inverse_fisher_z_transform(ci_mean_mae_corr[0]), inverse_fisher_z_transform(ci_mean_mae_corr[1]))
    #
    # df_mae_corr = pd.DataFrame({"MAE.": [f"{mean_mae_corr:.3f} ({ci_mean_mae_corr[0]:.3f}, "
    #                                        f"{ci_mean_mae_corr[1]:.3f}) +- {std_mae_corr:.3f} "
    #                                        f"({ci_std_mae_corr[0]:.3f}, {ci_std_mae_corr[1]:.3f})"]})
    #
    # return df_s_corr, df_mae_corr


def long_table(key_vals: Dict[Sequence, Sequence[Number]], col_names: Optional[Sequence[str]] = None):
    """
    Construct a long table from a dictionary containing sequences of the same lengths as keys and sequences of numbers
    as values. The long table will contain as many rows as the total number of numbers across all sequences of numbers.
    Each row has N+1 elements, where N is equal to the number of elements in the sequence of keys. The last element is
    the value.
    :param key_vals: a dictionary containing sequences of the same length as keys and sequences of numbers as values
    :param col_names: a sequence of names for the columns in the long table. The first name is assigned to the first
    element in the sequence of keys, the second is assigned to the second element in the sequence of keys, and so on
    until the last name, which is assigned to the values in the sequences of numbers. If None, the names assigned are
    `Col1`, `Col2` and so on. The value is assigned the `Val` name.
    :return: a long table
    """
    if not len(set([len(key) for key in key_vals.keys()])) == 1:
        raise ValueError('The keys in `key_vals` must have the same length')
    key_length = len(list(key_vals.keys())[0])
    if col_names is not None:
        if not len(col_names) == key_length + 1:
            raise ValueError("`col_names` must have the same length of each key in `key_vals` + 1")
    else:
        col_names = [f"Col{i+1}" for i in range(key_length)] + ["Val"]

    dict_df = {}

    for i, col_name in enumerate(col_names[:-1]):
        dict_df[col_name] = list(chain.from_iterable([[key[i]] * len(val) for key, val in key_vals.items()]))

    dict_df[col_names[-1]] = np.concatenate(list(key_vals.values()))

    return pd.DataFrame(dict_df)


def fisher_z_transform(x) -> np.ndarray:
    """
    Apply the fisher z-transform element-wise.
    :param x: an array-like input
    :return: a transformed numpy array
    """
    return np.arctanh(x)


def inverse_fisher_z_transform(x) -> np.ndarray:
    """
    Apply the inverse fisher z-transform element-wise.
    :param x: an array-like input
    :return: a transformed numpy array
    """
    return np.tanh(x)

##################################Utilities############################

##################################Actual data############################

def forgetting_stats():
    # the estimated forgetting range for all cifar-100 second experience exps
    forg_range_cifar_1_stats, forg_range_cifar_1_dist = forgetting_metric(ALL_CIFAR_1, "r")
    # the estimated forgetting range for all tinyimagenet second experience exps
    forg_range_tinyin_1_stats, forg_range_tinyin_1_dist = forgetting_metric(ALL_TINYIN_1, "r")
    # two-sided Welch’s t-test to test for statistical difference between the estimated mean of Forgetting Range for
    # the cifar100-second experience experiments vs the tinyimagenet second experience ones
    t_stat_forg_range_1, p_value_forg_range_1 = stats.ttest_ind(forg_range_cifar_1_dist, forg_range_tinyin_1_dist,
                                                                equal_var=False, random_state=123)
    # the estimated forgetting range for all cifar-100 third experience exps
    forg_range_cifar_2_stats, forg_range_cifar_2_dist = forgetting_metric(ALL_CIFAR_2, "r")
    # the estimated forgetting range for all tinyimagenet third experience exps
    forg_range_tinyin_2_stats, forg_range_tinyin_2_dist = forgetting_metric(ALL_TINYIN_2, "r")
    # the estimated forgetting range for all cifar-100 second experience exps; only 10, 5 increments
    forg_range_cifar_10_5_1_stats, forg_range_cifar_10_5_1_dist = forgetting_metric(ALL_CIFAR_10_5_1, "r")
    # the estimated forgetting range for all tinyimagenet second experience exps; only 10, 5 increments
    forg_range_tinyin_10_5_1_stats, forg_range_tinyin_10_5_1_dist = forgetting_metric(ALL_TINYIN_10_5_1, "r")


    # the estimated half-split forgetting gap for all cifar-100 second experience exps
    forg_hsg_cifar_1_stats, forg_hsg_cifar_1_dist = forgetting_metric(ALL_CIFAR_1, "hsg")
    # the estimated half-split forgetting gap for all tinyimagenet second experience exps
    forg_hsg_tinyin_1_stats, forg_hsg_tinyin_1_dist = forgetting_metric(ALL_TINYIN_1, "hsg")
    # two-sided Welch’s t-test to test for statistical difference between the estimated mean of Half-Split Forgetting
    # Gap for the cifar100 second experience experiments vs the tinyimagenet second experience ones
    t_stat_hsg_1, p_value_hsg_1 = stats.ttest_ind(forg_hsg_cifar_1_dist, forg_hsg_tinyin_1_dist, equal_var=False,
                                                  random_state=123)
    # the estimated half-split forgetting gap for all cifar-100 third experience exps
    forg_hsg_cifar_2_stats, forg_hsg_cifar_2_dist = forgetting_metric(ALL_CIFAR_2, "hsg")
    # the estimated half-split forgetting gap for all tinyimagenet third experience exps
    forg_hsg_tinyin_2_stats, forg_hsg_tinyin_2_dist = forgetting_metric(ALL_TINYIN_2, "hsg")
    # the estimated half-split forgetting gap for all cifar-100 second experience exps; only 10, 5 increments
    forg_hsg_cifar_10_5_1_stats, forg_hsg_cifar_10_5_1_dist = forgetting_metric(ALL_CIFAR_10_5_1, "hsg")
    # the estimated half-split forgetting gap for all tinyimagenet second experience exps; only 10, 5 increments
    forg_hsg_tinyin_10_5_1_stats, forg_hsg_tinyin_10_5_1_dist = forgetting_metric(ALL_TINYIN_10_5_1, "hsg")
    print("END")

def unnormalized_forgetting_stats():
    d_r = {}
    for name, benchmark in BENCHMARKS:
        d_r[name] = forgetting_metric(benchmark, "r", forgetting_name="UCF")[0]
        print(name)
        print(d_r[name])

    d_hsg = {}
    for name, benchmark in BENCHMARKS:
        d_hsg[name] = forgetting_metric(benchmark, "hsg", forgetting_name="UCF")[0]
        print(name)
        print(d_hsg[name])

    print("END")



def forgetting_sub_bench_stats():
    d_r = {}
    for name, benchmark in BENCHMARKS:
        d_r[name] = {}
        for b in benchmark:
            d_r[name][b] = forgetting_metric([b], "r")[0]

    d_hsg = {}
    for name, benchmark in BENCHMARKS:
        d_hsg[name] = {}
        for b in benchmark:
            d_hsg[name][b] = forgetting_metric([b], "hsg")[0]

    d_r_pooled = {}
    for name, benchmark in BENCHMARKS:
        d_r_pooled[name] = {}
        d_r_pooled[name]["N_EXPS"] = {}
        d_r_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_r_pooled[name]["N_EXPS"][n_exps] = forgetting_metric(modules,"r")[0]
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_r_pooled[name]["N_REH"][n_reh] = forgetting_metric([b for b in benchmark if b[4] == n_reh],
                                                                 "r")[0]

    d_hsg_pooled = {}
    for name, benchmark in BENCHMARKS:
        d_hsg_pooled[name] = {}
        d_hsg_pooled[name]["N_EXPS"] = {}
        d_hsg_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_hsg_pooled[name]["N_EXPS"][n_exps] = forgetting_metric(modules, "hsg")[0]
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_hsg_pooled[name]["N_REH"][n_reh] = forgetting_metric([b for b in benchmark if b[4] == n_reh],"hsg")[0]


    print("END")


def raw_pearson_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["SBI", "CBI", "NDI", "NDI-S", "CF"],
                              "pearson",
                              "raw"
                              )[0]
    print("END")


    # the estimated pairwise raw pearson correlations for all cifar-100 second experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"
    pairwise_raw_p_corr_cifar100_1_stats,  pairwise_raw_p_corr_cifar100_1_dist = correlation(ALL_CIFAR_1,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    # the estimated pairwise raw pearson correlations for all tinyimagenet second experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"
    pairwise_raw_p_corr_tinyin_1_stats, pairwise_raw_p_corr_tinyin_1_dist = correlation(ALL_TINYIN_1,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    # the estimated pairwise raw pearson correlations for all cifar-100 third experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"
    pairwise_raw_p_corr_cifar100_2_stats, pairwise_raw_p_corr_cifar100_2_dist = correlation(ALL_CIFAR_2,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    # the estimated pairwise raw pearson correlations for all tinyimagenet third experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"
    pairwise_raw_p_corr_tinyin_2_stats, pairwise_raw_p_corr_tinyin_2_dist = correlation(ALL_TINYIN_2,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    # the estimated pairwise raw pearson correlations for all cifar-100 second experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"; only 10, 5 increments
    pairwise_raw_p_corr_cifar100_10_5_1_stats, pairwise_raw_p_corr_cifar100_10_5_1_dist = correlation(ALL_CIFAR_10_5_1,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    # the estimated pairwise raw pearson correlations for all tinyimagenet second experience exps
    # using  "SBI", "CBI", "NDI", "NDI-S", "CF"; only 10, 5 increments
    pairwise_raw_p_corr_tinyin_10_5_1_stats, pairwise_raw_p_corr_tinyin_10_5_1_dist = correlation(ALL_TINYIN_10_5_1,
    ["SBI", "CBI", "NDI", "NDI-S", "CF"], "pearson", "raw")
    print("END")


def forgetting_normalized_forgetting_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["UCF", "CF", "SBI"],
                              "pearson",
                              "raw"
                              )[0]
        print(name, d[name])
    print("END")

def raw_spearman_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["R-SBI", "CBI", "NDI-S", "CF"],
                              "spearman",
                              "raw",
                              )[0]
    print("END")


def raw_spearman_correlation_NDI_S_one_epoch_vs_LOG_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["NDI-S", "LOG", "CF"],
                              "spearman",
                              "raw",
                              n_epochs=1,
                              )[0]
    print("END")


def pairwise_coeff_raw_spearman_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["SBI", "CBI", "NDI-S", "NDI", "OM"],
                              "spearman",
                              "raw"
                              )[0]
    print("END")


def raw_spearman_correlation_stats_OM_RP():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["RP", "OM", "CF"],
                              "spearman",
                              "raw",
                              )[0]
    print("END")

def raw_spearman_correlation_sub_bench_stats():
    d = {}
    d_pooled = {}
    for name, benchmark in BENCHMARKS:
        d[name] = {}
        for b in benchmark:
            d[name][b] = correlation([b],
                                     ["SBI", "CBI", "NDI-S", "CF"],
                                     "spearman",
                                     "raw"
                                    )[0]
        d_pooled[name] = {}
        d_pooled[name]["N_EXPS"] = {}
        d_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_pooled[name]["N_EXPS"][n_exps] = correlation(modules,
                                                               ["SBI", "CBI", "NDI-S", "CF"],
                                                                "spearman",
                                                                "raw"
                                                                )[0]
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_pooled[name]["N_REH"][n_reh] = correlation([b for b in benchmark if b[4] == n_reh],
                                                         ["SBI", "CBI", "NDI-S", "CF"],
                                                         "spearman",
                                                         "raw"
                                                         )[0]

    print("END")


def raw_spearman_correlation_sub_bench_stats_NDI_OM():
    d = {}
    d_pooled = {}
    for name, benchmark in BENCHMARKS:
        d[name] = {}
        for b in benchmark:
            d[name][b] = correlation([b],
                                     ["NDI-S", "OM", "CF"],
                                     "spearman",
                                     "raw"
                                    )[0]
        d_pooled[name] = {}
        d_pooled[name]["N_EXPS"] = {}
        d_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_pooled[name]["N_EXPS"][n_exps] = correlation(modules,
                                                               ["NDI-S", "OM", "CF"],
                                                                "spearman",
                                                                "raw"
                                                                )[0]
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_pooled[name]["N_REH"][n_reh] = correlation([b for b in benchmark if b[4] == n_reh],
                                                         ["NDI-S", "OM", "CF"],
                                                         "spearman",
                                                         "raw"
                                                         )[0]

    print("END")

def partial_pearson_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["SBI", "CBI", "NDI", "NDI-S", "CF"],
                              "pearson",
                              "partial",
                              non_included_covars={"NDI": ["NDI-S"], "NDI-S": ["NDI"]}
                              )[0]
    print("END")


def partial_spearman_correlation_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = correlation(benchmark,
                              ["R-SBI", "CBI", "NDI-S", "CF"],
                              "spearman",
                              "partial",
                              )[0]
    print("END")

def partial_spearman_correlation_NDIS_only_SBI_or_CBI_stats():
    d_only_SBI = {}
    for name, benchmark in BENCHMARKS:
        d_only_SBI[name] = correlation(benchmark,
                              ["SBI", "NDI-S", "CF"],
                              "spearman",
                              "partial",
                              )[0]

    d_only_CBI = {}
    for name, benchmark in BENCHMARKS:
        d_only_CBI[name] = correlation(benchmark,
                                       ["CBI", "NDI-S", "CF"],
                                       "spearman",
                                       "partial",
                                       )[0]
    print("END")


def partial_spearman_correlation_sub_bench_stats():
    d = {}
    d_pooled = {}
    for name, benchmark in BENCHMARKS:
        d[name] = {}
        for b in benchmark:
            d[name][b] = correlation([b],
                                     ["SBI", "CBI", "NDI-S", "CF"],
                                     "spearman",
                                     "partial"
                                    )[0]
        d_pooled[name] = {}
        d_pooled[name]["N_EXPS"] = {}
        d_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_pooled[name]["N_EXPS"][n_exps] = correlation(modules,
                                                               ["SBI", "CBI", "NDI-S", "CF"],
                                                                "spearman",
                                                                "partial"
                                                                )[0]
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_pooled[name]["N_REH"][n_reh] = correlation([b for b in benchmark if b[4] == n_reh],
                                                         ["SBI", "CBI", "NDI-S", "CF"],
                                                         "spearman",
                                                         "partial"
                                                         )[0]

    print("END")


def standardized_linear_regression_models_stats():
    d_NDI = {}
    for name, benchmark in BENCHMARKS:
        d_NDI[name] = standardized_linear_regression(benchmark,
                                                     ["SBI", "CBI", "NDI", "CF"],
                                                     y="CF")

    d_NDI_S = {}
    for name, benchmark in BENCHMARKS:
        d_NDI_S[name] = standardized_linear_regression(benchmark,
                                                       ["SBI", "CBI", "NDI-S", "CF"],
                                                       y="CF")
    print("END")


def standardized_linear_regression_SBI_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = standardized_linear_regression(benchmark,
                                                 ["SBI", "CF"],
                                                 y="CF")
    print("END")


def mixed_effects_linear_regression_stats():
    d_all = {}
    for name, benchmark in BENCHMARKS:
        d_all[name] = mixed_effects_linear_regression_leave_one_out(benchmark,
                                                                    ["R-SBI", "CBI", "NDI-S", "CF"],
                                                                    y="CF")
    d_SBI = {}
    for name, benchmark in BENCHMARKS:
        d_SBI[name] = mixed_effects_linear_regression_leave_one_out(benchmark,
                                                                    ["NDI-S","CF"],
                                                                    y="CF")
    print("END")


def mixed_effects_linear_regression_summary_stats():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = mixed_effects_linear_regression_summary(benchmark,
                                                         ["SBI", "CBI", "NDI-S", "CF"],
                                                          y="CF")
    print("END")


def mixed_effects_linear_regression_sub_bench_stats():
    d = {}
    d_pooled = {}
    for name, benchmark in BENCHMARKS:
        d[name] = {}
        for b in benchmark:
            d[name][b] = mixed_effects_linear_regression_leave_one_out([b],
                                                                        ["SBI", "CBI", "NDI-S", "CF"],
                                                                        y="CF")
        d_pooled[name] = {}
        d_pooled[name]["N_EXPS"] = {}
        d_pooled[name]["N_REH"] = {}
        for n_exps in [10, 5, 2]:
            modules = [b for b in benchmark if b[2] == n_exps]
            if len(modules) > 0:
                d_pooled[name]["N_EXPS"][n_exps] = mixed_effects_linear_regression_leave_one_out(
                    modules,
                    ["SBI", "CBI", "NDI-S", "CF"],
                    y="CF")
        for n_reh in N_REHEARSAL_PER_CLASS:
            d_pooled[name]["N_REH"][n_reh] = mixed_effects_linear_regression_leave_one_out(
                [b for b in benchmark if b[4] == n_reh],
                ["SBI", "CBI", "NDI-S", "CF"],
                "CF")

    print("END")


def correlation_line_plots_across_steps():
    cifar_benchmarks = [ALL_CIFAR_1, ALL_CIFAR_10_5_1, ALL_CIFAR_2]
    tinyin_benchmarks = [ALL_TINYIN_1, ALL_TINYIN_10_5_1, ALL_TINYIN_2]
    cifar_path = "SIC_CIC_NIC_across_steps_cifar.pkl"
    tinyin_path = "SIC_CIC_NIC_across_steps_tinyin.pkl"
    dfs_cifar = []
    dfs_tinyin = []

    if os.path.exists(cifar_path):
        with open(cifar_path, "rb") as f:
            dfs_cifar = pickle.load(f)
    else:
        for b in cifar_benchmarks:
            SBI_corr = []
            SBI_part_corr = []
            CBI_corr = []
            CBI_part_corr = []
            NDI_corr = []
            NDI_part_corr = []
            steps = []
            for n_epochs in range(1, 182):
                raw_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]
                SBI_corr.extend(raw_corr_matrices[:, 0, -1])
                CBI_corr.extend(raw_corr_matrices[:, 1, -1])
                NDI_corr.extend(raw_corr_matrices[:, 2, -1])

                part_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="partial",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]
                SBI_part_corr.extend(part_corr_matrices[:, 0, -1])
                CBI_part_corr.extend(part_corr_matrices[:, 1, -1])
                NDI_part_corr.extend(part_corr_matrices[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_cifar.append(pd.concat([pd.DataFrame({"values": SBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": SBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_cifar.append(pd.concat([pd.DataFrame({"values": CBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": CBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_cifar.append(pd.concat([pd.DataFrame({"values": NDI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": NDI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))

        with open(cifar_path, "wb") as f:
            pickle.dump(dfs_cifar, f)

    if os.path.exists(tinyin_path):
        with open(tinyin_path, "rb") as f:
            dfs_tinyin = pickle.load(f)
    else:
        for b in tinyin_benchmarks:
            SBI_corr = []
            SBI_part_corr = []
            CBI_corr = []
            CBI_part_corr = []
            NDI_corr = []
            NDI_part_corr = []
            steps = []
            for n_epochs in range(1, 102):
                raw_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]
                SBI_corr.extend(raw_corr_matrices[:, 0, -1])
                CBI_corr.extend(raw_corr_matrices[:, 1, -1])
                NDI_corr.extend(raw_corr_matrices[:, 2, -1])

                part_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="partial",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]
                SBI_part_corr.extend(part_corr_matrices[:, 0, -1])
                CBI_part_corr.extend(part_corr_matrices[:, 1, -1])
                NDI_part_corr.extend(part_corr_matrices[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": SBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": SBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": CBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": CBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": NDI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": NDI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))

        with open(tinyin_path, "wb") as f:
            pickle.dump(dfs_tinyin, f)


    line_plots(dfs=dfs_cifar, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", name_fig="SIC_CIC_NIC_across_steps_cifar.pdf",
               maxtick=181, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{C100}}$ (Progressive-Accumulation Analysis)", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{C100}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{C100}}$",
                           r"$\Omega^3_{\mathrm{C100}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)

    line_plots(dfs=dfs_tinyin, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(x))),
               errorbar="sd", name_fig="SIC_CIC_NIC_across_steps_tinyin.pdf",
               maxtick=101, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{TIN}}$ (Progressive-Accumulation Analysis)", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{TIN}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{TIN}}$",
                           r"$\Omega^3_{\mathrm{TIN}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)

def correlation_line_plots_across_steps_slid_window():
    cifar_benchmarks = [ALL_CIFAR_1, ALL_CIFAR_10_5_1, ALL_CIFAR_2]
    tinyin_benchmarks = [ALL_TINYIN_1, ALL_TINYIN_10_5_1, ALL_TINYIN_2]
    cifar_path = "SIC_CIC_NIC_across_steps_slid_window_cifar.pkl"
    tinyin_path = "SIC_CIC_NIC_across_steps_slid_window_tinyin.pkl"
    dfs_cifar = []
    dfs_tinyin = []

    if os.path.exists(cifar_path):
        with open(cifar_path, "rb") as f:
            dfs_cifar = pickle.load(f)
    else:
        for b in cifar_benchmarks:
            SBI_corr = []
            SBI_part_corr = []
            CBI_corr = []
            CBI_part_corr = []
            NDI_corr = []
            NDI_part_corr = []
            steps = []
            for n_epochs in range(1, 172):
                raw_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs,
                                                start_epoch=max(0, n_epochs -43))[1]
                SBI_corr.extend(raw_corr_matrices[:, 0, -1])
                CBI_corr.extend(raw_corr_matrices[:, 1, -1])
                NDI_corr.extend(raw_corr_matrices[:, 2, -1])

                part_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="partial",
                                                n_resamples=1,
                                                n_epochs=n_epochs,
                                                start_epoch=max(0, n_epochs -43))[1]
                SBI_part_corr.extend(part_corr_matrices[:, 0, -1])
                CBI_part_corr.extend(part_corr_matrices[:, 1, -1])
                NDI_part_corr.extend(part_corr_matrices[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_cifar.append(pd.concat([pd.DataFrame({"values": SBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": SBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_cifar.append(pd.concat([pd.DataFrame({"values": CBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": CBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_cifar.append(pd.concat([pd.DataFrame({"values": NDI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": NDI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))

        with open(cifar_path, "wb") as f:
            pickle.dump(dfs_cifar, f)

    if os.path.exists(tinyin_path):
        with open(tinyin_path, "rb") as f:
            dfs_tinyin = pickle.load(f)
    else:
        for b in tinyin_benchmarks:
            SBI_corr = []
            SBI_part_corr = []
            CBI_corr = []
            CBI_part_corr = []
            NDI_corr = []
            NDI_part_corr = []
            steps = []
            for n_epochs in range(1, 102):
                raw_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs,
                                                start_epoch=max(0, n_epochs - 25))[1]
                SBI_corr.extend(raw_corr_matrices[:, 0, -1])
                CBI_corr.extend(raw_corr_matrices[:, 1, -1])
                NDI_corr.extend(raw_corr_matrices[:, 2, -1])

                part_corr_matrices = correlation(b,
                                                label_measures=["SBI", "CBI", "NDI-S", "CF"],
                                                corr_op="spearman",
                                                corr_type="partial",
                                                n_resamples=1,
                                                n_epochs=n_epochs,
                                                start_epoch=max(0, n_epochs - 25))[1]
                SBI_part_corr.extend(part_corr_matrices[:, 0, -1])
                CBI_part_corr.extend(part_corr_matrices[:, 1, -1])
                NDI_part_corr.extend(part_corr_matrices[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": SBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": SBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": CBI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": CBI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))
            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": NDI_corr, "steps": steps, "group": r"$\rho$"}),
                                        pd.DataFrame({"values": NDI_part_corr, "steps": steps, "group": r"$\rho_p$"})]))

        with open(tinyin_path, "wb") as f:
            pickle.dump(dfs_tinyin, f)


    line_plots(dfs=dfs_cifar, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", name_fig="SIC_CIC_NIC_across_steps_slid_window_cifar.pdf",
               maxtick=171, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{C100}}$ (Sliding-Window Accumulation Analysis)", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{C100}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{C100}}$",
                           r"$\Omega^3_{\mathrm{C100}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)

    line_plots(dfs=dfs_tinyin, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(x))),
               errorbar="sd", name_fig="SIC_CIC_NIC_across_steps_slid_window_tinyin.pdf",
               maxtick=101, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{TIN}}$ (Sliding-Window Accumulation Analysis)", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{TIN}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{TIN}}$",
                           r"$\Omega^3_{\mathrm{TIN}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)


def correlation_line_plots_NIC_SIC_cifar_across_steps():
    cifar_benchmarks = [ALL_CIFAR_1, ALL_CIFAR_10_5_1, ALL_CIFAR_2]
    dfs_cifar = []

    for b in cifar_benchmarks:
        corr = []
        steps = []
        for n_epochs in range(1, 172):
            raw_corr_matrices = correlation(b,
                                            label_measures=["SBI", "NDI-S", "CF"],
                                            corr_op="spearman",
                                            corr_type="raw",
                                            n_resamples=1,
                                            n_epochs=n_epochs,
                                            apply_only=["NDI-S"])[1]
            corr.extend(raw_corr_matrices[:, 1, 0])
            steps.extend([n_epochs]*len(raw_corr_matrices))

        dfs_cifar.append(pd.DataFrame({"values": corr, "steps": steps, "group": r"$\rho$"}))

    line_plots(dfs=dfs_cifar, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", name_fig="NIC_SIC_across_steps_cifar.pdf",
               maxtick=171, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{C100}}$", col_titles=[r"$\Omega_{\mathrm{C100}}^2$",
                                                              r"$\Omega_{\mathrm{C100}}^{2;\,\{10\%,20\%\}}$",
                                                              r"$\Omega_{\mathrm{C100}}^3$"],
               row_titles=[r"", r"", r""], y_ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], y_min=0,
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)


def correlation_line_plots_NIC_SIC_tinyin_across_steps():
    tinyin_benchmarks = [ALL_TINYIN_1, ALL_TINYIN_10_5_1, ALL_TINYIN_2]
    dfs_tinyin = []

    for b in tinyin_benchmarks:
        corr = []
        steps = []
        for n_epochs in range(1, 102):
            raw_corr_matrices = correlation(b,
                                            label_measures=["SBI", "NDI-S", "CF"],
                                            corr_op="spearman",
                                            corr_type="raw",
                                            n_resamples=1,
                                            n_epochs=n_epochs,
                                            apply_only=["NDI-S"])[1]
            corr.extend(raw_corr_matrices[:, 1, 0])
            steps.extend([n_epochs]*len(raw_corr_matrices))

        dfs_tinyin.append(pd.DataFrame({"values": corr, "steps": steps, "group": r"$\rho$"}))

    line_plots(dfs=dfs_tinyin, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", name_fig="NIC_SIC_across_steps_tinyin.pdf",
               maxtick=101, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{TIN}}$", col_titles=[r"$\Omega_{\mathrm{TIN}}^2$",
                                                              r"$\Omega_{\mathrm{TIN}}^{2;\,\{10\%,20\%\}}$",
                                                              r"$\Omega_{\mathrm{TIN}}^3$"],
               row_titles=[r"", r"", r""], y_ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], y_min=0,
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)


def correlation_line_plots_across_steps_OM_RI_cifar():
    cifar_benchmarks = [ALL_CIFAR_1, ALL_CIFAR_10_5_1, ALL_CIFAR_2]
    cifar_path = "OM_RI_across_steps_cifar.pkl"
    dfs_cifar = []

    if os.path.exists(cifar_path):
        with open(cifar_path, "rb") as f:
            dfs_cifar = pickle.load(f)
    else:
        for b in cifar_benchmarks:
            OM_corr = []
            RI_corr = []
            SBI_corr = []
            OM_corr_single = []
            RI_corr_single = []
            SBI_corr_single = []
            steps = []
            for n_epochs in range(1, 172):
                raw_corr_matrices = correlation(b,
                                                label_measures=["OM", "RP", "SBI", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]

                raw_corr_matrices_single = correlation(b,
                                                       label_measures=["OM", "RP", "SBI", "CF"],
                                                       corr_op="spearman",
                                                       corr_type="raw",
                                                       n_resamples=1,
                                                       n_epochs=n_epochs,
                                                       start_epoch= max(0, n_epochs -43))[1]

                OM_corr.extend(-raw_corr_matrices[:, 0, -1])
                RI_corr.extend(raw_corr_matrices[:, 1, -1])
                SBI_corr.extend(raw_corr_matrices[:, 2, -1])
                OM_corr_single.extend(-raw_corr_matrices_single[:, 0, -1])
                RI_corr_single.extend(raw_corr_matrices_single[:, 1, -1])
                SBI_corr_single.extend(raw_corr_matrices_single[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_cifar.append(pd.concat([pd.DataFrame({"values": OM_corr, "steps": steps, "group": "OM"}),
                                        pd.DataFrame({"values": RI_corr, "steps": steps, "group": "-RI"}),
                                        pd.DataFrame({"values": SBI_corr, "steps": steps, "group": "SIC"})]))

            dfs_cifar.append(pd.concat([pd.DataFrame({"values": OM_corr_single, "steps": steps, "group": "OM"}),
                                        pd.DataFrame({"values": RI_corr_single, "steps": steps, "group": "-RI"}),
                                        pd.DataFrame({"values": SBI_corr_single, "steps": steps, "group": "SIC"})]))


        with open(cifar_path, "wb") as f:
            pickle.dump(dfs_cifar, f)


    line_plots(dfs=dfs_cifar, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", err_style="band", err_kws={"alpha": 0.1}, name_fig="OM_RI_across_steps_cifar.pdf",
               maxtick=171, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{C100}}$", col_titles=["Progressive Accumulation", "Sliding-Window Accumulation"],
               row_titles=[r"$\Omega^2_{\mathrm{C100}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{C100}}$",
                           r"$\Omega^3_{\mathrm{C100}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=2,
               )


def correlation_line_plots_across_steps_OM_RI_tinyin():
    tinyin_benchmarks = [ALL_TINYIN_1, ALL_TINYIN_10_5_1, ALL_TINYIN_2]
    tiyin_path = "OM_RI_across_steps_tinyin.pkl"
    dfs_tinyin = []

    if os.path.exists(tiyin_path):
        with open(tiyin_path, "rb") as f:
            dfs_tinyin = pickle.load(f)
    else:
        for b in tinyin_benchmarks:
            OM_corr = []
            RI_corr = []
            SBI_corr = []
            OM_corr_single = []
            RI_corr_single = []
            SBI_corr_single = []
            steps = []
            for n_epochs in range(1, 102):
                raw_corr_matrices = correlation(b,
                                                label_measures=["OM", "RP", "SBI", "CF"],
                                                corr_op="spearman",
                                                corr_type="raw",
                                                n_resamples=1,
                                                n_epochs=n_epochs)[1]

                raw_corr_matrices_single = correlation(b,
                                                       label_measures=["OM", "RP", "SBI", "CF"],
                                                       corr_op="spearman",
                                                       corr_type="raw",
                                                       n_resamples=1,
                                                       n_epochs=n_epochs,
                                                       start_epoch=max(0, n_epochs - 25))[1]

                OM_corr.extend(-raw_corr_matrices[:, 0, -1])
                RI_corr.extend(raw_corr_matrices[:, 1, -1])
                SBI_corr.extend(raw_corr_matrices[:, 2, -1])
                OM_corr_single.extend(-raw_corr_matrices_single[:, 0, -1])
                RI_corr_single.extend(raw_corr_matrices_single[:, 1, -1])
                SBI_corr_single.extend(raw_corr_matrices_single[:, 2, -1])

                steps.extend([n_epochs]*len(raw_corr_matrices))

            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": OM_corr, "steps": steps, "group": "OM"}),
                                        pd.DataFrame({"values": RI_corr, "steps": steps, "group": "-RI"}),
                                        pd.DataFrame({"values": SBI_corr, "steps": steps, "group": "SIC"})]))
            dfs_tinyin.append(pd.concat([pd.DataFrame({"values": OM_corr_single, "steps": steps, "group": "OM"}),
                                         pd.DataFrame({"values": RI_corr_single, "steps": steps, "group": "-RI"}),
                                         pd.DataFrame({"values": SBI_corr_single, "steps": steps, "group": "SIC"})]))

        with open(tiyin_path, "wb") as f:
            pickle.dump(dfs_tinyin, f)


    line_plots(dfs=dfs_tinyin, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(np.clip(x, -1 + 1e-7, 1-1e-7)))),
               errorbar="sd", err_style="band", err_kws={"alpha": 0.1}, name_fig="OM_RI_across_steps_tinyin.pdf",
               maxtick=101, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{TIN}}$", col_titles=["Progressive Accumulation", "Sliding-Window Accumulation"],
               row_titles=[r"$\Omega^2_{\mathrm{TIN}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{TIN}}$",
                           r"$\Omega^3_{\mathrm{TIN}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=2,
               )


def mixed_effects_linear_regression_across_steps():
    cifar_benchmarks = [ALL_CIFAR_1, ALL_CIFAR_10_5_1, ALL_CIFAR_2]
    tinyin_benchmarks = [ALL_TINYIN_1, ALL_TINYIN_10_5_1, ALL_TINYIN_2]
    cifar_path = "mixed_effects_across_steps_cifar.pkl"
    tinyin_path = "mixed_effects_across_steps_tinyin.pkl"
    dfs_cifar = []
    dfs_tinyin = []

    if os.path.exists(cifar_path):
        with open(cifar_path, "rb") as f:
            dfs_cifar = pickle.load(f)
    else:
        for b in cifar_benchmarks:
            corr = []
            steps = []
            for n_epochs in range(1, 182):
                out = mixed_effects_linear_regression_leave_one_out(b,
                                                                    ["SBI", "CBI", "NDI-S", "CF"],
                                                                    y="CF",
                                                                    n_epochs=n_epochs)[1]
                corr.extend(out)
                steps.extend([n_epochs] * len(out))

            dfs_cifar.append(pd.DataFrame({"values": corr, "steps": steps}))

        with open(cifar_path, "wb") as f:
            pickle.dump(dfs_cifar, f)

    if os.path.exists(tinyin_path):
        with open(tinyin_path, "rb") as f:
            dfs_tinyin = pickle.load(f)
    else:
        for b in tinyin_benchmarks:
            corr = []
            steps = []
            for n_epochs in range(1, 102):
                out = mixed_effects_linear_regression_leave_one_out(b,
                                                                    ["SBI", "CBI", "NDI-S", "CF"],
                                                                    y="CF",
                                                                    n_epochs=n_epochs)[1]
                corr.extend(out)
                steps.extend([n_epochs] * len(out))

            dfs_tinyin.append(pd.DataFrame({"values": corr, "steps": steps}))

        with open(tinyin_path, "wb") as f:
            pickle.dump(dfs_tinyin, f)


def anova_forgetting_metrics():
    d_r={}
    for name, benchmark in BENCHMARKS:
        dic = {b: forgetting_metric([b], "r")[1] for b in benchmark}
        tab = long_table(dic, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / ( anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
        d_r[name] = anova_table

    d_hsg = {}
    for name, benchmark in BENCHMARKS:
        dic = {b: forgetting_metric([b], "hsg")[1] for b in benchmark}
        tab = long_table(dic, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (
                    anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
        d_hsg[name] = anova_table

    print("END")


def anova_spearman_correlation_coefficient(index):
    d = {}
    for name, benchmark in BENCHMARKS:
        dic = {}
        dic_diff = {}
        for b in benchmark:
            raw_corr = correlation([b],
                                   ["SBI", "CBI", "NDI-S", "CF"],
                                   "spearman",
                                   "raw",
                                   n_resamples=1
                                   )[1][:, index, -1]

            part_corr = correlation([b],
                                   ["SBI", "CBI", "NDI-S", "CF"],
                                   "spearman",
                                   "partial",
                                    n_resamples=1
                                   )[1][:, index, -1]
            dic[b] = raw_corr
            dic_diff[b] = raw_corr - part_corr

        tab = long_table(dic, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (
                anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])

        tab_diff = long_table(dic_diff, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit_diff = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab_diff).fit()
        anova_table_diff = sm.stats.anova_lm(ols_fit_diff, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table_diff['partial_eta_sq'] = anova_table_diff['sum_sq'] / (
                anova_table_diff['sum_sq'] + anova_table_diff.loc['Residual', 'sum_sq'])

        d[name] = (anova_table, anova_table_diff)

    print("END")


def anova_spearman_correlation_SBI_NDIS():
    d = {}
    for name, benchmark in BENCHMARKS:
        dic = {}
        for b in benchmark:
            raw_corr = correlation([b],
                                   ["SBI", "CBI", "NDI-S", "CF"],
                                   "spearman",
                                   "raw",
                                   n_resamples=1
                                   )[1][:, 2, 0]
            dic[b] = raw_corr

        tab = long_table(dic, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (
                anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])

        d[name] = anova_table

    print("END")


def anova_mixed_effects_linear_regression():
    d = {}
    for name, benchmark in BENCHMARKS:
        dic = {}
        for b in benchmark:
            out = mixed_effects_linear_regression_leave_one_out([b],
                                                                ["SBI", "CBI", "NDI-S", "CF"],
                                                                y="CF"
                                                                )[1]
            dic[b] = out

        tab = long_table(dic, ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Exp_ID", "Val"])
        ols_fit = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)', data=tab).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3, robust="hc3")
        # Compute partial eta squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (
                anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])

        d[name] = anova_table

    print("END")


def top50_accuracy_NIC_S_forgetting():
    d = {}
    for name, benchmark in BENCHMARKS:
        d[name] = {}
        for b in benchmark:
            measures = collect_measures(["NDI-S", "CF"], modules_dict[b])
            _, C, _ = measures.shape
            idx = np.argsort(measures, axis=1)
            top50_idx = idx[:, C//2:, :]
            d[name][b] = np.asarray([len(np.intersect1d(a, b)) / (C/2) for a, b in zip(top50_idx[:,:,0], top50_idx[:,:,1])])

    print("END")
##################################Actual data############################

if __name__ == "__main__":
    #forgetting_stats()
    # unnormalized_forgetting_stats()
    # raw_pearson_correlation_stats()
    # raw_spearman_correlation_stats()
    #raw_spearman_correlation_stats_OM()
    #raw_spearman_correlation_stats_OM_RP()
    # partial_pearson_correlation_stats()
    # partial_spearman_correlation_stats()
    # standardized_linear_regression_models_stats()
    # standardized_linear_regression_SBI_stats()
    #mixed_effects_linear_regression_stats()
    # forgetting_sub_bench_stats()
    #raw_spearman_correlation_sub_bench_stats()
    #raw_spearman_correlation_sub_bench_stats_NDI_OM()
    # partial_spearman_correlation_sub_bench_stats()
    # mixed_effects_linear_regression_sub_bench_stats()
    # pairwise_coeff_raw_spearman_correlation_stats()
    correlation_line_plots_across_steps()
    #correlation_line_plots_across_steps_slid_window()
    # mixed_effects_linear_regression_across_steps()
    # anova_forgetting_metrics()
    # anova_spearman_correlation_coefficient(2)
    # anova_mixed_effects_linear_regression()
    # anova_spearman_correlation_SBI_NDIS()
    # partial_spearman_correlation_NDIS_only_SBI_or_CBI_stats()
    # mixed_effects_linear_regression_summary_stats()
    # correlation_line_plots_NIC_SIC_cifar_across_steps()
    # correlation_line_plots_NIC_SIC_tinyin_across_steps()
    #correlation_line_plots_across_steps_OM_RI_cifar()
    #correlation_line_plots_across_steps_OM_RI_tinyin()
    # raw_spearman_correlation_NDI_S_one_epoch_vs_LOG_stats()
    # top50_accuracy_NIC_S_forgetting()
    # forgetting_normalized_forgetting_correlation_stats()
    #
    # line_plots(dfs=[pd.concat([pd.DataFrame({"values": np.random.rand(1800), "group": "A", "R-SGD steps": np.repeat(np.arange(1, 181), 10) }),
    #                            pd.DataFrame({"values": np.random.rand(1800), "group": "B", "R-SGD steps": np.repeat(np.arange(1, 181), 10) })]) for _ in range(9)],
    #            x="R-SGD steps", y="values", estimator="mean", errorbar="sd", name_fig="pg.pdf", maxtick=180, xlabel="R-SGD steps",
    #            hue="group", title=r"$\Omega_{\mathrm{C100}}$", col_titles=["SIC", "CIC", "NIC"],
    #            row_titles=[r"$\Omega^2_{\mathrm{C100}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{C100}}$", r"$\Omega^3_{\mathrm{C100}}$"],
    #            figsize=(7.1, 8.8), sharey=True, sharex=True, nrows=3, ncols=3)
    #
    # swarm_plot_forgetting(
    #     ALL_CIFAR_1, n_exps=5, name_fig="cifar100_1.pdf", show=True,
    #     title=r"$\Omega_{\mathrm{C100}}^2$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', '50% Classes'],
    #     figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )
    #
    # swarm_plot_forgetting(
    #     ALL_CIFAR_2, n_exps=5, name_fig="cifar100_2.pdf", show=True,
    #     title=r"$\Omega_{\mathrm{C100}}^3$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', ''],
    #     figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )
    #
    # swarm_plot_forgetting(
    #     ALL_TINYIN_1, n_exps=5, name_fig="tinyimagenet_1.pdf", show=True,
    #     title=r"$\Omega_{\mathrm{TIN}}^2$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', '50% Classes'],
    #     figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )
    #
    # swarm_plot_forgetting(
    #     ALL_TINYIN_2, n_exps=5, name_fig="tinyimagenet_2.pdf", show=True,
    #     title=r"$\Omega_{\mathrm{TIN}}^3$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', ''],
    #     figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )

    # # the estimated forgetting range for all cifar-100 based exps
    # forg_range_cifar_stats, forg_range_cifar_dist = forgetting_metric(ALL_CIFAR, "r")
    # # the estimated forgetting range for all tinyimagenet based exps.
    # forg_range_tinyin_stats, forg_range_tinyin_dist = forgetting_metric(ALL_TINYIN, "r")
    # # two-sided Welch’s t-test to test for statistical difference between the estimated mean of Forgetting Range for
    # # the cifar100-based experiments vs the tinyimagenet ones
    # t_stat_forg_range, p_value_forg_range = stats.ttest_ind(forg_range_cifar_dist, forg_range_tinyin_dist,
    #                                                         equal_var=False, random_state=123)
    #
    # # the estimated half-split forgetting gap for all cifar-100 based exps
    # forg_hsg_cifar_stats, forg_hsg_cifar_dist = forgetting_metric(ALL_CIFAR, "hsg")
    # # the estimated half-split forgetting gap for all tinyimagenet based exps
    # forg_hsg_tinyin_stats, forg_hsg_tinyin_dist = forgetting_metric(ALL_TINYIN, "hsg")
    # # two-sided Welch’s t-test to test for statistical difference between the estimated mean of Half-Split Forgetting
    # # Gap for the cifar100-based experiments vs the tinyimagenet ones
    # t_stat_hsg, p_value_hsg = stats.ttest_ind(forg_hsg_cifar_dist, forg_hsg_tinyin_dist, equal_var=False,
    #                                           random_state=123)
    #
    # # a dictionary containing all the module keys in `ALL_CIFAR` as keys and the respective estimated
    # # forgetting ranges as values
    # forg_range_cifar_ind = {key: forgetting_metric([key], "r") for key in ALL_CIFAR}
    # # a long table containing all the experiments run in all the module keys in `ALL_CIFAR` along with their attrs. for
    # # the forgetting ranges
    # forg_range_cifar_long_tab = long_table({key: val[1] for key, val in forg_range_cifar_ind.items()},
    #                               ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Val"])
    # anova_model_forg_range_cifar = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)',
    #                                    data=forg_range_cifar_long_tab).fit()
    # anova_table_forg_range_cifar = sm.stats.anova_lm(anova_model_forg_range_cifar, typ=3, robust="hc3")
    # # Compute partial eta squared
    # anova_table_forg_range_cifar['partial_eta_sq'] = anova_table_forg_range_cifar['sum_sq'] / (
    #         anova_table_forg_range_cifar['sum_sq'] + anova_table_forg_range_cifar.loc['Residual', 'sum_sq']
    # )
    # forg_range_cifar_ind_df = pd.concat([x[0] for x in list(forg_range_cifar_ind.values())], axis=0)
    # forg_range_cifar_ind_df.index = list(forg_range_cifar_ind.keys())
    #
    # # a dictionary containing all the module keys in `ALL_TINYIN` as keys and the respective estimated
    # # forgetting ranges as values
    # forg_range_tinyin_ind = {key: forgetting_metric([key], "r") for key in ALL_TINYIN}
    # # a long table containing all the experiments run in all the module keys in `ALL_TINYIN` along with their attrs.
    # # for the forgetting ranges
    # forg_range_tinyin_long_tab = long_table({key: val[1] for key, val in forg_range_tinyin_ind.items()},
    #                                        ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Val"])
    # anova_model_forg_range_tinyin = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)',
    #                                     data=forg_range_tinyin_long_tab).fit()
    # anova_table_forg_range_tinyin = sm.stats.anova_lm(anova_model_forg_range_tinyin, typ=3, robust="hc3")
    # # Compute partial eta squared
    # anova_table_forg_range_tinyin['partial_eta_sq'] = anova_table_forg_range_tinyin['sum_sq'] / (
    #         anova_table_forg_range_tinyin['sum_sq'] + anova_table_forg_range_tinyin.loc['Residual', 'sum_sq']
    # )
    # forg_range_tinyin_ind_df = pd.concat([x[0] for x in list(forg_range_tinyin_ind.values())], axis=0)
    # forg_range_tinyin_ind_df.index = list(forg_range_tinyin_ind.keys())
    #
    # # a dictionary containing all the module keys in `ALL_CIFAR` as keys and the respective estimated
    # # half-split forgetting gap as values
    # forg_hsg_cifar_ind = {key: forgetting_metric([key], "hsg") for key in ALL_CIFAR}
    # # a long table containing all the experiments run in all the module keys in `ALL_CIFAR` along with their attrs. for
    # # the half-split forgetting gaps
    # forg_hsg_cifar_long_tab = long_table({key: val[1] for key, val in forg_hsg_cifar_ind.items()},
    #                                        ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Val"])
    # anova_model_forg_hsg_cifar = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)',
    #                                  data=forg_hsg_cifar_long_tab).fit()
    # anova_table_forg_hsg_cifar = sm.stats.anova_lm(anova_model_forg_hsg_cifar, typ=3, robust="hc3")
    # # Compute partial eta squared
    # anova_table_forg_hsg_cifar['partial_eta_sq'] = anova_table_forg_hsg_cifar['sum_sq'] / (
    #         anova_table_forg_hsg_cifar['sum_sq'] + anova_table_forg_hsg_cifar.loc['Residual', 'sum_sq']
    # )
    # forg_hsg_cifar_ind_df = pd.concat([x[0] for x in list(forg_hsg_cifar_ind.values())], axis=0)
    # forg_hsg_cifar_ind_df.index = list(forg_hsg_cifar_ind.keys())
    #
    # # a dictionary containing all the module keys in `ALL_TINYIN` as keys and the respective estimated
    # # half-split forgetting gap as values
    # forg_hsg_tinyin_ind = {key: forgetting_metric([key], "hsg") for key in ALL_TINYIN}
    # # a long table containing all the experiments run in all the module keys in `ALL_TINYIN` along with their attrs.
    # # for the half-split forgetting gaps
    # forg_hsg_tinyin_long_tab = long_table({key: val[1] for key, val in forg_hsg_tinyin_ind.items()},
    #                              ["Dataset", "Arch.", "N_Exps", "N_Cls", "N_Reh", "Val"])
    # anova_model_forg_hsg_tinyin = ols('Val ~ C(N_Cls) + C(N_Reh) + C(N_Cls):C(N_Reh)',
    #                                   data=forg_hsg_tinyin_long_tab).fit()
    # anova_table_forg_hsg_tinyin = sm.stats.anova_lm(anova_model_forg_hsg_tinyin, typ=3, robust="hc3")
    # # Compute partial eta squared
    # anova_table_forg_hsg_tinyin['partial_eta_sq'] = anova_table_forg_hsg_tinyin['sum_sq'] / (
    #         anova_table_forg_hsg_tinyin['sum_sq'] + anova_table_forg_hsg_tinyin.loc['Residual', 'sum_sq']
    # )
    # forg_hsg_tinyin_ind_df = pd.concat([x[0] for x in list(forg_hsg_tinyin_ind.values())], axis=0)
    # forg_hsg_tinyin_ind_df.index = list(forg_hsg_tinyin_ind.keys())
    #
    # # the estimated forgetting range for all cifar-100 based exps with 10 classes per experience
    # forg_range_cifar_10_stats, forg_range_cifar_10_dist = forgetting_metric(CIFAR100_10_CLS, "r")
    # # the estimated forgetting range for all cifar-100 based exps with 20 classes per experience.
    # forg_range_cifar_20_stats, forg_range_cifar_20_dist = forgetting_metric(CIFAR100_5_CLS, "r")
    # # the estimated forgetting range for all cifar-100 based exps with 50 classes per experience.
    # forg_range_cifar_50_stats, forg_range_cifar_50_dist = forgetting_metric(CIFAR100_2_CLS, "r")
    #
    # # the estimated forgetting range for all tinyimagenet based exps with 20 classes per experience
    # forg_range_tinyin_20_stats, forg_range_tinyin_20_dist = forgetting_metric(TINYIN_10_CLS, "r")
    # # the estimated forgetting range for all tinyimagenet based exps with 40 classes per experience
    # forg_range_tinyin_40_stats, forg_range_tinyin_40_dist = forgetting_metric(TINYIN_5_CLS, "r")
    # # the estimated forgetting range for all tinyimagenet based exps with 100 classes per experience
    # forg_range_tinyin_100_stats, forg_range_tinyin_100_dist = forgetting_metric(TINYIN_2_CLS, "r")
    #
    # # the estimated forgetting range for all cifar-100 based exps with 8% rehearsal
    # forg_range_cifar_8r_stats, forg_range_cifar_8r_dist = forgetting_metric(
    #     [l[0] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "r")
    # # the estimated forgetting range for all cifar-100 based exps with 20% rehearsal
    # forg_range_cifar_20r_stats, forg_range_cifar_20r_dist = forgetting_metric(
    #     [l[1] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "r")
    # # the estimated forgetting range for all cifar-100 based exps with 40% rehearsal
    # forg_range_cifar_40r_stats, forg_range_cifar_40r_dist = forgetting_metric(
    #     [l[2] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "r")
    #
    # # the estimated forgetting range for all tinyimagenet based exps with 8% rehearsal
    # forg_range_tinyin_8r_stats, forg_range_tinyin_8r_dist = forgetting_metric(
    #     [l[0] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "r")
    # # the estimated forgetting range for all tinyimagenet based exps with 20% rehearsal
    # forg_range_tinyin_20r_stats, forg_range_tinyin_20r_dist = forgetting_metric(
    #     [l[1] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "r")
    # # the estimated forgetting range for all tinyimagenet based exps with 40% rehearsal
    # forg_range_tinyin_40r_stats, forg_range_tinyin_40r_dist = forgetting_metric(
    #     [l[2] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "r")
    #
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 10 classes per experience
    # forg_hsg_cifar_10_stats, forg_hsg_cifar_10_dist = forgetting_metric(CIFAR100_10_CLS, "hsg")
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 20 classes per experience.
    # forg_hsg_cifar_20_stats, forg_hsg_cifar_20_dist = forgetting_metric(CIFAR100_5_CLS, "hsg")
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 50 classes per experience.
    # forg_hsg_cifar_50_stats, forg_hsg_cifar_50_dist = forgetting_metric(CIFAR100_2_CLS, "hsg")
    #
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 20 classes per experience
    # forg_hsg_tinyin_20_stats, forg_hsg_tinyin_20_dist = forgetting_metric(TINYIN_10_CLS, "hsg")
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 40 classes per experience
    # forg_hsg_tinyin_40_stats, forg_hsg_tinyin_40_dist = forgetting_metric(TINYIN_5_CLS, "hsg")
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 100 classes per experience
    # forg_hsg_tinyin_100_stats, forg_hsg_tinyin_100_dist = forgetting_metric(TINYIN_2_CLS, "hsg")
    #
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 8% rehearsal
    # forg_hsg_cifar_8r_stats, forg_hsg_cifar_8r_dist = forgetting_metric(
    #     [l[0] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "hsg")
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 20% rehearsal
    # forg_hsg_cifar_20r_stats, forg_hsg_cifar_20r_dist = forgetting_metric(
    #     [l[1] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "hsg")
    # # the estimated half-split forgetting gap for all cifar-100 based exps with 40% rehearsal
    # forg_hsg_cifar_40r_stats, forg_hsg_cifar_40r_dist = forgetting_metric(
    #     [l[2] for l in [CIFAR100_10_CLS, CIFAR100_5_CLS, CIFAR100_2_CLS]], "hsg")
    #
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 8% rehearsal
    # forg_hsg_tinyin_8r_stats, forg_hsg_tinyin_8r_dist = forgetting_metric(
    #     [l[0] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "hsg")
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 20% rehearsal
    # forg_hsg_tinyin_20r_stats, forg_hsg_tinyin_20r_dist = forgetting_metric(
    #     [l[1] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "hsg")
    # # the estimated half-split forgetting gap for all tinyimagenet based exps with 40% rehearsal
    # forg_hsg_tinyin_40r_stats, forg_hsg_tinyin_40r_dist = forgetting_metric(
    #     [l[2] for l in [TINYIN_10_CLS, TINYIN_5_CLS, TINYIN_2_CLS]], "hsg")
    #
    #
    #
    # swarm_plot_forgetting(
    #     ALL_CIFAR, n_exps=5, name_fig="cifar100.svg", show=True,
    #     title=r"Benchmark: $\Omega_{\text{CIFAR-100}}$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', '50% Classes'],
    #     index_y_label=3, figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )
    #
    # swarm_plot_forgetting(
    #     ALL_TINYIN, n_exps=5, name_fig="tinyimagenet.svg", show=True,
    #     title=r"Benchmark: $\Omega_{\text{TINY-IN}}$",
    #     col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #     row_titles=['10% Classes', '20% Classes', '50% Classes'],
    #     index_y_label=3, figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )
    # # the estimated pairwise raw pearson correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"
    # pairwise_raw_p_correlations_cifar100_stats,  pairwise_raw_p_correlations_cifar100_dist = correlation(ALL_CIFAR,
    #     ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson", "raw")
    #
    # # the estimated pairwise raw spearman correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"
    # pairwise_raw_s_correlations_cifar100_stats, pairwise_raw_s_correlations_cifar100_dist = correlation(ALL_CIFAR,
    #     ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman", "raw")
    #
    # # the estimated pairwise partial pearson correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # pairwise_part_p_correlations_cifar100_stats, pairwise_part_p_correlations_cifar100_dist = correlation(ALL_CIFAR,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson", "partial",
    #     non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"], "NDI": ["NDI-S"], "NDI-S": ["NDI"]})
    #
    # # the estimated pairwise partial spearman correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # pairwise_part_s_correlations_cifar100_stats, pairwise_part_s_correlations_cifar100_dist = correlation(ALL_CIFAR,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman", "partial",
    #     non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"], "NDI": ["NDI-S"], "NDI-S": ["NDI"]})
    #
    # # the estimated pairwise partial pearson correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "SBI" is not used as covariate for "NDI" and "NDI-S
    # pairwise_part_SBI_non_cov_p_correlations_cifar100_stats, pairwise_part_SBI_non_cov_p_correlations_cifar100_dist =(
    #     correlation(ALL_CIFAR, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                         "NDI": ["NDI-S", "SBI"], "NDI-S": ["NDI", "SBI"]}))
    #
    # # the estimated pairwise partial pearson correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "CBI" and "CBI-S" are not used as covariate for "NDI" and "NDI-S
    # pairwise_part_CBI_non_cov_p_correlations_cifar100_stats, pairwise_part_CBI_non_cov_p_correlations_cifar100_dist = (
    #     correlation(ALL_CIFAR, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "CBI", "CBI-S"], "NDI-S": ["NDI", "CBI", "CBI-S"]})
    # )
    #
    # # the estimated pairwise partial spearman correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "SBI" is not used as covariate for "NDI" and "NDI-S
    # pairwise_part_SBI_non_cov_s_correlations_cifar100_stats, pairwise_part_SBI_non_cov_s_correlations_cifar100_dist = (
    #     correlation(ALL_CIFAR, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "SBI"], "NDI-S": ["NDI", "SBI"]}))
    #
    # # the estimated pairwise partial spearman correlations for all cifar-100 based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "CBI" and "CBI-S" are not used as covariate for "NDI" and "NDI-S
    # pairwise_part_CBI_non_cov_s_correlations_cifar100_stats, pairwise_part_CBI_non_cov_s_correlations_cifar100_dist = (
    #     correlation(ALL_CIFAR, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "CBI", "CBI-S"], "NDI-S": ["NDI", "CBI", "CBI-S"]})
    # )
    #
    # # the estimated pairwise raw pearson correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"
    # pairwise_raw_p_correlations_tinyin_stats, pairwise_raw_p_correlations_tinyin_dist = correlation(ALL_TINYIN,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson", "raw")
    #
    # # the estimated pairwise raw spearman correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"
    # pairwise_raw_s_correlations_tinyin_stats, pairwise_raw_s_correlations_tinyin_dist = correlation(ALL_TINYIN,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman", "raw")
    #
    # # the estimated pairwise partial pearson correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # pairwise_part_p_correlations_tinyin_stats, pairwise_part_p_correlations_tinyin_dist = correlation(ALL_TINYIN,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson", "partial",
    #     non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"], "NDI": ["NDI-S"], "NDI-S": ["NDI"]})
    #
    # # the estimated pairwise partial spearman correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # pairwise_part_s_correlations_tinyin_stats, pairwise_part_s_correlations_tinyin_dist = correlation(ALL_TINYIN,
    # ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman", "partial",
    #     non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"], "NDI": ["NDI-S"], "NDI-S": ["NDI"]})
    #
    # # the estimated pairwise partial pearson correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "SBI" is not used as covariate for "NDI" and "NDI-S
    # pairwise_part_SBI_non_cov_p_correlations_tinyin_stats, pairwise_part_SBI_non_cov_p_correlations_tinyin_dist = (
    #     correlation(ALL_TINYIN, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "SBI"], "NDI-S": ["NDI", "SBI"]}))
    #
    # # the estimated pairwise partial pearson correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "CBI" and "CBI-S" are not used as covariate for "NDI" and "NDI-S
    # pairwise_part_CBI_non_cov_p_correlations_tinyin_stats, pairwise_part_CBI_non_cov_p_correlations_tinyin_dist = (
    #     correlation(ALL_TINYIN, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "pearson",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "CBI", "CBI-S"], "NDI-S": ["NDI", "CBI", "CBI-S"]})
    # )
    #
    # # the estimated pairwise partial spearman correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "SBI" is not used as covariate for "NDI" and "NDI-S
    # pairwise_part_SBI_non_cov_s_correlations_tinyin_stats, pairwise_part_SBI_non_cov_s_correlations_tinyin_dist = (
    #     correlation(ALL_TINYIN, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "SBI"], "NDI-S": ["NDI", "SBI"]}))
    #
    # # the estimated pairwise partial spearman correlations for all tinyimagenet based exps
    # # using  "SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF". The partial correlations do not include similar variables
    # # as covariates. For example, the partial correlation between "CF" and "NDI" does not use "NDI-S" as covariates.
    # # Furthermore, "CBI" and "CBI-S" are not used as covariate for "NDI" and "NDI-S
    # pairwise_part_CBI_non_cov_s_correlations_tinyin_stats, pairwise_part_CBI_non_cov_s_correlations_tinyin_dist = (
    #     correlation(ALL_TINYIN, ["SBI", "CBI", "CBI-S", "NDI", "NDI-S", "CF"], "spearman",
    #                 "partial", non_included_covars={"CBI": ["CBI-S"], "CBI-S": ["CBI"],
    #                                                 "NDI": ["NDI-S", "CBI", "CBI-S"], "NDI-S": ["NDI", "CBI", "CBI-S"]})
    # )
    #
    # grid_plot(data=np.stack((pairwise_raw_p_correlations_cifar100_dist[:, :-1, -1].T,
    #                          pairwise_part_p_correlations_cifar100_dist[:, :-1, -1].T,), axis=0),
    #           axes_level_sns_func=sns.histplot,
    #           save_name_fig="correlation_histograms_cifar100.svg",
    #           title=r"Benchmark: $\Omega_{\text{CIFAR-100}}$",
    #           col_titles=[r"Raw $p$", r"Partial $p$"],
    #           row_titles=["T-SBP", "T-CBP", "T-CBP-S", "T-NDP", "T-NDP-S"],
    #           ylabel="Percentage(%)",
    #           index_y_label=(2, 0),
    #           subplots_kwargs=dict(figsize=(7.1, 8.8), sharex=True, sharey=True),
    #           sns_kwargs=dict(stat="percent", kde=True)
    # )
    #
    # for modules in [ALL_CIFAR, ALL_TINYIN]:
    #     for measures in [["SBI"], ["CBI"], ["CBI-S"], ["NDI"], ["NDI-S"], ["SBI", "CBI", "NDI"],
    #                      ["SBI", "CBI", "NDI-S"], ["SBI", "CBI-S", "NDI"], ["SBI", "CBI-S", "NDI-S"]]:
    #         results = standardized_linear_regression(modules, label_measures=measures+["CF"], y="CF")

    # adj_r2_cifar100_stats, coeffs_cifar100_stats, adj_r2_cifar100_dist, coeffs_cifar100_dist = (
    #     standardized_linear_regression_leave_one_out_hold(ALL_CIFAR_1, ["SBI", "CBI", "NDI-S", "CF"], y="CF"))
    #
    # adj_r2_tinyin_stats, coeffs_tinyin_stats, adj_r2_tinyin_dist, coeffs_tinyin_dist = (
    #     standardized_linear_regression_leave_one_out_hold(ALL_TINYIN_1, ["SBI", "CBI", "NDI-S", "CF"], y="CF"))
    #
    # print("he")
    ##################################Trashed code############################

    # sns.pointplot(data=tab, x="N_Reh", y="Val", hue="N_Cls", dodge=True, capsize=.1)
    # plt.title("Interaction Plot: N_Cls × N_Reh")
    # plt.show()

    # p_values = []
    # for i, (_, dist1) in enumerate(list(forg_range_cifar_ind.values())):
    #     for j, (_, dist2) in enumerate(list(forg_range_cifar_ind.values())):
    #         if j > i:
    #             t_stat, p_value = stats.ttest_ind(dist1, dist2, equal_var=False, random_state=123)
    #             p_values.append(p_value)
    # rej_holm, p_holm, _, _ = multipletests(np.asarray(p_values), alpha=0.05, method='holm')
    ##################################Trashed code############################