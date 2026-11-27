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
P_REHEARSAL_PER_CLASS = [8, 20, 40]
"""
Proportion of rehearsal samples per class stored at the end of each incremental step
"""

TINYIN_10_EXPS = [("tinyimagenet", "resnet18", 10, 20, r) for r in P_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing experiments run with tinyimagenet, resnet18, 10 increments of 20 classes each.
"""

TINYIN_5_EXPS = [("tinyimagenet", "resnet18", 5, 40, r) for r in P_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing experiments run with tinyimagenet, resnet18, 5 increments of 40 classes each.
"""

ALL_TINYIN = TINYIN_10_EXPS + TINYIN_5_EXPS
"""
All the keys for the modules
"""

BENCHMARKS = [("ALL_TINYIN_1", 1, ALL_TINYIN), ("ALL_TINYIN_2", 2, ALL_TINYIN), ("ALL_TINYIN_3", 3, ALL_TINYIN),
              ("ALL_TINYIN_4", 4, ALL_TINYIN)]
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

    N_TRAIN_SAMPLES_PER_CLASS: int
    """
    Number of training samples per class
    """

    N_REHEARSAL: int
    """
    Total number of samples in the rehearsal buffer after the first incremental step
    """

    N_CLASS_ORDERS: int
    """
    Number of class orders in the experiments
    """

    ACTUAL_N_CLASS_ORDERS: int
    """
    The actual number of class orders
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
    Mini-batch size after the first experience
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
    training forgetting of class j at the end of the current experience relative to its original training accuracy
    for the ith seed. The ith, jth element in the second numpy array is the test forgetting of class j
    at the end of the current experience relative to its original test accuracy for the ith seed. 
    """

    normalized_forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
    where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
    number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
    normalized training forgetting of class j at the end of the current experience relative to its original training
    accuracy for the ith seed. The ith, jth element in the second numpy array is the normalized test forgetting of class j
    at the end of the current experience relative to its original test accuracy for the ith seed. 
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

    softmax_prob: Dict[int, Dict[int, np.ndarray]]
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

    logits: Dict[int, Dict[int, np.ndarray]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
    index 0, is skipped. Each array is a 3D numpy array of size NxExC, where N is equal to the number of seeds, E is equal
    to the number of training epochs, and C is equal to the number of unique classes encountered up to the current
    experience. The ith, eth, jth element in each array is the average logit for class j computed across all training
    samples in the current experience (only the new classes) for the ith seed and prior to start training if eth is 0,
    after the eth training epoch otherwise. The last training epoch is skipped.  
    """

    collected_data: Dict[int, Dict[int, List[Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]]]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective lists as values. The first experience, the experience with
    index 0, is skipped.  Each list contains as many elements as the number of training epochs used for that experience.
    Each element is a dictionary collected by the class `LLIFCDataCollection` (have a look at that class to see the content
    of each dictionary) (the `logits` key and its corresponding value are removed from each dictionary to save space)
    at the end of each epoch during that experience and prior to performing training on that experience.
    The last epoch is skipped.
    """
##################################Classes############################


##################################Collect the experiment modules############################

modules_dict: Dict[Tuple[str, str, int, int, int], ExperimentModule] = {}
"""
A dictionary containing 5-tuples as keys and respective modules as values.
The first element in each  5-tuple is the name of the dataset used by the respective module;
the second element is the name of the network architecture;
the third element is the number of total experiences;
the fourth element is the number of classes per experience;
the fifth element is the percentage of rehearsal samples stored at the end of each incremental step per class
"""
# fill `modules_dict` with the experiment modules present in this folder
for name in os.listdir():
    if name.endswith('data.py') and "tinyimagenet" in name:
        name = name.removesuffix(".py")
        module: ExperimentModule = importlib.import_module(name)
        key = (module.DATASET, module.NETWORK, module.N_EXPS_TOTAL, module.N_CLASSES_PER_EXP,
               int(module.N_REHEARSAL / module.N_CLASSES_PER_EXP / module.N_TRAIN_SAMPLES_PER_CLASS * 100))
        modules_dict[key] = module
##################################Collect the experiment modules############################

##################################Utilities############################
def collect_measures(label_measures: Sequence[Union[str, Tuple[str, int]]],
                     index_exp: int,
                     module: ExperimentModule,
                     n_epochs: Optional[int] = None,
                     apply_only: Optional[Sequence[Union[str, Tuple[str, int]]]] = None) -> np.ndarray:
    """
    Collect the given measures from the given module.
    :param label_measures: a sequence of measures that need to be collected
    :param index_exp: index of the experience from which the given measures need to be collected
    :param module: a module that contains the given measures
    :param n_epochs: number of epochs to include for computation of the measures. It does not apply to `forgetting` and
        `normalized_forgetting`. When None, all epochs are used. Default is None.
    :param: apply_only: if `n_epochs` is not None, it is only applied to the measures included here. If None, `n_epochs`
        is applied to all measures. Default is None.
    :return: a 3D numpy array of size NxCxM, where N is the number of training runs used in the module, C is the number
        of classes used in the module and M is the number of measures collected. Note that N=OxS, where O is the number
        of class orders used in the module and S is the number of seeds used per class order in the module.
    """
    if len(label_measures) == 0:
        raise ValueError("`label_measures` cannot be empty")
    if len(set(label_measures)) != len(label_measures):
        raise ValueError("`label_measures` cannot contain duplicates")
    if not all(isinstance(x, str) or (isinstance(x, tuple) and len(x) == 2) for x in label_measures):
        raise ValueError("Each element in `label_measures` must be either a string or a 2-tuple")
    if index_exp < 1:
        raise ValueError("`index_exp` must be greater than or equal to 1")
    if n_epochs is not None and n_epochs < 1:
        raise ValueError("`n_epochs` must be greater than or equal to 1 when it is not None")

    if apply_only is None:
        apply_only = [x for x in label_measures]

    measures_list: List[np.ndarray] = []

    for class_order in range(module.ACTUAL_N_CLASS_ORDERS):
        for seed in range(module.N_SEEDS):
            measures = []
            # append the measures in label_measures
            for l in label_measures:
                if l == "forgetting":
                    measure = module.forgetting[class_order][index_exp - 1][1][seed]
                elif l == "normalized_forgetting":
                    measure = module.normalized_forgetting[class_order][index_exp - 1][1][seed]
                elif l in ["softmax_prob", "logits"]:
                    d = module.softmax_prob if l == "softmax_prob" else module.logits
                    raw_measure = d[class_order][index_exp][seed, :, :module.N_CLASSES_PER_EXP*index_exp]
                    if n_epochs is None or l not in apply_only:
                        measure = raw_measure.sum(axis=0)
                    else:
                        measure = raw_measure[:n_epochs, :].sum(axis=0)
                elif l == "rel_sic_step_reh":
                    d_list = module.collected_data[class_order][index_exp]
                    sic_step_reh = np.stack([d["sic_step_reh"][0] for d in d_list], axis=0)
                    train_dataset_avg_grads_norm = np.stack([d["train_dataset_avg_grads_norm"][:module.N_CLASSES_PER_EXP*index_exp]
                                                             for d in d_list], axis=0)
                    if n_epochs is not None and l in apply_only:
                        sic_step_reh = sic_step_reh[:n_epochs]
                        train_dataset_avg_grads_norm = train_dataset_avg_grads_norm[:n_epochs]

                    measure = np.sum(sic_step_reh, axis=0) / np.sum(train_dataset_avg_grads_norm, axis=0)

                elif l == "rehearsal_projection":
                    d_list = module.collected_data[class_order][index_exp]
                    rehearsal_avg_grads_norm = np.stack([d["rehearsal_avg_grads_norm"] for d in d_list], axis=0)
                    cos_train_rehearsal_avg_grads = np.stack([d["cos_train_rehearsal_avg_grads"] for d in d_list], axis=0)
                    if n_epochs is not None and l in apply_only:
                        rehearsal_avg_grads_norm = rehearsal_avg_grads_norm[:n_epochs]
                        cos_train_rehearsal_avg_grads = cos_train_rehearsal_avg_grads[:n_epochs]
                    measure = (rehearsal_avg_grads_norm * cos_train_rehearsal_avg_grads).sum(axis=0)

                else:
                    d_list = module.collected_data[class_order][index_exp]
                    name = l if isinstance(l, str) else l[0]
                    if name in ["indices", "mb_indices", "orig_targets"]:
                        raise ValueError(f"`{name}` is a valid label measure but it is not supported")
                    if name not in d_list[0].keys():
                        raise ValueError(f"`{name}` is not a valid label measure")
                    value_list = [d[name] for d in d_list]
                    if isinstance(l, str):
                        if not isinstance(value_list[0], np.ndarray):
                            raise ValueError("A single string was provided for the given measure but the values "
                                             "associated with the given measure are tuples")
                    else:
                        if not isinstance(value_list[0], tuple):
                            raise ValueError("A tuple was provided for the given measure but the values associated with "
                                             "the given measure are just numpy arrays")

                    if not isinstance(l, str):
                        value_list = [v[l[1]] for v in value_list]

                    raw_measure = np.stack(value_list, axis=0)[:, :module.N_CLASSES_PER_EXP*index_exp]
                    if n_epochs is not None and l in apply_only:
                        raw_measure = raw_measure[:n_epochs]

                    measure = np.nansum(raw_measure, axis=0)

                measures.append(measure)

            measures = np.column_stack(measures)
            measures_list.append(measures)

    measures_list = np.stack(measures_list, axis=0)
    return measures_list

def forgetting_metric(module_keys: Sequence[Tuple[str, str, int, int, int]],
                      forgetting_type: Literal["forgetting", "normalized_forgetting"],
                      index_exp: int,
                      metric_name: Literal["r", "hsg"],
                      method: str = "BCa",
                      confidence: float = 0.95,
                      n_resamples: int = 1_000_000,
                      random_seed: int = 123,
                      **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute the mean and std of the given metric across all the experiments in the given modules. The CIs are
    computed as well for both the mean and std. The CI for the mean is computed using the t-distribution, whereas the CI
    for the std is computed using bootstrap. The metrics supported are the forgetting range (`r`) and the half-split
    forgetting gap (`hsg`). Both of these metrics support both `forgetting` and `normalized_forgetting`.
    """
    forg_list = [collect_measures([forgetting_type], index_exp, modules_dict[key]).squeeze()
                 for key in module_keys]
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

def correlation(module_keys: Sequence[Tuple[str, str, int, int, int]],
                label_measures: Sequence[Union[str, Tuple[str, int]]],
                index_exp: int,
                corr_op: Union[Literal["pearson"], Literal["spearman"]],
                corr_type: Union[Literal["raw"], Literal["partial"]],
                non_included_covars: Optional[Dict[Union[str, Tuple[str, int]], Iterable[Union[str, Tuple[str, int]]]]] = None,
                method: str = "BCa",
                confidence: float = 0.95,
                n_resamples: int = 1_000_000,
                random_seed: int = 123,
                n_epochs: Optional[int] = None,
                apply_only: Optional[Sequence[Union[str, Tuple[str, int]]]] = None,
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
    :param n_epochs: number of epochs to include for computation of the measures. It does not apply to `forgetting` and
        `normalized_forgetting`. When None, all epochs are used. Default is None.
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

    measures_list = [collect_measures(label_measures=label_measures, index_exp=index_exp, module=modules_dict[key],
                                      n_epochs=n_epochs, apply_only=apply_only)
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

def mixed_effects_linear_regression_leave_one_out(module_keys: Sequence[Tuple[str, str, int, int, int]],
                                                  label_measures: Sequence[Union[str, Tuple[str, int]]],
                                                  index_exp: int,
                                                  y: Union[str, Tuple[str, int]],
                                                  n_epochs: Optional[int] = None,
                                                  ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if y not in label_measures:
        raise ValueError("The dependent variable must be contained in `label_measures`")

    def name_convert(x):
        if isinstance(x, str):
            return x
        else:
            return x[0] + str(x[1])

    measures_list = [collect_measures(label_measures=label_measures, index_exp=index_exp, module=modules_dict[key],
                                      n_epochs=n_epochs) for key in module_keys]

    label_measures = [name_convert(x) for x in label_measures]
    y = name_convert(y)
    independent_vars = [x for x in label_measures if x != y]


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

def swarm_plot_forgetting(module_keys: Sequence[Tuple[str, str, int, int, int]],
                          n_exps: int,
                          index_exp: int,
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
    :param index_exp: index of experience
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
    forg_list = [collect_measures(["normalized_forgetting"], index_exp=index_exp,
                                  module=modules_dict[key]).squeeze() for key in module_keys]
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
def normalized_forgetting_stats():
    d_r = {}
    for name, index_exp, benchmark in BENCHMARKS:
        d_r[(name, index_exp)] = forgetting_metric(module_keys=benchmark, forgetting_type="normalized_forgetting",
                                                   index_exp=index_exp, metric_name="r")

    d_hsg = {}
    for name, index_exp, benchmark in BENCHMARKS:
        d_hsg[(name, index_exp)] = forgetting_metric(module_keys=benchmark, forgetting_type="normalized_forgetting",
                                                     index_exp=index_exp, metric_name="hsg")

    print("END")

def forgetting_stats():
    d_r = {}
    for name, index_exp, benchmark in BENCHMARKS:
        d_r[(name, index_exp)] = forgetting_metric(module_keys=benchmark, forgetting_type="forgetting",
                                                   index_exp=index_exp, metric_name="r")

    d_hsg = {}
    for name, index_exp, benchmark in BENCHMARKS:
        d_hsg[(name, index_exp)] = forgetting_metric(module_keys=benchmark, forgetting_type="forgetting",
                                                     index_exp=index_exp, metric_name="hsg")

    print("END")

def raw_spearman_correlation_stats():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=[("sic_step_reh", 0), ("cic_step_reh", 0),
                                                           ("nic_step_train", 0), "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="raw"
                                           )[0]
    print("END")

def partial_spearman_correlation_stats():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=[("sic_step_reh", 0), ("cic_step_reh", 0),
                                                           ("nic_step_train", 0), "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="partial",
                                           )[0]
    print("END")

def mixed_effects_linear_regression_stats():
    d_all = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d_all[(name, index_exp)] = mixed_effects_linear_regression_leave_one_out(
            module_keys=benchmark,
            label_measures=[("sic_step_reh", 0),("cic_step_reh", 0),("nic_step_train", 0), "normalized_forgetting"],
            index_exp=index_exp,
            y="normalized_forgetting")

    print("END")

def forgetting_normalized_forgetting_stats():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=[("sic_step_reh", 0), "forgetting", "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="raw"
                                           )[0]
        print(d[(name, index_exp)])

    print("END")

def correlation_line_plots_across_steps():
    imagenet_benchmarks = [(ALL_TINYIN_1, 1), (ALL_TINYIN_2, 1), (ALL_TINYIN_2, 2)]
    imagenet_path = "SIC_CIC_NIC_across_steps_imagenet.pkl"
    dfs_imagenet = []

    if os.path.exists(imagenet_path):
        with open(imagenet_path, "rb") as f:
            dfs_imagenet = pickle.load(f)
    else:
        for b, i in imagenet_benchmarks:
            SBI_corr = []
            SBI_part_corr = []
            CBI_corr = []
            CBI_part_corr = []
            NDI_corr = []
            NDI_part_corr = []
            steps = []
            for n_epochs in range(1, 101):
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
               title=r"$\Omega_{\mathrm{C100}}$", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{C100}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{C100}}$",
                           r"$\Omega^3_{\mathrm{C100}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)

    line_plots(dfs=dfs_tinyin, x="steps", y="values",
               estimator=lambda x: inverse_fisher_z_transform(np.mean(fisher_z_transform(x))),
               errorbar="sd", name_fig="SIC_CIC_NIC_across_steps_tinyin.pdf",
               maxtick=101, xlabel="R-SGD steps", hue="group",
               title=r"$\Omega_{\mathrm{TIN}}$", col_titles=["SIC", "CIC", "NIC"],
               row_titles=[r"$\Omega^2_{\mathrm{TIN}}$", r"$\Omega^{2;\,\{10\%, 20\%\}}_{\mathrm{TIN}}$",
                           r"$\Omega^3_{\mathrm{TIN}}$"],
               figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3)


def sic_analysis():
    for name, index_exp, benchmark in BENCHMARKS:
        for b in benchmark:
            module = modules_dict[b]
            for class_order in range(module.ACTUAL_N_CLASS_ORDERS):
                for seed in range(module.N_SEEDS):
                    for epoch in range(module.N_EPOCHS):
                        forgetting = module.normalized_forgetting[class_order][index_exp - 1][1][seed]
                        sic = module.collected_data[class_order][index_exp][epoch]["sic_step_reh"][0][:module.N_CLASSES_PER_EXP*index_exp]
                        orig_norm = module.collected_data[class_order][index_exp][epoch]["train_dataset_avg_grads_norm"][:module.N_CLASSES_PER_EXP*index_exp]
                        reh_norm = module.collected_data[class_order][index_exp][epoch]["rehearsal_avg_grads_norm"][:module.N_CLASSES_PER_EXP*index_exp]
                        cosine = module.collected_data[class_order][index_exp][epoch]["cos_train_rehearsal_avg_grads"]
                        print(f"{epoch}: {spearmanr(sic/orig_norm, forgetting)}")
                        print(f"{epoch}: {spearmanr(cosine, forgetting)}")
                        print(f"{epoch}: {spearmanr(reh_norm / orig_norm, forgetting)}")
                        plt.scatter(sic / orig_norm, forgetting)
                        plt.show()

def raw_spearman_correlation_stats_softmax_logit():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=["logits", "softmax_prob", "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="raw", n_epochs=1
                                           )[0]
    print("END")

def raw_spearman_correlation_stats_nic_all_nic():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=[("nic_step_train", 0), ("all_nic_step_train", 0),
                                                           "train_dataset_avg_grads_norm", "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="raw",
                                           )[0]
    print("END")

def raw_spearman_correlation_stats_OM_RI():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=["train_dataset_avg_grads_norm",
                                                           "rehearsal_projection", "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="raw",
                                           )[0]
    print("END")

def partial_spearman_correlation_stats_all_nic():
    d = {}
    for name, index_exp, benchmark in BENCHMARKS[2:]:
        d[(name, index_exp)] = correlation(module_keys=benchmark,
                                           label_measures=[("sic_step_reh", 0), ("cic_step_reh", 0),
                                                           ("nic_step_train", 0), ("all_nic_step_train", 0),
                                                           "normalized_forgetting"],
                                           index_exp=index_exp, corr_op="spearman", corr_type="partial",
                                           non_included_covars={("nic_step_train", 0): [("all_nic_step_train", 0)],
                                                                ("all_nic_step_train", 0): [("nic_step_train", 0)]}
                                           )[0]
    print("END")

##################################Actual data############################

if __name__ == "__main__":
    #normalized_forgetting_stats()
    #forgetting_stats()
    #raw_spearman_correlation_stats()
    #partial_spearman_correlation_stats()
    #mixed_effects_linear_regression_stats()
    #partial_spearman_correlation_stats2()
    #raw_spearman_correlation_stats2()
    forgetting_normalized_forgetting_stats()
    #raw_spearman_correlation_stats_softmax_logit()
    #raw_spearman_correlation_stats_nic_all_nic()
    #raw_spearman_correlation_stats_OM_RI()
    #partial_spearman_correlation_stats_all_nic()
    # swarm_plot_forgetting(ALL_IMGNT_1, n_exps=5, index_exp=1, name_fig="imagenet_1.pdf", show=True,
    # title=r"$\Omega_{\mathrm{IMN}}^2$",
    # col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    # row_titles=['10% Classes', '20% Classes', '50% Classes'],
    # figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )

    # swarm_plot_forgetting(ALL_IMGNT_2, n_exps=5, index_exp=2, name_fig="imagenet_2.pdf", show=True,
    #      title=r"$\Omega_{\mathrm{IMN}}^3$",
    #      col_titles=['8% Rehearsal', '20% Rehearsal', '40% Rehearsal'],
    #      row_titles=['10% Classes', '20% Classes', ''],
    #      figsize=(7.1, 8.8), sharey=True, nrows=3, ncols=3
    # )

