"""
This module includes a set of functions for the computation of the Last-Layer Imbalanced Forgetting Coefficients.
"""
from __future__ import annotations
from typing import Union, Tuple, Optional, TYPE_CHECKING
from llifc.utils import (all_array_tensor, all_with_len, all_with_shape, interference, get_xp,
                         grads_wrt_weights_biases_specific_class)

import torch
import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray


def SIC_step(rehearsal_mb_avg_grads: Union[Tensor, ndarray],
             train_dataset_avg_grads: Union[Tensor, ndarray],
             weight: Optional[Union[Tensor, ndarray]] = None,
             include_norm_cos: bool = False) \
        -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the SIC_step value for each past class at the current point in optimization space. SIC_step derives
    directly from the SIC coefficient in "Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental
    Learning". For each past class, it computes the bias vector between its average rehearsal/mini-batch gradient and
    the average original gradient, and then measures the interference (interference operator) of the latter onto the
    average original gradient. Unless `weight` is provided, the interference is not multiplied by any factor.
    :param rehearsal_mb_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over all the samples of that class currently stored in the rehearsal buffer or all the samples of that class in
        the current mini-batch.
    :param train_dataset_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over the entire training dataset of that class.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the bias vector of each past class and the cosine
        similarity between the bias vector of each past class and its average original gradient. Default is False.
    :return: a 1D array/tensor containing the SIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the SIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the bias vector of each past class, and a 1D
        array/tensor containing the cosine similarity between the bias vector of each past class and its average
        original gradient.

    .note::
        `rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, and `weight` (if not None) must be all numpy arrays or
        pytorch tensors, and have the same length. Additionally, `rehearsal_mb_avg_grads` and
        `train_dataset_avg_grads` must also have the same shape.
    """
    if not all_array_tensor(rehearsal_mb_avg_grads, train_dataset_avg_grads,
                            train_dataset_avg_grads if weight is None else weight):
        raise TypeError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, and `weight` (if not None) must be "
                        "all numpy arrays or pytorch tensors.")
    if (rehearsal_mb_avg_grads.ndim != 2 or train_dataset_avg_grads.ndim != 2 or
        (weight is not None and weight.ndim != 1)):
        raise ValueError("`rehearsal_mb_avg_grads` must be 2D; `train_dataset_avg_grads` must be 2D; `weight` must be "
                         "1D (if not None)")
    if not all_with_len(rehearsal_mb_avg_grads, train_dataset_avg_grads,
                        train_dataset_avg_grads if weight is None else weight):
        raise ValueError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, and `weight` (if not None) must all "
                        "have the same length.")
    if not all_with_shape(rehearsal_mb_avg_grads, train_dataset_avg_grads):
        raise ValueError("`rehearsal_mb_avg_grads` and `train_dataset_avg_grads` must have the same shape.")

    # compute the bias vector for each past class
    bias = rehearsal_mb_avg_grads - train_dataset_avg_grads
    interf = interference(bias, train_dataset_avg_grads, include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight


def NIC_step(train_dataset_mb_new_avg_grad: Union[Tensor, ndarray],
             train_dataset_past_avg_grads: Union[Tensor, ndarray],
             num_classes: int,
             weight: Optional[Union[Tensor, ndarray]] = None,
             include_norm_cos: bool = False) \
    -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the NIC_step value for each past class at the current point in optimization space. NIC_step derives
    directly from the NIC coefficient in "Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental
    Learning". For each past class, it computes the interference (interference operator) of the average
    original/mini-batch gradient of new classes onto the average original gradient of that class, both w.r.t. the
    weights and bias associated with the corresponding past class.
    Unless `weight` is provided, the interference is not multiplied by any factor.
    :param train_dataset_mb_new_avg_grad: a 1D array/tensor representing the average gradient of new classes, evaluated
        at the current point in optimization space over the entire training dataset of new classes or all the samples
        belonging to new classes in the current mini-batch.
    :param train_dataset_past_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where
        each ith row contains the average gradient for past class i, evaluated at the current point in optimization
        space over the entire training dataset of that class.
    :param num_classes: the total number of classes.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the average original/mini-batch gradient of new
        classes w.r.t. the weights and bias associated with each specific past class and the cosine similarity between
        the average original/mini-batch gradient of new classes w.r.t. the weights and bias associated with each
        specific past class and the average original gradient of the respective past class w.r.t. its specific weights
        and bias. Default is False.
    :return: a 1D array/tensor containing the NIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the NIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the average original/mini-batch gradient of new
        classes w.r.t. the weights and bias associated with each specific past class, and a 1D array/tensor containing
        the cosine similarity between the average original/mini-batch gradient of new classes w.r.t. the weights and
        bias associated with each specific past class and the average original gradient of the respective past class
        w.r.t. its specific weights and bias.

    .note::
        `train_dataset_mb_new_avg_grad`, `train_dataset_past_avg_grads`, and `weight` (if not None) must all be numpy
        arrays or pytorch tensors

    .note::
        Since NIC_step derives directly from NIC, the gradients in `train_dataset_mb_new_avg_grad` and
        `train_dataset_past_avg_grads` are assumed to be w.r.t. all weights and biases in the last layer.
        The order of the elements in each of the provided gradients must be as
        follows: W00, W01,..., W0(F-1), W10, W11,..., W1(F-1),...,W(C-1)0,...,W(C-1)(F-1),B0, B1,..., B(C-1),
        where F is the size of the embeddings (feature vectors), C the number of classes, WMN denotes the mth, nth
        element of the weight matrix, and BM denotes the mth element of the bias vector.
    """
    if not all_array_tensor(train_dataset_mb_new_avg_grad, train_dataset_past_avg_grads,
                            train_dataset_past_avg_grads if weight is None else weight):
        raise TypeError("`train_dataset_mb_new_avg_grad`, `train_dataset_past_avg_grads`, and `weight` (if not None) "
                        "must all be numpy arrays or pytorch tensors.")
    if (train_dataset_mb_new_avg_grad.ndim != 1 or train_dataset_past_avg_grads.ndim != 2 or
        (weight is not None and weight.ndim != 1)):
        raise ValueError("`train_dataset_mb_new_avg_grad` must be 1D; `train_dataset_past_avg_grads` must be 2D; "
                         "`weight` must be 1D (if not None)")
    if len(train_dataset_mb_new_avg_grad) != train_dataset_past_avg_grads.shape[1]:
        raise ValueError("The size of the gradients in `train_dataset_mb_new_avg_grad` and "
                         "`train_dataset_past_avg_grads` must be the same")
    if weight is not None and (not all_with_len(train_dataset_past_avg_grads, weight)):
        raise ValueError("`train_dataset_past_avg_grads` and `weight` (if not None) must have the same length")
    if num_classes <= 0:
        raise ValueError("`num_classes` must be strictly greater than 0")
    # get the number of past classes
    n_past_classes = len(train_dataset_past_avg_grads)
    if num_classes <= n_past_classes:
        raise ValueError("`num_classes` must be strictly greater than the number of past classes inferable from "
                         "`train_dataset_past_avg_grads`")

    # a 1D array/tensor containing the targets of past classes in ascending order
    y = get_xp(train_dataset_past_avg_grads).arange(n_past_classes)
    if isinstance(train_dataset_past_avg_grads, torch.Tensor):
        y = y.to(train_dataset_past_avg_grads.device)
    # get the average original gradient of each past class w.r.t. its associated weights and bias
    train_dataset_past_avg_grads = grads_wrt_weights_biases_specific_class(train_dataset_past_avg_grads, y,
                                                                           num_classes)
    # convert `train_dataset_mb_new_avg_grad` into a 2D array/tensor with `n_past_classes` rows, each containing the
    # average original/mini-batch gradient of new classes
    if isinstance(train_dataset_mb_new_avg_grad, torch.Tensor):
        train_dataset_mb_new_avg_grad = train_dataset_mb_new_avg_grad.unsqueeze(0).expand(n_past_classes,-1)
    else:
        train_dataset_mb_new_avg_grad = np.broadcast_to(train_dataset_mb_new_avg_grad,
                                                        (n_past_classes, len(train_dataset_mb_new_avg_grad)))

    # get the average original/mini-batch gradient of new classes w.r.t. the weights and bias associated with each past
    # class
    train_dataset_mb_new_avg_grad = grads_wrt_weights_biases_specific_class(train_dataset_mb_new_avg_grad, y,
                                                                            num_classes)
    interf = interference(train_dataset_mb_new_avg_grad, train_dataset_past_avg_grads,
                          include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight


def ALL_NIC_step(train_dataset_mb_new_avg_grad: Union[Tensor, ndarray],
                 train_dataset_past_avg_grads: Union[Tensor, ndarray],
                 weight: Optional[Union[Tensor, ndarray]] = None,
                 include_norm_cos: bool = False) \
    -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the ALL_NIC_step value for each past class at the current point in optimization space. ALL_NIC_step derives
    directly from the ALL_NIC coefficient in "Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental
    Learning". For each past class, it computes the interference (interference operator) of the average
    original/mini-batch gradient of new classes onto the average original gradient of that class.
    Unless `weight` is provided, the interference is not multiplied by any factor.
    :param train_dataset_mb_new_avg_grad: a 1D array/tensor representing the average gradient of new classes, evaluated
        at the current point in optimization space over the entire training dataset of new classes or all the samples
        belonging to new classes in the current mini-batch.
    :param train_dataset_past_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where
        each ith row contains the average gradient for past class i, evaluated at the current point in optimization
        space over the entire training dataset of that class.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the average original/mini-batch gradient of new
        classes and the cosine similarity between that average gradient and the average original gradient of each past
        class. Default is False.
    :return: a 1D array/tensor containing the ALL_NIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the ALL_NIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the average original/mini-batch gradient of new
        classes (the same value repeated for each past class), and a 1D array/tensor containing the cosine similarity
        between the average original/mini-batch gradient of new classes and the average original gradient of each past
        class.

    .note::
        `train_dataset_mb_new_avg_grad`, `train_dataset_past_avg_grads`, and `weight` (if not None) must be all numpy
        arrays or pytorch tensors.
    """
    if not all_array_tensor(train_dataset_mb_new_avg_grad, train_dataset_past_avg_grads,
                            train_dataset_past_avg_grads if weight is None else weight):
        raise TypeError("`train_dataset_mb_new_avg_grad`, `train_dataset_past_avg_grads`, and `weight` (if not None) "
                        "must all be numpy arrays or pytorch tensors.")
    if (train_dataset_mb_new_avg_grad.ndim != 1 or train_dataset_past_avg_grads.ndim != 2 or
            (weight is not None and weight.ndim != 1)):
        raise ValueError("`train_dataset_mb_new_avg_grad` must be 1D; `train_dataset_past_avg_grads` must be 2D; "
                         "`weight` must be 1D (if not None)")
    if len(train_dataset_mb_new_avg_grad) != train_dataset_past_avg_grads.shape[1]:
        raise ValueError("The size of the gradients in `train_dataset_mb_new_avg_grad` and "
                         "`train_dataset_past_avg_grads` must be the same")
    if weight is not None and (not all_with_len(train_dataset_past_avg_grads, weight)):
        raise ValueError("`train_dataset_past_avg_grads` and `weight` (if not None) must have the same length")

    # get the number of past classes
    n_past_classes = len(train_dataset_past_avg_grads)
    # convert `train_dataset_mb_new_avg_grad` into a 2D array/tensor with `n_past_classes` rows, each containing the
    # average original/mini-batch gradient of new classes
    if isinstance(train_dataset_mb_new_avg_grad, torch.Tensor):
        train_dataset_mb_new_avg_grad = train_dataset_mb_new_avg_grad.unsqueeze(0).expand(n_past_classes,-1)
    else:
        train_dataset_mb_new_avg_grad = np.broadcast_to(train_dataset_mb_new_avg_grad,
                                                        (n_past_classes, len(train_dataset_mb_new_avg_grad)))

    interf = interference(train_dataset_mb_new_avg_grad, train_dataset_past_avg_grads,
                          include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight


def CIC_step(rehearsal_mb_avg_grads: Union[Tensor, ndarray],
             train_dataset_avg_grads: Union[Tensor, ndarray],
             n_rehearsal_mb: Union[Tensor, ndarray],
             n_train_dataset: Union[Tensor, ndarray],
             weight: Optional[Union[Tensor, ndarray]] = None,
             include_norm_cos: bool = False)\
        -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the CIC_step value for each past class at the current point in optimization space. CIC_step derives
    directly from the CIC coefficient in "Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental
    Learning". For each past class, it computes the bias vector between the average rehearsal/mini-batch gradient of all
    other past classes and the average original gradient of all other past classes, and then measures the interference
    (interference operator) of the latter onto the average original gradient of that class.
    Unless `weight` is provided, the interference is not multiplied by any factor.
    :param rehearsal_mb_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over all the samples of that class currently stored in the rehearsal buffer or all the samples of that class in
        the current mini-batch. If there are no samples in the current rehearsal buffer or mini-batch for a given past
        class, any vector can be provided (including a vector of nan values), as this vector
        will not be considered in the computations.
    :param train_dataset_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over the entire training dataset of that class.
    :param n_rehearsal_mb: a 1D array/tensor containing as many elements as the number of past classes, where each ith
        element is the number of samples of class i currently stored in the rehearsal buffer or the number of samples
        of that class in the current mini-batch.
    :param n_train_dataset: a 1D array/tensor containing as many elements as the number of past classes, where each ith
        element is the number of samples in the entire training dataset of class i. It is assumed that the original
        training dataset of each past class contains at least one sample; a :class:`ValueError` is raised otherwise.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the bias vector of all other past classes for each
        past class and the cosine similarity between the bias vector of all other past classes and the average original
        gradient of each past class. Default is False.
    :return: a 1D array/tensor containing the CIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the CIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the bias vector of all other past classes
        for each past class, and a 1D array/tensor containing the cosine similarity between the bias vector of all
        other past classes and the average original gradient of each past class.

    .note::
        `rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`, `n_train_dataset`,
        and `weight` (if not None) must all be numpy arrays or pytorch tensors.

    .note::
        A :class:`ValueError` is raised if only one past class is present, as CIC_step is undefined when only one past
        class exists.

    .note::
        The CIC_step value for a past class is NaN if there are no samples in the current rehearsal buffer or
        mini-batch belonging to other past classes.
    """
    if not all_array_tensor(rehearsal_mb_avg_grads, train_dataset_avg_grads, n_rehearsal_mb, n_train_dataset,
                            n_train_dataset if weight is None else weight):
        raise TypeError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`, `n_train_dataset`, "
                        "and `weight`(if not None) must all be numpy arrays or pytorch tensors")
    if (rehearsal_mb_avg_grads.ndim != 2 or train_dataset_avg_grads.ndim != 2 or n_rehearsal_mb.ndim != 1 or
        n_train_dataset.ndim != 1 or (weight is not None and weight.ndim != 1)):
        raise ValueError("`rehearsal_mb_avg_grads` must be 2D; `train_dataset_avg_grads` must be 2D;"
                         "`n_rehearsal_mb` must be 1D; `n_train_dataset` must be 1D; `weight` must be 1D (if not None)")
    if not all_with_len(rehearsal_mb_avg_grads, train_dataset_avg_grads, n_rehearsal_mb, n_train_dataset,
                        n_train_dataset if weight is None else weight):
        raise ValueError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`, `n_train_dataset`, and "
                         "`weight` (if not None) must all have the same length")
    if not all_with_shape(rehearsal_mb_avg_grads, train_dataset_avg_grads):
        raise ValueError("`rehearsal_mb_avg_grads` and `train_dataset_avg_grads` must have the same shape")
    if len(rehearsal_mb_avg_grads) < 2:
        raise ValueError("At least two past classes must be present")
    if (n_rehearsal_mb < 0).any():
        raise ValueError("`n_rehearsal_mb` must contain positive values (0 included) only")
    if (n_train_dataset <= 0).any():
        raise ValueError("`n_train_dataset` must contain strictly positive values (>0) only")

    # multiply the gradient of each class in `train_dataset_avg_grads` by the corresponding number of samples in the
    # original training dataset of that class
    weighted_train_dataset_avg_grads = n_train_dataset[:, None] * train_dataset_avg_grads
    # compute the average original gradient of all other past classes for each past class, i.e. the vector at index i
    # represents the average original gradient of all other past classes for past class i
    others_train_dataset_avg_grads = ((weighted_train_dataset_avg_grads.sum(0) - weighted_train_dataset_avg_grads) /
                                      (n_train_dataset.sum() - n_train_dataset)[:, None])
    # create a 2D array/tensor replacing the average gradients in `rehearsal_mb_avg_grads` of past classes containing no
    # samples in the current rehearsal buffer or mini-batch with 0 vectors (this is performed because those average
    # gradients might be NaN vectors, giving wrongful results for the subsequent operations); the other average
    # gradients are not changed
    rehearsal_mb_avg_grads = get_xp(rehearsal_mb_avg_grads).where((n_rehearsal_mb == 0)[:, None], 0.,
                                                                  rehearsal_mb_avg_grads)
    # multiply the gradient of each class in `rehearsal_mb_avg_grads` by the corresponding number of samples in the
    # current rehearsal buffer or mini-batch
    weighted_rehearsal_mb_avg_grads = n_rehearsal_mb[:, None] * rehearsal_mb_avg_grads
    # compute the average rehearsal/mini-batch gradient of all other past classes for each past class, i.e. the vector
    # at index i represents the average rehearsal/mini-batch gradient of all other past classes for past class i
    others_rehearsal_mb_avg_grads = ((weighted_rehearsal_mb_avg_grads.sum(0) - weighted_rehearsal_mb_avg_grads) /
                                     (n_rehearsal_mb.sum() - n_rehearsal_mb)[:, None])
    bias = others_rehearsal_mb_avg_grads - others_train_dataset_avg_grads
    interf = interference(bias, train_dataset_avg_grads, include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight


def OIC_step(train_dataset_avg_grads: Union[Tensor, ndarray],
             n_train_dataset: Union[Tensor, ndarray],
             weight: Optional[Union[Tensor, ndarray]] = None,
             include_norm_cos: bool = False)\
        -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the OIC_step value for each past class at the current point in optimization space.
    For each past class, it computes the interference (interference operator) of the average original gradient of all
    other past classes onto the average original gradient of that class.
    Unless `weight` is provided, the interference is not multiplied by any factor.
    :param train_dataset_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over the entire training dataset of that class.
    :param n_train_dataset: a 1D array/tensor containing as many elements as the number of past classes, where each ith
        element is the number of samples in the entire training dataset of class i. It is assumed that the original
        training dataset of each past class contains at least one sample; a :class:`ValueError` is raised otherwise.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the average original gradient of all
        other past classes for each past class and the cosine similarity between the average original gradient of all
        other past classes and the average original gradient of each past class. Default is False.
    :return: a 1D array/tensor containing the OIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the OIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the average original gradient of all
        other past classes for each past class, and a 1D array/tensor containing the cosine similarity between the
        average original gradient of all other past classes and the average original gradient of each past class

    .note::
        `train_dataset_avg_grads`, `n_train_dataset`, and `weight` (if not None) must all be numpy arrays or pytorch
        tensors.

    .note::
        A :class:`ValueError` is raised if only one past class is present, as OIC_step is undefined when only one past
        class exists.
    """
    if not all_array_tensor(train_dataset_avg_grads, n_train_dataset,
                            n_train_dataset if weight is None else weight):
        raise TypeError("`train_dataset_avg_grads`, `n_train_dataset`, and `weight`(if not None) must all be numpy "
                        "arrays or pytorch tensors")
    if (train_dataset_avg_grads.ndim != 2 or n_train_dataset.ndim != 1 or (weight is not None and weight.ndim != 1)):
        raise ValueError("`train_dataset_avg_grads` must be 2D; `n_train_dataset` must be 1D; `weight` must be 1D "
                         "(if not None)")
    if not all_with_len(train_dataset_avg_grads, n_train_dataset,
                        n_train_dataset if weight is None else weight):
        raise ValueError("`train_dataset_avg_grads`, `n_train_dataset`, and "
                         "`weight` (if not None) must all have the same length")
    if len(train_dataset_avg_grads) < 2:
        raise ValueError("At least two past classes must be present")
    if (n_train_dataset <= 0).any():
        raise ValueError("`n_train_dataset` must contain strictly positive values (>0) only")

    # multiply the gradient of each class in `train_dataset_avg_grads` by the corresponding number of samples in the
    # original training dataset of that class
    weighted_train_dataset_avg_grads = n_train_dataset[:, None] * train_dataset_avg_grads
    # compute the average original gradient of all other past classes for each past class, i.e. the vector at index i
    # represents the average original gradient of all other past classes for past class i
    others_train_dataset_avg_grads = ((weighted_train_dataset_avg_grads.sum(0) - weighted_train_dataset_avg_grads) /
                                      (n_train_dataset.sum() - n_train_dataset)[:, None])

    interf = interference(others_train_dataset_avg_grads, train_dataset_avg_grads, include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight


def RIC_step(rehearsal_mb_avg_grads: Union[Tensor, ndarray],
             train_dataset_avg_grads: Union[Tensor, ndarray],
             n_rehearsal_mb: Union[Tensor, ndarray],
             weight: Optional[Union[Tensor, ndarray]] = None,
             include_norm_cos: bool = False)\
        -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray]]]:
    """
    Compute the RIC_step value for each past class at the current point in optimization space. For each past class,
    it computes the interference (interference operator) of the average rehearsal/mini-batch gradient of all
    other past classes onto the average original gradient of that class.
    Unless `weight` is provided, the interference is not multiplied by any factor.
    :param rehearsal_mb_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over all the samples of that class currently stored in the rehearsal buffer or all the samples of that class in
        the current mini-batch. If there are no samples in the current rehearsal buffer or mini-batch for a given past
        class, any vector can be provided (including a vector of nan values), as this vector
        will not be considered in the computations.
    :param train_dataset_avg_grads: a 2D array/tensor containing as many rows as the number of past classes, where each
        ith row contains the average gradient for past class i, evaluated at the current point in optimization space
        over the entire training dataset of that class.
    :param n_rehearsal_mb: a 1D array/tensor containing as many elements as the number of past classes, where each ith
        element is the number of samples of class i currently stored in the rehearsal buffer or the number of samples
        of that class in the current mini-batch.
    :param weight: a 1D array/tensor, one element for each past class, weighting each interference (weight*interference).
        If None, no weighting is applied to the interference measured on each past class. Default is None.
    :param include_norm_cos: whether to also return the L2 norm of the average rehearsal/mini-batch gradient of all
        other past classes for each past class and the cosine similarity between the average rehearsal/mini-batch
        gradient of all other past classes and the average original gradient of each past class. Default is False.
    :return: a 1D array/tensor containing the RIC_step value for each past class if `include_norm_cos` is False.
        Otherwise, a tuple containing the three following elements: a 1D array/tensor containing the RIC_step value
        for each past class, a 1D array/tensor containing the L2 norm of the average rehearsal/mini-batch
        gradient of all other past classes for each past class, and a 1D array/tensor containing the cosine similarity
        between the average rehearsal/mini-batch gradient of all other past classes and the average original gradient
        of each past class.

    .note::
        `rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`,
        and `weight` (if not None) must all be numpy arrays or pytorch tensors.

    .note::
        A :class:`ValueError` is raised if only one past class is present, as RIC_step is undefined when only one past
        class exists.

    .note::
        The RIC_step value for a past class is NaN if there are no samples in the current rehearsal buffer or
        mini-batch belonging to other past classes.
    """
    if not all_array_tensor(rehearsal_mb_avg_grads, train_dataset_avg_grads, n_rehearsal_mb,
                            n_rehearsal_mb if weight is None else weight):
        raise TypeError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`,  "
                        "and `weight`(if not None) must all be numpy arrays or pytorch tensors")
    if (rehearsal_mb_avg_grads.ndim != 2 or train_dataset_avg_grads.ndim != 2 or n_rehearsal_mb.ndim != 1 or
        (weight is not None and weight.ndim != 1)):
        raise ValueError("`rehearsal_mb_avg_grads` must be 2D; `train_dataset_avg_grads` must be 2D;"
                         "`n_rehearsal_mb` must be 1D; `weight` must be 1D (if not None)")
    if not all_with_len(rehearsal_mb_avg_grads, train_dataset_avg_grads, n_rehearsal_mb,
                        n_rehearsal_mb if weight is None else weight):
        raise ValueError("`rehearsal_mb_avg_grads`, `train_dataset_avg_grads`, `n_rehearsal_mb`,  and "
                         "`weight` (if not None) must all have the same length")
    if not all_with_shape(rehearsal_mb_avg_grads, train_dataset_avg_grads):
        raise ValueError("`rehearsal_mb_avg_grads` and `train_dataset_avg_grads` must have the same shape")
    if len(rehearsal_mb_avg_grads) < 2:
        raise ValueError("At least two past classes must be present")
    if (n_rehearsal_mb < 0).any():
        raise ValueError("`n_rehearsal_mb` must contain positive values (0 included) only")

    # create a 2D array/tensor replacing the average gradients in `rehearsal_mb_avg_grads` of past classes containing no
    # samples in the current rehearsal buffer or mini-batch with 0 vectors (this is performed because those average
    # gradients might be NaN vectors, giving wrongful results for the subsequent operations); the other average
    # gradients are not changed
    rehearsal_mb_avg_grads = get_xp(rehearsal_mb_avg_grads).where((n_rehearsal_mb == 0)[:, None], 0.,
                                                                  rehearsal_mb_avg_grads)
    # multiply the gradient of each class in `rehearsal_mb_avg_grads` by the corresponding number of samples in the
    # current rehearsal buffer or mini-batch
    weighted_rehearsal_mb_avg_grads = n_rehearsal_mb[:, None] * rehearsal_mb_avg_grads
    # compute the average rehearsal/mini-batch gradient of all other past classes for each past class, i.e. the vector
    # at index i represents the average rehearsal/mini-batch gradient of all other past classes for past class i
    others_rehearsal_mb_avg_grads = ((weighted_rehearsal_mb_avg_grads.sum(0) - weighted_rehearsal_mb_avg_grads) /
                                     (n_rehearsal_mb.sum() - n_rehearsal_mb)[:, None])
    interf = interference(others_rehearsal_mb_avg_grads, train_dataset_avg_grads, include_norm_cos=include_norm_cos)
    if weight is None:
        return interf
    if include_norm_cos:
        return interf[0] * weight, interf[1], interf[2]

    return interf * weight