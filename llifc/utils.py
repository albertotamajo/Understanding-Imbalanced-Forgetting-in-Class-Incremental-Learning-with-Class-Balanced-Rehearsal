"""
This module includes a set of utilities for the given package
"""
from __future__ import annotations
from typing import Union, Iterable, TYPE_CHECKING
from numbers import Integral

import torch
import numpy as np
from scipy.special import softmax

if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray
    from collections.abc import Sized


def all_with_len(*args: Sized) -> bool:
    """
    Check whether all arguments have the same length.
    :param args: some arguments
    :return: True if all arguments have the same length, or no argument is provided; False otherwise.
    """
    if not args:  # If no arguments, return True
        return True

    first_len = len(args[0])
    return all(len(arg) == first_len for arg in args)


def all_array_tensor(*args) -> bool:
    """
    Check whether all arguments are numpy arrays or pytorch tensors.
    :param args: some arguments.
    :return: True if all arguments are numpy arrays, pytorch tensors, or no argument is provided; False otherwise.
    """
    if not args:  # If no arguments, return True
        return True

    return all(isinstance(arg, np.ndarray) for arg in args) or all(isinstance(arg, torch.Tensor) for arg in args)


def all_with_dim(*args: Union[Tensor, ndarray], dim: Union[Integral, Iterable[Integral]]) -> bool:
    """
    Check whether all provided numpy arrays or pytorch tensors have the same dimensionality and this is included
    in `dim`.
    :param args: some numpy arrays or pytorch tensors.
    :param dim: an iterable of dimensionalities or a single dimensionality.
    :return: True if all provided numpy arrays or pytorch tensors have the same dimensionality and this is included
        in `dim`, or no arguments are provided; False otherwise.
    """
    if not args:  # If no arguments, return True
        return True
    dim = {int(dim)} if isinstance(dim, Integral) else {int(d) for d in dim}
    first_ndim = args[0].ndim
    return first_ndim in dim and all(arg.ndim == first_ndim for arg in args)


def all_with_shape(*args: Union[Tensor, ndarray]) -> bool:
    """
    Check whether all provided numpy arrays or pytorch tensors have the same shape.
    :param args: some numpy arrays or pytorch tensors.
    :return: True if all provided numpy arrays or pytorch tensors have the same shape, or no arguments are provided;
    False otherwise.
    """
    if not args:  # If no arguments, return True
        return True
    first_shape = args[0].shape
    return  all(arg.shape == first_shape for arg in args)


def get_xp(x: Union[Tensor, ndarray]):
    """
    Get the `torch` module or the `np` module according to the provided input.
    :param x: input
    :return: the `torch` module if an instance of :class:`torch.Tensor` or the `np` module if an instance
        of :class:`np.ndarray`.

    .note::
        A :class:`TypeError` is raised if an instance of an unsupported type is provided.
    """
    if isinstance(x, torch.Tensor):
        return torch
    elif isinstance(x, np.ndarray):
        return np
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def interference(v1: Union[Tensor, ndarray],
                 v2: Union[Tensor, ndarray],
                 include_norm_cos: bool = False):
    """
    Compute the interference of `v1` onto `v2` (if both of them are 1D arrays/tensors) or the interference of each row
    in `v1` onto the corresponding row in `v2` (if both of them are 2D arrays/tensors).
    The interference of a vector onto another is computed as described in
    "Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental Learning" (interference operator), i.e.
    let a and b be two vectors, the interference of a onto b is measured as: -<a,b>/||b||.
    :param v1: a 1D or 2D array/tensor.
    :param v2: a 1D or 2D array/tensor.
    :param include_norm_cos: whether to also return the L2 norm of `v1` (if it is a 1D array/tensor) or the L2 norm of
        each row in `v1` (if it is a 2D array/tensor) and the cosine similarity between `v1` and `v2`
        (if both are 1D arrays/tensors) or the cosine similarity between  each row in `v1` and the corresponding row in
        `v2` (if both are 2D arrays/tensors). Default is `False`.
    :return: if `include_norm_cos` is False, the interference of `v1` onto `v2` (if both of them are 1D arrays/tensors)
        or the interference of each row in `v1` onto the corresponding row in `v2`
        (if both of them are 2D arrays/tensors). Otherwise, a tuple with the following three elements:
        the interference of `v1` onto `v2` (if both of them are 1D arrays/tensors) or the interference of each row in
        `v1` onto the corresponding row in `v2` (if both of them are 2D arrays/tensors), the L2 norm of `v1`
        (if it is a 1D array/tensor) or the L2 norm of each row in `v1` (if it is a 2D array/tensor), and the cosine
        similarity between `v1` and `v2` (if both are 1D arrays/tensors) or the cosine similarity between each row in
        `v1` and the corresponding row in `v2` (if both are 2D arrays/tensors).

    .note::
        `v1` and `v2` **must** be both numpy arrays or pytorch tensors, be both 1D or 2D and have the same shape.
    """
    if not all_array_tensor(v1, v2):
        raise TypeError("`v1` and `v2` must be both numpy arrays or pytorch tensors")
    if not all_with_dim(v1, v2, dim=[1, 2]):
        raise ValueError("`v1` and `v2` must be both 1D or 2D")
    if not all_with_shape(v1, v2):
        raise ValueError("`v1` and `v2` must have the same shape")

    xp = get_xp(v1)
    inn_product = (v1 * v2).sum(-1)  # sum over the last dimension/axis
    normv2 = xp.linalg.norm(v2, 2, -1)  # compute the L2-norm over the last dimension/axis
    interf = -inn_product / normv2
    if include_norm_cos:
        normv1 = xp.linalg.norm(v1, 2, -1)  # compute the L2-norm over the last dimension/axis
        cos_sim = inn_product / (normv1 * normv2)
        return interf, normv1, cos_sim

    return interf


def sample_wise_mce_grads_wrt_weights_biases(predicted_y: Union[Tensor, ndarray],
                                             true_y: Union[Tensor, ndarray],
                                             embeds: Union[Tensor, ndarray],
                                             logits: bool = True) -> Union[Tensor, ndarray]:
    """
    Compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the classification head
    (last layer) for each sample.
    :param predicted_y: the predicted raw logits if `logits` is True, they must be the raw logits before passing
        through the softmax layer, or the predicted softmax probabilities if `logits` is False. It must be a
        2D array/tensor.
    :param true_y: the ground truth target labels. It must be a 1D array/tensor.
    :param embeds: the embeddings, also known as feature vectors, being fed into the classification head. It must be a
        2D array/tensor.
    :param logits: if True `predicted_y` is assumed to contain raw logits, they must be the raw logits before passing
        through the softmax layer; otherwise it is assumed to contain softmax probabilities. Default is True.
    :return: a 2D array/tensor, i.e. the gradients of the multi-class cross-entropy loss w.r.t the weights and biases
        in the classification head (last layer) for each sample.

    .note::
        For each sample, the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
        classification head (last layer) is computed as follows: flatten((Y_hat - Y) H^T) || (Y_hat - Y),
        where Y_hat is the vector of softmax probabilities of the given sample, Y is the one-hot encoding vector of the
        true class for the given sample, H is the embedding vector of the given sample , also known as feature vector,
        being fed into the classification head, || is the concatenation operator (it concatenates the gradient wrt
        the weights[left] with the gradient wrt the bias[right]), and flatten is the classic operator flattening a 2D
        vector into a 1D vector. The flattening operation is performed using the standard row-major order, i.e. ....
        the order of the elements in the returned gradients for each sample is the following: W00, W01,..., W0(F-1),
        W10, W11,..., W1(F-1),...,W(C-1)0,...,W(C-1)(F-1),B0, B1,..., B(C-1), where F is the size of the feature
        vector, C the number of classes, WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth
        element of the bias vector.
    .note::
        The inputs must be all numpy arrays or pytorch tensors, and must have the same length.
    """
    if not all_array_tensor(predicted_y, true_y, embeds):
        raise TypeError("Inputs must be all numpy arrays or pytorch tensors")
    if predicted_y.ndim != 2 or true_y.ndim != 1 or embeds.ndim != 2:
        raise ValueError("`predicted_y` must be 2D; `true_y` must be 1D; `embeds` must be 2D")
    if not all_with_len(predicted_y, true_y, embeds):
        raise ValueError("`predicted_y`, `true_y` and `embeds` must have the same length")
    if (true_y < 0).any():
        raise ValueError("`true_y` must contain positive values (>=0) only")
    # get the number of classes from the size of the logit/softmax vector output for each sample
    num_classes = predicted_y.shape[1]
    if num_classes < true_y.max() + 1:
        raise ValueError("The number of classes inferable from `predicted_y` must be larger than or equal to the one "
                         "inferable from `true_y` (`true_y` might not contain all ground truth targets)")

    if isinstance(predicted_y, torch.Tensor):
        if logits:
            predicted_y = predicted_y.softmax(dim=1)
        gradients_biases = predicted_y - torch.eye(num_classes, device=predicted_y.device,
                                                   dtype=predicted_y.dtype)[true_y]
        gradients_weights = torch.bmm(gradients_biases.unsqueeze(2), embeds.unsqueeze(1))
        gradients_weights = gradients_weights.reshape(len(gradients_biases), -1)
        return torch.cat((gradients_weights, gradients_biases), dim=1)

    else:
        if logits:
            predicted_y = softmax(predicted_y, axis=1)
        gradients_biases = predicted_y - np.eye(num_classes, dtype=predicted_y.dtype)[true_y]
        gradients_biases_expand = np.expand_dims(gradients_biases, axis=2)
        embeds_expand = np.expand_dims(embeds, axis=1)
        grads_weights = np.matmul(gradients_biases_expand, embeds_expand)
        grads_weights = grads_weights.reshape(len(gradients_biases), -1)
        return np.concatenate((grads_weights, gradients_biases), axis=1)


def class_avg_grads(grads: Union[Tensor, ndarray],
                    true_y: Union[Tensor, ndarray],
                    num_classes: int) -> Union[Tensor, ndarray]:
    """
    Compute the average gradient for each class from sample-wise gradients.
    :param grads: sample-wise gradients. It must be a 2D array/tensor.
    :param true_y: the ground truth target labels for each sample. It must be a 1D array/tensor.
    :param num_classes: the total number of classes. This number must be larger than or equal to the number of classes
        inferable from `true_y` (true_y.max()+1, as indexing starts from 0); otherwise, a :class:`ValueError` is raised.
    :return: a 2D array/tensor of size (num_classes x grads_size), where grads_size is the size of each gradient vector.
        The ith row is the average gradient for the ith target class. If a target class is not present in `true_y`,
        the corresponding average gradient will be a nan values vector.

    .note::
        `grads` and `true_y` must be both numpy arrays or pytorch tensors, and must have the same length.
    """
    if not all_array_tensor(grads, true_y):
        raise TypeError("`grads` and `true_y` must be both numpy arrays or pytorch tensors")
    if grads.ndim != 2 or true_y.ndim != 1:
        raise ValueError("`grads` must be 2D; `true_y` must be 1D")
    if not all_with_len(grads, true_y):
        raise ValueError("`grads` and `true_y` must have the same length")
    if (true_y < 0).any():
        raise ValueError("`true_y` must contain positive values (>=0) only")
    if num_classes < true_y.max() + 1:
        raise ValueError("`num_classes` must be larger than or equal to the number of classes inferable from `true_y`")

    was_numpy = False
    # if `grads` and `true_y` are numpy arrays, convert them into pytorch tensors.
    if isinstance(grads, np.ndarray):
        grads = torch.from_numpy(grads)
        true_y = torch.from_numpy(true_y)
        was_numpy = True

    # get the size of each gradient
    size_grads = grads.shape[1]
    # get the number of times each target appears in `true_y`
    counts = torch.bincount(true_y, minlength=num_classes)
    acc_dtype = torch.float32 if grads.dtype in (torch.float16, torch.bfloat16) else grads.dtype

    # create a `num_classes` x `size_grads` tensor filled initially with 0
    class_avg_grads_t = torch.zeros((num_classes, size_grads), device=grads.device, dtype=acc_dtype)
    # compute for each row i, the sum of gradients of class i using `grads`; if samples from class i are not present,
    # the sum of gradients for class i will be a vector of 0
    class_avg_grads_t.index_add_(dim=0, index=true_y, source=grads.to(acc_dtype))

    # get a boolean mask indicating which targets appear at least once in `true_y`
    present = counts > 0
    # compute for each row i, the average gradient of class i if samples from class i are present
    class_avg_grads_t[present] /= counts[present].to(acc_dtype)[:, None]
    # if samples from class i are not present, the average gradient of class i is a vector of nans
    class_avg_grads_t[~present] = torch.nan

    class_avg_grads_t = class_avg_grads_t.to(grads.dtype)

    # return a numpy array if the inputs were numpy arrays, a pytorch tensor otherwise
    return class_avg_grads_t.numpy() if was_numpy else class_avg_grads_t


def grads_wrt_weights_biases_specific_class(grads: Union[Tensor, ndarray],
                                            y: Union[Tensor, ndarray],
                                            num_classes: int) -> Union[Tensor, ndarray]:
    """
    Given gradients w.r.t. **all** the weights and biases in the classification head (last layer), extract sub-gradients
    w.r.t. the weights and bias associated with a specific class.
    :param grads: a 2D array/tensor of gradients. The order of the elements in each of these gradients must be as
        follows: W00, W01,..., W0(F-1), W10, W11,..., W1(F-1),...,W(C-1)0,...,W(C-1)(F-1),B0, B1,..., B(C-1),
        where F is the size of the embeddings (feature vectors), C the number of classes, WMN denotes the mth, nth
        element of the weight matrix, and BM denotes the mth element of the bias vector.
    :param y: a 1D array/tensor of target labels, one for each gradient in `grads`. Each target label is used to extract
        from the corresponding gradient in `grads`, the sub-gradient w.r.t. the weights and bias associated with the
        corresponding class. The order of the elements in a sub-gradient w.r.t the weights and bias associated with
        target t are the following: Wt0, Wt1,..., Wt(F-1),Bt.
    :param num_classes: the total number of classes. A :class:`ValueError` is raised if this number is less than the
        number of classes inferable from `y` (y.max()+1, as indexing starts from 0) or yields a wrongful embedding
        (feature vector) size, i.e. (size_gradient - num_classes) mod num_classes is not 0.
    :return: a 2D array/tensor with the same number of rows as `grads` and F+1 columns, where F is the size of the
        embeddings (feature vectors).

    .note::
        `grads` and `y` must be both numpy arrays or pytorch tensors, and must have the same length.
    """
    if not all_array_tensor(grads, y):
        raise TypeError("`grads` and `y` must be both numpy arrays or pytorch tensors")
    if grads.ndim != 2 or y.ndim != 1:
        raise ValueError("`grads` must be 2D; `y` must be 1D")
    if not all_with_len(grads, y):
        raise ValueError("`grads` and `y` must have the same length")
    if (y < 0).any():
        raise ValueError("`y` must contain positive values (>=0) only")
    if num_classes <= 0:
        raise ValueError("`num_classes` must be strictly positive (>0)")
    if num_classes < y.max() + 1:
        raise ValueError("`num_classes` must be larger than or equal to the number of classes inferable from `y`")

    n_rows, n_cols = grads.shape
    # get the total number of elements in the weight matrix
    n_cols_without_biases = n_cols - num_classes
    if n_cols_without_biases <= 0:
        raise ValueError("The size of each gradient must be strictly greater than `num_classes`")
    if n_cols_without_biases % num_classes != 0:
        raise ValueError("`num_classes` produces a wrongful embedding (feature vector) size")

    # get the size of the embedding (feature vector)
    embed_size = n_cols_without_biases // num_classes
    # convert it into a 2D array/tensor with only one column
    y = y[:, None]
    # get the index of the bias that needs to be retrieved for each row. (n_cols - num_classes) is the index of the
    # bias for class target 0, and summing y acts as an offset to retrieve the correct bias element
    bias_indexes = n_cols_without_biases + y

    if isinstance(grads, torch.Tensor):
        # get the indexes of the weight elements that need to be retrieved for each row. (embed_size * y) is the
        # index of the first weight element to be retrieved for that row. To retrieve all weight elements, this is then
        # summed by 0,1,2...,`embed_size`-1, thereby producing a `n_rows`x`embed_size` 2D tensor.
        weights_indexes = (embed_size * y) + torch.arange(embed_size, device=grads.device)[None, :]
        col_indexes = torch.cat((weights_indexes, bias_indexes), dim=1)
        row_indexes = torch.arange(n_rows, device=grads.device)[:, None]
    else:
        # get the indexes of the weight elements that need to be retrieved for each row. (embed_size * y) is the
        # index of the first weight element to be retrieved for that row. To retrieve all weight elements, this is then
        # summed by 0,1,2...,`embed_size`-1, thereby producing a `n_rows`x`embed_size` 2D array.
        weights_indexes = (embed_size * y) + np.arange(embed_size)[None, :]
        col_indexes = np.concatenate((weights_indexes, bias_indexes), axis=1)
        row_indexes = np.arange(n_rows)[:, None]

    return grads[row_indexes, col_indexes]


def unflatten_grads_wrt_weights_biases(grads: Union[Tensor, ndarray],
                                       num_classes: int) -> Union[Tensor, ndarray]:
    """
    Unflatten 1D gradients w.r.t. the weights and biases in the classification head (last layer) into 2D gradients.
    :param grads: a 2D array/tensor containing gradients w.r.t. the weights and biases in the classification head in
        each row. The order of the elements in each of these gradients **must** be as
        follows: W00, W01,..., W0(F-1), W10, W11,..., W1(F-1),...,W(C-1)0,...,W(C-1)(F-1),B0, B1,..., B(C-1),
        where F is the size of the embeddings (feature vectors), C the number of classes, WMN denotes the mth, nth
        element of the weight matrix, and BM denotes the mth element of the bias vector.
    :param num_classes: the total number of classes.
    :return: a 3D array/tensor containing the unflattened gradients. The structure of each 2D gradient is as follows:
        [ [W00, W01,..., W0(F-1), B0],
          [W10, W11,..., W1(F-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F-1), B(C-1)] ]
    """
    if not all_array_tensor(grads):
        raise TypeError("`grads` must be a numpy array or a pytorch tensor")
    if grads.ndim != 2:
        raise ValueError("`grads` must be 2D")
    if not num_classes > 0:
        raise ValueError("`num_classes` must be strictly positive (>0)")
    n_rows, n_cols = grads.shape
    # get the total number of elements in the weight matrix
    n_cols_without_biases = n_cols - num_classes
    if n_cols_without_biases <= 0:
        raise ValueError("The size of each gradient must be strictly greater than `num_classes`")
    if n_cols_without_biases % num_classes != 0:
        raise ValueError("`num_classes` produces a wrongful embedding (feature vector) size")

    grads_wrt_weights = grads[:, :-num_classes].reshape(n_rows, num_classes, -1)
    grads_wrt_biases = grads[:, -num_classes:].reshape(n_rows, num_classes, -1)
    return get_xp(grads).concatenate((grads_wrt_weights, grads_wrt_biases), axis=-1)


def update_rows_by_id(t1: Tensor,
                      t2: Tensor,
                      id1: Tensor,
                      id2: Tensor,
                      in_place: bool = True) -> Tensor:
    """
    Update rows in `t1` with rows from `t2` that have matching IDs.

    The first dimension of `t1` and `t2` is treated as the row dimension. For every ID that appears in both `id1` and
    `id2`, the corresponding row from `t2` replaces the row in `t1` with the same ID.
    Rows in `t1` whose IDs are not present in `id2` are left unchanged. Rows in `t2` whose IDs are not present in
    `id1` are ignored.
    :param t1: a tensor to update
    :param t2: a source tensor
    :param id1: a 1D tensor containing IDs, one for each row in `t1`
    :param id2: a 1D tensor containing IDs, one for each row in `t2`
    :param in_place: If True, `t1` is modified in place. If False, a clone is updated, leaving the original `t1`
        unchanged
    :return: a tensor with the same shape as `t1`. If `in_place` is True, this is `t1` itself after replacing the rows
        whose IDs also appear in `id2`. If `in_place` is False, this is a cloned copy of `t1` with those replacements
        applied, leaving the original `t1` unchanged

    .note::
        `t1` and `t2` **must** be at least one-dimensional; a :class:`ValueError` is raised otherwise

    .note::
        `t1` and `t2` **must** have the same number of dimensions and every corresponding dimension, apart from the
        first one, **must** have the same size; a :class:`ValueError` is raised otherwise

    .note::
        This function **assumes** that both `id1` and `id2` contain no duplicate elements
    """
    if not (all_array_tensor(t1, t2, id1, id2) and isinstance(t1, torch.Tensor)):
        raise TypeError("`t1`, `t2`, `id1` and `id2` must be all pytorch tensors")
    if t1.ndim < 1 or t2.ndim < 1:
        raise ValueError("`t1` and `t2` must be both at least 1D")
    if t1.shape[1:] != t2.shape[1:]:
        raise ValueError("`t1` and `t2` must have the same number of dimensions and every corresponding dimension, "
                         "apart from the first one, must have the same size")
    if id1.ndim != 1 or id2.ndim != 1:
        raise ValueError("`id1` and `id2` must be 1D")
    if not (all_with_len(t1, id1) and all_with_len(t2, id2)):
        raise ValueError("`id1` and `id2` must have the same length of `t1` and `t2`, respectively")

    if not in_place:
        t1 = t1.clone()

    # get a mask of which elements in `id1` are also in `id2`
    mask_id1 = torch.isin(id1, id2, assume_unique=True)
    # if there are no common elements between `id1` and `id2`
    if not mask_id1.any():
        return t1

    # get the elements that `id1` and `id2` have in common, sorted according to their order of appearance in `id1`
    common_elems_id1 = id1[mask_id1]
    # get a mask of which elements in `id2` are in common with those in `id1`
    mask_id2 = torch.isin(id2, common_elems_id1, assume_unique=True)
    # get the corresponding elements in `t2` from `mask_id2`
    t2 = t2[mask_id2]
    # get the elements that `id1` and `id2` have in common, sorted according to their order of appearance in `id2`
    common_elems_id2 = id2[mask_id2]
    # get the indices that would sort `common_elems_id2` into ascending order
    order = torch.argsort(common_elems_id2)
    # get the indices that would sort `common_elems_id2` so to match the same order of values in `common_elems_id1`,
    # i.e.  common_elems_id2[idx] = common_elems_id1
    idx = order[torch.searchsorted(common_elems_id2, common_elems_id1, sorter=order)]
    # sort `t2` so to align it with the same order in t1[mask_id1]
    t2 = t2[idx]
    # insert into `t1` the corresponding elements of `t2`
    t1[mask_id1] = t2

    return t1
