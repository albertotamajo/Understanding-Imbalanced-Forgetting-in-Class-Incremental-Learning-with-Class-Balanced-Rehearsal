from __future__ import annotations
from typing import Union, Dict, Tuple, Callable, Sequence, List, Optional, TYPE_CHECKING

from training.plugins import ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from llifc.utils import (all_array_tensor, all_with_len, all_with_shape, get_xp,
                         sample_wise_mce_grads_wrt_weights_biases, class_avg_grads,
                         grads_wrt_weights_biases_specific_class, unflatten_grads_wrt_weights_biases,
                         update_rows_by_id)
from llifc.computation import SIC_step, CIC_step, NIC_step, ALL_NIC_step, OIC_step, RIC_step

import torch
import torch.nn.functional as tF

if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


def interf_projection_affine_hyperplane(v1: Union[Tensor, ndarray],
                                        v2: Union[Tensor, ndarray],
                                        interf_target: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Compute for each row in `v1` the closest vector in Euclidean space v* such that the interference of v* onto the
    respective row in `v2` is exactly equal to the corresponding value in `interf_target`. Interference is measured
    using the interference operator as described in "Understanding Imbalanced Forgetting in Rehearsal-Based
    Class-Incremental Learning", i.e., let a and b be two vectors, the interference of a onto b is measured as:
    -<a,b>/||b||. More formally, for each row v1_i, the following constrained minimization problem is performed:
    min ||v1_i - v1_i*|| s.t. -<v1_i*, v2_i> / ||v2_i|| = interf_target_i. This is simply the Euclidean projection of
    v1_i onto the affine hyperplane {v: <v,v2_i> = -interf_target_i ||v2_i||} and is given by
    v1_i* = v1_i - [(<v1_i,v2_i> + interf_target_i ||v2_i||)/||v2_i||^2]v2_i.
    :param v1: a 2D array/tensor
    :param v2: a 2D array/tensor
    :param interf_target: a 1D array/tensor of interference targets.
    :return: a 2D array/tensor
    """
    if not all_array_tensor(v1, v2, interf_target):
        raise TypeError("`v1`, `v2` and `interf_target` must all be numpy arrays or pytorch tensors")
    if v1.ndim != 2 or v2.ndim != 2 or interf_target.ndim != 1:
        raise ValueError("`v1` must be 2D; `v2` must be 2D; `interf_target` must be 1D")
    if not all_with_len(v1, v2, interf_target):
        raise ValueError("`v1`, `v2` and `interf_target` must have the same length")
    if not all_with_shape(v1, v2):
        raise ValueError("`v1` and `v2` must have the same shape")

    norm_v2 = get_xp(v2).linalg.norm(v2, 2, 1)
    weight = ((v1 * v2).sum(1) + interf_target * norm_v2) / norm_v2**2
    return v1 - weight[:, None] * v2


def project_to_frobenius_hyperplane(G: Union[Tensor, ndarray],
                                    F: Union[Tensor, ndarray],
                                    V: Union[Tensor, ndarray],
                                    b: Union[Tensor, ndarray],
                                    k: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    For each matrix G_i, F_i, V_i and scalar b_i, k_i, solve the following constrained minimization problem:

    X_i* = argmin ||X_i - G_i ||_F s.t. 1/k_i < X_i^T F_i, V_i>_F = b_i

    where || ||_F is the Frobenius norm and <>_F is the Frobenius inner product.

    The solution to this constrained problem is the following:
    X_i* = G_i + [(k_ib_i - <G_i,B_i>_F) / ||B_i||^2_F]B_i, where B_i = F_iV_i^T

    :param G: a 3D array/tensor containing N 2D matrices of size PxC
    :param F: a 3D array/tensor containing N 2D matrices of size PxD
    :param V: a 3D array/tensor containing N 2D matrices of size CxD
    :param b: a 1D array/tensor containing N elements
    :param k: a 1D array/tensor containing N elements, with all elements being strictly positive
    :return: a 3D array/tensor containing N 2D matrices of size PxC, i.e. the X_i* matrices

    .note::
        This function raises a :class:`ValueError` when at least one of the B_i matrices is the 0 matrix. This is
        performed because if k_ib_i = 0, then the optimal solution is simply G_i; if k_ib_i != 0, the problem is
        infeasible
    """
    if not all_array_tensor(G, F, V, b, k):
        raise TypeError("`G`, `F`, `V`, `b`, and `k` must all be numpy arrays or pytorch tensors")
    if G.ndim != 3 or F.ndim != 3 or V.ndim != 3 or b.ndim != 1 or k.ndim != 1:
        raise ValueError("`G` must be 3D; `F` must be 3D; `V` must be 3D; `b` must be 1D; `k` must be 1D")
    if not all_with_len(G, F, V, b, k):
        raise ValueError("`G`, `F`, `V`, `b`, and `k` must have the same length")
    if G.shape[1] != F.shape[1] or G.shape[2] != V.shape[1] or F.shape[2] != V.shape[2]:
        raise ValueError("The shapes of `G`, `F` and `V` do not match the expected ones")
    if not (k > 0).all():
        raise ValueError("`k` must contain strictly positive (>0) elements only")

    B = F @ V.swapaxes(-1, -2)
    denom = (B**2).sum((-2,-1))
    if (denom == 0).any():
        raise ValueError("None of the B_i matrices is allowed to be the 0 matrix")
    weight = ( (b*k) - (G*B).sum((-2,-1)) ) / denom
    return G + weight[:, None, None]*B


def minimize_last_layer_bias_interf_via_softmax(S: Tensor,
                                                H: Tensor,
                                                F: Tensor,
                                                V: Tensor,
                                                m: Tensor,
                                                epsilon: Tensor,
                                                eta: Tensor,
                                                expansion_steps: int = 20,
                                                bisection_steps: int = 20) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    For each matrix S_n, H_n, F_n, V_n and scalar m_n, epsilon_n, eta_n, this function minimizes the interference of the
    bias matrix B_n := 1/m_n (S_n - H_n)^T F_n - V_n onto the matrix V_n, provided such initial interference is greater
    than or equal to eta_n.
    The interference is measured using the interference operator as described in "Understanding Imbalanced
    Forgetting in Rehearsal-Based Class-Incremental Learning", i.e., let A and B be two matrices, the interference of A
    onto B is measured as -<A,B>_F/||B||_F, where <.,.>_F is the Frobenius inner product and ||.||_F the Frobenius norm.
    The interference of the bias matrix B_n onto the matrix V_n is minimized by replacing the matrix S_n with an
    alternative matrix S*_n satisfying certain constraints. More formally, the following constrained optimization
    problem is solved:

    S*_n = argmin_{S'_n} - [ (<1/m_n (S'_n - H_n)^T F_n - V_n, V_n>_F) / ||V_n||_F ]
         = argmin_{S'_n} -1/m_n <S'_n - H_n, Q_n>_F + ||V_n||_F
         = argmin_{S'_n} -1/m_n <S'_n - H_n, Q_n>_F
    s.t.
        - each row in S'_n sums to 1
        - each element in S'_n > 0
        - 1/m_n sum_{i=1}^I_n KL((S'_n)_i || (S_n)_i) <= epsilon_n   (KL constraint), where I_n is the total number of
            rows in S_n
        - -1/m_n <S'_n - H_n, Q_n>_F + ||V_n||_F >= eta_n  (interference constraint)
    where Q_n := F_n(V_hat)_n^T, (V_hat)_n := V_n / ||V_n||_F, and KL denotes the KL divergence. The first two
    constraints ensure that S_n (a matrix of row-wise softmax probabilities) is replaced with a matrix S*_n that also
    encodes row-wise softmax probabilities.  The penultimate constraint (KL constraint) ensures that the sum of
    row-wise KL divergence values between S_n and S*_n, weighted by 1/m_n, is less than or equal to epsilon_n. The last
    constraint (interference constraint) ensures that the interference of the bias matrix
    B*_n := 1/m_n (S*_n - H_n)^T F_n - V_n onto the matrix V_n is greater than or equal to eta_n.

    The solution to this problem is the following:

    (S*_n)_i = softmax(log (S_n)_i+ gamma*_n (Q_n)_i), where gamma*_n >= 0 is the largest real-valued scalar satisfying
    the last two constraints. This function finds gamma*_n using bisection.


    :param S: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the vector of softmax probabilities for sample i.
    :param H: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the one-hot encoding vector for sample i.
    :param F: a 3D tensor containing N 2D matrices of size IxG, where I is the number of samples and G=F'+1 is the size
        of feature vectors + 1. The ith row in each nth matrix contains the feature vector for sample i appended with a
        value of 1 at the end, i.e., the last column in each nth matrix must only contain 1s.
    :param V: a 3D tensor containing N 2D matrices of size CxG, where C is the total number of classes and G=F'+1 is the
        size of feature vectors + 1. Each nth matrix is a gradient w.r.t. the weights and bias in the last layer.
        The structure of each gradient must be as follows:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.
    :param m: a 1D tensor containing N strictly positive real-valued scalars.
    :param epsilon: a 1D tensor containing N strictly positive real-valued scalars.
    :param eta: a 1D tensor containing N real-valued scalars.
    :param expansion_steps: the number of expansion steps used to find an infeasible upper bound for bisection. It must
        be greater than or equal to 1. Default is 20.
    :param bisection_steps: the number of bisection steps used to solve each constrained optimization problem. It must
        be greater than or equal to 1. Default is 20.
    :return: a tuple of 8 tensors. The first element is a 3D tensor containing N 2D matrices of size IxC, where each nth
        matrix corresponds to S*_n. The second element is a 1D tensor containing N values, where each nth value
        corresponds to gamma*_n. The third element is a 1D tensor containing N values, where each nth value corresponds
        to 1/m_n sum_{i=1}^I_n KL((S*_n)_i || (S_n)_i), i.e. the left hand side of the KL constraint.
        The fourth element is a 1D tensor containing N values, where each nth value corresponds to
        -1/m_n <S*_n - H_n, Q_n>_F + ||V_n||_F, i.e. the left hand side of the interference constraint.
        The fifth element is a 1D tensor containing N values, where each nth value corresponds to ||V_n||_F.
        The sixth element is a 1D tensor containing N values, where each nth value corresponds to ||Q_n||_F.
        The seventh element is a 1D tensor containing N values, where each nth value is True if the interference of the
        bias matrix B_n onto the matrix V_n was initially greater than or equal to eta_n. The last element is a 1D
        tensor containing N values, where each nth value is True if the upper-bound gamma found by the expansion loop
        for the nth constrained optimization problem produces a matrix of softmax probabilites that satifies the last
        two constraints, False otherwise. For bisection to find the optimal gamma value, the upper-bound gamma
        **must not** satisfy the last two constraints. If they are satisfied, the optimal gamma value is actually
        greater than the returned one.

    .note::
        1/m_n (S_n - H_n)^T F_n corresponds to the sum of sample-wise gradients of the multi-class cross entropy loss
        w.r.t. the weights and bias in the last layer, weighted by 1/m_n. It is a 2D matrix of size CxG, where C is the
        total number of classes and G=F'+1 is the size of feature vectors + 1, and has the following structure:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.

    .note::
        This functions **assumes** that all tensors have the same data type. It is highly recommended to use at least
        float32 to prevent the clamping operations from distorting too much the constrained problems.

    .note::
        This function **assumes** elements in S are strictly positive. Entries are clamped only as a numerical
        safeguard; if clamping materially changes S, the computed KL is no longer exactly KL(S* || S).

    .note::
        The implementation of this function is heavily optimized to work efficiently with torch.compile, i.e., feeding
        this function to torch.compile should produce substantial gain in computational performance, especially for CUDA
        tensors. For this reason, the function performs no checks on the input tensors. Users are responsible for
        ensuring that inputs comply with the protocol outlined for each input tensor.

    .note::
        If the interference of the bias matrix B_n onto the matrix V_n is initially less than eta_n, the problem is
        infeasible provided gamma*_n >= 0. This function returns the original matrix S_n as the optimal matrix of
        softmax probabilities (although it violates the constraint) and 0 as the optimal gamma value.
        The returned matrix of softmax probabilities might slightly differ from the original one due to floating-point
        errors or clamping.

    .note::
        If Q_n turns out to be the zero matrix, it happens either because V_n is the 0 matrix or because F_n(V_hat)_n^T
        produces the 0 matrix. In the first case the problem is undefined while it is defined in the second
        case. For both cases, this function simply returns S_n as S*_n.
        The returned matrix of softmax probabilities might slightly differ from the original one due to floating-point
        errors or clamping. The returned optimal gamma value will be very close to the upper bound found by the
        expansion loop if the interference constraint is initially satisfied, otherwise will be 0.

    .note::
        Provided an infeasible upperbound has been found by the expansion loop (larger number of iterations make it more
        likely to find it), the larger the number of bisection steps, the closer the returned gamma*_n is to the actual
        optimal gamma value; thus the closer the returned S*_n is to the actual optimal matrix of softmax probabilities.
    """
    dtype = S.dtype
    # get the smallest positive “normal” number that the dtype of the input tensors can safely represent
    tiny = torch.finfo(dtype).tiny

    # compute the Frobenius norm of each matrix in V
    V_norm = torch.linalg.norm(V, dim=(1,2))
    # normalize each matrix in V; `clamp_min(tiny)` ensures no division by 0 or very small numbers that underflow to 0
    V_hat = V / V_norm.clamp_min(tiny)[:, None, None]
    # compute Q so that each Q_n = F_n(V_hat)_n^T
    Q = torch.bmm(F, V_hat.transpose(1, 2))
    # compute the Frobenius norm of each matrix in Q
    Q_norm = torch.linalg.norm(Q, dim=(1,2))
    # center each row of each matrix in Q. Softmax is invariant to row-wise constant shifts, so this is safe and
    # performed for stable computation
    Q_centered = Q - Q.mean(dim=2, keepdim=True)

    # compute the log of each element in S; `clamp_min(tiny)` ensures stable log computations
    log_S = torch.log(S.clamp_min(tiny))

    def evaluate(gamma: Tensor):
        # compute the logits from gamma
        logits = log_S + gamma[:, None, None] * Q_centered
        # compute the log(softmax()) row-wise for each matrix in `logits`. This is computed for computation of the
        # weighted KL-divergence. log_softmax is faster and more stable than applying the log after computing the
        # softmax values.
        log_P = tF.log_softmax(logits, dim=2)
        # get back to the softmax probabilities by exponentiating element-wise `log_P`
        P = torch.exp(log_P)
        # compute the weighted KL-divergence per batch
        weighted_kl = torch.sum(P * (log_P - log_S), dim=(1, 2)) / m
        # compute the interference per batch
        interf = -(torch.sum((P - H) * Q, dim=(1, 2)) / m) + V_norm

        return P, weighted_kl, interf

    # get the initial gamma values, all set to 0, for each batch
    zero_gamma = S.new_zeros(S.shape[0])
    # get the initial softmax probabilities, kl-divergence and interference values resulting from using zero gamma, i.e.
    # no update to the original softmax probabilities is performed
    P0, kl0, interf0 = evaluate(zero_gamma)
    # check whether each batch optimization problem is actually feasible; it is not feasible when the interference
    # constraint is already violated by the current softmax probabilities
    feasible0 = interf0 >= eta

    # ------------------------------------------------------------
    # Local curvature approximation for initial high
    # ------------------------------------------------------------
    EQ = torch.sum(S * Q, dim=2, keepdim=True)
    var_Q = torch.sum(S * (Q - EQ) ** 2, dim=2)
    mean_var_Q = (var_Q.sum(dim=1) / m).clamp_min(tiny)
    # approximate gamma where the KL divergence constraint becomes active
    gamma_kl = torch.sqrt(2.0 * epsilon / mean_var_Q)
    # approximate gamma where the interference constraint becomes active
    gamma_interf = (interf0 - eta) / mean_var_Q
    # use `gamma_interf` if the interference constraint is not violated by the current softmax probabilities;
    # otherwise use `gamma_kl`. This is done to prevent negative gamma values. Obviously if the current softmax
    # probabilities already violate the interference constraint, then any positive gamma also violates it.
    gamma_interf = torch.where(feasible0, gamma_interf, gamma_kl)
    # make a guess on the high gamma not satisfying the constraints, i.e. the minimum between `gamma_kl`
    # and `gamma_interf`
    gamma_guess = torch.minimum(gamma_kl, gamma_interf).clamp_min(S.new_tensor(1e-12))

    low_gamma = zero_gamma
    # multiply the previous guess by 2 to start the expansion steps with a safer initial high gamma
    high_gamma = 2.0 * gamma_guess
    # the highest gamma values so far satisfying the kl-divergence and align constraints simultaneously
    best_gamma = low_gamma
    # the softmax probabilities resulting from the best gamma
    best_P = P0
    # the kl-divergence values resulting from the best gamma
    best_kl = kl0
    # the interference values resulting from the best gamma
    best_interf = interf0

    # fixed expansion loop. Finds an infeasible upper bound per batch element.
    for _ in range(expansion_steps):
        # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
        P_high, kl_high, interf_high = evaluate(high_gamma)
        # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_high = (kl_high <= epsilon) & (interf_high >= eta)
        # create a 3D mask from `feasible_high`
        mask = feasible_high[:, None, None]
        # if `high_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_high, high_gamma, best_gamma)
        # if `high_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_high, kl_high, best_kl)
        # if `high_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_high, interf_high, best_interf)
        # if `high_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_high, best_P)
        # if `high_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_high, high_gamma, low_gamma)
        # if `high_gamma` satisfies both constraints then the new high gamma is twice as much; otherwise keep the
        # previous one
        high_gamma = torch.where(feasible_high, 2.0 * high_gamma, high_gamma)

    # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
    _, kl_high, interf_high = evaluate(high_gamma)
    # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
    feasible_high = (kl_high <= epsilon) & (interf_high >= eta)

    # fixed bisection loop. Finds the best gamma per batch element
    for _ in range(bisection_steps):
        # get the middle gamma
        mid_gamma = 0.5 * (low_gamma + high_gamma)
        # get the softmax probabilities, kl-divergence and interference values resulting from the middle gamma
        P_mid, kl_mid, interf_mid = evaluate(mid_gamma)
        # check whether `mid_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_mid = (kl_mid <= epsilon) & (interf_mid >= eta)
        # create a 3D mask from `feasible_mid`
        mask = feasible_mid[:, None, None]
        # if `mid_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_mid, mid_gamma, best_gamma)
        # if `mid_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_mid, kl_mid, best_kl)
        # if `mid_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_mid, interf_mid, best_interf)
        # if `mid_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_mid, best_P)
        # if `mid_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_mid, mid_gamma, low_gamma)
        # if `mid_gamma` satisfies both constraints then keep the old high gamma; otherwise use `mid_gamma`
        high_gamma = torch.where(feasible_mid, high_gamma, mid_gamma)

    return (best_P, best_gamma, best_kl, best_interf, V_norm, Q_norm,
            feasible0, feasible_high)


def min_max_last_layer_interf_via_softmax(S: Tensor,
                                          H: Tensor,
                                          F: Tensor,
                                          V: Tensor,
                                          m: Tensor,
                                          epsilon: Tensor,
                                          eta: Tensor,
                                          min_max: Tensor,
                                          expansion_steps: int = 20,
                                          bisection_steps: int = 20) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    For each matrix S_n, H_n, F_n, V_n and scalar m_n, epsilon_n, eta_n, (min_max)_n this function performs the
    following:
        - If (min_max)_n = 1, it minimizes the interference of 1/m_n (S_n - H_n)^T F_n onto the matrix V_n, provided
          such initial interference is greater than or equal to eta_n.
        - If (min_max)_n = -1, it maximizes the interference of 1/m_n (S_n - H_n)^T F_n onto the matrix V_n, provided
          such initial interference is less than or equal to eta_n.

    The interference is measured using the interference operator as described in "Understanding Imbalanced
    Forgetting in Rehearsal-Based Class-Incremental Learning", i.e., let A and B be two matrices, the interference of A
    onto B is measured as -<A,B>_F/||B||_F, where <.,.>_F is the Frobenius inner product and ||.||_F the Frobenius norm.
    The interference of 1/m_n (S_n - H_n)^T F_n onto the matrix V_n is minimized/maximized by replacing the matrix S_n
    with an alternative matrix S*_n satisfying certain constraints. More formally, one of the following constrained
    optimization problems is solved according to the value of (min_max)_n:

    S*_n = argmin_{S'_n}/argmax_{S'_n}    - [ (<1/m_n (S'_n - H_n)^T F_n, V_n>_F) / ||V_n||_F ]
         = argmin_{S'_n}/argmax_{S'_n}    -1/m_n <S'_n - H_n, Q_n>_F
    s.t.
        - each row in S'_n sums to 1
        - each element in S'_n > 0
        - 1/m_n sum_{i=1}^I_n KL((S'_n)_i || (S_n)_i) <= epsilon_n   (KL constraint), where I_n is the total number of
          rows in S_n
        - -1/m_n <S'_n - H_n, Q_n>_F  >= eta_n  if  (min_max)_n = 1   (interference constraint)
          -1/m_n <S'_n - H_n, Q_n>_F  <= eta_n  if  (min_max)_n = -1  (interference constraint)
    where Q_n := F_n(V_hat)_n^T, (V_hat)_n := V_n / ||V_n||_F, and KL denotes the KL divergence. The first two
    constraints ensure that S_n (a matrix of row-wise softmax probabilities) is replaced with a matrix S*_n that also
    encodes row-wise softmax probabilities.  The penultimate constraint (KL constraint) ensures that the sum of
    row-wise KL divergence values between S_n and S*_n, weighted by 1/m_n, is less than or equal to epsilon_n. The last
    constraint (interference constraint) ensures that the interference of
    1/m_n (S*_n - H_n)^T F_n  onto the matrix V_n is greater than or equal to eta_n in case of minimization or less
    than or equal to eta_n in case of maximization.

    The solutions to the above optimization problems are the following:

        - If (min_max)_n = 1, (S*_n)_i = softmax(log (S_n)_i + gamma*_n (Q_n)_i), where gamma*_n >= 0 is the largest
          real-valued scalar satisfying the last two constraints.
        - If (min_max)_n = -1, (S*_n)_i = softmax(log (S_n)_i - gamma*_n (Q_n)_i), where gamma*_n >= 0 is the largest
          real-valued scalar satisfying the last two constraints.

    For both minimization and maximization problems, this function finds gamma*_n using bisection.

    :param S: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the vector of softmax probabilities for sample i.
    :param H: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the one-hot encoding vector for sample i.
    :param F: a 3D tensor containing N 2D matrices of size IxG, where I is the number of samples and G=F'+1 is the size
        of feature vectors + 1. The ith row in each nth matrix contains the feature vector for sample i appended with a
        value of 1 at the end, i.e., the last column in each nth matrix must only contain 1s.
    :param V: a 3D tensor containing N 2D matrices of size CxG, where C is the total number of classes and G=F'+1 is the
        size of feature vectors + 1. Each nth matrix is a gradient w.r.t. the weights and bias in the last layer.
        The structure of each gradient must be as follows:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.
    :param m: a 1D tensor containing N strictly positive real-valued scalars.
    :param epsilon: a 1D tensor containing N strictly positive real-valued scalars.
    :param eta: a 1D tensor containing N real-valued scalars.
    :param min_max: a 1D tensor of size N containing 1 at index n if the nth constrained optimization problem needs to
        be minimized,  -1 if it needs to be maximized.
    :param expansion_steps: the number of expansion steps used to find an infeasible upper bound for bisection. It must
        be greater than or equal to 1. Default is 20.
    :param bisection_steps: the number of bisection steps used to solve each constrained optimization problem. It must
        be greater than or equal to 1. Default is 20.
    :return: a tuple of 8 tensors. The first element is a 3D tensor containing N 2D matrices of size IxC, where each nth
        matrix corresponds to S*_n. The second element is a 1D tensor containing N values, where each nth value
        corresponds to gamma*_n * (min_max)_n. The third element is a 1D tensor containing N values, where each nth
        value corresponds to 1/m_n sum_{i=1}^I_n KL((S*_n)_i || (S_n)_i), i.e. the left hand side of the KL constraint.
        The fourth element is a 1D tensor containing N values, where each nth value corresponds to
        -1/m_n <S*_n - H_n, Q_n>_F, i.e. the left hand side of the interference constraint.
        The fifth element is a 1D tensor containing N values, where each nth value corresponds to ||V_n||_F.
        The sixth element is a 1D tensor containing N values, where each nth value is True if each row in Q_n has
        constant values (the values need to be constant along each row but not across rows), False otherwise.
        The seventh element is a 1D tensor containing N values, where each nth value is True if the interference
        constraint of the nth constraint optimization problem was initially satisfied by the provided softmax
        probabilities, False otherwise. The last element is a 1D tensor containing N values, where each nth value is
        True if the upper-bound gamma found by the expansion loop for the nth constrained optimization problem produces
        a matrix of softmax probabilities that satisfies the last two constraints, False otherwise.
        For bisection to find the optimal gamma value, the upper-bound gamma **must not** satisfy the last two
        constraints. If they are satisfied, the optimal gamma value is actually greater than the returned one.

    .note::
        1/m_n (S_n - H_n)^T F_n corresponds to the sum of sample-wise gradients of the multi-class cross entropy loss
        w.r.t. the weights and bias in the last layer, weighted by 1/m_n. It is a 2D matrix of size CxG, where C is the
        total number of classes and G=F'+1 is the size of feature vectors + 1, and has the following structure:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.

    .note::
        This functions **assumes** that all tensors have the same data type. It is highly recommended to use at least
        float32 to prevent the clamping operations from distorting too much the constrained problems.

    .note::
        This function **assumes** elements in S are strictly positive. Entries are clamped only as a numerical
        safeguard; if clamping materially changes S, the computed KL is no longer exactly KL(S* || S).

    .note::
        The implementation of this function is heavily optimized to work efficiently with torch.compile, i.e., feeding
        this function to torch.compile should produce substantial gain in computational performance, especially for CUDA
        tensors. For this reason, the function performs no checks on the input tensors. Users are responsible for
        ensuring that inputs comply with the protocol outlined for each input tensor.

    .note::
        If the interference constraint is initially already violated, this function returns the original matrix S_n
        as the optimal matrix of softmax probabilities (although it violates the constraint) and 0 as the optimal gamma
        value. The returned matrix of softmax probabilities might slightly differ from the original one due to
        floating-point errors or clamping.

    .note::
        If each row in Q_n has constant values (the values need to be constant along each row but not across rows),
        different values of gamma will result in the same exact softmax probabilities provided as input. Therefore, this
        function simply returns S_n as S*_n (this also occurs when the problem is undefined, i.e., V_n is the 0 matrix);
        however, the returned matrix of softmax probabilities might slightly differ from the original one due to
        floating-point errors or clamping. The returned optimal gamma value will be very close to the upper bound
        found by the expansion loop if the interference constraint is initially satisfied, otherwise will be 0.

    .note::
        Provided an infeasible upperbound has been found by the expansion loop (larger number of iterations make it more
        likely to find it), the larger the number of bisection steps, the closer the returned gamma*_n is to the actual
        optimal gamma value; thus the closer the returned S*_n is to the actual optimal matrix of softmax probabilities.
    """
    dtype = S.dtype
    # get the smallest positive “normal” number that the dtype of the input tensors can safely represent
    tiny = torch.finfo(dtype).tiny

    # compute the Frobenius norm of each matrix in V
    V_norm = torch.linalg.norm(V, dim=(1,2))
    # normalize each matrix in V; `clamp_min(tiny)` ensures no division by 0 or very small numbers that underflow to 0
    V_hat = V / V_norm.clamp_min(tiny)[:, None, None]
    # compute Q so that each Q_n = F_n(V_hat)_n^T
    Q = torch.bmm(F, V_hat.transpose(1, 2))
    # check whether each row in each matrix of Q has constant values
    constant_Q = (Q == Q[:,:,0:1]).all(dim=2).all(dim=1)
    # center each row of each matrix in Q. Softmax is invariant to row-wise constant shifts, so this is safe and
    # performed for stable computation
    Q_centered = Q - Q.mean(dim=2, keepdim=True)

    # compute the log of each element in S; `clamp_min(tiny)` ensures stable log computations
    log_S = torch.log(S.clamp_min(tiny))

    def evaluate(gamma: Tensor):
        # negate `gamma` for those constrained optimization problems that need to be maximized
        gamma = gamma * min_max
        # compute the logits from gamma
        logits = log_S + gamma[:, None, None] * Q_centered
        # compute the log(softmax()) row-wise for each matrix in `logits`. This is computed for computation of the
        # weighted KL-divergence. log_softmax is faster and more stable than applying the log after computing the
        # softmax values.
        log_P = tF.log_softmax(logits, dim=2)
        # get back to the softmax probabilities by exponentiating element-wise `log_P`
        P = torch.exp(log_P)
        # compute the weighted KL-divergence per batch
        weighted_kl = torch.sum(P * (log_P - log_S), dim=(1, 2)) / m
        # compute the interference per batch
        interf = -(torch.sum((P - H) * Q, dim=(1, 2)) / m)

        return P, weighted_kl, interf

    def interf_feasible(interf: Tensor):
        return min_max * (interf - eta) >= 0

    # get the initial gamma values, all set to 0, for each batch
    zero_gamma = S.new_zeros(S.shape[0])
    # get the initial softmax probabilities, kl-divergence and interference values resulting from using zero gamma, i.e.
    # no update to the original softmax probabilities is performed
    P0, kl0, interf0 = evaluate(zero_gamma)
    # check whether each batch optimization problem is actually feasible; it is not feasible when the interference
    # constraint is already violated by the current softmax probabilities
    feasible0 = interf_feasible(interf0)

    # ------------------------------------------------------------
    # Local curvature approximation for initial high
    # ------------------------------------------------------------
    EQ = torch.sum(S * Q, dim=2, keepdim=True)
    var_Q = torch.sum(S * (Q - EQ) ** 2, dim=2)
    mean_var_Q = (var_Q.sum(dim=1) / m).clamp_min(tiny)
    # approximate gamma where the KL divergence constraint becomes active
    gamma_kl = torch.sqrt(2.0 * epsilon / mean_var_Q)
    # approximate gamma where the interference constraint becomes active
    gamma_interf = (min_max * (interf0 - eta)) / mean_var_Q
    # use `gamma_interf` if the interference constraint is not violated by the current softmax probabilities;
    # otherwise use `gamma_kl`. This is done to prevent negative gamma values. Obviously if the current softmax
    # probabilities already violate the interference constraint, then any positive gamma also violates it.
    gamma_interf = torch.where(feasible0, gamma_interf, gamma_kl)
    # make a guess on the high gamma not satisfying the constraints, i.e. the minimum between `gamma_kl`
    # and `gamma_interf`
    gamma_guess = torch.minimum(gamma_kl, gamma_interf).clamp_min(S.new_tensor(1e-12))

    low_gamma = zero_gamma
    # multiply the previous guess by 2 to start the expansion steps with a safer initial high gamma
    high_gamma = 2.0 * gamma_guess
    # the highest gamma values so far satisfying the kl-divergence and align constraints simultaneously
    best_gamma = low_gamma
    # the softmax probabilities resulting from the best gamma
    best_P = P0
    # the kl-divergence values resulting from the best gamma
    best_kl = kl0
    # the interference values resulting from the best gamma
    best_interf = interf0

    # fixed expansion loop. Finds an infeasible upper bound per batch element.
    for _ in range(expansion_steps):
        # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
        P_high, kl_high, interf_high = evaluate(high_gamma)
        # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_high = (kl_high <= epsilon) & interf_feasible(interf_high)
        # create a 3D mask from `feasible_high`
        mask = feasible_high[:, None, None]
        # if `high_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_high, high_gamma, best_gamma)
        # if `high_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_high, kl_high, best_kl)
        # if `high_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_high, interf_high, best_interf)
        # if `high_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_high, best_P)
        # if `high_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_high, high_gamma, low_gamma)
        # if `high_gamma` satisfies both constraints then the new high gamma is twice as much; otherwise keep the
        # previous one
        high_gamma = torch.where(feasible_high, 2.0 * high_gamma, high_gamma)

    # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
    _, kl_high, interf_high = evaluate(high_gamma)
    # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
    feasible_high = (kl_high <= epsilon) & interf_feasible(interf_high)

    # fixed bisection loop. Finds the best gamma per batch element
    for _ in range(bisection_steps):
        # get the middle gamma
        mid_gamma = 0.5 * (low_gamma + high_gamma)
        # get the softmax probabilities, kl-divergence and interference values resulting from the middle gamma
        P_mid, kl_mid, interf_mid = evaluate(mid_gamma)
        # check whether `mid_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_mid = (kl_mid <= epsilon) & interf_feasible(interf_mid)
        # create a 3D mask from `feasible_mid`
        mask = feasible_mid[:, None, None]
        # if `mid_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_mid, mid_gamma, best_gamma)
        # if `mid_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_mid, kl_mid, best_kl)
        # if `mid_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_mid, interf_mid, best_interf)
        # if `mid_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_mid, best_P)
        # if `mid_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_mid, mid_gamma, low_gamma)
        # if `mid_gamma` satisfies both constraints then keep the old high gamma; otherwise use `mid_gamma`
        high_gamma = torch.where(feasible_mid, high_gamma, mid_gamma)

    return best_P, best_gamma * min_max, best_kl, best_interf, V_norm, constant_Q, feasible0, feasible_high


def min_max_last_layer_interf_via_softmax_eq_constraints(S: Tensor,
                                                         H: Tensor,
                                                         F: Tensor,
                                                         V: Tensor,
                                                         EQ: Tensor,
                                                         m: Tensor,
                                                         epsilon: Tensor,
                                                         eta: Tensor,
                                                         min_max: Tensor,
                                                         expansion_steps: int = 20,
                                                         bisection_steps: int = 20) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    For each matrix S_n, H_n, F_n, V_n, EQ_n and scalar m_n, epsilon_n, eta_n, (min_max)_n this function performs the
    following:
        - If (min_max)_n = 1, it minimizes (subject to constraints) the interference of 1/m_n (S_n - H_n)^T F_n onto
          the matrix V_n, provided such initial interference is greater than or equal to eta_n.
        - If (min_max)_n = -1, it maximizes (subject to constraints) the interference of 1/m_n (S_n - H_n)^T F_n onto
          the matrix V_n, provided such initial interference is less than or equal to eta_n.

    The interference is measured using the interference operator as described in "Understanding Imbalanced
    Forgetting in Rehearsal-Based Class-Incremental Learning", i.e., let A and B be two matrices, the interference of A
    onto B is measured as -<A,B>_F/||B||_F, where <.,.>_F is the Frobenius inner product and ||.||_F the Frobenius norm.
    The interference of 1/m_n (S_n - H_n)^T F_n onto the matrix V_n is minimized/maximized by replacing the matrix S_n
    with an alternative matrix S*_n satisfying certain constraints. More formally, one of the following constrained
    optimization problems is solved according to the value of (min_max)_n:

    S*_n = argmin_{S'_n}/argmax_{S'_n}    - [ (<1/m_n (S'_n - H_n)^T F_n, V_n>_F) / ||V_n||_F ]
         = argmin_{S'_n}/argmax_{S'_n}    -1/m_n <S'_n - H_n, Q_n>_F
    s.t.
        - each row in S'_n sums to 1
        - each element in S'_n > 0
        - 1/m_n sum_{i=1}^I_n KL((S'_n)_i || (S_n)_i) <= epsilon_n   (KL constraint), where I_n is the total number of
          rows in S_n
        - -1/m_n <S'_n - H_n, Q_n>_F  >= eta_n  if  (min_max)_n = 1   (interference constraint)
          -1/m_n <S'_n - H_n, Q_n>_F  <= eta_n  if  (min_max)_n = -1  (interference constraint)
        - (S'_n)_i,j = (S_n)_i,j if (EQ_n)_i,j = True  (equality constraint)
    where Q_n := F_n(V_hat)_n^T, (V_hat)_n := V_n / ||V_n||_F, and KL denotes the KL divergence. The first two
    constraints ensure that S_n (a matrix of row-wise softmax probabilities) is replaced with a matrix S*_n that also
    encodes row-wise softmax probabilities.  The third constraint (KL constraint) ensures that the sum of
    row-wise KL divergence values between S_n and S*_n, weighted by 1/m_n, is less than or equal to epsilon_n. The
    penultimate constraint (interference constraint) ensures that the interference of
    1/m_n (S*_n - H_n)^T F_n  onto the matrix V_n is greater than or equal to eta_n in case of minimization or less
    than or equal to eta_n in case of maximization. Finally, the last constraint (equality constraint) ensures that
    each (S*_n)_i,j is equal to (S_n)_i,j when (EQ_n)_i,j is True.

    The solutions to the above optimization problems are the following:

        - If (min_max)_n = 1, (S*_n)_i[(EQ_n)_i] = (S_n)_i[(EQ_n)_i]
          and (S*_n)_i[~(EQ_n)_i] = (S_n)_i[~(EQ_n)_i].sum() softmax(log (S_n)_i[~(EQ_n)_i] + gamma*_n (Q_n)_i[~(EQ_n)_i])
          where gamma*_n >= 0 is the largest real-valued scalar satisfying the KL and interference constraints.

        - If (min_max)_n = -1, (S*_n)_i[(EQ_n)_i] = (S_n)_i[(EQ_n)_i]
          and (S*_n)_i[~(EQ_n)_i] = (S_n)_i[~(EQ_n)_i].sum() softmax(log (S_n)_i[~(EQ_n)_i] - gamma*_n (Q_n)_i[~(EQ_n)_i])
          where gamma*_n >= 0 is the largest real-valued scalar satisfying the KL and interference constraints.


    For both minimization and maximization problems, this function finds gamma*_n using bisection.

    :param S: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the vector of softmax probabilities for sample i.
    :param H: a 3D tensor containing N 2D matrices of size IxC, where I is the number of samples and C the total number
        of classes. The ith row in each nth matrix contains the one-hot encoding vector for sample i.
    :param F: a 3D tensor containing N 2D matrices of size IxG, where I is the number of samples and G=F'+1 is the size
        of feature vectors + 1. The ith row in each nth matrix contains the feature vector for sample i appended with a
        value of 1 at the end, i.e., the last column in each nth matrix must only contain 1s.
    :param V: a 3D tensor containing N 2D matrices of size CxG, where C is the total number of classes and G=F'+1 is the
        size of feature vectors + 1. Each nth matrix is a gradient w.r.t. the weights and bias in the last layer.
        The structure of each gradient must be as follows:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.
    :param EQ: a 3D boolean tensor containing N 2D matrices of size IxC, where I is the number of samples and C the
        total number of classes. It encodes the equality constraints: (EQ_n)_i,j = True ensures that
        (S*_n)_i,j will be equal to (S_n)_i,j; otherwise (S*_n)_i,j is allowed to differ from (S_n)_i,j.
    :param m: a 1D tensor containing N strictly positive real-valued scalars.
    :param epsilon: a 1D tensor containing N strictly positive real-valued scalars.
    :param eta: a 1D tensor containing N real-valued scalars.
    :param min_max: a 1D tensor of size N containing 1 at index n if the nth constrained optimization problem needs to
        be minimized,  -1 if it needs to be maximized.
    :param expansion_steps: the number of expansion steps used to find an infeasible upper bound for bisection. It must
        be greater than or equal to 1. Default is 20.
    :param bisection_steps: the number of bisection steps used to solve each constrained optimization problem. It must
        be greater than or equal to 1. Default is 20.
    :return: a tuple of 8 tensors. The first element is a 3D tensor containing N 2D matrices of size IxC, where each nth
        matrix corresponds to S*_n. The second element is a 1D tensor containing N values, where each nth value
        corresponds to gamma*_n * (min_max)_n. The third element is a 1D tensor containing N values, where each nth
        value corresponds to 1/m_n sum_{i=1}^I_n KL((S*_n)_i || (S_n)_i), i.e. the left hand side of the KL constraint.
        The fourth element is a 1D tensor containing N values, where each nth value corresponds to
        -1/m_n <S*_n - H_n, Q_n>_F, i.e. the left hand side of the interference constraint.
        The fifth element is a 1D tensor containing N values, where each nth value corresponds to ||V_n||_F.
        The sixth element is a 1D tensor containing N values, where each nth value is True if for **each** row i in the
        nth optimization problem, (EQ_n)_i contains more than or equal to C-1 True values, i.e., (EQ_n)_i sets more than
        or equal to C-1 equality constraints, or the values in (Q_n)_i associated with changeable entries, i.e.,
        (Q_n)_i[~(EQ_n)_i], are all equal; False otherwise.
        The seventh element is a 1D tensor containing N values, where each nth value is True if the interference
        constraint of the nth constraint optimization problem was initially satisfied by the provided softmax
        probabilities, False otherwise. The last element is a 1D tensor containing N values, where each nth value is
        True if the upper-bound gamma found by the expansion loop for the nth constrained optimization problem produces
        a matrix of softmax probabilities that satisfies the last two constraints, False otherwise.
        For bisection to find the optimal gamma value, the upper-bound gamma **must not** satisfy the last two
        constraints. If they are satisfied, the optimal gamma value is actually greater than the returned one.

    .note::
        1/m_n (S_n - H_n)^T F_n corresponds to the sum of sample-wise gradients of the multi-class cross entropy loss
        w.r.t. the weights and bias in the last layer, weighted by 1/m_n. It is a 2D matrix of size CxG, where C is the
        total number of classes and G=F'+1 is the size of feature vectors + 1, and has the following structure:
        [ [W00, W01,..., W0(F'-1), B0],
          [W10, W11,..., W1(F'-1), B1],
                    ...
          [W(C-1)0,...,W(C-1)(F'-1), B(C-1)] ]
        where WMN denotes the mth, nth element of the weight matrix, and BM denotes the mth element of the bias vector.

    .note::
        This functions **assumes** that all tensors (apart from EQ, which needs to be boolean) have the same data type.
        It is highly recommended to use at least float32 to prevent the clamping operations from distorting too much
        the constrained problems.

    .note::
        This function **assumes** elements in S are strictly positive. Entries are clamped only as a numerical
        safeguard; if clamping materially changes S, the computed KL is no longer exactly KL(S* || S).

    .note::
        The implementation of this function is heavily optimized to work efficiently with torch.compile, i.e., feeding
        this function to torch.compile should produce substantial gain in computational performance, especially for CUDA
        tensors. For this reason, the function performs no checks on the input tensors. Users are responsible for
        ensuring that inputs comply with the protocol outlined for each input tensor.

    .note::
        If the interference constraint is initially already violated, this function returns the original matrix S_n
        as the optimal matrix of softmax probabilities (although it violates the constraint) and 0 as the optimal gamma
        value. The returned matrix of softmax probabilities might slightly differ from the original one due to
        floating-point errors or clamping (this only holds for those entries that are allowed to change).

    .note::
        If for **each** row i in the nth optimization problem, (EQ_n)_i contains more than or equal to C-1 True
        values, i.e., (EQ_n)_i sets more than or equal to C-1 equality constraints, or the values in (Q_n)_i associated
        with changeable entries, i.e., (Q_n)_i[~(EQ_n)_i], are all equal, different values of gamma will result in the
        same exact softmax probabilities provided as input. Therefore, this function simply returns S_n as S*_n
        (this also occurs when the problem is undefined, i.e., V_n is the 0 matrix); however, the returned matrix of
        softmax probabilities might slightly differ from the original one due to floating-point errors or clamping
        (this only holds for those entries that are allowed to change).
        The returned optimal gamma value will be very close to the upper bound found by the expansion loop if the
        interference constraint is initially satisfied, otherwise will be 0.

    .note::
        Provided an infeasible upperbound has been found by the expansion loop (larger number of iterations make it more
        likely to find it), the larger the number of bisection steps, the closer the returned gamma*_n is to the actual
        optimal gamma value; thus the closer the returned S*_n is to the actual optimal matrix of softmax probabilities.
    """
    dtype = S.dtype
    # get the smallest positive “normal” number that the dtype of all input tensors (except for EQ) can safely represent
    tiny = torch.finfo(dtype).tiny
    # negate the input tensor EQ; it contains True for those elements that are allowed to change; False for the others
    DF = ~EQ
    # get the probability mass to be redistributed among the changeable elements in each row
    free_prob_mass = torch.sum(S * DF, dim=2, keepdim=True)
    # a mask containing True for those rows containing at least one changeable element
    has_changeable = DF.any(dim=2, keepdim=True)

    # compute the Frobenius norm of each matrix in V
    V_norm = torch.linalg.norm(V, dim=(1,2))
    # normalize each matrix in V; `clamp_min(tiny)` ensures no division by 0 or very small numbers that underflow to 0
    V_hat = V / V_norm.clamp_min(tiny)[:, None, None]
    # compute Q so that each Q_n = F_n(V_hat)_n^T
    Q = torch.bmm(F, V_hat.transpose(1, 2))
    # compute the S-weighted conditional mean for each row in each matrix in Q
    Q_changeable_mean = torch.sum(S * Q * DF, dim=2, keepdim=True) / free_prob_mass.clamp_min(tiny)
    # center each row of each matrix in Q (only those elements that are allowed to change); for those elements
    # that are not allowed to change fill with 0. Softmax is invariant to row-wise constant shifts, so this is safe and
    # performed for stable computation
    Q_centered = torch.where(DF, Q - Q_changeable_mean, torch.zeros_like(Q))
    # get the index of the first entry allowed to change along the second dimension; if there is no entry allowed to
    # change index 0 will be returned
    first_idx = DF.long().argmax(dim=2, keepdim=True)
    # get the values of Q stored in the indices in `first_idx`
    reference = Q.gather(dim=2, index=first_idx)
    same_softmax_probs = ((Q == reference) | EQ).all(dim=2).all(dim=1)

    # compute the log of each element in S; `clamp_min(tiny)` ensures stable log computations
    log_S = torch.log(S.clamp_min(tiny))

    def evaluate(gamma: Tensor):
        # negate `gamma` for those constrained optimization problems that need to be maximized
        gamma = gamma * min_max
        # compute the logits from gamma
        logits = log_S + gamma[:, None, None] * Q_centered
        # set the logits of those elements whose softmax probabilities are not allowed to change to -inf. This completely
        # excludes those logits from the conditional softmax
        masked_logits = logits.masked_fill(EQ, float("-inf"))
        # log_softmax(all -inf) produces NaNs. Rows with no changeable entries are temporarily assigned finite dummy
        # logits. Specifically they are assigned with 0.
        safe_logits = torch.where(has_changeable, masked_logits, torch.zeros_like(masked_logits))
        # compute the log(softmax()) row-wise for each matrix in `safe_logits`. This is computed for computation of the
        # weighted KL-divergence. log_softmax is faster and more stable than applying the log after computing the
        # softmax values.
        log_conditional_P = tF.log_softmax(safe_logits, dim=2)
        # get the softmax probabilities associated with changeable entries by both exponentiating element-wise
        # `log_conditional_P` and multiplying by `free_prob_mass`. Multiplying by `free_prob_mass` makes their total
        # equal the mass available after preserving the fixed entries. This tensor contains rubish values for those
        # entries whose softmax probabilities are not allowed to change
        P_free = free_prob_mass * torch.exp(log_conditional_P)
        # merge the fixed softmax values with those associated with changeable entries
        P = torch.where(EQ, S, P_free)
        # compute the log of the new softmax probabilities associated with changeable entries. This is obtained summing
        # `log_conditional_p` to the log of `free_prob_mass`. This tensor contains rubish values for those entries whose
        # softmax probabilities are not allowed to change
        log_P_free = torch.log(free_prob_mass.clamp_min(tiny)) + log_conditional_P
        # compute the log ratio matrices for computation of the weighted KL-divergence. The log ratio of fixed entries
        # is set to 0 as those elements are fixed
        log_ratio = torch.where(DF, log_P_free - log_S, torch.zeros_like(log_S))
        # compute the weighted KL-divergence per batch
        weighted_kl = torch.sum(P * log_ratio, dim=(1, 2)) / m
        # compute the interference per batch
        interf = -(torch.sum((P - H) * Q, dim=(1, 2)) / m)

        return P, weighted_kl, interf

    def interf_feasible(interf: Tensor):
        return min_max * (interf - eta) >= 0

    # get the initial gamma values, all set to 0, for each batch
    zero_gamma = S.new_zeros(S.shape[0])
    # get the initial softmax probabilities, kl-divergence and interference values resulting from using zero gamma, i.e.
    # no update to the original softmax probabilities is performed
    P0, kl0, interf0 = evaluate(zero_gamma)
    # check whether each batch optimization problem is actually feasible; it is not feasible when the interference
    # constraint is already violated by the current softmax probabilities
    feasible0 = interf_feasible(interf0)

    # ------------------------------------------------------------
    # Local curvature approximation for initial high
    # ------------------------------------------------------------
    var_Q = torch.sum(S * (Q_centered ** 2), dim=2)
    mean_var_Q = (var_Q.sum(dim=1) / m).clamp_min(tiny)
    # approximate gamma where the KL divergence constraint becomes active
    gamma_kl = torch.sqrt(2.0 * epsilon / mean_var_Q)
    # approximate gamma where the interference constraint becomes active
    gamma_interf = (min_max * (interf0 - eta)) / mean_var_Q
    # use `gamma_interf` if the interference constraint is not violated by the current softmax probabilities;
    # otherwise use `gamma_kl`. This is done to prevent negative gamma values. Obviously if the current softmax
    # probabilities already violate the interference constraint, then any positive gamma also violates it.
    gamma_interf = torch.where(feasible0, gamma_interf, gamma_kl)
    # make a guess on the high gamma not satisfying the constraints, i.e. the minimum between `gamma_kl`
    # and `gamma_interf`
    gamma_guess = torch.minimum(gamma_kl, gamma_interf).clamp_min(S.new_tensor(1e-12))

    low_gamma = zero_gamma
    # multiply the previous guess by 2 to start the expansion steps with a safer initial high gamma
    high_gamma = 2.0 * gamma_guess
    # the highest gamma values so far satisfying the kl-divergence and align constraints simultaneously
    best_gamma = low_gamma
    # the softmax probabilities resulting from the best gamma
    best_P = P0
    # the kl-divergence values resulting from the best gamma
    best_kl = kl0
    # the interference values resulting from the best gamma
    best_interf = interf0

    # fixed expansion loop. Finds an infeasible upper bound per batch element.
    for _ in range(expansion_steps):
        # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
        P_high, kl_high, interf_high = evaluate(high_gamma)
        # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_high = (kl_high <= epsilon) & interf_feasible(interf_high)
        # create a 3D mask from `feasible_high`
        mask = feasible_high[:, None, None]
        # if `high_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_high, high_gamma, best_gamma)
        # if `high_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_high, kl_high, best_kl)
        # if `high_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_high, interf_high, best_interf)
        # if `high_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_high, best_P)
        # if `high_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_high, high_gamma, low_gamma)
        # if `high_gamma` satisfies both constraints then the new high gamma is twice as much; otherwise keep the
        # previous one
        high_gamma = torch.where(feasible_high, 2.0 * high_gamma, high_gamma)

    # get the softmax probabilities, kl-divergence and interference values resulting from using `high_gamma`
    _, kl_high, interf_high = evaluate(high_gamma)
    # check whether `high_gamma` satisfies the kl-divergence and interference constraints simultaneously
    feasible_high = (kl_high <= epsilon) & interf_feasible(interf_high)

    # fixed bisection loop. Finds the best gamma per batch element
    for _ in range(bisection_steps):
        # get the middle gamma
        mid_gamma = 0.5 * (low_gamma + high_gamma)
        # get the softmax probabilities, kl-divergence and interference values resulting from the middle gamma
        P_mid, kl_mid, interf_mid = evaluate(mid_gamma)
        # check whether `mid_gamma` satisfies the kl-divergence and interference constraints simultaneously
        feasible_mid = (kl_mid <= epsilon) & interf_feasible(interf_mid)
        # create a 3D mask from `feasible_mid`
        mask = feasible_mid[:, None, None]
        # if `mid_gamma` satisfies both constraints then this is the new best gamma; otherwise keep the previous one
        best_gamma = torch.where(feasible_mid, mid_gamma, best_gamma)
        # if `mid_gamma` satisfies both constraints then use its corresponding kl-divergence value; otherwise keep the
        # previous one
        best_kl = torch.where(feasible_mid, kl_mid, best_kl)
        # if `mid_gamma` satisfies both constraints then use its corresponding interference value; otherwise keep the
        # previous one
        best_interf = torch.where(feasible_mid, interf_mid, best_interf)
        # if `mid_gamma` satisfies both constraints then use its corresponding softmax probabilities; otherwise keep
        # the previous ones
        best_P = torch.where(mask, P_mid, best_P)
        # if `mid_gamma` satisfies both constraints then this is the new low gamma; otherwise keep the previous one
        low_gamma = torch.where(feasible_mid, mid_gamma, low_gamma)
        # if `mid_gamma` satisfies both constraints then keep the old high gamma; otherwise use `mid_gamma`
        high_gamma = torch.where(feasible_mid, high_gamma, mid_gamma)

    return best_P, best_gamma * min_max, best_kl, best_interf, V_norm, same_softmax_probs, feasible0, feasible_high


def SIC_step_intervention_via_softmax(logits: Tensor,
                                      embeds: Tensor,
                                      targets: Tensor,
                                      indices: Tensor,
                                      strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      minimize_SIC: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, int, int]]) \
        -> Union[Tuple[None, None, None], Tuple[None, None, Dict[str, Union[ndarray, Tuple[ndarray, ndarray, ndarray]]]]]:
    """
    A function whose partially applied version with `minimize_SIC` already filled in, such as
    functools.partial(SIC_step_intervention_via_softmax, minimize_SIC={callable here}), is intended to be
    passed as argument for `intervention` when constructing an instance
    of :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The latter calls this function each time an SGD iteration is scheduled to incur an intervention in the last-layer
    gradient.

    According to the output of `minimize_SIC`, this function minimizes the unweighted rehearsal SIC_step of certain
    past classes at the current point in optimisation space through the function
    `minimize_last_layer_bias_interf_via_softmax`. Note that the latter minimizes the unweighted rehearsal SIC_step of a
    given past class by replacing the original softmax probabilities output by the network for the
    rehearsal samples of that class with alternative ones found by means of a constrained minimization problem.
    Afterward, this function attaches a hook on the logits output by the network for the current mini-batch
    (strategy.mb_output_incremental_classifier) so that to modify their gradients during the backward pass.
    The gradient w.r.t. logits of new training samples are left untouched as well as those of rehearsal samples
    belonging to past classes whose unweighted rehearsal SIC_step was not to be minimized according to `minimize_SIC`.
    In contrast, those of rehearsal samples belonging to past classes whose unweighted rehearsal SIC_step was to be
    minimized according to `minimize_SIC` are modified according to the alternative softmax probabilities found by
    running `minimize_last_layer_bias_interf_via_softmax`. Unlike intervening on the gradient w.r.t weights and bias in
    the last layer, which only affects that layer, this sort of intervention on the gradient w.r.t logits will
    backpropagate throughout the entire network, including the last layer (for this reason, this function does not
    output tensors for the gradient w.r.t weights and bias in the last-layer; it outputs None for them, which
    :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    interprets as no hook needs to be attached on the weights and bias parameters in the last layer at the current SGD
    step).
    Additionally, this function computes the following metrics for each past class:
        - the unweighted rehearsal SIC_step
        - the post-intervention unweighted rehearsal SIC_step
        - the unweighted mini-batch SIC_step
        - the post-intervention unweighted mini-batch SIC_step
        - the unweighted rehearsal CIC_step
        - the post-intervention unweighted rehearsal CIC_step
        - the unweighted mini-batch CIC_step
        - the post-intervention unweighted mini-batch CIC_step
        - the unweighted full-training NIC_step
        - the unweighted mini-batch NIC_step
        - the unweighted full-training ALL_NIC_step
        - the unweighted mini-batch ALL_NIC_step
    These metrics, and additional data, are collected via a dictionary and returned by this function. Such a dictionary
    will be appended into the `data` property of
    :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The code of this function needs to be inspected to find out what data the returned dictionary contains.
    Note that the metrics, additional data, and the interventions are not computed during the first experience,
    as there are no past classes during the first experience. This is performed by checking the ID of the current
    experience, assuming that the ID of the first experience is 0.

    :param logits: a 2D tensor of predicted raw logits
    :param embeds: a 2D tensor of corresponding embeddings
    :param targets: a 1D tensor of corresponding targets
    :param indices: a 1D tensor of corresponding indices
    :param strategy: the underlying strategy template
    :param minimize_SIC: a callable that takes in a 1D tensor and returns a tuple of five elements.
        The input tensor stores at index i the actual unweighted rehearsal SIC_step value of past class i at the
        current point in optimization space, i.e. the interference (interference operator) exerted by the bias between
        the average gradient of past class i over its rehearsal samples and the average gradient of past class i over
        its original training dataset onto the average gradient of past class i over its original training dataset,
        with all gradients computed w.r.t. the weights and bias in the last layer. The size of the input tensor is
        `n_past_classes`, where `n_past_classes` denotes the total number of past classes.
        The first element of the returned tuple needs to be a 1D boolean tensor, of same size as the input tensor,
        storing at index i whether the unweighted rehearsal SIC_step value of past class i needs to be minimized at the
        current point in optimization space. If True, the unweighted rehearsal SIC_step value of past class i is
        minimized  through the function `minimize_last_layer_bias_interf_via_softmax`.
        The second element of the returned tuple needs to be a 1D tensor of same size as the number of True values in
        the first element of the returned tuple. This tensor will be provided as `epsilon` to
        `minimize_last_layer_bias_interf_via_softmax`. The third element of the returned tuple needs to be a 1D tensor
        of same size as the number of True values in the first element of the returned tuple. This tensor will be
        provided as `eta` to `minimize_last_layer_bias_interf_via_softmax`. The fourth element of the returned tuple
        needs to be an integer; it will be provided as `expansion_steps` to
        `minimize_last_layer_bias_interf_via_softmax`. The fifth element of the returned tuple needs to be an integer;
        it will be provided as `bisection_steps` to `minimize_last_layer_bias_interf_via_softmax`. Note that the tensors
        in the returned tuple do not need to be in the same device as the input tensor; this function will move them to
        the appropriate device.
    :return: a 3-tuple of Nones during the first experience; afterward, a 3-tuple containing None in the first two
        indices and a dictionary in the third one. This dictionary contains strings as keys and tuples or numpy
        arrays as values, and is appended into the `data` property of
        :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    .note::
        As described in :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`,
        the tensor inputs are collected for **all** training samples of **all** the classes encountered so far (both the
        classes in the current experience and the classes in the previously encountered experiences [if any]) prior to
        performing the actual update step at that point in optimisation space.

    .note::
        This function is called by :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        within a torch.no_grad() block

    .note::
        This function **assumes** that the loss used is the standard multi-class cross entropy.

    .note::
        This function **assumes** that an instance of
        :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is present as a
        plugin in the given strategy. A :class:`RuntimeError` is raised, otherwise.

    .note::
        This function **assumes** that `logits`, `embeds`, `targets`, `indices` contain no NaN values.

    .note::
        This function **assumes** that the original training dataset of each past class and the new-class training
        dataset contain at least one sample. A :class:`RuntimeError` is raised, otherwise.

    .note::
        This function **assumes** that the same number of samples is stored in the rehearsal buffer for each past class,
        with at least one sample per class. A :class:`RuntimeError` is raised, otherwise.

    .note::
        This function **assumes** that each mini-batch contains samples from both the rehearsal buffer and the new-class
        training dataset. A :class:`RuntimeError` is raised, otherwise.
    """
    if strategy.experience.current_experience == 0:
        return None, None, None

    # get the total number of classes
    n_classes = logits.shape[1]
    if n_classes != targets.max() + 1:
        raise RuntimeError("The number of classes inferable from `logits` does not match the one inferable from "
                           "`targets`")
    # get the number of old classes by subtracting to the total number the number of classes in the new experience
    n_old_classes = n_classes - len(strategy.experience.classes_in_this_experience)
    # retrieve the indices of all the samples in the rehearsal buffer; raise a RuntimeError if an instance of
    # ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate is not present in the
    # underlying strategy as a plugin
    replay = next((x for x in strategy.plugins
                   if isinstance(x, ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)),
                  None)
    if replay is None:
        raise RuntimeError("The underlying strategy must have an instance of "
                           "`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` as "
                           "a plugin")
    rehearsal_indices = torch.tensor(replay.storage_policy.buffer.dataset_indices, device=logits.device)
    # retrieve the indices of all the samples in the current mini-batch
    mb_indices = strategy.mb_dataset_indices
    # compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
    # classification head (last layer) for all training samples of all the classes encountered so far
    sample_wise_grads = sample_wise_mce_grads_wrt_weights_biases(logits, targets, embeds)
    # a mask for retrieving data of all the samples in the rehearsal buffer only
    mask_rehearsal = torch.isin(indices, rehearsal_indices)
    # a mask for retrieving data of all the samples in the current mini-batch only
    mask_mb = torch.isin(indices, mb_indices)
    # construct a new `targets` tensor containing the same targets as before for the past classes but assigns all
    # the new classes the same target, i.e. the value of `n_old_classes`. This way, below, it will be easy to
    # compute an overall average gradient for the new classes. It is as all new classes belong to the same overall
    # class "NEW CLASSES"
    targets = torch.where(targets >= n_old_classes, n_old_classes, targets)
    # compute the number of samples in the original training dataset of each past class and that of the overall
    # new classes class; it is a 1D tensor of length n_old_classes+1
    n_train_samples = torch.bincount(targets, minlength=n_old_classes + 1)
    if (n_train_samples == 0).any():
        raise RuntimeError("The original training dataset of each past class and the new-class training dataset "
                           "must contain at least one sample")
    # compute the number of samples stored in the rehearsal buffer for each past class; it is a 1D tensor of
    # length n_old_classes
    n_reh_samples = torch.bincount(targets[mask_rehearsal], minlength=n_old_classes)
    if (n_reh_samples == 0).any() or (n_reh_samples != n_reh_samples[0]).any():
        raise RuntimeError("The rehearsal buffer must store the same number of samples for each past class, "
                           "with at least one sample per class")
    # compute the number of samples in the current mini-batch for each past class and the overall new classes class;
    # it is a 1D tensor of length n_old_classes + 1
    n_mb_samples = torch.bincount(targets[mask_mb], minlength=n_old_classes + 1)
    if n_mb_samples[:-1].sum() == 0 or n_mb_samples[-1] == 0:
        raise RuntimeError("Each mini-batch must contain samples from both the rehearsal buffer and the new-class "
                           "training dataset")

    # compute the average gradient for each past class and the overall new classes class, evaluated over the
    # entire training dataset of that class. It is a 2D tensor of shape ([n_old_classes + 1] x grad_size), where
    # grad_size is the size of each gradient
    train_dataset_avg_grads = class_avg_grads(sample_wise_grads, targets, n_old_classes + 1)
    # compute the average gradient for each past class, evaluated over all the samples of that class currently
    # stored in the rehearsal buffer. It is a 2D tensor of shape (n_old_classes x grad_size), where
    # grad_size is the size of each gradient.
    rehearsal_avg_grads = class_avg_grads(sample_wise_grads[mask_rehearsal], targets[mask_rehearsal], n_old_classes)
    # compute the average gradient for each past class and the overall new classes class, evaluated over all the
    # samples of that class in the current mini-batch. It is a 2D tensor of shape ([n_old_classes+1] x grad_size),
    # where grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if a
    # class has no samples in the current mini-batch, then its average gradient is a vector of nan.
    mb_avg_grads = class_avg_grads(sample_wise_grads[mask_mb], targets[mask_mb], n_old_classes + 1)

    # compute the unweighted rehearsal SIC_step for each past class
    sic_step_reh = SIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], include_norm_cos=True)
    # compute the unweighted mini-batch SIC_step for each past class
    sic_step_mb = SIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
    # compute the unweighted rehearsal CIC_step for each past class
    cic_step_reh = CIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], n_reh_samples, n_train_samples[:-1],
                            include_norm_cos=True)
    # compute the unweighted mini-batch CIC_step for each past class
    cic_step_mb = CIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], n_mb_samples[:-1], n_train_samples[:-1],
                           include_norm_cos=True)
    # compute the unweighted full-training NIC_step for each past class
    nic_step_train = NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes,
                              include_norm_cos=True)
    # compute the unweighted mini-batch NIC_step for each past class
    nic_step_mb = NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes, include_norm_cos=True)
    # compute the unweighted full-training ALL_NIC_step for each past class
    all_nic_step_train = ALL_NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1],
                                      include_norm_cos=True)
    # compute the unweighted mini-batch ALL_NIC_step for each past class
    all_nic_step_mb = ALL_NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], include_norm_cos=True)

    # create a dictionary that will be returned by this function; all the tensors are converted to numpy arrays
    data = dict(sic_step_reh=tuple(x.detach().cpu().numpy() for x in sic_step_reh),
                sic_step_mb=tuple(x.detach().cpu().numpy() for x in sic_step_mb),
                cic_step_reh=tuple(x.detach().cpu().numpy() for x in cic_step_reh),
                cic_step_mb=tuple(x.detach().cpu().numpy() for x in cic_step_mb),
                nic_step_train=tuple(x.detach().cpu().numpy() for x in nic_step_train),
                nic_step_mb=tuple(x.detach().cpu().numpy() for x in nic_step_mb),
                all_nic_step_train=tuple(x.detach().cpu().numpy() for x in all_nic_step_train),
                all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in all_nic_step_mb),
                n_train_samples=n_train_samples.detach().cpu().numpy(),
                n_reh_samples=n_reh_samples.detach().cpu().numpy(),
                n_mb_samples=n_mb_samples.detach().cpu().numpy()
                )

    # get a mask of which past classes need their unweighted rehearsal SIC_step to be minimized, along with `epsilon`,
    # `eta`, `expansion_steps` and `bisection_steps` to be passed to `minimize_last_layer_bias_interf_via_softmax`
    mask_minimize, epsilon, eta, expansion_steps, bisection_steps = minimize_SIC(sic_step_reh[0])
    if mask_minimize.any():
        # move the tensors to the appropriate device
        mask_minimize, epsilon, eta = (mask_minimize.to(sic_step_reh[0].device),
                                       epsilon.to(sic_step_reh[0].device),
                                       eta.to(sic_step_reh[0].device))
        # get the id of which past classes need their unweighted rehearsal SIC_step to be minimized
        id_past_classes_minimized = torch.nonzero(mask_minimize, as_tuple=True)[0]
        # a mask for retrieving only data of all the samples in the rehearsal buffer belonging to past classes that need
        # their unweighted rehearsal SIC_step to be minimized
        mask_rehearsal_minimized = mask_rehearsal & torch.isin(targets, id_past_classes_minimized)
        # get the logits of all the samples in the rehearsal buffer belonging to past classes that need
        # their unweighted rehearsal SIC_step to be minimized, and transform them into softmax values
        reh_min_softmax = logits[mask_rehearsal_minimized].softmax(dim=1)
        # get the respective embeddings of all the samples in the rehearsal buffer belonging to past classes that need
        # their unweighted rehearsal SIC_step to be minimized
        reh_min_embeds = embeds[mask_rehearsal_minimized]
        # get the respective targets of all the samples in the rehearsal buffer belonging to past classes that need
        # their unweighted rehearsal SIC_step to be minimized
        reh_min_targets = targets[mask_rehearsal_minimized]
        # get the respective indices of all the samples in the rehearsal buffer belonging to past classes that need
        # their unweighted rehearsal SIC_step to be minimized
        reh_min_indices = indices[mask_rehearsal_minimized]
        # get the indices that sort `reh_min_targets` into ascending order
        sort_idx = torch.argsort(reh_min_targets)
        # use `sort_idx` to sort `reh_min_softmax`
        reh_min_softmax = reh_min_softmax[sort_idx]
        # use `sort_idx` to sort `reh_min_embeds`
        reh_min_embeds = reh_min_embeds[sort_idx]
        # use `sort_idx` to sort `reh_min_targets`
        reh_min_targets = reh_min_targets[sort_idx]
        # use `sort_idx` to sort `reh_min_indices`
        reh_min_indices = reh_min_indices[sort_idx]

        # get the number of past classes that need their unweighted rehearsal SIC_step to be minimized
        n_past_classes_minimized = id_past_classes_minimized.shape[0]
        # get the number of rehearsal samples stored per class
        n_reh_samples_per_class = n_reh_samples[0].item()
        # create the inputs for `minimize_last_layer_bias_interf_via_softmax`
        S = reh_min_softmax.reshape(n_past_classes_minimized, n_reh_samples_per_class, -1)
        H = torch.eye(n_classes, device=S.device, dtype=S.dtype)[reh_min_targets].reshape(
                n_past_classes_minimized, n_reh_samples_per_class, -1)
        F = torch.cat((reh_min_embeds, reh_min_embeds.new_ones((reh_min_embeds.shape[0], 1))), dim=1).reshape(
                n_past_classes_minimized, n_reh_samples_per_class, -1)
        V = unflatten_grads_wrt_weights_biases(train_dataset_avg_grads[id_past_classes_minimized], n_classes)
        m = n_reh_samples[mask_minimize]
        # run `minimize_last_layer_bias_interf_via_softmax`
        result_minimization = minimize_last_layer_bias_interf_via_softmax(S=S, H=H, F=F, V=V, m=m, epsilon=epsilon,
                                                                          eta=eta,expansion_steps=expansion_steps,
                                                                          bisection_steps=bisection_steps)
        # get the new matrix of softmax probabilities for each past class that had their unweighted rehearsal SIC_step
        # minimized
        new_S = result_minimization[0]
        # get the average vector of softmax probabilities for each past class that needs its unweighted rehearsal
        # SIC_step to be minimized, computed prior to running `minimize_last_layer_bias_interf_via_softmax`; it is a 2D
        # tensor of size (`n_past_classes_minimized` x `n_classes`)
        avg_reh_min_softmax_per_class = S.mean(dim=1)  #TODO add to dictionary
        # get the average vector of softmax probabilities for each past class that needs its unweighted rehearsal
        # SIC_step to be minimized, resulting from the new matrix of softmax probabilities returned by
        # `minimize_last_layer_bias_interf_via_softmax`;  it is a 2D tensor of size
        # (`n_past_classes_minimized` x `n_classes`)
        post_avg_reh_min_softmax_per_class = new_S.mean(dim=1) #TODO add to dictionary
        # compute the gradient of the multi-class cross-entropy loss w.r.t the logits for all the samples in the
        # rehearsal buffer belonging to past classes that had their unweighted rehearsal SIC_step minimized, using the
        # softmax probabilities returned by `minimize_last_layer_bias_interf_via_softmax`
        post_reh_min_sample_wise_log_grads = (new_S - H).reshape(n_past_classes_minimized * n_reh_samples_per_class, -1)
        # attach a hook to the logits output by the network so to replace the original sample-wise gradient wrt logits,
        # divided by the number of samples in the current mini-batch, of each sample from the rehearsal buffer
        # belonging to past classes that had their unweighted rehearsal SIC_step minimized with the respective one in
        # `post_reh_min_sample_wise_log_grads` The hook does not need to be removed as the logits output is a non-leaf
        # tensor, i.e. each SGD step produces a new logits tensor
        strategy.mb_output_incremental_classifier.register_hook(lambda grad: update_rows_by_id(
            t1=grad,
            t2=post_reh_min_sample_wise_log_grads/grad.shape[0],
            id1=mb_indices,
            id2=reh_min_indices,
            in_place=False
        ))
        # reshape `new_S` into a 2D tensor of same size as `reh_min_softmax`
        post_reh_min_softmax = new_S.reshape(n_past_classes_minimized * n_reh_samples_per_class, -1)
        # compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
        # classification head (last layer) for all the samples in the rehearsal buffer belonging to past classes that
        # had their unweighted rehearsal SIC_step minimized using `post_reh_min_softmax`;
        # torch.log(post_reh_min_softmax) is used to transform the softmax values back into logits as
        # `sample_wise_mce_grads_wrt_weights_biases` requires logits as input
        post_reh_min_sample_wise_grads = sample_wise_mce_grads_wrt_weights_biases(torch.log(post_reh_min_softmax),
                                                                                  reh_min_targets, reh_min_embeds)
        post_sample_wise_grads = update_rows_by_id(t1=sample_wise_grads, t2=post_reh_min_sample_wise_grads,
                                                   id1=indices, id2=reh_min_indices, in_place=True)
        post_rehearsal_avg_grads = class_avg_grads(post_sample_wise_grads[mask_rehearsal], targets[mask_rehearsal],
                                                   n_old_classes)
        post_mb_avg_grads = class_avg_grads(post_sample_wise_grads[mask_mb], targets[mask_mb], n_old_classes + 1)

        # compute the post-intervention unweighted rehearsal SIC_step for each past class
        post_sic_step_reh = SIC_step(post_rehearsal_avg_grads, train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the post-intervention unweighted mini-batch SIC_step for each past class
        post_sic_step_mb = SIC_step(post_mb_avg_grads[:-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the post-intervention unweighted rehearsal CIC_step for each past class
        post_cic_step_reh = CIC_step(post_rehearsal_avg_grads, train_dataset_avg_grads[:-1], n_reh_samples,
                                     n_train_samples[:-1], include_norm_cos=True)
        # compute the post-intervention unweighted mini-batch CIC_step for each past class
        post_cic_step_mb = CIC_step(post_mb_avg_grads[:-1], train_dataset_avg_grads[:-1], n_mb_samples[:-1],
                                    n_train_samples[:-1], include_norm_cos=True)

        data_add = dict(post_sic_step_reh=tuple(x.detach().cpu().numpy() for x in post_sic_step_reh),
                        post_sic_step_mb=tuple(x.detach().cpu().numpy() for x in post_sic_step_mb),
                        post_cic_step_reh=tuple(x.detach().cpu().numpy() for x in post_cic_step_reh),
                        post_cic_step_mb=tuple(x.detach().cpu().numpy() for x in post_cic_step_mb),
                        id_past_classes_minimized=id_past_classes_minimized.detach().cpu().numpy(),
                        avg_reh_min_softmax_per_class=avg_reh_min_softmax_per_class.detach().cpu().numpy(),
                        post_avg_reh_min_softmax_per_class=post_avg_reh_min_softmax_per_class.detach().cpu().numpy()
                        )
        data = {**data, **data_add}
        print("SIC_step_reh: ", sic_step_reh)
        print("Post Sic_step reh:", post_sic_step_reh)
        print("Softmax avg.:", avg_reh_min_softmax_per_class)
        print("Post Softmax avg.", post_avg_reh_min_softmax_per_class)
    return None, None, data


class NewClassesLastLayerInterferenceInterventionViaSoftmax:
    """
    A class whose instance is intended to be passed as argument for `intervention` when constructing an instance
    of :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The latter will call the `__call__` method of such an instance each time an SGD iteration is scheduled to
    incur an intervention in the last-layer gradient.

    According to the output of `optimize_R_NIC_ALL_NIC`, calling `__call__` minimizes/maximizes the average unweighted
    mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step computed over a set of past classes at the current point
    in optimisation space through the function `min_max_last_layer_interf_via_softmax` or its compiled counterpart
    (compiled with torch.compile(mode="reduce-overhead")) if `eq_constraints` is None; otherwise,
    `min_max_last_layer_interf_via_softmax_eq_constraints` or its compiled counterpart
    (compiled with torch.compile(mode="reduce-overhead")) is used.
    Note that the latter functions minimize/maximize the average unweighted mini-batch
    NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step computed over a set of past classes by replacing the original
    softmax probabilities output by the network for the new-class samples in the current mini-batch with alternative
    ones found by means of a constrained minimization/maximization problem. These new softmax probabilities are then
    used during the backward pass, thereby performing an intervention on the last-layer gradient at the current point
    in optimisation space. If the eighth element output by `optimize_R_NIC_ALL_NIC` is True, such intervention will
    backpropagate to earlier layers as well; otherwise, earlier layers will be updated as if no intervention was
    performed in the last layer.

    Additionally, calling `__call__` computes the following metrics for each past class:
        - the unweighted rehearsal SIC_step
        - the unweighted mini-batch SIC_step
        - the unweighted rehearsal CIC_step
        - the unweighted mini-batch CIC_step
        - the unweighted full-training NIC_step
        - the post-intervention unweighted full-training NIC_step (if any intervention is performed)
        - the unweighted mini-batch NIC_step
        - the post-intervention unweighted mini-batch NIC_step (if any intervention is performed)
        - the unweighted full-training ALL_NIC_step
        - the post-intervention unweighted full-training ALL_NIC_step (if any intervention is performed)
        - the unweighted mini-batch ALL_NIC_step
        - the post-intervention unweighted mini-batch ALL_NIC_step (if any intervention is performed)
        - the unweighted full-training R_NIC_step
        - the post-intervention unweighted full-training R_NIC_step (if any intervention is performed)
        - the unweighted mini-batch R_NIC_step
        - the post-intervention unweighted mini-batch R_NIC_step (if any intervention is performed)
        - the unweighted full-training R_ALL_NIC_step
        - the post-intervention unweighted full-training R_ALL_NIC_step (if any intervention is performed)
        - the unweighted mini-batch R_ALL_NIC_step
        - the post-intervention unweighted mini-batch R_ALL_NIC_step (if any intervention is performed)

    These metrics, and additional data, are collected via a dictionary and returned by the `__call__` method.
    Such a dictionary will be appended into the `data` property of
    :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The code of the `__call__` method needs to be inspected to find out what data the returned dictionary contains.
    Note that the metrics, additional data, and the interventions are not computed during the first experience,
    as there are no past classes during the first experience. This is performed by checking the ID of the current
    experience, assuming that the ID of the first experience is 0.

    .note::
        The metrics computed by the `__call__` method reflect the real metric values for **all** past classes **only**
        if the tensor inputs received from an instance of :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        are collected for **all** training samples of **all** the classes encountered so far (both the
        classes in the current experience and the classes in the previously encountered experiences). In all other
        cases, one or more metrics for **at least one past class** will not reflect the real value.
        The classes from which the tensor inputs are collected depend on the `only_classes` attribute of
        :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        As a bare minimum, the tensor inputs are **always** collected for the samples in the current training
        mini-batch.

    .note::
        R_NIC_step and R_ALL_NIC_step of a past class are the counterparts of NIC_step and ALL_NIC_step, respectively,
        using the interference onto the average rehearsal gradient of that class w.r.t. the weight and bias in the last
        layer  rather than onto its original last-layer average gradient, i.e. the gradient of that class w.r.t. the
        weight and bias in the last layer evaluated over all its original training samples.
    """

    def __init__(self,
                 optimize_R_NIC_ALL_NIC: Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Sequence[int], Union[float, int], Union[float, int], Union[float, int], int, int, str, bool]],
                 compiled: bool = False,
                 eq_constraints: Optional[str] = None):
        """
        Construct an instance of `NewClassesLastLayerInterferenceInterventionViaSoftmax`

        :param optimize_R_NIC_ALL_NIC: a callable that takes in four 1D tensors and returns a tuple of eight elements.
            The first input tensor stores at index i the actual unweighted mini-batch NIC_step value of past class i at
            the current point in optimization space, the second input tensor stores at index i the actual unweighted
            mini-batch ALL_NIC_step value of past class i at the current point in optimization space, the third input
            tensor stores at index i the actual unweighted mini-batch R_NIC_step value of past class i at the current
            point in optimization space, the fourth input tensor stores at index i the actual unweighted mini-batch
            R_ALL_NIC_step value of past class i at the current point in optimization space. The size of the input
            tensors is `n_past_classes`, where `n_past_classes` denotes the total number of past classes. These input
            tensors may have some NaN entries. The first element of the returned tuple needs to be a 1D sequence of
            **different** integers representing the indexes of the past classes whose average unweighted mini-batch
            NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value needs to be minimized/maximized at the current point
            in optimization space through the function `min_max_last_layer_interf_via_softmax`
            or its compiled counterpart if `eq_constraints` is None; otherwise,
            `min_max_last_layer_interf_via_softmax_eq_constraints` or its compiled counterpart (compiled with
            torch.compile(mode="reduce-overhead")) is used. If an empty sequence, no average unweighted mini-batch
            NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value is minimized/maximized.
            The second element of the returned tuple needs to be either a float or an int; it is converted to a
            single-element 1D tensor, of same data type as `logits`, and passed as `epsilon` to
            `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
            compiled counterparts. The third element of the returned tuple needs to be either a float or an int
            representing the lower-bound (for minimization) or upper-bound (for maximization) of the average
            unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step; it is converted to a single-element
            1D tensor, of same data type as `logits`, transformed and finally passed as `eta` to
            `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
            compiled counterparts (it is transformed prior to being fed into those functions because those solve a
            minimization/maximization problem that is equivalent to the one that needs to be solved but it is scaled
            by a constant). The fourth element of the returned tuple needs to be either a float or an int; it is
            converted to a single-element 1D tensor, of same data type as `logits`, and passed as `min_max` to
            `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
            compiled counterparts. The fifth element of the returned tuple needs to be an integer; it is passed as
            `expansion_steps` to `min_max_last_layer_interf_via_softmax`,
            `min_max_last_layer_interf_via_softmax_eq_constraints` or their compiled counterparts. The sixth element of
            the returned tuple needs to be an integer; it is passed as `bisection_steps` to
            `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
            compiled counterparts. The seventh element of the returned tuple needs to be a string; `NIC_step` if the
            average unweighted mini-batch NIC_step of the given past classes needs to be minimized/maximized,
            `ALL_NIC_step` if their average unweighted mini-batch ALL_NIC_step needs to be minimized/maximized,
            `R_NIC_step` if their average unweighted mini-batch R_NIC_step needs to be minimized/maximized,
            `R_ALL_NIC_step` if their unweighted mini-batch R_ALL_NIC_step needs to be minimized/maximized.
            The eighth element of the returned tuple needs to be a boolean; True if the interventions affecting the
            last-layer parameters need to backpropagate to earlier layers, False otherwise.
        :param compiled: whether to use the compiled counterpart of `min_max_last_layer_interf_via_softmax`, i.e.,
            `torch.compile(min_max_last_layer_interf_via_softmax, mode="reduce-overhead")` if `eq_constraints` is None
            or the compiled counterpart of `min_max_last_layer_interf_via_softmax_eq_constraints`, i.e.,
            `torch.compile(min_max_last_layer_interf_via_softmax_eq_constraints, mode="reduce-overhead")`,
            if `eq_constraints` is not None. Default is False.
        :param eq_constraints: `correct_class` or None. If `correct_class`, the average unweighted
            mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value is minimized/maximized by allowing all
            softmax probabilities output by the network for a given new-class sample to change except for that
            corresponding to the correct class. If None, no equality constraints are set while minimizing/maximizing the
            average unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value. Default is None.
        """
        if eq_constraints is not None:
            if eq_constraints not in ["correct_class"]:
                raise ValueError("`eq_constraints` must be either None or `correct_class`")

        self.optimize_R_NIC_ALL_NIC: Callable[[Tensor, Tensor, Tensor, Tensor],
            Tuple[Sequence[int], Union[float, int], Union[float, int], Union[float, int], int, int, str, bool]] = optimize_R_NIC_ALL_NIC
        """
        a callable that takes in four 1D tensors and returns a tuple of eight elements.
        The first input tensor stores at index i the actual unweighted mini-batch NIC_step value of past class i at
        the current point in optimization space, the second input tensor stores at index i the actual unweighted
        mini-batch ALL_NIC_step value of past class i at the current point in optimization space, the third input
        tensor stores at index i the actual unweighted mini-batch R_NIC_step value of past class i at the current
        point in optimization space, the fourth input tensor stores at index i the actual unweighted mini-batch
        R_ALL_NIC_step value of past class i at the current point in optimization space. The size of the input
        tensors is `n_past_classes`, where `n_past_classes` denotes the total number of past classes. These input
        tensors may have some NaN entries. The first element of the returned tuple needs to be a 1D sequence of
        **different** integers representing the indexes of the past classes whose average unweighted mini-batch
        NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value needs to be minimized/maximized at the current point in
        optimization space through the function `min_max_last_layer_interf_via_softmax` or its compiled counterpart
        if `self.eq_constraints` is None;
        otherwise, `min_max_last_layer_interf_via_softmax_eq_constraints` or its compiled counterpart (compiled with
        torch.compile(mode="reduce-overhead")) is used. If an empty sequence, no average unweighted mini-batch
        NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value is minimized/maximized.
        The second element of the returned tuple needs to be either a float or an int; it is converted to a
        single-element 1D tensor, of same data type as `logits`, and passed as `epsilon` to
        `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
        compiled counterparts. The third element of the returned tuple needs to be either a float or an int
        representing the lower-bound (for minimization) or upper-bound (for maximization) of the average
        unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step; it is converted to a single-element
        1D tensor, of same data type as `logits`, transformed and finally passed as `eta` to
        `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
        compiled counterparts (it is transformed prior to being fed into those functions because those solve a
        minimization/maximization problem that is equivalent to the one that needs to be solved but it is scaled
        by a constant). The fourth element of the returned tuple needs to be either a float or an int; it is
        converted to a single-element 1D tensor, of same data type as `logits`, and passed as `min_max` to
        `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
        compiled counterparts. The fifth element of the returned tuple needs to be an integer; it is passed as
        `expansion_steps` to `min_max_last_layer_interf_via_softmax`,
        `min_max_last_layer_interf_via_softmax_eq_constraints` or their compiled counterparts. The sixth element of
        the returned tuple needs to be an integer; it is passed as `bisection_steps` to
        `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or their
        compiled counterparts. The seventh element of the returned tuple needs to be a string; `NIC_step` if the
        average unweighted mini-batch NIC_step of the given past classes needs to be minimized/maximized,
        `ALL_NIC_step` if their average unweighted mini-batch ALL_NIC_step needs to be minimized/maximized,
        `R_NIC_step` if their average unweighted mini-batch R_NIC_step needs to be minimized/maximized,
        `R_ALL_NIC_step` if their unweighted mini-batch R_ALL_NIC_step needs to be minimized/maximized.
        The eighth element of the returned tuple needs to be a boolean; True if the interventions affecting the
        last-layer parameters need to backpropagate to earlier layers, False otherwise.
        """

        self.compiled: bool = compiled
        """
        whether to use the compiled counterpart of `min_max_last_layer_interf_via_softmax`, i.e.,
        `torch.compile(min_max_last_layer_interf_via_softmax, mode="reduce-overhead")`, if `self.eq_constraints` is None
        or the compiled counterpart of `min_max_last_layer_interf_via_softmax_eq_constraints`, i.e.,
        `torch.compile(min_max_last_layer_interf_via_softmax_eq_constraints, mode="reduce-overhead")`,
        if `self.eq_constraints` is not None.
        """

        self.opt = None
        """
        It is either None, `torch.compile(min_max_last_layer_interf_via_softmax, mode="reduce-overhead")` if
        `self.eq_constraint` is None,
        or `torch.compile(min_max_last_layer_interf_via_softmax_eq_constraints, mode="reduce-overhead")` otherwise
        """

        self.eq_constraints: Optional[str] = eq_constraints
        """
        `correct_class` or None. If `correct_class`, the average unweighted
        mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value is minimized/maximized by allowing all
        softmax probabilities output by the network for a given new-class sample to change except for that
        corresponding to the correct class. If None, no equality constraints are set while minimizing/maximizing the
        average unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step value.
        """


    def __getstate__(self):
        """
        During object serialization, let `self.opt` be serialized as None; this prevents serialization of
        `torch.compile(min_max_last_layer_interf_via_softmax, mode="reduce-overhead")` or
        `torch.compile(min_max_last_layer_interf_via_softmax_eq_constraints, mode="reduce-overhead")`,
        which cannot be performed.
        """
        state = self.__dict__.copy()
        state["opt"] = None
        return state


    def __call__(self,
                 logits: Tensor,
                 embeds: Tensor,
                 targets: Tensor,
                 indices: Tensor,
                 strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate) \
            -> Union[Tuple[None, None, None], Tuple[None, None, Dict[str, Union[ndarray, Tuple[ndarray, ...]]]], Tuple[Tensor, Tensor, Dict[str, Union[ndarray, Tuple[ndarray, ...]]]]]:
        """
        Minimize/maximize the average unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step
        computed over a set of past classes at the current point in optimisation space through the function
        `min_max_last_layer_interf_via_softmax`, `min_max_last_layer_interf_via_softmax_eq_constraints` or one of their
        compiled counterparts (compiled with torch.compile(mode="reduce-overhead")).

        :param logits: a 2D tensor of predicted raw logits.
        :param embeds: a 2D tensor of corresponding embeddings.
        :param targets: a 1D tensor of corresponding targets.
        :param indices: a 1D tensor of corresponding indices.
        :param strategy: the underlying strategy template.
        :return: a 3-tuple of Nones during the first experience. For all subsequent experiences, a 3-tuple is returned,
        where the last element is a dictionary containing strings as keys and tuples of numpy arrays or numpy arrays as
        values (this dictionary is appended into the `data` property of
        :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`)
        and the first two elements differ according to the following: they are both None if the first element returned
        by `optimize_R_NIC_ALL_NIC` is an empty sequence or the eighth element is True; they are tensors if the first
        element returned by `optimize_R_NIC_ALL_NIC` is not an empty sequence and the eighth element is False.

        .note::
            As described in :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`,
            the tensor inputs are collected prior to performing the actual update step at that point in optimisation
            space. The classes from which the tensor inputs are collected depend on the `only_classes` attribute of
            :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
            As a bare minimum, the tensor inputs are **always** collected for the samples in the current training
            mini-batch.

        .note::
            If no tensor inputs are received from at least one of the past classes returned by the first output element
            of `optimize_R_NIC_ALL_NIC`, the average unweighted mini-batch
            NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step over the
            given set of past classes cannot be computed and thereby cannot be maximized/minimized.
            A :class`RuntimeError` is raised in these cases.

        .note::
            If, for at least one of the past classes returned by the first output element of `optimize_R_NIC_ALL_NIC`,
            tensor inputs are received but none of them are associated with samples of that class currently stored in
            the rehearsal buffer, the average unweighted mini-batch R_NIC_step/R_ALL_NIC_step over the given set of
            past classes cannot be computed and thereby cannot be maximized/minimized.
            A :class`RuntimeError` is raised in these cases.

        .note::
            This method is called by :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            within a torch.no_grad() block

        .note::
            This method **assumes** that the loss used is the standard multi-class cross entropy.

        .note::
            This method **assumes** that an instance of
            :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is present as a
            plugin in the given strategy. A :class:`RuntimeError` is raised, otherwise.

        .note::
            This method **assumes** that `logits`, `embeds`, `targets`, `indices` contain no NaN values.

        .note::
            This method **assumes** that each mini-batch contains samples from both the rehearsal buffer and the new-class
            training dataset. A :class:`RuntimeError` is raised, otherwise.
        """
        if strategy.experience.current_experience == 0:
            return None, None, None

        # get the total number of classes
        n_classes = logits.shape[1]
        # get the number of old classes by subtracting to the total number the number of classes in the new experience
        n_old_classes = n_classes - len(strategy.experience.classes_in_this_experience)
        # retrieve the indices of all the samples in the rehearsal buffer; raise a RuntimeError if an instance of
        # ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate is not present in the
        # underlying strategy as a plugin
        replay = next((x for x in strategy.plugins
                       if isinstance(x,
                                     ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)),
                      None)
        if replay is None:
            raise RuntimeError("The underlying strategy must have an instance of "
                               "`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` as "
                               "a plugin")
        rehearsal_indices = torch.tensor(replay.storage_policy.buffer.dataset_indices, device=logits.device)
        # retrieve the indices of all the samples in the current mini-batch
        mb_indices = strategy.mb_dataset_indices
        # compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
        # classification head (last layer) for all training samples received
        sample_wise_grads = sample_wise_mce_grads_wrt_weights_biases(logits, targets, embeds)
        # a mask for retrieving data from the training samples received that are in the rehearsal buffer
        mask_rehearsal = torch.isin(indices, rehearsal_indices)
        # a mask for retrieving data from the training samples received that are in the current mini-batch
        mask_mb = torch.isin(indices, mb_indices)
        # construct a new `targets` tensor containing the same targets as before for the past classes but assigns all
        # the new classes the same target, i.e. the value of `n_old_classes`. This way, below, it will be easy to
        # compute an overall average gradient for the new classes. It is as all new classes belong to the same overall
        # class "NEW CLASSES"
        targets = torch.where(targets >= n_old_classes, n_old_classes, targets)
        # compute the number of samples received from each past class and the overall new classes class;
        # it is a 1D tensor of length n_old_classes+1
        n_train_samples = torch.bincount(targets, minlength=n_old_classes + 1)
        # compute the number of samples received from each past class that are currently stored in the rehearsal buffer;
        # it is a 1D tensor of length n_old_classes
        n_reh_samples = torch.bincount(targets[mask_rehearsal], minlength=n_old_classes)
        # compute the number of samples received from each past class and the overall new classes class that are in the
        # current training mini-batch; it is a 1D tensor of length n_old_classes + 1
        n_mb_samples = torch.bincount(targets[mask_mb], minlength=n_old_classes + 1)
        if n_mb_samples[:-1].sum() == 0 or n_mb_samples[-1] == 0:
            raise RuntimeError("Each mini-batch must contain samples from both the rehearsal buffer and the new-class "
                               "training dataset")

        # compute the average gradient for each past class and the overall new classes class, evaluated over all the
        # training samples received from that class. It is a 2D tensor of shape ([n_old_classes + 1] x grad_size), where
        # grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if no
        # samples are received from a class, then its average gradient is a vector of nan.
        train_dataset_avg_grads = class_avg_grads(sample_wise_grads, targets, n_old_classes + 1)
        # compute the average gradient for each past class, evaluated over all the samples received from that class that
        # are currently stored in the rehearsal buffer. It is a 2D tensor of shape (n_old_classes x grad_size), where
        # grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if no
        # samples currently stored in the rehearsal buffer are received from a class, then its average gradient is a
        # vector of nan.
        rehearsal_avg_grads = class_avg_grads(sample_wise_grads[mask_rehearsal], targets[mask_rehearsal], n_old_classes)
        # compute the average gradient for each past class and the overall new classes class, evaluated over all the
        # samples of that class in the current mini-batch. It is a 2D tensor of shape ([n_old_classes+1] x grad_size),
        # where grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if a
        # class has no samples in the current mini-batch, then its average gradient is a vector of nan.
        mb_avg_grads = class_avg_grads(sample_wise_grads[mask_mb], targets[mask_mb], n_old_classes + 1)

        # compute the unweighted rehearsal SIC_step for each past class
        sic_step_reh = SIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the unweighted mini-batch SIC_step for each past class
        sic_step_mb = SIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
        # if at least one training sample is received from each past class, it is possible to compute both rehearsal and
        # mini-batch CIC_step for each past class
        if (n_train_samples[:-1] > 0).all():
            # compute the unweighted rehearsal CIC_step for each past class
            cic_step_reh = CIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], n_reh_samples, n_train_samples[:-1],
                                    include_norm_cos=True)
            # compute the unweighted mini-batch CIC_step for each past class
            cic_step_mb = CIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], n_mb_samples[:-1], n_train_samples[:-1],
                                   include_norm_cos=True)
        # otherwise it is not possible to compute those CIC-based metrics, as a consequence both the rehearsal and
        # mini-batch CIC_step for each past class are assigned NaN
        else:
            cic_step_reh = tuple(torch.full_like(sic_step_reh[0], float("nan")) for _ in range(3))
            cic_step_mb = tuple(torch.full_like(sic_step_reh[0], float("nan")) for _ in range(3))
        # compute the unweighted full-training NIC_step for each past class
        nic_step_train = NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes,
                                  include_norm_cos=True)
        # compute the unweighted mini-batch NIC_step for each past class
        nic_step_mb = NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes, include_norm_cos=True)
        # compute the unweighted full-training ALL_NIC_step for each past class
        all_nic_step_train = ALL_NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1],
                                          include_norm_cos=True)
        # compute the unweighted mini-batch ALL_NIC_step for each past class
        all_nic_step_mb = ALL_NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the unweighted full_training R_NIC_step for each past class
        r_nic_step_train = NIC_step(train_dataset_avg_grads[-1], rehearsal_avg_grads, n_classes, include_norm_cos=True)
        # compute the unweighted mini-batch R_NIC_step for each past class
        r_nic_step_mb = NIC_step(mb_avg_grads[-1], rehearsal_avg_grads, n_classes, include_norm_cos=True)
        # compute the unweighted full-training R_ALL_NIC_step for each past class
        r_all_nic_step_train = ALL_NIC_step(train_dataset_avg_grads[-1], rehearsal_avg_grads, include_norm_cos=True)
        # compute the unweighted mini-batch R_ALL_NIC_step for each past class
        r_all_nic_step_mb = ALL_NIC_step(mb_avg_grads[-1], rehearsal_avg_grads, include_norm_cos=True)

        # create a dictionary that will be returned by this function; all the tensors are converted to numpy arrays
        data = dict(sic_step_reh=tuple(x.detach().cpu().numpy() for x in sic_step_reh),
                    sic_step_mb=tuple(x.detach().cpu().numpy() for x in sic_step_mb),
                    cic_step_reh=tuple(x.detach().cpu().numpy() for x in cic_step_reh),
                    cic_step_mb=tuple(x.detach().cpu().numpy() for x in cic_step_mb),
                    nic_step_train=tuple(x.detach().cpu().numpy() for x in nic_step_train),
                    nic_step_mb=tuple(x.detach().cpu().numpy() for x in nic_step_mb),
                    all_nic_step_train=tuple(x.detach().cpu().numpy() for x in all_nic_step_train),
                    all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in all_nic_step_mb),
                    r_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_nic_step_train),
                    r_nic_step_mb=tuple(x.detach().cpu().numpy() for x in r_nic_step_mb),
                    r_all_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_all_nic_step_train),
                    r_all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in r_all_nic_step_mb),
                    train_dataset_avg_grads_norm=torch.linalg.norm(train_dataset_avg_grads,
                                                                   dim=1).detach().cpu().numpy(),
                    rehearsal_avg_grads_norm=torch.linalg.norm(rehearsal_avg_grads, dim=1).detach().cpu().numpy(),
                    mb_avg_grads_norm=torch.linalg.norm(mb_avg_grads, dim=1).detach().cpu().numpy(),
                    cos_train_rehearsal_avg_grads=tF.cosine_similarity(train_dataset_avg_grads[:-1], rehearsal_avg_grads,
                                                                       dim=1).detach().cpu().numpy(),
                    cos_train_mb_avg_grads=tF.cosine_similarity(train_dataset_avg_grads, mb_avg_grads,
                                                                dim=1).detach().cpu().numpy(),
                    n_train_samples=n_train_samples.detach().cpu().numpy(),
                    n_reh_samples=n_reh_samples.detach().cpu().numpy(),
                    n_mb_samples=n_mb_samples.detach().cpu().numpy(),
                    indices=indices.detach().cpu().numpy(),
                    mb_indices=mb_indices.detach().cpu().numpy()
                    )

        # set the returned gradient w.r.t. the weights in the last layer to None by default
        avg_grad_wrt_weights = None
        # set the returned gradient w.r.t. the bias in the last layer to None by default
        avg_grad_wrt_bias = None

        # get the output of `optimize_R_NIC_ALL_NIC`
        id_past_classes, epsilon, eta, min_max, expansion_steps, bisection_steps, metric_target, is_backpropagated = (
            self.optimize_R_NIC_ALL_NIC(nic_step_mb[0], all_nic_step_mb[0], r_nic_step_mb[0], r_all_nic_step_mb[0]))

        if len(id_past_classes) > 0:
            allowed_metric_targets = ["NIC_step", "ALL_NIC_step", "R_NIC_step", "R_ALL_NIC_step"]
            if metric_target not in allowed_metric_targets:
                raise RuntimeError("The metric provided by `optimize_R_NIC_ALL_NIC` is not valid")

            # transform into 1D tensors
            id_past_classes, epsilon, eta, min_max = (torch.tensor(id_past_classes, device=logits.device,
                                                                 dtype=torch.int64),
                                                    torch.tensor([epsilon], device=logits.device, dtype=logits.dtype),
                                                    torch.tensor([eta], device=logits.device, dtype=logits.dtype),
                                                    torch.tensor([min_max], device=logits.device, dtype=logits.dtype))
            # a mask for retrieving only data of the samples in the current mini-batch that belong to new classes
            mask_mb_new_classes = strategy.mb_y_incremental_classifier >= n_old_classes
            # get the logits of all the samples in the current mini-batch belonging to new classes, and transform them into
            # softmax values
            mb_new_classes_softmax = strategy.mb_output_incremental_classifier[mask_mb_new_classes].softmax(dim=1)
            # get the respective embeddings of all the samples in the current mini-batch belonging to new classes
            mb_new_classes_embeds = strategy.mb_feature_incremental_classifier[mask_mb_new_classes]
            # get the respective targets of all the samples in the current mini-batch belonging to new classes
            mb_new_classes_targets = strategy.mb_y_incremental_classifier[mask_mb_new_classes]
            # get the respective indices of all the samples in the current mini-batch belonging to new classes
            mb_new_classes_indices = mb_indices[mask_mb_new_classes]
            # create the inputs for `min_max_last_layer_interf_via_softmax`,
            # `min_max_last_layer_interf_via_softmax_eq_constraints` or one of their compiled counterparts
            S = mb_new_classes_softmax.unsqueeze(0)
            H = torch.eye(n_classes, device=S.device, dtype=S.dtype)[mb_new_classes_targets].unsqueeze(0)
            F = torch.cat((mb_new_classes_embeds,
                           mb_new_classes_embeds.new_ones((mb_new_classes_embeds.shape[0], 1))), dim=1).unsqueeze(0)
            # if the average unweighted mini-batch NIC_step/ALL_NIC_step computed over the given past classes needs to
            # be intervened upon
            if metric_target in allowed_metric_targets[:2]:
                if (n_train_samples[id_past_classes] == 0).any():
                    raise RuntimeError(
                        "No tensor inputs are received from at least one of the past classes returned by "
                        "the first output element of `optimize_R_NIC_ALL_NIC`. Consequently, the average "
                        "unweighted mini-batch NIC_step/ALL_NIC_step over the given set of past classes "
                        "cannot be maximized/minimized.")
                train_dataset_reh_avg_grads = train_dataset_avg_grads
            else:
                if (n_reh_samples[id_past_classes] == 0).any():
                    raise RuntimeError(
                        "No tensor inputs associated with samples in the rehearsal buffer are received from at least "
                        "one of the past classes returned by the first output element of `optimize_R_NIC_ALL_NIC`. "
                        "Consequently, the average unweighted mini-batch R_NIC_step/R_ALL_NIC_step over the given set "
                        "of past classes cannot be maximized/minimized.")
                train_dataset_reh_avg_grads = rehearsal_avg_grads

            # convert the average gradient vector w.r.t. the weights and bias in the last layer for each of the given
            # past classes, each computed over that class' received training or rehearsal samples,
            # into a 2D matrix. The result is a 3D tensor, whose first dimension has size equal to
            # `len(id_past_classes)`.
            unflattened_grads = unflatten_grads_wrt_weights_biases(train_dataset_reh_avg_grads[id_past_classes],
                                                                  n_classes)
            # if the average unweighted mini-batch NIC_step/R_NIC_step of the given past classes needs to be intervened
            # upon
            if metric_target in [allowed_metric_targets[0], allowed_metric_targets[2]]:
                # construct a tensor of zeros of same shape as `unflattened_grads`
                V = torch.zeros_like(unflattened_grads)
                # construct a tensor for indexing along the first dimension of `V`
                idx = torch.arange(V.shape[0], device=V.device)
                # for each matrix in `V`, fill the row corresponding to the weights and bias associated with the given
                # past class with the corresponding row in the respective matrix in `unflattened_grads`
                V[idx, id_past_classes] = unflattened_grads[idx, id_past_classes]
            else:  # otherwise
                V = unflattened_grads

            if len(id_past_classes) > 1:
                tiny = torch.finfo(V.dtype).tiny
                # compute the Frobenius norm of each matrix in V
                V_norm = torch.linalg.norm(V, dim=(1, 2))
                # normalize each matrix in V; `clamp_min(.)` ensures no division by 0 or very small numbers that
                # underflow to 0
                V = V / V_norm.clamp_min(tiny)[:, None, None]
                # compute the mean over each matrix in V
                V = V.mean(dim=0, keepdim=True)
                # To remain consistent, the lower-bound (for minimization) or upper-bound (for maximization) of the
                # average unweighted mini-batch NIC_step/ALL_NIC_step/R_NIC_step/R_ALL_NIC_step must be weighted by the
                # inverse of the Frobenius norm of `V`.
                eta = eta / torch.linalg.norm(V, dim=(1, 2)).clamp_min(tiny)

            m = n_mb_samples[-1:]
            inputs = dict(S=S, H=H, F=F, V=V, m=m, epsilon=epsilon, eta=eta, min_max=min_max,
                          expansion_steps=expansion_steps, bisection_steps=bisection_steps)
            if self.eq_constraints == "correct_class":
                inputs = {**inputs, "EQ": H.bool()}
            # run the optimization problem
            result_optimization = self.get_opt()(**inputs)

            # get the new matrix of softmax probabilities for all the samples in the current mini-batch belonging to new
            # classes
            new_S = result_optimization[0]
            # get the average vector of softmax probabilities for all the samples in the current mini-batch belonging
            # to new classes, computed prior to running the optimization problem; it is a 1D tensor of size `n_classes`
            avg_mb_new_classes_softmax = S.mean(dim=1).squeeze(0)
            # get the average vector of softmax probabilities for all the samples in the current mini-batch belonging to
            # new classes, resulting from the new matrix of softmax probabilities returned by
            # the optimization problem;  it is a 1D tensor of size `n_classes`
            post_avg_mb_new_classes_softmax = new_S.mean(dim=1).squeeze(0)
            # compute the gradient of the multi-class cross-entropy loss w.r.t the logits for all the samples in the
            # current mini-batch belonging to new classes, using the
            # softmax probabilities returned by the optimization problem
            post_mb_new_classes_sample_wise_log_grads = (new_S - H).squeeze(0)
            post_mb_new_classes_softmax = new_S.squeeze(0)
            # compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
            # classification head (last layer) for all the samples in the current mini-batch belonging to new classes
            # using `post_mb_new_classes_softmax`;
            post_mb_new_classes_sample_wise_grads = sample_wise_mce_grads_wrt_weights_biases(
                post_mb_new_classes_softmax, mb_new_classes_targets, mb_new_classes_embeds, logits=False)
            # update rows in `sample_wise_grads` with rows from `post_mb_new_classes_sample_wise_grads`
            post_sample_wise_grads = update_rows_by_id(t1=sample_wise_grads, t2=post_mb_new_classes_sample_wise_grads,
                                                       id1=indices, id2=mb_new_classes_indices, in_place=True)
            # compute the average post-intervention gradient of the multi-class cross-entropy loss w.r.t the weights and
            # biases in the classification head (last layer), evaluated over all the samples in the current mini-batch
            # belonging to new classes
            avg_post_mb_new_classes_grad = post_mb_new_classes_sample_wise_grads.mean(dim=0)
            # compute the average post-intervention gradient of the multi-class cross-entropy loss w.r.t the weights and
            # biases in the classification head (last layer), evaluated over all training samples received from
            # new classes
            avg_post_train_new_classes_grad = post_sample_wise_grads[targets == n_old_classes].mean(dim=0)

            # compute the post-intervention unweighted full-training NIC_step for each past class
            post_nic_step_train = NIC_step(avg_post_train_new_classes_grad, train_dataset_avg_grads[:-1],
                                           n_classes, include_norm_cos=True)
            # compute the post-intervention unweighted mini-batch NIC_step for each past class
            post_nic_step_mb = NIC_step(avg_post_mb_new_classes_grad, train_dataset_avg_grads[:-1],
                                        n_classes, include_norm_cos=True)
            # compute the post-intervention unweighted full-training ALL_NIC_step for each past class
            post_all_nic_step_train = ALL_NIC_step(avg_post_train_new_classes_grad, train_dataset_avg_grads[:-1],
                                                   include_norm_cos=True)
            # compute the post-intervention unweighted mini-batch ALL_NIC_step for each past class
            post_all_nic_step_mb = ALL_NIC_step(avg_post_mb_new_classes_grad, train_dataset_avg_grads[:-1],
                                                include_norm_cos=True)
            # compute the post-intervention unweighted full-training R_NIC_step for each past class
            post_r_nic_step_train = NIC_step(avg_post_train_new_classes_grad, rehearsal_avg_grads,
                                             n_classes, include_norm_cos=True)
            # compute the post-intervention unweighted mini-batch R_NIC_step for each past class
            post_r_nic_step_mb = NIC_step(avg_post_mb_new_classes_grad, rehearsal_avg_grads,
                                          n_classes, include_norm_cos=True)
            # compute the post-intervention unweighted full-training R_ALL_NIC_step for each past class
            post_r_all_nic_step_train = ALL_NIC_step(avg_post_train_new_classes_grad, rehearsal_avg_grads,
                                                     include_norm_cos=True)
            # compute the post-intervention unweighted mini-batch R_ALL_NIC_step for each past class
            post_r_all_nic_step_mb = ALL_NIC_step(avg_post_mb_new_classes_grad, rehearsal_avg_grads,
                                                  include_norm_cos=True)

            data_add = dict(result_optimization=tuple(x.detach().cpu().numpy() for x in result_optimization[1:]),
                            post_nic_step_train=tuple(x.detach().cpu().numpy() for x in post_nic_step_train),
                            post_nic_step_mb=tuple(x.detach().cpu().numpy() for x in post_nic_step_mb),
                            post_all_nic_step_train=tuple(x.detach().cpu().numpy() for x in post_all_nic_step_train),
                            post_all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in post_all_nic_step_mb),
                            post_r_nic_step_train=tuple(x.detach().cpu().numpy() for x in post_r_nic_step_train),
                            post_r_nic_step_mb=tuple(x.detach().cpu().numpy() for x in post_r_nic_step_mb),
                            post_r_all_nic_step_train=tuple(
                                x.detach().cpu().numpy() for x in post_r_all_nic_step_train),
                            post_r_all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in post_r_all_nic_step_mb),
                            id_past_classes=id_past_classes.detach().cpu().numpy(),
                            avg_mb_new_classes_softmax=avg_mb_new_classes_softmax.detach().cpu().numpy(),
                            post_avg_mb_new_classes_softmax=post_avg_mb_new_classes_softmax.detach().cpu().numpy()
                            )
            data = {**data, **data_add}

            if is_backpropagated:
                # attach a hook to the logits output by the network so to replace the original sample-wise gradient wrt
                # logits, divided by the number of samples in the current mini-batch, of each sample in the current
                # mini-batch belonging to new classes with the respective one in
                # `post_mb_new_classes_sample_wise_log_grads`. The hook does not need to be removed as the logits output is
                # a non-leaf tensor, i.e. each SGD step produces a new logits tensor
                strategy.mb_output_incremental_classifier.register_hook(lambda grad: update_rows_by_id(
                    t1=grad,
                    t2=post_mb_new_classes_sample_wise_log_grads / grad.shape[0],
                    id1=mb_indices,
                    id2=mb_new_classes_indices,
                    in_place=False
                ))
            else:
                # compute the average post-intervention gradient of the multi-class cross-entropy loss w.r.t the weights and
                # biases in the classification head (last layer), evaluated over all the samples in the current mini-batch
                avg_grad_wrt_weights_bias = post_sample_wise_grads[mask_mb].mean(dim=0)
                # unflatten `avg_grad_wrt_weights_bias`
                avg_grad_wrt_weights_bias = unflatten_grads_wrt_weights_biases(avg_grad_wrt_weights_bias.unsqueeze(0),
                                                                               n_classes).squeeze(0)
                avg_grad_wrt_weights = avg_grad_wrt_weights_bias[:, :-1]
                avg_grad_wrt_bias = avg_grad_wrt_weights_bias[:, -1]

        return avg_grad_wrt_weights, avg_grad_wrt_bias, data


    def _get_opt_func(self):
        if self.eq_constraints is None:
            return min_max_last_layer_interf_via_softmax
        if self.eq_constraints == "correct_class":
            return min_max_last_layer_interf_via_softmax_eq_constraints

        raise ValueError("`self.eq_constraints` must be either None or `correct_class`")


    def get_opt(self):
        opt_f = self._get_opt_func()

        if self.compiled:
            if self.opt is None:
                self.opt = torch.compile(opt_f, mode="reduce-overhead")
            return self.opt
        else:
            return opt_f


class LLIFCDataCollection:
    """
    A class whose instance is intended to be passed as argument for `intervention` when constructing an instance
    of :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The latter will call the `__call__` method of such an instance each time an SGD iteration is scheduled to
    incur an intervention in the last-layer gradient.

    Calling `__call__` computes the following metrics for each past class:
        - the unweighted rehearsal SIC_step
        - the unweighted mini-batch SIC_step
        - the unweighted rehearsal CIC_step
        - the unweighted mini-batch CIC_step
        - the unweighted OIC_step
        - the unweighted rehearsal RIC_step
        - the unweighted mini-batch RIC_step
        - the unweighted full-training NIC_step
        - the unweighted mini-batch NIC_step
        - the unweighted full-training ALL_NIC_step
        - the unweighted mini-batch ALL_NIC_step
        - the unweighted full-training R_NIC_step
        - the unweighted mini-batch R_NIC_step
        - the unweighted full-training R_ALL_NIC_step
        - the unweighted mini-batch R_ALL_NIC_step

    These metrics, and additional data, are collected via a dictionary and returned by the `__call__` method.
    Such a dictionary will be appended into the `data` property of
    :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The code of the `__call__` method needs to be inspected to find out what data the returned dictionary contains. No
    intervention is performed by the `__call__` method.
    Note that the metrics and additional data are not computed during the first experience,
    as there are no past classes during the first experience. This is performed by checking the ID of the current
    experience, assuming that the ID of the first experience is 0.

    .note::
        The metrics computed by the `__call__` method reflect the real metric values for **all** past classes **only**
        if the tensor inputs received from an instance of :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        are collected for **all** training samples of **all** the classes encountered so far (both the
        classes in the current experience and the classes in the previously encountered experiences). In all other
        cases, one or more metrics for **at least one past class** will not reflect the real value.
        The classes from which the tensor inputs are collected depend on the `only_classes` attribute of
        :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        As a bare minimum, the tensor inputs are **always** collected for the samples in the current training
        mini-batch.

    .note::
        R_NIC_step and R_ALL_NIC_step of a past class are the counterparts of NIC_step and ALL_NIC_step, respectively,
        using the interference onto the average rehearsal gradient of that class w.r.t. the weight and bias in the last
        layer  rather than onto its original last-layer average gradient, i.e. the gradient of that class w.r.t. the
        weight and bias in the last layer evaluated over all its original training samples.
    """
    def __call__(self,
                 logits: Tensor,
                 embeds: Tensor,
                 targets: Tensor,
                 indices: Tensor,
                 strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate) \
            -> Union[Tuple[None, None, Dict[str, Union[ndarray, Tuple[ndarray, ...]]]], Tuple[None, None, None]]:
        """
        Collect metrics and additional data.

        :param logits: a 2D tensor of predicted raw logits.
        :param embeds: a 2D tensor of corresponding embeddings.
        :param targets: a 1D tensor of corresponding targets.
        :param indices: a 1D tensor of corresponding indices.
        :param strategy: the underlying strategy template.
        :return: a 3-tuple of Nones during the first experience. For all subsequent experiences, a 3-tuple is returned,
        where the last element is a dictionary containing strings as keys and tuples of numpy arrays or numpy arrays as
        values (this dictionary is appended into the `data` property of
        :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`)
        and the first two elements are None.

        .note::
            As described in :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`,
            the tensor inputs are collected prior to performing the actual update step at that point in optimisation
            space. The classes from which the tensor inputs are collected depend on the `only_classes` attribute of
            :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
            As a bare minimum, the tensor inputs are **always** collected for the samples in the current training
            mini-batch.

        .note::
            This method is called by :class:`training.plugins.LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            within a torch.no_grad() block

        .note::
            This method **assumes** that the loss used is the standard multi-class cross entropy.

        .note::
            This method **assumes** that an instance of
            :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is present as a
            plugin in the given strategy. A :class:`RuntimeError` is raised, otherwise.

        .note::
            This method **assumes** that `logits`, `embeds`, `targets`, `indices` contain no NaN values.

        .note::
            This method **assumes** that each mini-batch contains samples from both the rehearsal buffer and the new-class
            training dataset. A :class:`RuntimeError` is raised, otherwise.
        """
        if strategy.experience.current_experience == 0:
            return None, None, None

        # get the total number of classes
        n_classes = logits.shape[1]
        # get the number of old classes by subtracting to the total number the number of classes in the new experience
        n_old_classes = n_classes - len(strategy.experience.classes_in_this_experience)
        # retrieve the indices of all the samples in the rehearsal buffer; raise a RuntimeError if an instance of
        # ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate is not present in the
        # underlying strategy as a plugin
        replay = next((x for x in strategy.plugins
                       if isinstance(x,
                                     ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)),
                      None)
        if replay is None:
            raise RuntimeError("The underlying strategy must have an instance of "
                               "`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` as "
                               "a plugin")
        rehearsal_indices = torch.tensor(replay.storage_policy.buffer.dataset_indices, device=logits.device)
        # retrieve the indices of all the samples in the current mini-batch
        mb_indices = strategy.mb_dataset_indices
        # compute the gradient of the multi-class cross-entropy loss w.r.t the weights and biases in the
        # classification head (last layer) for all training samples received
        sample_wise_grads = sample_wise_mce_grads_wrt_weights_biases(logits, targets, embeds)
        # a mask for retrieving data from the training samples received that are in the rehearsal buffer
        mask_rehearsal = torch.isin(indices, rehearsal_indices)
        # a mask for retrieving data from the training samples received that are in the current mini-batch
        mask_mb = torch.isin(indices, mb_indices)
        # store the current `targets` tensor into this new variable
        orig_targets = targets
        # construct a new `targets` tensor containing the same targets as before for the past classes but assigns all
        # the new classes the same target, i.e. the value of `n_old_classes`. This way, below, it will be easy to
        # compute an overall average gradient for the new classes. It is as all new classes belong to the same overall
        # class "NEW CLASSES"
        targets = torch.where(targets >= n_old_classes, n_old_classes, targets)
        # compute the number of samples received from each past class and the overall new classes class;
        # it is a 1D tensor of length n_old_classes+1
        n_train_samples = torch.bincount(targets, minlength=n_old_classes + 1)
        # compute the number of samples received from each past class that are currently stored in the rehearsal buffer;
        # it is a 1D tensor of length n_old_classes
        n_reh_samples = torch.bincount(targets[mask_rehearsal], minlength=n_old_classes)
        # compute the number of samples received from each past class and the overall new classes class that are in the
        # current training mini-batch; it is a 1D tensor of length n_old_classes + 1
        n_mb_samples = torch.bincount(targets[mask_mb], minlength=n_old_classes + 1)
        if n_mb_samples[:-1].sum() == 0 or n_mb_samples[-1] == 0:
            raise RuntimeError("Each mini-batch must contain samples from both the rehearsal buffer and the new-class "
                               "training dataset")

        # compute the average gradient for each past class and the overall new classes class, evaluated over all the
        # training samples received from that class. It is a 2D tensor of shape ([n_old_classes + 1] x grad_size), where
        # grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if no
        # samples are received from a class, then its average gradient is a vector of nan.
        train_dataset_avg_grads = class_avg_grads(sample_wise_grads, targets, n_old_classes + 1)
        # compute the average gradient for each past class, evaluated over all the samples received from that class that
        # are currently stored in the rehearsal buffer. It is a 2D tensor of shape (n_old_classes x grad_size), where
        # grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if no
        # samples currently stored in the rehearsal buffer are received from a class, then its average gradient is a
        # vector of nan.
        rehearsal_avg_grads = class_avg_grads(sample_wise_grads[mask_rehearsal], targets[mask_rehearsal], n_old_classes)
        # compute the average gradient for each past class and the overall new classes class, evaluated over all the
        # samples of that class in the current mini-batch. It is a 2D tensor of shape ([n_old_classes+1] x grad_size),
        # where grad_size is the size of each gradient. By the nature of the output returned by `class_avg_grads`, if a
        # class has no samples in the current mini-batch, then its average gradient is a vector of nan.
        mb_avg_grads = class_avg_grads(sample_wise_grads[mask_mb], targets[mask_mb], n_old_classes + 1)

        # compute the unweighted rehearsal SIC_step for each past class
        sic_step_reh = SIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the unweighted mini-batch SIC_step for each past class
        sic_step_mb = SIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
        # if at least one training sample is received from each past class, it is possible to compute both rehearsal and
        # mini-batch CIC_step for each past class, as well as OIC_step
        if (n_train_samples[:-1] > 0).all():
            # compute the unweighted rehearsal CIC_step for each past class
            cic_step_reh = CIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], n_reh_samples, n_train_samples[:-1],
                                    include_norm_cos=True)
            # compute the unweighted mini-batch CIC_step for each past class
            cic_step_mb = CIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], n_mb_samples[:-1], n_train_samples[:-1],
                                   include_norm_cos=True)
            # compute the unweighted OIC_step for each past class
            oic_step = OIC_step(train_dataset_avg_grads[:-1], n_train_samples[:-1], include_norm_cos=True)
        # otherwise it is not possible to compute those CIC-based metrics and OIC, as a consequence they are assigned
        # NaN
        else:
            cic_step_reh = tuple(torch.full_like(sic_step_reh[0], float("nan")) for _ in range(3))
            cic_step_mb = tuple(torch.full_like(sic_step_reh[0], float("nan")) for _ in range(3))
            oic_step = tuple(torch.full_like(sic_step_reh[0], float("nan")) for _ in range(3))
        # compute the unweighted rehearsal RIC_step
        ric_step_reh = RIC_step(rehearsal_avg_grads, train_dataset_avg_grads[:-1], n_reh_samples, include_norm_cos=True)
        # compute the unweighted mini-batch RIC_step
        ric_step_mb = RIC_step(mb_avg_grads[:-1], train_dataset_avg_grads[:-1], n_mb_samples[:-1],
                               include_norm_cos=True)
        # compute the unweighted full-training NIC_step for each past class
        nic_step_train = NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes,
                                  include_norm_cos=True)
        # compute the unweighted mini-batch NIC_step for each past class
        nic_step_mb = NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], n_classes, include_norm_cos=True)
        # compute the unweighted full-training ALL_NIC_step for each past class
        all_nic_step_train = ALL_NIC_step(train_dataset_avg_grads[-1], train_dataset_avg_grads[:-1],
                                          include_norm_cos=True)
        # compute the unweighted mini-batch ALL_NIC_step for each past class
        all_nic_step_mb = ALL_NIC_step(mb_avg_grads[-1], train_dataset_avg_grads[:-1], include_norm_cos=True)
        # compute the unweighted full_training R_NIC_step for each past class
        r_nic_step_train = NIC_step(train_dataset_avg_grads[-1], rehearsal_avg_grads, n_classes, include_norm_cos=True)
        # compute the unweighted mini-batch R_NIC_step for each past class
        r_nic_step_mb = NIC_step(mb_avg_grads[-1], rehearsal_avg_grads, n_classes, include_norm_cos=True)
        # compute the unweighted full-training R_ALL_NIC_step for each past class
        r_all_nic_step_train = ALL_NIC_step(train_dataset_avg_grads[-1], rehearsal_avg_grads, include_norm_cos=True)
        # compute the unweighted mini-batch R_ALL_NIC_step for each past class
        r_all_nic_step_mb = ALL_NIC_step(mb_avg_grads[-1], rehearsal_avg_grads, include_norm_cos=True)

        # create a dictionary that will be returned by this function; all the tensors are converted to numpy arrays
        data = dict(sic_step_reh=tuple(x.detach().cpu().numpy() for x in sic_step_reh),
                    sic_step_mb=tuple(x.detach().cpu().numpy() for x in sic_step_mb),
                    cic_step_reh=tuple(x.detach().cpu().numpy() for x in cic_step_reh),
                    cic_step_mb=tuple(x.detach().cpu().numpy() for x in cic_step_mb),
                    oic_step=tuple(x.detach().cpu().numpy() for x in oic_step),
                    ric_step_reh=tuple(x.detach().cpu().numpy() for x in ric_step_reh),
                    ric_step_mb=tuple(x.detach().cpu().numpy() for x in ric_step_mb),
                    nic_step_train=tuple(x.detach().cpu().numpy() for x in nic_step_train),
                    nic_step_mb=tuple(x.detach().cpu().numpy() for x in nic_step_mb),
                    all_nic_step_train=tuple(x.detach().cpu().numpy() for x in all_nic_step_train),
                    all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in all_nic_step_mb),
                    r_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_nic_step_train),
                    r_nic_step_mb=tuple(x.detach().cpu().numpy() for x in r_nic_step_mb),
                    r_all_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_all_nic_step_train),
                    r_all_nic_step_mb=tuple(x.detach().cpu().numpy() for x in r_all_nic_step_mb),
                    train_dataset_avg_grads_norm=torch.linalg.norm(train_dataset_avg_grads,
                                                                   dim=1).detach().cpu().numpy(),
                    rehearsal_avg_grads_norm=torch.linalg.norm(rehearsal_avg_grads, dim=1).detach().cpu().numpy(),
                    mb_avg_grads_norm=torch.linalg.norm(mb_avg_grads, dim=1).detach().cpu().numpy(),
                    cos_train_rehearsal_avg_grads=tF.cosine_similarity(train_dataset_avg_grads[:-1], rehearsal_avg_grads,
                                                                       dim=1).detach().cpu().numpy(),
                    cos_train_mb_avg_grads=tF.cosine_similarity(train_dataset_avg_grads, mb_avg_grads,
                                                                dim=1).detach().cpu().numpy(),
                    n_train_samples=n_train_samples.detach().cpu().numpy(),
                    n_reh_samples=n_reh_samples.detach().cpu().numpy(),
                    n_mb_samples=n_mb_samples.detach().cpu().numpy(),
                    indices=indices.detach().cpu().numpy(),
                    mb_indices=mb_indices.detach().cpu().numpy(),
                    logits=logits.detach().cpu().numpy(),
                    orig_targets=orig_targets.detach().cpu().numpy()
                    )

        return None, None, data


class InterferenceCoefficientsDataCollectionFromFullGrads:
    """
    A class whose instance is intended to be passed as argument for `analyzer` when constructing an instance
    of :class:`training.plugins.ClassWiseReplayFullDataAvgFullGradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The latter will call the `__call__` method of such an instance each time computation of average full-gradients is
    scheduled.

    Calling `__call__` computes the following interference-based past-class-wise metrics multiple times, each time using
    subgradients (extracted from the full-gradients) wrt to a different set of parameters:
        - the unweighted rehearsal SIC_step
        - the unweighted rehearsal CIC_step
        - the unweighted OIC_step
        - the unweighted rehearsal RIC_step
        - the unweighted full-training ALL_NIC_step
        - the unweighted full-training R_ALL_NIC_step
    If subgradients are wrt the last-layer parameters, the following interference-based past-class-wise metrics can be
    optionally computed:
        - the unweighted full-training NIC_step
        - the unweighted full-training R_NIC_step

    These metrics, and additional data, are collected via a dictionary and returned by the `__call__` method.
    Such a dictionary will be appended into the `data` property of
    :class:`training.plugins.ClassWiseReplayFullDataAvgFullGradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The code of the `__call__` method needs to be inspected to find out what data the returned dictionary contains.
    Note that the metrics and additional data are not computed during the first experience,
    as there are no past classes during the first experience. This is performed by checking the shape of
    `reh_avg_full_grads`, which is one of the inputs to the `__call__` method.

    .note::
        R_NIC_step and R_ALL_NIC_step of a past class are the counterparts of NIC_step and ALL_NIC_step, respectively,
        using the interference onto the average rehearsal gradient of that class rather than onto its original
        average gradient, i.e. the gradient of that class evaluated over all its original training samples.
    """
    def __init__(self, params: Sequence[Tuple[str, ...]], index_last_layer: Optional[int] = None):
        """
        Create a new InterferenceCoefficientsDataCollectionFromFullGrads

        :param params: a sequence containing tuples of parameter names. For each tuple of parameter names,
            `self.__call__` computes interference-based past-class-wise metrics using subgradients
            (extracted from the full-gradients) wrt that set of parameters. If a parameter name does not exist in the
            underlying model, a :class:`RuntimeError` will be raised by `self.__call__`. If this sequence is empty or
            any of the inner tuples is empty, a :class:`ValueError` is raised.
        :param index_last_layer: index of which tuple of parameter names in `params` represents the parameters in the
            last layer. For that set of parameters, `self.__call__` will additionally compute NIC_step and R_NIC_step.
            Importantly, the tuple of parameter names in `params` representing the parameters in the last layer
            must first contain the name associated with the weight matrix in the last layer followed by the name
            associated with the bias vector in the last layer. A :class:`ValueError` is raised if this index is larger
            than or equal to the number of elements in `params`. A :class:`ValueError` is raised if this index
            is negative. Use None if no tuple of parameter names in `params` represents the parameters in the last
            layer or if you do not want NIC_step and R_NIC_step to be computed for the last-layer parameters.
            Default is None.
        """

        # conver into a list
        params = list(params)
        if len(params) == 0:
            raise ValueError("`params` must not be empty.")
        if not all(isinstance(t, tuple) for t in params):
            raise ValueError("All the elements in `params` must be tuples.")
        if any(len(t) == 0 for t in params):
            raise ValueError("All the inner tuples in `params` must not be empty.")
        if index_last_layer is not None:
            if index_last_layer >= len(params):
                raise ValueError("`index_last_layer` must be less than the number of elements in `params`.")
            if index_last_layer < 0:
                raise ValueError("`index_last_layer` must be greater than or equal to 0.")

        self._params: List[Tuple[str, ...]] = params
        """
        a list containing tuples of parameter names. For each tuple of parameter names,
        `self.__call__` computes interference-based past-class-wise metrics using subgradients
        (extracted from the full-gradients) wrt that set of parameters. If a parameter name does not exist in the
        underlying model, a :class:`RuntimeError` will be raised by `self.__call__`.
        """

        self._index_last_layer: Optional[int] = index_last_layer
        """
        index of which tuple of parameter names in `self._params` represents the parameters in the
        last layer. For that set of parameters, `self.__call__` will additionally compute NIC_step and R_NIC_step. 
        Importantly, the tuple of parameter names in `self._params` representing the parameters in the last layer
        must first contain the name associated with the weight matrix in the last layer followed by the name
        associated with the bias vector in the last layer. None if no tuple of parameter names in `self._params`
        represents the parameters in the last layer or if you do not want NIC_step and R_NIC_step to be computed for the
        last-layer parameters
        """

    def __call__(self,
                 train_avg_full_grads: Tensor,
                 reh_avg_full_grads: Tensor,
                 n_train: Tensor,
                 n_reh: Tensor,
                 parameter_names: List[str],
                 parameter_sizes: Tensor,
                 strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate) \
            -> Union[None, Dict[Tuple[str, ...], Dict[str, Union[ndarray, Tuple[ndarray, ...]]]]]:
        """
        Collect metrics and additional data.

        :param train_avg_full_grads: a 2D tensor containing in each ith row the average full-gradient of class i
            over **all** its training samples; its shape is (n_classes x grad_size), where `n_classes` is the total
            number of classes encountered so far (old classes +  new classes) and `grad_size` is the size of the
            average full-gradient.
        :param reh_avg_full_grads: a 2D tensor containing in each ith row the average full-gradient of past class i
            over its training samples currently stored in the replay buffer; its shape is (n_past_classes x grad_size),
            where `n_past_classes` is the number of **past** classes encountered so far and `grad_size` is the size
            of the average full-gradient. During the first experience, `n_past_classes` is obviously 0.
        :param n_train: a 1D tensor, of length equal to `n_classes`, containing at each ith index the total number
            of samples in the original training dataset of class i.
        :param n_reh: a 1D tensor, of length equal to `n_past_classes`, containing at each ith index the number of
            samples currently stored in the replay buffer for past class i.
        :param parameter_names: a list containing the names of the network parameters used to compute the
            full-gradients; the order of the names matches the order used to compute the full-gradients.
        :param parameter_sizes: a 1D tensor, of same length as the list of parameter names, containing at each ith
            index the number of elements (weights) present in the respective parameter name.
        :param strategy: the underlying strategy template.
        :return:  None when there are no past classes; a dictionary otherwise. This dictionary contains the inner
            tuples of parameter names in `self._params` as keys and dictionaries as values. Each inner dictionary
            contains strings as keys and tuples of numpy arrays or numpy arrays as values. This dictionary is appended
            into the `data` property of :class:`training.plugins.ClassWiseReplayFullDataAvgFullGradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        """

        # get the total number of classes
        n_classes = train_avg_full_grads.shape[0]
        # get the number of old classes
        n_old_classes = reh_avg_full_grads.shape[0]
        # if there are no past classes
        if n_old_classes == 0:
            return None
        # if the length of `parameter_names` does not match the length of `parameter_sizes`
        if len(parameter_names) != len(parameter_sizes):
            raise RuntimeError("The length of `parameter_names` must match the length of `parameter_sizes`")

        # compute the total number of training samples belonging to any of the new classes
        new_cls_n_train = n_train[n_old_classes:].sum()
        # compute the average full-gradient over all the new-class training samples; it is a 1D tensor
        new_cls_avg_full_grad = (train_avg_full_grads[n_old_classes:] *
                                 n_train[n_old_classes:, None]).sum(dim=0) / new_cls_n_train
        # transform `train_avg_full_grads` into a (`n_old_classes` + 1, `grad_size`) tensor; the average full-gradients
        # of individual past classes are kept while those of new classes are replaced by a unique
        # overall new-class average
        train_avg_full_grads = torch.cat((train_avg_full_grads[:n_old_classes], new_cls_avg_full_grad[None, :]),
                                         dim=0)
        # transform `n_train` into a (`n_old_classes` + 1) tensor; the number of samples in the original training
        # dataset of individual past classes are kept while those of individual new classes are replaced by a unique
        # total
        n_train = torch.cat((n_train[:n_old_classes], new_cls_n_train[None]), dim=0)

        d = {}
        # create a dictionary containing the names in `parameter_names` as keys and their respective indices in
        # `parameter_names` as values
        idx_map: Dict[str, int] = {v: i for i, v in enumerate(parameter_names)}
        # for each parameter group in `self._params`, compute the respective subgradient-based metrics
        for n, group in enumerate(self._params):
            group_train_avg_grads: List[Tensor] = []
            group_reh_avg_grads: List[Tensor] = []
            for name in group:
                # if the given parameter name does not exist in the underlying model
                if name not in idx_map.keys():
                    raise RuntimeError("The provided parameter name does not exist")
                # get the index of the given parameter name in `parameter_names`
                idx = idx_map[name]
                # get the starting index for the subgradient wrt the weights associated with the given parameter name
                start_slice = parameter_sizes[:idx].sum()
                # get the ending index for the subgradient wrt the weights associated with the given parameter name
                end_slice = start_slice + parameter_sizes[idx]
                # append into `group_train_avg_grads` the subgradients wrt the weights associated with the given
                # parameter names extracted from `train_avg_full_grads`
                group_train_avg_grads.append(train_avg_full_grads[:, start_slice:end_slice])
                # append into `group_reh_avg_grads` the subgradients wrt the weights associated with the given
                # parameter names extracted from `reh_avg_full_grads`
                group_reh_avg_grads.append(reh_avg_full_grads[:, start_slice:end_slice])

            # form the subgradient wrt all parameters in the given parameter group extracted from `train_avg_full_grads`
            group_train_avg_grads = torch.cat(group_train_avg_grads, dim=1)
            # form the subgradient wrt all parameters in the given parameter group extracted from `reh_avg_full_grads`
            group_reh_avg_grads = torch.cat(group_reh_avg_grads, dim=1)

            # compute the unweighted rehearsal SIC_step for each past class
            sic_step_reh = SIC_step(group_reh_avg_grads, group_train_avg_grads[:-1], include_norm_cos=True)
            # compute the unweighted rehearsal CIC_step for each past class
            cic_step_reh = CIC_step(group_reh_avg_grads, group_train_avg_grads[:-1], n_reh, n_train[:-1],
                                    include_norm_cos=True)
            # compute the unweighted OIC_step for each past class
            oic_step = OIC_step(group_train_avg_grads[:-1], n_train[:-1], include_norm_cos=True)
            # compute the unweighted rehearsal RIC_step
            ric_step_reh = RIC_step(group_reh_avg_grads, group_train_avg_grads[:-1], n_reh, include_norm_cos=True)
            # compute the unweighted full-training ALL_NIC_step for each past class
            all_nic_step_train = ALL_NIC_step(group_train_avg_grads[-1], group_train_avg_grads[:-1],
                                              include_norm_cos=True)
            # compute the unweighted full-training R_ALL_NIC_step for each past class
            r_all_nic_step_train = ALL_NIC_step(group_train_avg_grads[-1], group_reh_avg_grads, include_norm_cos=True)

            # create a sub-dictionary that will be returned by this function; all the tensors are converted to
            # numpy arrays
            data = dict(sic_step_reh=tuple(x.detach().cpu().numpy() for x in sic_step_reh),
                        cic_step_reh=tuple(x.detach().cpu().numpy() for x in cic_step_reh),
                        oic_step=tuple(x.detach().cpu().numpy() for x in oic_step),
                        ric_step_reh=tuple(x.detach().cpu().numpy() for x in ric_step_reh),
                        all_nic_step_train=tuple(x.detach().cpu().numpy() for x in all_nic_step_train),
                        r_all_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_all_nic_step_train),
                        train_dataset_avg_grads_norm=torch.linalg.norm(group_train_avg_grads,
                                                                       dim=1).detach().cpu().numpy(),
                        rehearsal_avg_grads_norm=torch.linalg.norm(group_reh_avg_grads, dim=1).detach().cpu().numpy(),
                        cos_train_rehearsal_avg_grads=tF.cosine_similarity(group_train_avg_grads[:-1],
                                                                           group_reh_avg_grads,
                                                                           dim=1).detach().cpu().numpy(),
                        n_train_samples=n_train.detach().cpu().numpy(),
                        n_reh_samples=n_reh.detach().cpu().numpy(),
                        )

            # if the current parameter group corresponds to the set of weights and biases in the last layer, then
            # compute NIC-based metrics too
            if self._index_last_layer is not None and self._index_last_layer == n:
                # compute the unweighted full-training NIC_step for each past class
                nic_step_train = NIC_step(group_train_avg_grads[-1], group_train_avg_grads[:-1], n_classes,
                                          include_norm_cos=True)
                # compute the unweighted full_training R_NIC_step for each past class
                r_nic_step_train = NIC_step(group_train_avg_grads[-1], group_reh_avg_grads, n_classes,
                                            include_norm_cos=True)

                data_ = dict(nic_step_train=tuple(x.detach().cpu().numpy() for x in nic_step_train),
                             r_nic_step_train=tuple(x.detach().cpu().numpy() for x in r_nic_step_train))

                data = {**data, **data_}

            d[group] = data

        return d
