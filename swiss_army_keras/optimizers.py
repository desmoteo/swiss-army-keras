from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


import abc
import contextlib
import functools

import six
import tensorflow as tf
import numpy as np
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


from typeguard import typechecked

from typing import Union, Callable, List, Dict


# TODO: Remove once https://github.com/tensorflow/tensorflow/issues/44613 is resolved
if tf.__version__[:3] > "2.5":
    from keras.engine import keras_tensor
else:
    from tensorflow.python.keras.engine import keras_tensor


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable,
                    tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable,
                    tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


def _solve(a, b, c):
    """Return solution of a quadratic minimization.
    The optimization equation is:
         f(a, b, c) = argmin_w{1/2 * a * w^2 + b * w + c * |w|}
    we get optimal solution w*:
         w* = -(b - sign(b)*c)/a if |b| > c else w* = 0
    REQUIRES: Dimensionality of a and b must be same
    Args:
      a: A Tensor
      b: A Tensor
      c: A Tensor with one element.
    Returns:
      A Tensor w, which is solution for the equation
    """
    w = (c * tf.sign(b) - b) / a
    w = tf.cast(tf.abs(b) > c, dtype=b.dtype) * w
    return w


#@tf.keras.utils.register_keras_serializable(package="Addons")


class Yogi(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Yogi algorithm in Keras.
    See Algorithm 2 of
    https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf.
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.01,
        beta1: FloatTensorLike = 0.9,
        beta2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-3,
        l1_regularization_strength: FloatTensorLike = 0.0,
        l2_regularization_strength: FloatTensorLike = 0.0,
        initial_accumulator_value: FloatTensorLike = 1e-6,
        activation: str = "sign",
        name: str = "Yogi",
        **kwargs,
    ):
        """Construct a new Yogi optimizer.
        Args:
          learning_rate: A Tensor or a floating point value.
            The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A constant trading off adaptivity and noise.
          l1_regularization_strength: A float value, must be greater than or
            equal to zero.
          l2_regularization_strength: A float value, must be greater than or
            equal to zero.
          initial_accumulator_value: The starting value for accumulators.
            Only positive values are allowed.
          activation: Use hard sign or soft tanh to determin sign.
          name: Optional name for the operations created when applying
            gradients. Defaults to "Yogi".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
            `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
            is clip gradients by value, `decay` is included for backward
            compatibility to allow time inverse decay of learning rate. `lr`
            is included for backward compatibility, recommended to use
            `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta1)
        self._set_hyper("beta_2", beta2)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("l1_regularization_strength",
                        l1_regularization_strength)
        self._set_hyper("l2_regularization_strength",
                        l2_regularization_strength)

        self._beta1 = beta1
        self._activation = activation
        self._initial_accumulator_value = initial_accumulator_value
        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength

    def _create_slots(self, var_list):
        """See `tf.train.Optimizer._create_slots()`."""
        # Create slots for the first and second moments, and maximum second moments.
        for var in var_list:
            init = tf.constant_initializer(self._initial_accumulator_value)
            self.add_slot(var, "v", init)
            if self._beta1 > 0.0:
                self.add_slot(var, "m")

    def _resource_apply_dense(self, grad, var):
        """See `tf.train.Optimizer._apply_dense()`."""
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta1_t = self._get_hyper("beta_1", var_dtype)
        beta2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = self._get_hyper("epsilon", var_dtype)
        l1_t = self._get_hyper("l1_regularization_strength", var_dtype)
        l2_t = self._get_hyper("l2_regularization_strength", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(beta1_t, local_step)
        beta2_power = tf.pow(beta2_t, local_step)

        lr = lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        update_vs = []
        if self._beta1 == 0.0:
            # v_t = v + sign(g_t^2-v)(g_t^2)
            v = self.get_slot(var, "v")
            grad2 = grad * grad
            if self._activation == "sign":
                sign = tf.sign(grad2 - v)
            elif self._activation == "tanh":
                sign = tf.tanh(10 * (grad2 - v))
            else:
                raise NotImplementedError(
                    "Activation function can be sign or tanh")
            v_t = v.assign_add(
                (1 - beta2_t) * sign * grad2, use_locking=self._use_locking
            )
            v_sqrt = tf.sqrt(v_t)

            # Yogi effective LR
            per_coord_lr = lr / (v_sqrt + epsilon_t)

            # Variable update
            # Step 1: Gradient descent
            new_var = var - per_coord_lr * grad
            # Step 2: Prox operator
            if self._l1_regularization_strength > 0:
                new_var = _solve(1 + l2_t * per_coord_lr, -
                                 new_var, l1_t * per_coord_lr)
            elif self._l2_regularization_strength > 0:
                new_var = new_var / (1 + l2_t * per_coord_lr)
            # Step 3: Update
            var_update = var.assign(new_var, use_locking=self._use_locking)

            update_vs.append(var_update)
            update_vs.append(v_t)

        else:
            # m_t = beta1 * m + (1 - beta1) * g_t
            m = self.get_slot(var, "m")
            m_t = m.assign(
                m * beta1_t + grad * (1 - beta1_t), use_locking=self._use_locking
            )

            # v_t = v + sign(g_t^2-v)(g_t^2)
            v = self.get_slot(var, "v")
            grad2 = grad * grad
            if self._activation == "sign":
                sign = tf.sign(grad2 - v)
            elif self._activation == "tanh":
                sign = tf.tanh(10 * (grad2 - v))
            else:
                raise NotImplementedError(
                    "Activation function can be sign or tanh")
            v_t = v.assign_add(
                (1 - beta2_t) * sign * grad2, use_locking=self._use_locking
            )
            v_sqrt = tf.sqrt(v_t)

            # Yogi effective LR
            per_coord_lr = lr / (v_sqrt + epsilon_t)

            # Variable update
            # Step 1: Gradient descent
            new_var = var - per_coord_lr * m_t
            # Step 2: Prox operator
            if self._l1_regularization_strength > 0:
                new_var = _solve(1 + l2_t * per_coord_lr, -
                                 new_var, l1_t * per_coord_lr)
            elif self._l2_regularization_strength > 0:
                new_var = new_var / (1 + l2_t * per_coord_lr)
            # Step 3: Update
            var_update = var.assign(new_var, use_locking=self._use_locking)
            update_vs.append(var_update)
            update_vs.append(m_t)
            update_vs.append(v_t)

        # Create an op that groups all the above operations
        return tf.group(*update_vs)

    def _resource_apply_sparse(self, grad, var, indices):
        """Applies sparse gradients to a variable.
        Args:
          grad: A tensor for the `values` of `tf.IndexedSlices`.
          var: A `tf.Variable` object.
          indices: A tensor for the `indices` of `tf.IndexedSlices`.
        Returns:
          An op which updates `var` with `grad` and `indices`.
        """

        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta1_t = self._get_hyper("beta_1", var_dtype)
        beta2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = self._get_hyper("epsilon", var_dtype)
        l1_t = self._get_hyper("l1_regularization_strength", var_dtype)
        l2_t = self._get_hyper("l2_regularization_strength", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(beta1_t, local_step)
        beta2_power = tf.pow(beta2_t, local_step)

        lr = lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        update_vs = []
        if self._beta1 == 0.0:
            # v_t = v + sign(g_t^2-v)(g_t^2)
            v = self.get_slot(var, "v")
            grad2 = grad * grad
            v_slice = tf.gather(v, indices)
            if self._activation == "sign":
                sign = tf.sign(grad2 - v_slice)
            elif self._activation == "tanh":
                sign = tf.tanh(10 * (grad2 - v_slice))
            else:
                raise NotImplementedError(
                    "Activation function can be sign or tanh")
            v_scaled_g_values = v_slice + (1 - beta2_t) * sign * grad2
            v_t = self._resource_scatter_update(v, indices, v_scaled_g_values)
            v_sqrt = tf.sqrt(v_scaled_g_values)

            # Yogi effective LR
            per_coord_lr = lr / (v_sqrt + epsilon_t)

            # Variable update
            # Step 1: Gradient descent
            var_slice = tf.gather(var, indices)
            new_var = var_slice - per_coord_lr * grad
            # Step 2: Prox operator
            if self._l1_regularization_strength > 0:
                new_var = _solve(1 + l2_t * per_coord_lr, -
                                 new_var, l1_t * per_coord_lr)
            elif self._l2_regularization_strength > 0:
                new_var = new_var / (1 + l2_t * per_coord_lr)
            # Step 3: Update
            var_update = self._resource_scatter_update(var, indices, new_var)
            update_vs.append(var_update)
            update_vs.append(v_t)

        else:
            # m_t = beta1 * m + (1 - beta1) * g_t
            m = self.get_slot(var, "m")
            m_scaled_g_values = grad * (1 - beta1_t)
            m_t = m.assign(m * beta1_t, use_locking=self._use_locking)
            with tf.control_dependencies([m_t]):
                m_slice = tf.gather(m, indices) + m_scaled_g_values
                m_t = self._resource_scatter_update(m, indices, m_slice)

            # v_t = v + sign(g_t^2-v)(g_t^2)
            v = self.get_slot(var, "v")
            grad2 = grad * grad
            v_slice = tf.gather(v, indices)
            if self._activation == "sign":
                sign = tf.sign(grad2 - tf.gather(v, indices))
            elif self._activation == "tanh":
                sign = tf.tanh(10 * (grad2 - tf.gather(v, indices)))
            else:
                raise NotImplementedError(
                    "Activation function can be sign or tanh")
            v_scaled_g_values = v_slice + (1 - beta2_t) * sign * grad2
            v_t = self._resource_scatter_update(v, indices, v_scaled_g_values)
            v_sqrt = tf.sqrt(v_scaled_g_values)

            # Yogi effective LR
            per_coord_lr = lr / (v_sqrt + epsilon_t)

            # Variable update
            # Step 1: Gradient descent
            var_slice = tf.gather(var, indices)
            new_var = var_slice - per_coord_lr * m_slice
            # Step 2: Prox operator
            if self._l1_regularization_strength > 0:
                new_var = _solve(1 + l2_t * per_coord_lr, -
                                 new_var, l1_t * per_coord_lr)
            elif self._l2_regularization_strength > 0:
                new_var = new_var / (1 + l2_t * per_coord_lr)
            # Step 3: Update
            var_update = self._resource_scatter_update(var, indices, new_var)
            update_vs.append(var_update)
            update_vs.append(m_t)
            update_vs.append(v_t)

        # Create an op that groups all the above operations
        return tf.group(*update_vs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "beta1": self._serialize_hyperparameter("beta_1"),
                "beta2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self._serialize_hyperparameter("epsilon"),
                "l1_regularization_strength": self._serialize_hyperparameter(
                    "l1_regularization_strength"
                ),
                "l2_regularization_strength": self._serialize_hyperparameter(
                    "l2_regularization_strength"
                ),
                "activation": self._activation,
                "initial_accumulator_value": self._initial_accumulator_value,
            }
        )
        return config


#@tf.keras.utils.register_keras_serializable(package="Addons")


class RectifiedAdam2(tf.keras.optimizers.Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    Example of usage:
    ```python
    opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = tfa.optimizers.RectifiedAdam(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    ```
    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.
    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
    which is announced by Less Wright and the new combined optimizer can also
    be called "Ranger". The mechanism can be enabled by using the lookahead
    wrapper. For example:
    ```python
    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable, Dict] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: Union[FloatTensorLike, Callable, Dict] = 0.0,
        amsgrad: bool = False,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "RectifiedAdam",
        **kwargs,
    ):
        r"""Construct a new RAdam optimizer.
        Args:
            learning_rate: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                Weight decay for each parameter.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer value. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        if isinstance(learning_rate, Dict):
            learning_rate = tf.keras.optimizers.schedules.deserialize(
                learning_rate)

        if isinstance(weight_decay, Dict):
            weight_decay = tf.keras.optimizers.schedules.deserialize(
                weight_decay)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", float(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        one_minus_beta_1_t = 1.0 - beta_1_t
        recip_one_minus_beta_1_power = 1.0 / (1.0 - beta_1_power)
        one_minus_beta_2_t = 1.0 - beta_2_t
        recip_one_minus_beta_2_power = 1.0 / (1.0 - beta_2_power)
        sma_inf = 2.0 / one_minus_beta_2_t - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power * recip_one_minus_beta_2_power
        r_t = tf.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )
        sma_threshold = self._get_hyper("sma_threshold", var_dtype)
        sma_t_ge_sma_threshold = sma_t >= sma_threshold
        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * \
                self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate *
                tf.minimum(local_step - warmup_steps, decay_steps),
            )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr_t=lr_t,
                wd_t=wd_t,
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                epsilon_t=tf.convert_to_tensor(self.epsilon, var_dtype),
                local_step=local_step,
                beta_1_power=beta_1_power,
                beta_2_power=beta_2_power,
                sma_inf=sma_inf,
                sma_t=sma_t,
                one_minus_beta_1_t=one_minus_beta_1_t,
                recip_one_minus_beta_1_power=recip_one_minus_beta_1_power,
                one_minus_beta_2_t=one_minus_beta_2_t,
                recip_one_minus_beta_2_power=recip_one_minus_beta_2_power,
                r_t=r_t,
                sma_t_ge_sma_threshold=sma_t_ge_sma_threshold,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coef = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = m.assign(
            coef["beta_1_t"] * m + coef["one_minus_beta_1_t"] * grad,
            use_locking=self._use_locking,
        )
        m_corr_t = m_t * coef["recip_one_minus_beta_1_power"]

        v_t = v.assign(
            coef["beta_2_t"] * v +
            coef["one_minus_beta_2_t"] * tf.square(grad),
            use_locking=self._use_locking,
        )
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t),
                                 use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t * coef["recip_one_minus_beta_2_power"])
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t * coef["recip_one_minus_beta_2_power"])

        var_t = tf.where(
            coef["sma_t_ge_sma_threshold"],
            coef["r_t"] * m_corr_t / (v_corr_t + coef["epsilon_t"]),
            m_corr_t,
        )

        if self._has_weight_decay:
            var_t += coef["wd_t"] * var

        var_update = var.assign_sub(
            coef["lr_t"] * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coef = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coef["one_minus_beta_1_t"]
        m_t = m.assign(m * coef["beta_1_t"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t * coef["recip_one_minus_beta_1_power"]

        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * coef["one_minus_beta_2_t"]
        v_t = v.assign(v * coef["beta_2_t"], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t),
                                 use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t * coef["recip_one_minus_beta_2_power"])
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t * coef["recip_one_minus_beta_2_power"])

        var_t = tf.where(
            coef["sma_t_ge_sma_threshold"],
            coef["r_t"] * m_corr_t / (v_corr_t + coef["epsilon_t"]),
            m_corr_t,
        )

        if self._has_weight_decay:
            var_t += coef["wd_t"] * var

        with tf.control_dependencies([var_t]):
            var_update = self._resource_scatter_add(
                var, indices, tf.gather(-coef["lr_t"] * var_t, indices)
            )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "total_steps": int(self._serialize_hyperparameter("total_steps")),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )


@tf_export(v1=["train.AdaBeliefOptimizer"])
class AdaBeliefOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.
    References:
    Adam - A Method for Stochastic Optimization:
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)
      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 use_locking=False,
                 name="AdaBelief", amsgrad=False):
        r"""Construct a new Adam optimizer.
        Initialization:
        $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
        $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
        $$t := 0 \text{(Initialize timestep)}$$
        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section 2 of the paper:
        $$t := t + 1$$
        $$\text{lr}_t := \mathrm{learning_rate} *
          \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
        $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
        $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
        $$\text{variable} := \text{variable} -
          \text{lr}_t * m_t / (\sqrt{v_t} + \epsilon)$$
        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
        formulation just before Section 2.1 of the Kingma and Ba paper rather than
        the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
        hat" in the paper.
        The sparse implementation of this algorithm (used when the gradient is an
        IndexedSlices object, typically because of `tf.gather` or an embedding
        lookup in the forward pass) does apply momentum to variable slices even if
        they were not used in the forward pass (meaning they have a gradient equal
        to zero). Momentum decay (beta1) is also applied to the entire momentum
        accumulator. This means that the sparse behavior is equivalent to the dense
        behavior (in contrast to some momentum implementations which ignore momentum
        unless a variable slice was actually used).
        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        @compatibility(eager)
        When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
        `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """
        super(AdaBeliefOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self.amsgrad = amsgrad

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr = self._lr
        beta1 = self._beta1
        beta2 = self._beta2
        epsilon = self._epsilon

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable(
            "beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable(
            "beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(
            m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(
            v, beta2_t * v + v_scaled_g_values + epsilon_t, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * step_size

        var_update = state_ops.assign_sub(
            var, bounded_lr, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable(
            "beta1_power", graph=graph), grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable(
            "beta2_power", graph=graph), grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(
            m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(
            v, beta2_t * v + v_scaled_g_values + epsilon_t, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * step_size

        var_update = state_ops.assign_sub(
            var, bounded_lr, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values + epsilon_t)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        var_update = state_ops.assign_sub(
            var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope)


class AdaHessian(optimizer_v2.OptimizerV2):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-4,
                 weight_decay=0.,
                 hessian_power=1.0,
                 name='AdaHessian',
                 average_size_1d=None,
                 average_size_2d=None,
                 average_size_3d=-1,
                 average_size_4d=-1,
                 **kwargs):
        """Construct a new AdaHessian optimizer.
        Args:
            learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.1.
            beta_1: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use. The exponential decay
            rate for the 1st moment estimates. Defaults to 0.9.
            beta_2: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use, The exponential decay
            rate for the 2nd moment estimates. Defaults to 0.999.
            epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
            weight_decay: We are using AdamW's weight decay scheme. Defaults to 0.
            name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
            hessian_power: Hessian power to control the optimizer more similar to first/second 
            order method (default: 1). You can also try 0.5. For some tasks we found this 
            to result in better performance.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
            # average_size_{1,2,3,4}d: 
                        None: use no spatial averaging
                        -1: use suggested spatial averaging (recommended for conv kernels)
                        >= 1: use customized size
        """

        super(AdaHessian, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.weight_decay = weight_decay
        self.hessian_power = hessian_power
        self.average_size_1d = average_size_1d
        self.average_size_2d = average_size_2d
        self.average_size_3d = average_size_3d
        self.average_size_4d = average_size_4d

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaHessian, self)._prepare_local(
            var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (
            apply_state[(var_device, var_dtype)]['lr_t'] *
            (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdaHessian, self).set_weights(weights)

    def get_gradients_hessian(self, loss, params):
        """Returns gradients and Hessian of `loss` with respect to `params`.
        Arguments:
            loss: Loss tensor.
            params: List of variables.
        Returns:
            List of gradient and Hessian tensors.
        Raises:
            ValueError: In case any gradient cannot be computed (e.g. if gradient
            function not implemented).
        """
        params = nest.flatten(params)
        with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                                  "/gradients"):
            grads = gradients.gradients(loss, params)
            for grad, param in zip(grads, params):
                if grad is None:
                    raise ValueError("Variable {} has `None` for gradient. "
                                     "Please make sure that all of your ops have a "
                                     "gradient defined (i.e. are differentiable). "
                                     "Common ops without gradient: "
                                     "K.argmax, K.round, K.eval.".format(param))

            # WARNING: for now we do not support gradient clip
            # grads = self._clip_gradients(grads)

            v = [np.random.uniform(0, 1, size=p.shape) for p in params]
            for vi in v:
                vi[vi < 0.5] = -1
                vi[vi >= 0.5] = 1
            v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]

            vprod = tf.reduce_sum([tf.reduce_sum(vi * grad)
                                  for vi, grad in zip(v, grads)])

            Hv = gradients.gradients(vprod, params)

            Hd = [tf.abs(Hvi * vi) for Hvi, vi in zip(Hv, v)]

        return grads, Hd

    def _filter_grads_hessian(self, grads_hessian_and_vars):
        """Filter out iterable with grad equal to None."""
        grads_hessian_and_vars = tuple(grads_hessian_and_vars)
        if not grads_hessian_and_vars:
            return grads_hessian_and_vars
        filtered = []
        vars_with_empty_grads = []
        for grad, hessian, var in grads_hessian_and_vars:
            if grad is None:
                vars_with_empty_grads.append(var)
            else:
                filtered.append((grad, hessian, var))
        filtered = tuple(filtered)

        if not filtered:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([v.name for _, v in grads_and_vars],))
        if vars_with_empty_grads:
            logging.warning(
                ("Gradients do not exist for variables %s when minimizing the loss."),
                ([v.name for v in vars_with_empty_grads]))
        return filtered

    def apply_gradients_hessian(self,
                                grads_hessian_and_vars,
                                name=None,
                                experimental_aggregate_gradients=True):
        grads_hessian_and_vars = self._filter_grads_hessian(
            grads_hessian_and_vars)
        var_list = [v for (_, _, v) in grads_hessian_and_vars]

        with backend.name_scope(self._name):
            # Create iteration if necessary.
            with ops.init_scope():
                self._create_all_weights(var_list)

        if not grads_hessian_and_vars:
            # Distribution strategy does not support reducing an empty list of
            # gradients
            return control_flow_ops.no_op()

        if distribute_ctx.in_cross_replica_context():
            raise RuntimeError(
                "`apply_gradients() cannot be called in cross-replica context. "
                "Use `tf.distribute.Strategy.run` to enter replica "
                "context.")

        strategy = distribute_ctx.get_strategy()
        if (not experimental_aggregate_gradients and strategy and isinstance(
            strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended)):
            raise NotImplementedError(
                "`experimental_aggregate_gradients=False is not supported for "
                "ParameterServerStrategy and CentralStorageStrategy")

        apply_state = self._prepare(var_list)
        if experimental_aggregate_gradients:
            reduced_grads, reduced_hessian = self._aggregate_gradients_hessian(
                grads_hessian_and_vars)
            var_list = [v for _, _, v in grads_hessian_and_vars]
            grads_hessian_and_vars = list(
                zip(reduced_grads, reduced_hessian, var_list))

        return distribute_ctx.get_replica_context().merge_call(
            functools.partial(self._distributed_apply,
                              apply_state=apply_state),
            args=(grads_hessian_and_vars,),
            kwargs={
                "name": name,
            })

    def _aggregate_gradients_hessian(self, grads_hessian_and_vars):
        """Returns all-reduced gradients.
        Args:
        grads_and_vars: List of (gradient, hessian, variable) pairs.
        Returns:
        Two lists of all-reduced gradients and Hessian.
        """
        grads_hessian_and_vars = list(grads_hessian_and_vars)
        filtered_grads_hessian_and_vars = self._filter_grads_hessian(
            grads_hessian_and_vars)

        # split the list so that we can use the all_recude_fn
        filtered_grads_and_vars = tuple(
            [(g, v) for (g, h, v) in filtered_grads_hessian_and_vars])
        filtered_hessian_and_vars = tuple(
            [(h, v) for (g, h, v) in filtered_grads_hessian_and_vars])

        def all_reduce_fn(distribution, grads_hessian_and_vars):
            # WARNING: this ReduceOp.SUM can only support two entries, for now we have three.
            # So far now, we do it for two steps to make life easier.
            return distribution.extended.batch_reduce_to(
                ds_reduce_util.ReduceOp.SUM, grads_hessian_and_vars)

        if filtered_grads_hessian_and_vars:
            reduced_part1 = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_grads_and_vars,))
            reduced_part2 = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_hessian_and_vars,))
        else:
            reduced = []

        # Copy 'reduced' but add None gradients back in
        reduced_with_nones_grads = []
        reduced_with_nones_hessian = []

        reduced_pos = 0
        for g, h, _ in grads_hessian_and_vars:
            if g is None:
                reduced_with_nones_grads.append(None)
                reduced_with_nones_hessian.append(None)
            else:
                reduced_with_nones_grads.append(reduced_part1[reduced_pos])
                reduced_with_nones_hessian.append(reduced_part2[reduced_pos])
                reduced_pos += 1

        return reduced_with_nones_grads, reduced_with_nones_hessian

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, hess, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
        # this part need to be changed for spatial averaging

        if len(v.shape) == 1:
            resize = self.average_size_1d
        elif len(v.shape) == 2:
            resize = self.average_size_2d
        elif len(v.shape) == 3:
            resize = self.average_size_3d
        elif len(v.shape) == 4:
            resize = self.average_size_4d
        else:
            raise Exception(
                'You need to define the spatial average size by yourself!')

        if resize == None:
            v.assign_add((math_ops.square(hess) - v) *
                         (1 - coefficients['beta_2_t']))
        elif resize == -1:
            if len(v.shape) == 1:
                v.assign_add((math_ops.square(hess) - v) *
                             (1 - coefficients['beta_2_t']))
            elif len(v.shape) == 2:
                hess_average = tf.reduce_mean(hess, [0], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v)
                             * (1 - coefficients['beta_2_t']))
            elif len(v.shape) == 3:
                hess_average = tf.reduce_mean(hess, [0], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v)
                             * (1 - coefficients['beta_2_t']))
            elif len(v.shape) == 4:
                hess_average = tf.reduce_mean(hess, [0, 1], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v)
                             * (1 - coefficients['beta_2_t']))
        else:
            if resize <= 0:
                raise Exception(
                    'You need to define the spatial average size >= 1!')
            hess_average = tf.reshape(hess, [resize, -1])
            hess_average = tf.reduce_mean(hess_average, [0])
            hess_average = tf.repeat(hess_average, resize)
            hess_average = tf.reshape(hess_average, v.shape)
            v.assign_add((math_ops.square(hess_average) - v)
                         * (1 - coefficients['beta_2_t']))

        bias_correct1 = 1 - coefficients['beta_1_power']
        bias_correct2 = 1 - coefficients['beta_2_power']

        if self.weight_decay != 0:
            var.assign_sub(coefficients['lr_t'] * self.weight_decay * var)

        # denom = np.power(math_ops.sqrt(v / bias_correct2), self.hessian_power) + coefficients['epsilon']
        denom = tf.math.pow(math_ops.sqrt(v / bias_correct2),
                            self.hessian_power) + coefficients['epsilon']

        var.assign_sub(coefficients['lr_t'] * m / bias_correct1 / denom)

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise Exception('For now, we do not support sparse update yet.')

    def get_config(self):
        config = super(AdaHessian, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        })
        return config

    def _distributed_apply(self, distribution, grads_hessian_and_vars, name, apply_state):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad, hess):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            update_op = self._resource_apply_dense(
                grad, hess, var, **apply_kwargs)

            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, hess, var in grads_hessian_and_vars:

                def _assume_mirrored(grad, hess):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values), ds_values.Mirrored(hess.values)
                    return grad, hess

                grad, hess = nest.map_structure(_assume_mirrored, grad, hess)
                # Colocate the update with variables to avoid unnecessary communication
                # delays. See b/136304694.
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                        "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                            var, apply_grad_to_update_var, args=(grad, hess), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)
