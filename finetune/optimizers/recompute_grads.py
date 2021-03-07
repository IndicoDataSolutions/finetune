import functools

import tensorflow as tf
from tensorflow.python.framework import function, ops


def fn_with_custom_grad(grad_fn, use_global_vars=False, use_entire_scope=False, train_vars=None):
    """Decorator to create a subgraph with a custom gradient function.
    The subgraph created by the decorated function is NOT put in a Defun and so
    does not suffer from the limitations of the Defun (all subgraph ops on the
    same device, no summaries).
    Args:
      grad_fn: function with signature
        (inputs, variables, outputs, output_grads) -> (grad_inputs, grad_vars),
        all of which are lists of Tensors.
      use_global_vars: if True, variables will be the global variables created.
        If False, will be the trainable variables.
    Returns:
      Decorator for function such that the gradient is defined by grad_fn.
    """

    def dec(fn):
        @functools.wraps(fn)
        def wrapped(*args):
            return _fn_with_custom_grad(
                fn, args, grad_fn, use_global_vars=use_global_vars, use_entire_scope=use_entire_scope, train_vars=None)

        return wrapped

    return dec


def _fn_with_custom_grad(fn, inputs, grad_fn, use_global_vars=False, use_entire_scope=False, train_vars=None):
    """Create a subgraph with a custom gradient.
    Args:
      fn: function that takes inputs as arguments and produces 1 or more Tensors.
      inputs: list<Tensor>, will be passed as fn(*inputs).
      grad_fn: function with signature
        (inputs, vars, outputs, output_grads) -> (grad_inputs, grad_vars),
        all of which are lists of Tensors.
      use_global_vars: if True, variables will be the global variables created.
        If False, will be the trainable variables.
    Returns:
      fn(*inputs)
    """
    vs = tf.compat.v1.get_variable_scope()

    # Use a function attribute to store the local variables. This means that on graph rebuild,
    # When variables already exist we know what variables relate to this function.
    get_vars_fn = (vs.global_variables if use_global_vars else vs.trainable_variables)
    len_before_vars = len(get_vars_fn())
    inputs = list(inputs)
    outputs = fn(*inputs)
    if not train_vars:
        if use_entire_scope:
            train_vars = get_vars_fn()
        else:
            train_vars = get_vars_fn()[len_before_vars:]

    if grad_fn is None:
        return outputs

    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    outputs = list(outputs)
    raw_outputs = outputs
    outputs = [o for o in outputs if isinstance(o, tf.Tensor)]

    defun_inputs = [inputs, train_vars, outputs]

    def custom_grad_fn(op, *dys):
        """Custom grad fn applying grad_fn for identity Defun."""
        fn_inputs, fn_vars, fn_outputs = tf.nest.pack_sequence_as(
            defun_inputs, list(op.inputs))
        dys = list(dys)
        assert len(fn_outputs) == len(outputs)
        assert len(fn_outputs) == len(dys)

        grad_inputs, grad_vars = grad_fn(fn_inputs, fn_vars, fn_outputs, dys)
        grad_outputs = [None] * len(fn_outputs)
        return tuple(grad_inputs + grad_vars + grad_outputs)

    # The Defun takes as input the original inputs, the trainable variables
    # created in fn, and the outputs. In the forward it passes through the
    # outputs. In the backwards, it produces gradients for the original inputs
    # and the trainable variables.
    in_types = [t.dtype for t in inputs]
    out_types = [t.dtype for t in outputs]
    var_types = [t.dtype for t in train_vars]

    @function.Defun(
        *(in_types + var_types + out_types),
        func_name="identity_custom_grad%d" % ops.uid(),
        python_grad_func=custom_grad_fn,
        shape_func=lambda _: [t.get_shape() for t in outputs])
    def identity(*args):
        _, _, outs = tf.nest.pack_sequence_as(defun_inputs, args)
        return tuple([tf.identity(t) if isinstance(t, tf.Tensor) else t for t in outs])

    flat_inputs = tf.nest.flatten(defun_inputs)
    id_out = identity(*flat_inputs)
    if isinstance(id_out, tuple):
        id_out = list(id_out)
    else:
        id_out = [id_out]
    combined_out = []
    for o in raw_outputs:
        if isinstance(o, tf.Tensor):
            combined_out.append(id_out.pop(0))
        else:
            combined_out.append(o)
    assert len(id_out) == 0, str(id_out)
    return tuple(combined_out)


def recompute_grad(fn, use_entire_scope=False, train_vars=None):
    """Decorator that recomputes the function on the backwards pass.
    Args:
      fn: a function that takes Tensors (all as positional arguments) and returns
        a tuple of Tensors.
    Returns:
      A wrapped fn that is identical to fn when called, but its activations will
      be discarded and recomputed on the backwards pass (i.e. on a call to
      tf.gradients).
      :param use_entire_scope:
    """

    @functools.wraps(fn)
    def wrapped(*args):
        out = _recompute_grad(fn, args, use_entire_scope=use_entire_scope, train_vars=train_vars)
        if isinstance(out, tuple) and len(out) == 1:
            return out[0]
        return out

    return wrapped


def underlying_variable_ref(t):
    """Find the underlying variable ref.
    Traverses through Identity, ReadVariableOp, and Enter ops.
    Stops when op type has Variable or VarHandle in name.
    Args:
      t: a Tensor
    Returns:
      a Tensor that is a variable ref, or None on error.
    """
    while t.op.type in ["Identity", "ReadVariableOp", "Enter"]:
        t = t.op.inputs[0]

    op_type = t.op.type
    if "Variable" in op_type or "VarHandle" in op_type:
        return t
    else:
        return None


def _recompute_grad(fn, args, use_entire_scope, train_vars=None):
    """See recompute_grad."""

    cached_vs = []

    def grad_fn(inputs, variables, outputs, output_grads):
        """Recompute outputs for gradient computation."""
        del outputs
        variables = [underlying_variable_ref(v) for v in variables]
        # Recompute outputs
        with tf.control_dependencies([o for o in output_grads if o is not None]):
            with tf.compat.v1.variable_scope(cached_vs[0], reuse=True):
                outputs = fn(*inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        outputs = [o for o in outputs if isinstance(o, tf.Tensor)]
        input_vars = inputs + variables
        grads = tf.gradients(ys=outputs, xs=input_vars, grad_ys=output_grads)

        grad_inputs = grads[:len(inputs)]
        grad_vars = grads[len(inputs):]
        return grad_inputs, grad_vars

    @fn_with_custom_grad(grad_fn, use_entire_scope=use_entire_scope, train_vars=train_vars)
    def fn_with_recompute(*args):
        cached_vs.append(tf.compat.v1.get_variable_scope())
        return fn(*args)

    return fn_with_recompute(*args)


def recompute_grads_w_kwargs(fn, use_entire_scope=False, train_vars=None, name=None):
    def inner_recompute_grads_w_kwargs(*args, **kwargs):
        # Force an activation cache just before it enters the function, meaning output of 1 never == input of the other.
        args = [tf.identity(a) if isinstance(a, tf.Tensor) else a for a in args]
        kwargs = {k: tf.identity(v) if isinstance(v, tf.Tensor) else v for k, v in kwargs.items()}
        output_keywords = None
        output_local = None
        def remapped_fn(*_):
            nonlocal output_keywords
            nonlocal output_local
            out = fn(*args, **kwargs)
            if isinstance(out, dict):
                output_keywords = []
                output_local = dict()
                for k, v in out.items():
                    if isinstance(v, tf.Tensor):
                        output_keywords.append(k)
                    else:
                        output_local[k] = v
                return [out[k] for k in output_keywords]
            return out
        tensor_args_kwargs = [a for a in list(args) + list(kwargs.values()) if isinstance(a, tf.Tensor)]
        out = _recompute_grad(remapped_fn, tensor_args_kwargs, use_entire_scope=use_entire_scope, train_vars=train_vars)
        if output_keywords is not None:
            return {**{k: v for k, v in zip(output_keywords, out)}, **output_local}
        return out
    return inner_recompute_grads_w_kwargs
