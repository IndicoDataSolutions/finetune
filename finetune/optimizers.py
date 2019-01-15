import math
import tensorflow as tf
from itertools import zip_longest


def warmup_cosine(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1


def warmup_linear(x, warmup=0.002, *args):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

def exponential_cyclic(x, lr, global_step, total_num_steps, warmup = 0.002, 
                       gamma = 0.999, max_lr = 0.006, min_lr = 0.001, num_cycles = 20):
    #exponential decay of upper/lower learning_rate boundaries
    decay = math.pow(gamma,global_step)
    max_lr = max_lr * decay
    min_lr = min_lr * decay
    
    #normal triangular cycle
    steps_per_cycle = math.floor(total_num_steps/num_cycles)
    current_cycle = math.floor(1 + x*num_cycles)
    steps_completed_current_cycle = global_step-steps_per_cycle*(current_cycle-1)
    fraction_through_cycle = (steps_completed_current_cycle/steps_per_cycle)
    increasing = fraction_through_cycle <= 0.5
    print(x)
    if increasing:
        new_lr = min_lr + (fraction_through_cycle/0.5)*(max_lr-min_lr)
    else:
        fraction_through_decrease = (fraction_through_cycle-0.5)/0.5
        new_lr = min_lr + (1 - fraction_through_decrease)*(max_lr-min_lr)
    
    
    return new_lr/lr

def triangular_cyclic(x, lr, global_step, total_num_steps, warmup = 0.002, 
                       max_lr = 0.0006, min_lr = 0.0001, num_cycles = 20):
    
    steps_per_cycle = math.floor(total_num_steps/num_cycles)
    current_cycle = math.floor(1 + x*num_cycles)
    steps_completed_current_cycle = global_step-steps_per_cycle*(current_cycle-1)
    fraction_through_cycle = (steps_completed_current_cycle/steps_per_cycle)
    increasing = fraction_through_cycle <= 0.5
    print(x)
    if increasing:
        new_lr = min_lr + (fraction_through_cycle/0.5)*(max_lr-min_lr)
    else:
        fraction_through_decrease = (fraction_through_cycle-0.5)/0.5
        new_lr = min_lr + (1 - fraction_through_decrease)*(max_lr-min_lr)
    
    
    return new_lr/lr
    
schedules = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
    'triangular_cyclic': triangular_cyclic,
    'exponential_cyclic': exponential_cyclic,
    'none': lambda x, *args, **kwargs: x,
}