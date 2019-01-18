import math
import tensorflow as tf
from itertools import zip_longest


def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1


def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)


def exponential_cyclic(x, lr, global_step, total_num_steps, warmup = 0.002, 
                       gamma = 0.999, max_lr = 0.0005, min_lr = 0.00001, num_cycles = 20):
    x = tf.cast(x, tf.float32)
    lr = tf.cast(lr, tf.float32)
    global_step = tf.cast(global_step, tf.float32)
    total_num_steps = tf.cast(total_num_steps, tf.float32)
    print("after casting")
    #exponential decay of upper/lower learning_rate boundaries
    decay = tf.pow(gamma,global_step)
    print("after power")
    max_lr = max_lr * decay
    min_lr = min_lr * decay
    
    print("after decay")
    #normal triangular cycle
    steps_per_cycle = tf.floor(total_num_steps/num_cycles)
    current_cycle = tf.floor(1 + x*num_cycles)
    steps_completed_current_cycle = global_step-steps_per_cycle*(current_cycle-1)
    fraction_through_cycle = (steps_completed_current_cycle/steps_per_cycle)
    increasing = fraction_through_cycle <= 0.5
    print("after fraction")
    def increase(): return min_lr + (fraction_through_cycle/0.5)*(max_lr-min_lr)
    def decrease(): 
        fraction_through_decrease = (fraction_through_cycle-0.5)/0.5
        return min_lr + (1 - fraction_through_decrease)*(max_lr-min_lr)
    new_lr = tf.cond(increasing, increase, decrease)
    
    print("return")
    return new_lr/lr

def triangular_cyclic(x, lr, global_step, total_num_steps, warmup = 0.002, 
                       max_lr = 0.0006, min_lr = 0.0001, num_cycles = 20):
    x = tf.cast(x, tf.float32)
    lr = tf.cast(lr, tf.float32)
    global_step = tf.cast(global_step, tf.float32)
    total_num_steps = tf.cast(total_num_steps, tf.float32)
    steps_per_cycle = tf.floor(total_num_steps/num_cycles)
    current_cycle = tf.floor(1 + x*num_cycles)
    steps_completed_current_cycle = global_step-steps_per_cycle*(current_cycle-1)
    fraction_through_cycle = (steps_completed_current_cycle/steps_per_cycle)
    increasing = fraction_through_cycle <= 0.5
    def increase(): return min_lr + (fraction_through_cycle/0.5)*(max_lr-min_lr)
    def decrease(): 
        fraction_through_decrease = (fraction_through_cycle-0.5)/0.5
        return min_lr + (1 - fraction_through_decrease)*(max_lr-min_lr)
    new_lr = tf.cond(increasing, increase, decrease)
    
    return new_lr/lr
    
schedules = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
    'triangular_cyclic': triangular_cyclic,
    'exponential_cyclic': exponential_cyclic,
    'none': lambda x, *args, **kwargs: x,
}