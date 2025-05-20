import gc
import tensorflow as tf

def cleanup_sessions():
    """
    Clears the tensorflow session and resets the default graph.
    Also runs gc as some of the released objects are in cycles and very large.
    Need to be careful about using this alongside the scheduler / cache predict as it will clear the session.
    """
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()