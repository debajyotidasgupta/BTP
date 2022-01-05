import tensorflow as tf


def load_policy(policy):
    saved_policy = tf.saved_model.load(f'policies/{policy}')
    return saved_policy
