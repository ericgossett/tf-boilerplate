import tensorflow as tf

class Model:
    def __init__(self, config):
        self.config = config
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step'
            )

    def model_constructor(self):
        raise NotImplementedError
