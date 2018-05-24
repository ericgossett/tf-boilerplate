import tensorflow as tf
from .validators import config_validator

class Train:
    """Base class for training a model.

    Args:
        sess (tf.Session): The current session.
        model (Model): The model to train.
        data (tf.data.Dataset): The training data.
    """
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        config_validator(config)
        self.config = config
        self.logger = logger
        self.sess.run(
            tf.group(
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            )
        )

    def train(self):
        """Iterates over epochs preforming the training_step for each epoch."""
        for epoch in range(self.config['max_epoch']):
            print('epoch %d/%d' % (epoch+1, self.config['max_epoch']))
            self.training_step()

    def training_step(self):
        """Abstract method to define the operations for each epoch."""
        raise NotImplementedError
