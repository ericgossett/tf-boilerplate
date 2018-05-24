import tensorflow as tf

class Train:
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.sess.run(
            tf.group(
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            )
        )

    def train(self):
        for epoch in range(self.config['max_epoch']):
            print('epoch %d/%d' % (epoch+1, self.config['max_epoch']))
            self.training_step()

    def training_step(self):
        raise NotImplementedError
