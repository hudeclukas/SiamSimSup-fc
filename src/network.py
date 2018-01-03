import tensorflow as tf

class siamese_fc:
    def __init__(self, margin=5, image_size=(100,100,3)):
        self.margin = margin
        self.x1 = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="x1")
        self.x2 = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="x2")
        self.network1 = None
        self.network2 = None


        with tf.variable_scope("siamese-fc") as scope:
            self.network1 = self.network(self.x1)
            scope.reuse_variables()
            self.network2 = self.network(self.x2)

        self.y = tf.placeholder(tf.float32, [None, ], name="labels")
        self.loss = self.loss_contrastive()

    def network(self, x):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="conv1"
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2,
            name="pool1"
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="conv2"
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=2,
            name="pool2"
        )
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=164,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="conv3"
        )

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=164,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="conv4"
        )
        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=128,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="conv5"
        )

        # pool3 = tf.layers.max_pooling2d(
        #     inputs=conv5,
        #     pool_size=[3,3],
        #     strides=3,
        #     name="pool3"
        # )
        fc1 = tf.layers.conv2d(
            inputs=conv5,
            filters=384,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="fc1"
        )
        fc2 = tf.layers.conv2d(
            inputs=fc1,
            filters=10,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_uniform_initializer(),
            name="fc2"
        )
        flatten = tf.reshape(fc2, [-1, 8 * 8 * 10])
        return flatten


    def loss_contrastive(self):
        # zero label means similar, one is value of dissimilarity
        y_t = tf.subtract(1.0, tf.convert_to_tensor(self.y, dtype=tf.float32, name="labels"), name="dissimilarity")
        margin = tf.constant(self.margin, name="margin", dtype=tf.float32)

        eucd2 = tf.reduce_sum(tf.pow(tf.subtract(self.network1, self.network2), 2), 1, name="euclid2")
        eucd = tf.sqrt(eucd2, name="euclid")

        tf.summary.histogram('euclidean_distance', eucd)

        y_f = tf.subtract(1.0, y_t, name="1-y")
        half_f = tf.multiply(y_f, 0.5, name="y_f/2")
        similar = tf.multiply(half_f, eucd2, name="con_l")
        half_t = tf.multiply(y_t, 0.5, name="y_t/2")
        dissimilar = tf.multiply(half_t, tf.maximum(0.0, tf.subtract(margin, eucd)))

        losses = tf.add(similar, dissimilar, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        try:
            tf.check_numerics(loss, 'Check of the loss: ')
        except tf.errors.InvalidArgumentError:
            print('InvalidArgumentError in loss')
        else:
            tf.summary.histogram('loss_s', loss)

        return loss

def similarity(vec1, vec2):
    eucd2 = tf.reduce_sum(tf.pow(tf.subtract(vec1, vec2), 2), 1, name="euclid2_test")
    eucd = tf.sqrt(eucd2, name="euclid_test")
    return eucd
