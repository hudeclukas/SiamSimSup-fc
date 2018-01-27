import tensorflow as tf
from tensorflow.contrib import slim as slim

class siamese_fc:
    def __init__(self, margin=5, image_size=(150,150,1)):
        self._margin = margin
        self.x1 = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="x1")
        self.x2 = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="x2")
        self.network1 = None
        self.network2 = None
        self._image_size = image_size
        self.training = True
        self.dropout_prob = 0.5

        with tf.variable_scope("siamese-fc") as scope:
            self.network1 = self._network_slim(self.x1)
            scope.reuse_variables()
            self.network2 = self._network_slim(self.x2)

        self.y = tf.placeholder(tf.float32, [None, ], name="labels")
        self.loss = self.loss_contrastive()

    def _network_slim(self, x):
        with tf.variable_scope('input_images','slim_net',reuse=True):
            batch_shape = x.shape
            iyr = int(450 / batch_shape[1].value)
            ixr = int(450 / batch_shape[2].value)
            ix = batch_shape[1].value
            iy = batch_shape[2].value
            batch_d = tf.slice(x, (0, 0, 0, 0), (1, -1, -1, -1))
            batch_s = tf.slice(x, (3, 0, 0, 0), (1, -1, -1, -1))
            batch_d = tf.image.resize_images(
                images=batch_d,
                size=(iyr * iy, ixr * ix),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            batch_s = tf.image.resize_images(
                images=batch_s,
                size=(iyr * iy, ixr * ix),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            slim.summary.image("batch_d_input", batch_d)
            slim.summary.image("batch_s_input", batch_s)
        conv1 = slim.conv2d(
            inputs=x,
            num_outputs=96,
            kernel_size=[5,5],
            scope='conv1',
            reuse=tf.AUTO_REUSE
        )
        pool1 = slim.max_pool2d(
            inputs=conv1,
            kernel_size=[3, 3],
            stride=3,
            scope='pool1'
        )
        with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
            b1, b2 = tf.split(pool1, 2, 3)
            b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
            # The original implementation has bias terms for all convolution, but
            # it actually isn't necessary if the convolution layer is followed by a batch
            # normalization layer since batch norm will subtract the mean.
            b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
            conv2 = tf.concat([b1, b2], 3)
        pool2 = slim.max_pool2d(
            inputs=conv2,
            kernel_size=[3, 3],
            stride=2,
            scope='pool2'
        )
        conv3 = slim.conv2d(
            inputs=pool2,
            num_outputs=384,
            kernel_size=[3, 3],
            stride=1,
            scope='conv3',
            reuse=tf.AUTO_REUSE
        )
        with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE):
            b1, b2 = tf.split(conv3, 2, 3)
            b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
            b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
            conv4 = tf.concat([b1, b2], 3)
        # Conv 5 with only convolution, has bias
        with tf.variable_scope('conv5',reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=None):
                b1, b2 = tf.split(conv4, 2, 3)
                b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
                b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
            conv5 = tf.concat([b1, b2], 3)

        with tf.variable_scope('out_image','slim_net',reuse=tf.AUTO_REUSE):
            batch_shape = conv5.shape
            iyr = int(650 / batch_shape[1].value)
            ixr = int(650 / batch_shape[2].value)
            ix = batch_shape[1].value
            iy = batch_shape[2].value
            image_d = tf.slice(conv5,(0,0,0,0),(1,-1,-1,-1))
            image_s = tf.slice(conv5,(3,0,0,0),(1,-1,-1,-1))
            image_d = tf.reshape(image_d, (iy, ix, batch_shape[3].value))
            image_s = tf.reshape(image_s, (iy, ix, batch_shape[3].value))
            ix+=2
            iy+=2
            image_d = tf.image.resize_image_with_crop_or_pad(image_d, iy, ix)
            image_s = tf.image.resize_image_with_crop_or_pad(image_s, iy, ix)
            image_d = tf.reshape(image_d, (iy, ix, 16, 16))
            image_s = tf.reshape(image_s, (iy, ix, 16, 16))
            image_d = tf.transpose(image_d,(2,0,3,1))
            image_s = tf.transpose(image_s,(2,0,3,1))
            image_d = tf.reshape(image_d, (1, 16 * iy, 16 * ix, 1))
            image_s = tf.reshape(image_s, (1, 16 * iy, 16 * ix, 1))
            image_d = tf.image.resize_images(
                images=image_d,
                size=(iyr * iy, ixr * ix),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            image_s = tf.image.resize_images(
                images=image_s,
                size=(iyr * iy, ixr * ix),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            tf.summary.image("out_d",image_d)
            tf.summary.image("out_s",image_s)

        fc1 = slim.conv2d(
            inputs=conv5,
            num_outputs=1024,
            kernel_size=[1, 1],
            stride=1,
            scope='fc1',
            reuse=tf.AUTO_REUSE
        )
        dropout = slim.dropout(
            fc1,
            scope='dropout',
            keep_prob=self.dropout_prob,
            is_training=self.training
        )
        fc2 = slim.conv2d(
            inputs=dropout,
            num_outputs=128,
            kernel_size=[1, 1],
            stride=1,
            scope='fc2',
            reuse=tf.AUTO_REUSE
        )
        flatten = slim.flatten(fc2, scope='flatten')
        return flatten

    def distanceEuclid(self, in1:tf.Tensor, in2:tf.Tensor, name:str):
        with tf.variable_scope('Euclid_dw_'+name, reuse=tf.AUTO_REUSE):
            eucd2 = tf.reduce_sum(tf.pow(tf.subtract(in1, in2), 2), 1, name="e2_" + name)
            eucd = tf.sqrt(eucd2, name="e_" + name)
            return eucd, eucd2

    def distanceCanberra(self, in1:tf.Tensor, in2:tf.Tensor, name:str):
        with tf.variable_scope('Canberra_dw_'+name, reuse=tf.AUTO_REUSE):
            in1_abs = tf.abs(in1,name+'_a_in1')
            in2_abs = tf.abs(in2,name+'_a_in2')
            in1_in2_abs = tf.abs(tf.subtract(in1, in2, name + '_a_in1_in2'))
            canbd = tf.reduce_sum(tf.divide(in1_in2_abs,tf.add(tf.add(in1_abs,in2_abs),tf.constant(0.00001,dtype=tf.float32))),axis=1,name='canberra_'+name)
            return canbd

    def loss_contrastive(self):
        # one label means similar, zero is value of dissimilarity
        y_t = tf.subtract(1.0, tf.convert_to_tensor(self.y, dtype=tf.float32, name="labels"), name="dissimilarity")
        margin = tf.constant(self._margin, name="margin", dtype=tf.float32)

        # canbd , canbd2 = self.distanceEuclid(self.network1, self.network2, 'eucd-loss')

        canbd = tf.divide(tf.reduce_sum(tf.divide(tf.abs(tf.subtract(self.network1, self.network2)),
                                        tf.add(tf.add(tf.abs(self.network1), tf.abs(self.network2)), tf.constant(0.0001,dtype=tf.float32))), axis=1
                                        ), tf.constant(1000, tf.float32),
                                        name='canberra_dst')
        canbd2 = tf.pow(canbd, 2, 'canb_2')

        try:
            tf.check_numerics(canbd, 'Check of the Canberra distance (canbd): ')
        except tf.errors.InvalidArgumentError:
            print('InvalidArgumentError in Canberra distance "canbd"')
        else:
            tf.summary.histogram('canberra_distance', canbd)

        y_f = tf.subtract(1.0, y_t, name="1-y")
        half_f = tf.multiply(y_f, 0.5, name="y_f/2")
        similar = tf.multiply(half_f, canbd2, name="con_l")
        half_t = tf.multiply(y_t, 0.5, name="y_t/2")
        dissimilar = tf.multiply(half_t, tf.maximum(0.0, tf.subtract(margin, canbd)))

        losses = tf.add(similar, dissimilar, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        # loss=tf.reduce_mean(canbd)
        try:
            tf.check_numerics(loss, 'Check of the loss: ')
        except tf.errors.InvalidArgumentError:
            print('InvalidArgumentError in loss')
        else:
            tf.summary.histogram('loss_s', loss)

        return loss

def similarityEc(vec1, vec2):
    eucd2 = tf.reduce_sum(tf.pow(tf.subtract(vec1, vec2), 2), 1, name="euclid2_test")
    eucd = tf.sqrt(eucd2, name="euclid_test")
    return eucd

def similarityCb(vec1, vec2):
    canbd = tf.divide(tf.reduce_sum(tf.divide(tf.abs(tf.subtract(vec1, vec2)),
                                              tf.add(tf.add(tf.abs(vec1), tf.abs(vec2)),
                                                     tf.constant(0.0001, dtype=tf.float32))), axis=1
                                    ), tf.constant(1000, tf.float32),
                      name='canberra_dst')
    return canbd