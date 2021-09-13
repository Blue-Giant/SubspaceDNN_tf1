# -*- coding: utf-8 -*-
"""
Created on 2021.09.08
@author: xi'an Li
"""
import tensorflow as tf
import numpy as np


def pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: tensor (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def np_pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: numpy (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = tf.nn.top_k(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def np_knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = np.argpartition(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def knn_excludeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors index: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    k_neighbors = k+1
    _, knn_idx = tf.nn.top_k(neg_dist, k=k_neighbors)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    nn_idx = knn_idx[:, 1: k_neighbors]
    return nn_idx


def get_kneighbors_3D_4DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (batch_size, num_points, 1, dim)
        nn_idx: (batch_size, num_points, k)
        k: int

        Returns:
        neighbors features: (batch_size, num_points, k, dim)
      """
    og_batch_size = point_set.get_shape().as_list()[0]
    og_num_dims = point_set.get_shape().as_list()[-1]
    point_set = tf.squeeze(point_set)
    if og_batch_size == 1:
        point_set = tf.expand_dims(point_set, 0)
    if og_num_dims == 1:
        point_set = tf.expand_dims(point_set, -1)

    point_set_shape = point_set.get_shape()
    batch_size = point_set_shape[0].value
    num_points = point_set_shape[1].value
    num_dims = point_set_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_set_flat = tf.reshape(point_set, [-1, num_dims])
    point_set_neighbors = tf.gather(point_set_flat, nn_idx + idx_)

    return point_set_neighbors


def get_kneighbors_2DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (num_points, dim)
        nn_idx: (num_points, k_num)
        num_points: the number of point
        k_num: the number of neighbor

        Returns:
        neighbors features: (num_points, k_num, dim)
      """
    shape2point_set = point_set.get_shape().as_list()
    assert(len(shape2point_set) == 2)
    point_set_neighbors = tf.gather(point_set, nn_idx)
    return point_set_neighbors


def cal_attends2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)            # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1)   # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    exp_dis = tf.exp(-norm2edges)                   # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=-1)
    atten_ceof = tf.expand_dims(normalize_exp_dis, axis=-2)   # (num_points, 1, k_neighbors)
    return atten_ceof


def cal_edgesNorm_attends2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)            # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1, keepdims=True)   # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    normalize_edgeNrom = tf.nn.softmax(norm2edges, axis=1)
    exp_dis = tf.exp(-norm2edges)                   # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=1)
    atten_ceof = tf.transpose(normalize_exp_dis, perm=[0, 2, 1])
    return normalize_edgeNrom, atten_ceof


class np_GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(np_GaussianNormalizer, self).__init__()

        self.mean2x = np.mean(x, axis=-1, keepdims=True)
        self.std2x = np.std(x, axis=-1, keepdims=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean2x) / (self.std2x * self.std2x + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std2x * self.std2x + self.eps)) + self.mean2x
        return x


# ---------------------------------------------- my activations -----------------------------------------------
class my_actFunc(object):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tf.nn.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)*tf.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tf.nn.elu(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = tf.exp(-1.0*x_input*x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = tf.sin(x_input)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tf.nn.sigmoid(x_input)
        elif str.lower(self.actName) == 'softplus':
            out_x = tf.nn.softplus(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = x_input*tf.tanh(tf.math.log(1+tf.exp(x_input)))
        else:
            out_x = x_input
        return out_x

    # The following activation functions are previous works in the old version of DNN_base
    def sinAddcos(self, x):
        return 0.5*(tf.sin(x) + tf.cos(x))
        # return tf.sin(x) + tf.cos(x)

    def sinAddcos_sReLu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*(tf.sin(2*np.pi*x) + tf.cos(2*np.pi*x))

    def s3relu(self, x):
        # return 0.5*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
        # return 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
        # return tf.nn.relu(1 - x) * tf.nn.relu(x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x))   # (work不好)
        # return tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x)) #（不work）
        return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))      # work 不如 s2relu
        # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)            # work 不如 s2relu
        # return 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(np.pi*x)
        # return tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)

    def csrelu(self, x):
        # return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.cos(np.pi*x)
        return 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
        # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)

    def stanh(self, x):
        # return tf.tanh(x)*tf.sin(2*np.pi*x)
        return tf.sin(2*np.pi*tf.tanh(x))

    def mexican(self, x):
        return (1-x*x)*tf.exp(-0.5*x*x)

    def modify_mexican(self, x):
        # return 1.25*x*tf.exp(-0.25*x*x)
        # return x * tf.exp(-0.125 * x * x)
        return x * tf.exp(-0.075*x * x)
        # return -1.25*x*tf.exp(-0.25*x*x)

    def sm_mexican(x):
        # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
        # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
        return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)

    def singauss(self, x):
        # return 0.6 * tf.exp(-4 * x * x) * tf.sin(np.pi * x)
        # return 0.6 * tf.exp(-5 * x * x) * tf.sin(np.pi * x)
        # return 0.75*tf.exp(-5*x*x)*tf.sin(2*np.pi*x)
        # return tf.exp(-(x-0.5) * (x - 0.5)) * tf.sin(np.pi * x)
        # return 0.25 * tf.exp(-3.5 * x * x) * tf.sin(2 * np.pi * x)
        # return 0.225*tf.exp(-2.5 * (x - 0.5) * (x - 0.5)) * tf.sin(2*np.pi * x)
        return 0.225 * tf.exp(-2 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
        # return 0.4 * tf.exp(-10 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
        # return 0.45 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(np.pi * x)
        # return 0.3 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(2 * np.pi * x)
        # return tf.sin(2*np.pi*tf.exp(-0.5*x*x))

    def powsin_srelu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)

    def sin2_srelu(self, x):
        return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)

    def slrelu(self, x):
        return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)

    def pow2relu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)

    def selu(self, x):
        return tf.nn.elu(1-x)*tf.nn.elu(x)

    def wave(self, x):
        return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
               2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)

    def phi(self, x):
        return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
               - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


class Dense_Net(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
            if str.lower(self.name2Model) == 'fourier_dnn':
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** 0.5
                Win = tf.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                      dtype=self.float_type)
                Bin = tf.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=self.float_type, trainable=False)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units)-1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                    if i_layer == 0:
                        W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                            shape=(hidden_units[i_layer] * 2, hidden_units[i_layer + 1]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                        B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                    else:
                        W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                            shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                        B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)
            else:
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** 0.5
                Win = tf.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                      dtype=self.float_type)
                Bin = tf.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=self.float_type, trainable=True)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units) - 1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                    if i_layer == 0:
                        W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                            shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                        B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                    else:
                        W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                            shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                        B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.get_variable(name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)
            Bout = tf.get_variable(name=str(scope2B) + '_out', shape=(outdim,),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        if str.lower(self.name2Model) == 'fourier_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            if str.lower(self.actName) == 's2relu':
                H = 0.5 * tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
            else:
                H = tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
        elif str.lower(self.name2Model) == 'scale_dnn' or str.lower(self.name2Model) == 'wavelet_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            H = self.actFunc_in(H * repeat_scale)
        else:  # name2Model:DNN, RBF_DNN
            H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_ScaleNet(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** 0.5
            Win = tf.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                  initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                  dtype=self.float_type)
            Bin = tf.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                  dtype=self.float_type, trainable=True)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                if i_layer == 0:
                    W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                        shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                    B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                else:
                    W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                        shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                    B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.get_variable(name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)
            Bout = tf.get_variable(name=str(scope2B) + '_out', shape=(outdim,),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        H = self.actFunc_in(H * repeat_scale)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_Fourier_Net(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_Fourier_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** 0.5
            Win = tf.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                  initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                  dtype=self.float_type)
            Bin = tf.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                  dtype=self.float_type, trainable=False)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                if i_layer == 0:
                    W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                        shape=(hidden_units[i_layer] * 2, hidden_units[i_layer + 1]),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                    B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                else:
                    W = tf.get_variable(name=str(scope2W) + str(i_layer),
                                        shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                    B = tf.get_variable(name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                        dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.get_variable(name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)
            Bout = tf.get_variable(name=str(scope2B) + '_out', shape=(outdim,),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                   dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        if str.lower(self.actName) == 's2relu':
            H = 0.5 * tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
        else:
            H = tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


if __name__ == "__main__":
    input_dim = 3
    out_dim = 1
    hidden_layer = (5, 10, 10, 15, 20)
    name2base_model = 'DNN'
    actFun = 'tanh'

    model = Dense_Net(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=name2base_model,
                      actName=actFun)
    batch_size = 10
    x = np.random.rand(batch_size, input_dim)
    freq = [1, 2, 3, 4, 5, 6, 7, 8]
    with tf.device('/gpu:%s' % ('0')):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='XYit2train', shape=[None, input_dim])      # [N, D]
            Y = model(X, scale=freq)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        for i_epoch in range(5):
            sess.run(tf.global_variables_initializer())
            y = sess.run(Y, feed_dict={X: x})
            print('Y:', y)