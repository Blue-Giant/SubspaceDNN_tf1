# -*- coding: utf-8 -*-
"""
@author: xi'an Li
Created on 2020.05.31
Modified on 2020.06.17
Modified on 2021.10.15
Modified and formed the final version on 2022.5.15
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
    square_edges = tf.square(edge_point_set)                           # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1, keepdims=True)   # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    exp_dis = tf.exp(-norm2edges)                                      # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=1)
    atten_ceof = tf.transpose(normalize_exp_dis, perm=[0, 2, 1])       # (num_points, 1, k_neighbors)
    return atten_ceof


def cal_edgesNorm_attends2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)                           # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1, keepdims=True)   # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    normalize_edgeNrom = tf.nn.softmax(norm2edges, axis=1)
    exp_dis = tf.exp(-norm2edges)                                      # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=1)
    atten_ceof = tf.transpose(normalize_exp_dis, perm=[0, 2, 1])
    return normalize_edgeNrom, atten_ceof


# ---------------------------------------------- my activations -----------------------------------------------
def linear(x):
    return x


def mysin(x):
    # return tf.sin(2*np.pi*x)
    return tf.sin(x)
    # return 0.5*tf.sin(x)


def srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)


def s2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.5*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.25*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)


def sinAddcos(x):
    return 0.5*(tf.sin(x) + tf.cos(x))
    # return tf.sin(x) + tf.cos(x)


def sinAddcos_sReLu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*(tf.sin(2*np.pi*x) + tf.cos(2*np.pi*x))


def s3relu(x):
    # return 0.5*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x))   # (work不好)
    # return tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x)) #（不work）
    return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))      # work 不如 s2relu
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)            # work 不如 s2relu
    # return 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)


def csrelu(x):
    # return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.cos(np.pi*x)
    return 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)


def stanh(x):
    return tf.tanh(x)*tf.sin(2*np.pi*x)


def gauss(x):
    return tf.exp(-1.0 * x * x)
    # return 0.2*tf.exp(-4*x*x)
    # return 0.25*tf.exp(-7.5*(x-0.5)*(x-0.5))


def mexican(x):
    return (1-x*x)*tf.exp(-0.5*x*x)


def modify_mexican(x):
    # return 1.25*x*tf.exp(-0.25*x*x)
    # return x * tf.exp(-0.125 * x * x)
    return x * tf.exp(-0.075*x * x)
    # return -1.25*x*tf.exp(-0.25*x*x)


def sm_mexican(x):
    # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
    # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
    return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)


def singauss(x):
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


def powsin_srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)


def sin2_srelu(x):
    return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)


def slrelu(x):
    return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)


def pow2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)


def selu(x):
    return tf.nn.elu(1-x)*tf.nn.elu(x)


def wave(x):
    return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
           2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)


def phi(x):
    return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
           - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


def gelu(x):
    out = x*tf.exp(x)/(1+tf.exp(x))
    return out


def mgelu(x):
    temp2x = np.sqrt(2 / np.pi) * (x + 0.044715 * x * x * x)
    # out = 0.5*+ 0.5*x*tf.tanh(temp2x)
    out = 0.25 * x * tf.tanh(temp2x)
    return out


# ------------------------------------------------  初始化权重和偏置 --------------------------------------------
# 生成DNN的权重和偏置
# tf.random_normal(): 用于从服从指定正太分布的数值中取出随机数
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# hape: 输出张量的形状，必选.--- mean: 正态分布的均值，默认为0.----stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32 ----seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样---name: 操作的名称
def Generally_Init_NN(in_size, out_size, hidden_layers, Flag='flag'):
    n_hiddens = len(hidden_layers)
    Weights = []  # 权重列表，用于存储隐藏层的权重
    Biases = []  # 偏置列表，用于存储隐藏层的偏置
    # 隐藏层：第一层的权重和偏置，对输入数据做变换
    W = tf.compat.v1.Variable(0.1 * tf.random.normal([in_size, hidden_layers[0]]), dtype='float32',
                              name='W_transInput' + str(Flag))
    B = tf.compat.v1.Variable(0.1 * tf.random.uniform([1, hidden_layers[0]]), dtype='float32',
                              name='B_transInput' + str(Flag))
    Weights.append(W)
    Biases.append(B)
    # 隐藏层：第二至倒数第二层的权重和偏置
    for i_layer in range(n_hiddens - 1):
        W = tf.compat.v1.Variable(0.1 * tf.random.normal([hidden_layers[i_layer], hidden_layers[i_layer+1]]),
                                  dtype='float32', name='W_hidden' + str(i_layer + 1) + str(Flag))
        B = tf.compat.v1.Variable(0.1 * tf.random.uniform([1, hidden_layers[i_layer+1]]), dtype='float32',
                                  name='B_hidden' + str(i_layer + 1) + str(Flag))
        Weights.append(W)
        Biases.append(B)

    # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
    W = tf.compat.v1.Variable(0.1 * tf.random.normal([hidden_layers[-1], out_size]), dtype='float32',
                              name='W_outTrans' + str(Flag))
    B = tf.compat.v1.Variable(0.1 * tf.random.uniform([1, out_size]), dtype='float32',
                              name='B_outTrans' + str(Flag))
    Weights.append(W)
    Biases.append(B)

    return Weights, Biases


# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，
# 均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
# 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
# truncated_normal(
#     shape,
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.float32,
#     seed=None,
#     name=None)
def truncated_normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.compat.v1.Variable(scale_coef*tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32,
                              name=weight_name)
    return V


# tf.random_uniform()
# 默认是在 0 到 1 之间产生随机数，也可以通过 minval 和 maxval 指定上下界
def uniform_init(in_dim, out_dim, weight_name='weight'):
    V = tf.compat.v1.Variable(tf.random_uniform([in_dim, out_dim], dtype=tf.float32), dtype=tf.float32,
                              name=weight_name)
    return V


# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 从正态分布中输出随机值。
# 参数:
# shape: 一维的张量, 也是输出的张量。
# mean: 正态分布的均值。
# stddev: 正态分布的标准差。
# dtype: 输出的类型。
# seed: 一个整数，当设置之后，每次生成的随机数都一样。
# name: 操作的名字。
def normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    stddev2normal = np.sqrt(2.0/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.compat.v1.Variable(scale_coef*tf.random_normal([in_dim, out_dim], mean=0, stddev=stddev2normal,
                                                          dtype=tf.float32), dtype=tf.float32, name=weight_name)
    return V


def Truncated_normal_init_NN(in_size, out_size, hidden_layers, Flag='flag'):
    with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
        scale = 5.0
        n_hiddens = len(hidden_layers)
        Weights = []                  # 权重列表，用于存储隐藏层的权重
        Biases = []                   # 偏置列表，用于存储隐藏层的偏置

        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        W = truncated_normal_init(in_size, hidden_layers[0], scale_coef=scale, weight_name='W-transInput' + str(Flag))
        B = uniform_init(1, hidden_layers[0], weight_name='B-transInput' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            W = truncated_normal_init(hidden_layers[i_layer], hidden_layers[i_layer + 1], scale_coef=scale,
                                      weight_name='W-hidden' + str(i_layer + 1) + str(Flag))
            B = uniform_init(1, hidden_layers[i_layer + 1], weight_name='B-hidden' + str(i_layer + 1) + str(Flag))
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        W = truncated_normal_init(hidden_layers[-1], out_size, scale_coef=scale, weight_name='W-outTrans' + str(Flag))
        B = uniform_init(1, out_size, weight_name='B-outTrans' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def Xavier_init_NN(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []   # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.compat.v1.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.compat.v1.get_variable(
                name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                          initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.compat.v1.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def Xavier_init_NN_Fourier(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []   # 权重列表，用于存储隐藏层的权重
        Biases = []    # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.compat.v1.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)

        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            if 0 == i_layer:
                W = tf.compat.v1.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer]*2, hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            else:
                W = tf.compat.v1.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.compat.v1.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


# ----------------------------------- 正则化 -----------------------------------------------
def regular_weights_biases_L1(weights, biases):
    # L1正则化权重和偏置
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.abs(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.abs(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


# L2正则化权重和偏置
def regular_weights_biases_L2(weights, biases):
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.square(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.square(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


#  --------------------------------------------  网络模型 ------------------------------------------------------
def DNN(variable_input, Weights, Biases, hiddens, activateIn_name='tanh', activate_name='tanh', activateOut_name='linear'):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
    return:
        output data, dim:NxD', generally D'=1
    """
    if str.lower(activateIn_name) == 'relu':
        act_in = tf.nn.relu
    elif str.lower(activateIn_name) == 'leaky_relu':
        act_in = tf.nn.leaky_relu(0.2)
    elif str.lower(activateIn_name) == 'srelu':
        act_in = srelu
    elif str.lower(activateIn_name) == 's2relu':
        act_in = s2relu
    elif str.lower(activateIn_name) == 'elu':
        act_in = tf.nn.elu
    elif str.lower(activateIn_name) == 'sin':
        act_in = mysin
    elif str.lower(activateIn_name) == 'sinaddcos':
        act_in = sinAddcos
    elif str.lower(activateIn_name) == 'tanh':
        act_in = tf.tanh
    elif str.lower(activateIn_name) == 'gauss':
        act_in = gauss
    elif str.lower(activateIn_name) == 'softplus':
        act_in = tf.nn.softplus
    elif str.lower(activateIn_name) == 'sigmoid':
        act_in = tf.nn.sigmoid
    elif str.lower(activateIn_name) == 'gelu':
        act_in = gelu
    elif str.lower(activateIn_name) == 'mgelu':
        act_in = mgelu
    else:
        act_in = linear

    if str.lower(activate_name) == 'relu':
        act_func = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        act_func = tf.nn.leaky_relu(0.2)
    elif str.lower(activate_name) == 'srelu':
        act_func = srelu
    elif str.lower(activate_name) == 's2relu':
        act_func = s2relu
    elif str.lower(activate_name) == 'elu':
        act_func = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        act_func = mysin
    elif str.lower(activate_name) == 'sinaddcos':
        act_func = sinAddcos
    elif str.lower(activate_name) == 'tanh':
        act_func = tf.tanh
    elif str.lower(activate_name) == 'gauss':
        act_func = gauss
    elif str.lower(activate_name) == 'softplus':
        act_func = tf.nn.softplus
    elif str.lower(activate_name) == 'sigmoid':
        act_func = tf.nn.sigmoid
    elif str.lower(activate_name) == 'gelu':
        act_func = gelu
    elif str.lower(activate_name) == 'mgelu':
        act_func = mgelu
    else:
        act_func = linear

    if str.lower(activateOut_name) == 'relu':
        act_out = tf.nn.relu
    elif str.lower(activateOut_name) == 'leaky_relu':
        act_out = tf.nn.leaky_relu(0.2)
    elif str.lower(activateOut_name) == 'srelu':
        act_out = srelu
    elif str.lower(activateOut_name) == 's2relu':
        act_out = s2relu
    elif str.lower(activateOut_name) == 'elu':
        act_out = tf.nn.elu
    elif str.lower(activateOut_name) == 'sin':
        act_out = mysin
    elif str.lower(activateOut_name) == 'sinaddcos':
        act_out = sinAddcos
    elif str.lower(activateOut_name) == 'tanh':
        act_out = tf.nn.tanh
    elif str.lower(activateOut_name) == 'gauss':
        act_out = gauss
    elif str.lower(activateOut_name) == 'softplus':
        act_out = tf.nn.softplus
    elif str.lower(activateOut_name) == 'sigmoid':
        act_out = tf.nn.sigmoid
    elif str.lower(activateOut_name) == 'gelu':
        act_out = gelu
    elif str.lower(activateOut_name) == 'mgelu':
        act_out = mgelu
    else:
        act_out = linear

    layers = len(hiddens) + 1               # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    hidden_record = 0
    for k in range(layers-1):
        H_pre = H
        W = Weights[k]
        B = Biases[k]
        if k == 0:
            H = act_in(tf.add(tf.matmul(H, W), B))
        else:
            H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k] == hidden_record:
            H = H+H_pre
        hidden_record = hiddens[k]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    output = act_out(output)
    return output


def DNN_scale(variable_input, Weights, Biases, hiddens, freq_frag, activateIn_name='tanh', activate_name='tanh',
              activateOut_name='linear', repeat_Highfreq=True):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
    return:
        output data, dim:NxD', generally D'=1
    """
    if str.lower(activateIn_name) == 'relu':
        act_in = tf.nn.relu
    elif str.lower(activateIn_name) == 'leaky_relu':
        act_in = tf.nn.leaky_relu(0.2)
    elif str.lower(activateIn_name) == 'srelu':
        act_in = srelu
    elif str.lower(activateIn_name) == 's2relu':
        act_in = s2relu
    elif str.lower(activateIn_name) == 'elu':
        act_in = tf.nn.elu
    elif str.lower(activateIn_name) == 'sin':
        act_in = tf.sin
    elif str.lower(activateIn_name) == 'sinaddcos':
        act_in = sinAddcos
    elif str.lower(activateIn_name) == 'tanh':
        act_in = tf.tanh
    elif str.lower(activateIn_name) == 'gauss':
        act_in = gauss
    elif str.lower(activateIn_name) == 'softplus':
        act_in = tf.nn.softplus
    elif str.lower(activateIn_name) == 'sigmoid':
        act_in = tf.nn.sigmoid
    elif str.lower(activateIn_name) == 'gelu':
        act_in = gelu
    elif str.lower(activateIn_name) == 'mgelu':
        act_in = mgelu
    else:
        act_in = linear

    if str.lower(activate_name) == 'relu':
        act_func = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        act_func = tf.nn.leaky_relu(0.2)
    elif str.lower(activate_name) == 'srelu':
        act_func = srelu
    elif str.lower(activate_name) == 's2relu':
        act_func = s2relu
    elif str.lower(activate_name) == 'elu':
        act_func = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        act_func = mysin
    elif str.lower(activate_name) == 'sinaddcos':
        act_func = sinAddcos
    elif str.lower(activate_name) == 'tanh':
        act_func = tf.tanh
    elif str.lower(activate_name) == 'gauss':
        act_func = gauss
    elif str.lower(activate_name) == 'softplus':
        act_func = tf.nn.softplus
    elif str.lower(activate_name) == 'sigmoid':
        act_func = tf.nn.sigmoid
    elif str.lower(activate_name) == 'gelu':
        act_func = gelu
    elif str.lower(activate_name) == 'mgelu':
        act_func = mgelu
    else:
        act_func = linear

    if str.lower(activateOut_name) == 'relu':
        act_out = tf.nn.relu
    elif str.lower(activateOut_name) == 'leaky_relu':
        act_out = tf.nn.leaky_relu(0.2)
    elif str.lower(activateOut_name) == 'srelu':
        act_out = srelu
    elif str.lower(activateOut_name) == 's2relu':
        act_out = s2relu
    elif str.lower(activateOut_name) == 'elu':
        act_out = tf.nn.elu
    elif str.lower(activateOut_name) == 'sin':
        act_out = mysin
    elif str.lower(activateOut_name) == 'sinaddcos':
        act_out = sinAddcos
    elif str.lower(activateOut_name) == 'tanh':
        act_out = tf.nn.tanh
    elif str.lower(activateOut_name) == 'gauss':
        act_out = gauss
    elif str.lower(activateOut_name) == 'softplus':
        act_out = tf.nn.softplus
    elif str.lower(activateOut_name) == 'sigmoid':
        act_out = tf.nn.sigmoid
    elif str.lower(activateOut_name) == 'gelu':
        act_out = gelu
    elif str.lower(activateOut_name) == 'mgelu':
        act_out = mgelu
    else:
        act_out = linear

    Unit_num = int(hiddens[0] / len(freq_frag))

    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    # 这个的作用是什么？
    if repeat_Highfreq==True:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0], mixcoe))

    mixcoe = mixcoe.astype(np.float32)

    layers = len(hiddens) + 1  # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)

    H = act_in(H)

    hidden_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k+1] == hidden_record:
            H = H + H_pre
        hidden_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    output = act_out(output)
    return output


def DNN_adapt_scale(variable_input, Weights, Biases, hiddens, freq_frag, activateIn_name='tanh', activate_name='tanh',
                    activateOut_name='linear', repeat_Highfreq=True):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
    return:
        output data, dim:NxD', generally D'=1
    """
    if str.lower(activateIn_name) == 'relu':
        act_in = tf.nn.relu
    elif str.lower(activateIn_name) == 'leaky_relu':
        act_in = tf.nn.leaky_relu(0.2)
    elif str.lower(activateIn_name) == 'srelu':
        act_in = srelu
    elif str.lower(activateIn_name) == 's2relu':
        act_in = s2relu
    elif str.lower(activateIn_name) == 'elu':
        act_in = tf.nn.elu
    elif str.lower(activateIn_name) == 'sin':
        act_in = tf.sin
    elif str.lower(activateIn_name) == 'sinaddcos':
        act_in = sinAddcos
    elif str.lower(activateIn_name) == 'tanh':
        act_in = tf.tanh
    elif str.lower(activateIn_name) == 'gauss':
        act_in = gauss
    elif str.lower(activateIn_name) == 'softplus':
        act_in = tf.nn.softplus
    elif str.lower(activateIn_name) == 'sigmoid':
        act_in = tf.nn.sigmoid
    elif str.lower(activateIn_name) == 'gelu':
        act_in = gelu
    elif str.lower(activateIn_name) == 'mgelu':
        act_in = mgelu
    else:
        act_in = linear

    if str.lower(activate_name) == 'relu':
        act_func = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        act_func = tf.nn.leaky_relu(0.2)
    elif str.lower(activate_name) == 'srelu':
        act_func = srelu
    elif str.lower(activate_name) == 's2relu':
        act_func = s2relu
    elif str.lower(activate_name) == 'elu':
        act_func = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        act_func = mysin
    elif str.lower(activate_name) == 'tanh':
        act_func = tf.tanh
    elif str.lower(activate_name) == 'gauss':
        act_func = gauss
    elif str.lower(activate_name) == 'softplus':
        act_func = tf.nn.softplus
    elif str.lower(activate_name) == 'sigmoid':
        act_func = tf.nn.sigmoid
    elif str.lower(activate_name) == 'gelu':
        act_func = gelu
    elif str.lower(activate_name) == 'mgelu':
        act_func = mgelu
    else:
        act_func = linear

    if str.lower(activateOut_name) == 'relu':
        act_out = tf.nn.relu
    elif str.lower(activateOut_name) == 'leaky_relu':
        act_out = tf.nn.leaky_relu(0.2)
    elif str.lower(activateOut_name) == 'srelu':
        act_out = srelu
    elif str.lower(activateOut_name) == 's2relu':
        act_out = s2relu
    elif str.lower(activateOut_name) == 'elu':
        act_out = tf.nn.elu
    elif str.lower(activateOut_name) == 'sin':
        act_out = mysin
    elif str.lower(activateOut_name) == 'sinaddcos':
        act_out = sinAddcos
    elif str.lower(activateOut_name) == 'tanh':
        act_out = tf.nn.tanh
    elif str.lower(activateOut_name) == 'gauss':
        act_out = gauss
    elif str.lower(activateOut_name) == 'softplus':
        act_out = tf.nn.softplus
    elif str.lower(activateOut_name) == 'sigmoid':
        act_out = tf.nn.sigmoid
    elif str.lower(activateOut_name) == 'gelu':
        act_out = gelu
    elif str.lower(activateOut_name) == 'mgelu':
        act_out = mgelu
    else:
        act_out = linear

    Unit_num = int(hiddens[0] / len(freq_frag))

    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    init_mixcoe = np.repeat(freq_frag, Unit_num)

    # 这个的作用是什么？
    if repeat_Highfreq==True:
        init_mixcoe = np.concatenate((init_mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        init_mixcoe = np.concatenate((init_mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0]))

    # 将 int 型的 mixcoe 转化为 发np.flost32 型的 mixcoe，mixcoe[: units[1]]省略了行的维度
    init_mixcoe = init_mixcoe.astype(np.float32)

    layers = len(hiddens)+1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    B_in = Biases[0]
    mixcoe = tf.get_variable(name='M0', initializer=init_mixcoe)
    # mixcoe = tf.exp(mixcoe)

    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)

    H = act_in(H)

    hidden_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k+1] == hidden_record:
            H = H + H_pre
        hidden_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    output = act_out(output)
    return output


# FourierBase 代表 cos concatenate sin according to row（i.e. the number of sampling points）
def DNN_FourierBase(variable_input, Weights, Biases, hiddens, freq_frag, activate_name='tanh', activateOut_name='linear',
                    repeat_Highfreq=True, sFourier=0.5):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
        sFourier：a scale factor for adjust the range of input-layer
    return:
        output data, dim:NxD', generally D'=1
    """
    if str.lower(activate_name) == 'relu':
        act_func = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        act_func = tf.nn.leaky_relu(0.2)
    elif str.lower(activate_name) == 'srelu':
        act_func = srelu
    elif str.lower(activate_name) == 's2relu':
        act_func = s2relu
    elif str.lower(activate_name) == 'elu':
        act_func = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        act_func = mysin
    elif str.lower(activate_name) == 'sinaddcos':
        act_func = sinAddcos
    elif str.lower(activate_name) == 'tanh':
        act_func = tf.tanh
    elif str.lower(activate_name) == 'gauss':
        act_func = gauss
    elif str.lower(activate_name) == 'softplus':
        act_func = tf.nn.softplus
    elif str.lower(activate_name) == 'sigmoid':
        act_func = tf.nn.sigmoid
    elif str.lower(activate_name) == 'gelu':
        act_func = gelu
    elif str.lower(activate_name) == 'mgelu':
        act_func = mgelu
    else:
        act_func = linear

    if str.lower(activateOut_name) == 'relu':
        act_out = tf.nn.relu
    elif str.lower(activateOut_name) == 'leaky_relu':
        act_out = tf.nn.leaky_relu(0.2)
    elif str.lower(activateOut_name) == 'srelu':
        act_out = srelu
    elif str.lower(activateOut_name) == 's2relu':
        act_out = s2relu
    elif str.lower(activateOut_name) == 'elu':
        act_out = tf.nn.elu
    elif str.lower(activateOut_name) == 'sin':
        act_out = mysin
    elif str.lower(activateOut_name) == 'sinaddcos':
        act_out = sinAddcos
    elif str.lower(activateOut_name) == 'tanh':
        act_out = tf.nn.tanh
    elif str.lower(activateOut_name) == 'gauss':
        act_out = gauss
    elif str.lower(activateOut_name) == 'softplus':
        act_out = tf.nn.softplus
    elif str.lower(activateOut_name) == 'sigmoid':
        act_out = tf.nn.sigmoid
    elif str.lower(activateOut_name) == 'gelu':
        act_out = gelu
    elif str.lower(activateOut_name) == 'mgelu':
        act_out = mgelu
    else:
        act_out = linear

    layers = len(hiddens) + 1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层

    # 计算第一个隐藏单元和尺度标记的比例
    Unit_num = int(hiddens[0] / len(freq_frag))

    # 然后，频率标记按按照比例复制
    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    if repeat_Highfreq == True:
        # 如果第一个隐藏单元的长度大于复制后的频率标记，那就按照最大的频率在最后补齐
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0]))

    mixcoe = mixcoe.astype(np.float32)

    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        # H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)
        H = tf.matmul(H, W_in) * mixcoe

    H = sFourier * (tf.concat([tf.cos(H), tf.sin(H)], axis=-1))
    # H = sfactor * (tf.concat([tf.cos(np.pi * H), tf.sin(np.pi * H)], axis=-1))
    # H = sfactor * tf.concat([tf.cos(2 * np.pi * H), tf.sin(2 * np.pi * H)], axis=-1)

    hiddens_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if (hiddens[k+1] == hiddens_record) and (k != 0):
            H = H + H_pre
        hiddens_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    output = act_out(output)
    return output
