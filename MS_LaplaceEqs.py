import tensorflow as tf
import numpy as np
import matData2pLaplace


# 这里注意一下: 对于 np.ones_like(x), x要是一个有实际意义的树或数组或矩阵才可以。不可以是 tensorflow 占位符
# 如果x是占位符，要使用 tf.ones_like
# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数


def get_infos2pLaplace_1D(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps=0.01, equa_name=None):
    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    utrue = lambda x: x - tf.square(x) + eps * (
            1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
        np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    if index2p == 2:
        f = lambda x: tf.ones_like(x)
    elif index2p == 3:
        f = lambda x: abs(2 * x - 1) * (
                4 * eps + 2 * eps * tf.cos(2 * np.pi * x / eps) + np.pi * (1 - 2 * x) * tf.sin(2 * np.pi * x / eps)) / (
                              2 * eps)
    elif index2p == 4:
        f = lambda x: ((1 - 2 * x) ** 2) * (2 + tf.cos(2 * np.pi * x / eps)) * (
                6 * eps + 3 * eps * tf.cos(2 * np.pi * x / eps) - 2 * np.pi * (2 * x - 1) * tf.sin(
            2 * np.pi * x / eps)) / (
                              4 * eps)
    elif index2p == 5:
        f = lambda x: -1.0*abs((2*x-1)**3)*((2+tf.cos(2*np.pi*x/eps))**2)*(3*np.pi*(2*x-1)*tf.sin(2*np.pi*x/eps)-4*eps*tf.cos(2*np.pi*x/eps)-8*eps)/(8*eps)
    elif index2p == 8:
        f = lambda x: ((1 - 2 * x) ** 6) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 5) * (
                7 * eps * tf.cos(2 * np.pi * x / eps) + 2 * (
                7 * eps - 3 * np.pi * (2 * x - 1) * tf.sin(2 * np.pi * x / eps))) / (
                              64 * eps)
    else:

        f = lambda x: (np.power(abs(1 - 2 * x), index2p) * np.power(2 + tf.cos(2 * np.pi * x / eps), index2p) * (
                eps * (index2p - 1) * (2 + tf.cos(2 * np.pi * x / eps)) - np.pi * (index2p - 2) * (2 * x - 1) * tf.sin(
            2 * np.pi * x / eps))) / (
                              np.power(2, index2p - 2) * eps * ((1 - 2 * x) ** 2) * (
                                  (2 + tf.cos(2 * np.pi * x / eps)) ** 3))

    return utrue, f, aeps, u_l, u_r


def get_infos2pLaplace_1D_2(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps1=0.02, eps2=0.01, equa_name=None):
    aeps = lambda x: (2 + tf.cos(2 * np.pi * x / eps1)) * (2 + tf.cos(2 * np.pi * x / eps2))

    utrue = lambda x: x - tf.square(x) + (eps1/(4*np.pi))*tf.sin(2*np.pi*x/eps1) + (eps2/(4*np.pi))*tf.sin(2*np.pi*x/eps2)

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    f = lambda x: tf.ones_like(x)

    return utrue, f, aeps, u_l, u_r


def get_infos2pLaplace_1D_3(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps1=0.02, eps2=0.01, equa_name=None):
    aeps = lambda x: 1.0/((2 + tf.cos(2 * np.pi * x / eps1)) * (2 + tf.cos(2 * np.pi * x / eps2)))

    utrue = lambda x: x - tf.square(x) + (eps1/(4*np.pi))*tf.sin(2*np.pi*x/eps1) + (eps2/(4*np.pi))*tf.sin(2*np.pi*x/eps2)

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    ax = lambda x: -(2*np.pi/eps1)*tf.sin(2 * np.pi * x / eps1)*(1 + tf.cos(2 * np.pi * x / eps2)) - \
                   (2*np.pi/eps2)*tf.sin(2 * np.pi * x / eps2)*(1 + tf.cos(2 * np.pi * x / eps1))

    ux = lambda x: 1 -2*x + 0.5*tf.cos(2 * np.pi * x / eps1) + 0.5*tf.cos(2 * np.pi * x / eps2)

    uxx = lambda x: -2-(np.pi/eps1)*tf.sin(2 * np.pi * x / eps1)-(np.pi/eps2)*tf.sin(2 * np.pi * x / eps2)

    f = lambda x: tf.ones_like(x)

    return utrue, f, aeps, u_l, u_r


def get_infos2pLaplace_1D_4(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps1=0.02, eps2=0.01, equa_name=None):
    aeps = lambda x: (2 + tf.cos(2 * np.pi * x / eps1)) + (2 + tf.cos(2 * np.pi * x / eps2))

    utrue = lambda x: x - tf.square(x) + (eps1/(4*np.pi))*tf.sin(2*np.pi*x/eps1) + (eps2/(4*np.pi))*tf.sin(2*np.pi*x/eps2)

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    ax = lambda x: -(2*np.pi/eps1)*tf.sin(2 * np.pi * x / eps1)*(1 + tf.cos(2 * np.pi * x / eps2)) - \
                   (2*np.pi/eps2)*tf.sin(2 * np.pi * x / eps2)*(1 + tf.cos(2 * np.pi * x / eps1))

    ux = lambda x: 1 -2*x + 0.5*tf.cos(2 * np.pi * x / eps1) + 0.5*tf.cos(2 * np.pi * x / eps2)

    uxx = lambda x: -2-(np.pi/eps1)*tf.sin(2 * np.pi * x / eps1)-(np.pi/eps2)*tf.sin(2 * np.pi * x / eps2)

    f = lambda x: tf.ones_like(x)

    return utrue, f, aeps, u_l, u_r


def force_sice_3scale2(x, eps1=0.02, eps2=0.01):
    aeps = (2 + tf.cos(2 * np.pi * x / eps1)) * (2 + tf.cos(2 * np.pi * x / eps2))

    aepsx = -(2 * np.pi / eps1) * tf.sin(2 * np.pi * x / eps1) * (2 + tf.cos(2 * np.pi * x / eps2)) - \
                   (2 * np.pi / eps2) * tf.sin(2 * np.pi * x / eps2) * (2 + tf.cos(2 * np.pi * x / eps1))

    ux = 1 - 2 * x + 0.5 * tf.cos(2 * np.pi * x / eps1) + 0.5 * tf.cos(2 * np.pi * x / eps2)

    uxx = -2 - (np.pi / eps1) * tf.sin(2 * np.pi * x / eps1) - (np.pi / eps2) * tf.sin(2 * np.pi * x / eps2)

    fside = -1.0*(aepsx * ux + aeps * uxx)

    return fside


def force_sice_3scale3(x, eps1=0.02, eps2=0.01):

    aeps = 1.0/((2 + tf.cos(2 * np.pi * x / eps1)) * (2 + tf.cos(2 * np.pi * x / eps2)))

    ax1 = (2 + tf.cos(2 * np.pi * x / eps1))*(2 + tf.cos(2 * np.pi * x / eps1))*(2 + tf.cos(2 * np.pi * x / eps2))
    ax2 = (2 + tf.cos(2 * np.pi * x / eps1)) * (2 + tf.cos(2 * np.pi * x / eps2)) * (2 + tf.cos(2 * np.pi * x / eps2))
    ax = -(2 * np.pi / eps1) * tf.sin(2 * np.pi * x / eps1) * (1/ax1) - \
                   (2 * np.pi / eps2) * tf.sin(2 * np.pi * x / eps2) * (1/ax2)

    ux = 1 - 2 * x + 0.5 * tf.cos(2 * np.pi * x / eps1) + 0.5 * tf.cos(2 * np.pi * x / eps2)

    uxx = -2 - (np.pi / eps1) * tf.sin(2 * np.pi * x / eps1) - (np.pi / eps2) * tf.sin(2 * np.pi * x / eps2)

    fside = -1.0*(ax * ux + aeps * uxx)

    return fside


#  例一
def true_solution2E1(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E1(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E1(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E1(input_dim=None, output_dim=None):
    a_eps = lambda x, y: 1.0*tf.ones_like(x)
    return a_eps


#  例二
def true_solution2E2(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E2(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E2(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E2(input_dim=None, output_dim=None):
    a_eps = lambda x, y: 2.0 + tf.multiply(tf.sin(3 * np.pi * x), tf.cos(5 * np.pi * y))
    return a_eps


# 例三
def true_solution2E3(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E3(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E3(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E3(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+tf.sin(2*np.pi*x/e1))/(1.1+tf.sin(2*np.pi*y/e1)) +
                              (1.1+tf.sin(2*np.pi*y/e2))/(1.1+tf.cos(2*np.pi*x/e2)) +
                              (1.1+tf.cos(2*np.pi*x/e3))/(1.1+tf.sin(2*np.pi*y/e3)) +
                              (1.1+tf.sin(2*np.pi*y/e4))/(1.1+tf.cos(2*np.pi*x/e4)) +
                              (1.1+tf.cos(2*np.pi*x/e5))/(1.1+tf.sin(2*np.pi*y/e5)) +
                              tf.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例四
def true_solution2E4(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E4(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E4(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E4(input_dim=None, output_dim=None, mesh_num=2):
    if mesh_num == 2:
        a_eps = lambda x, y: (1+0.5*tf.cos(2*np.pi*(x+y)))*(1+0.5*tf.sin(2*np.pi*(y-3*x))) * \
                             (1+0.5*tf.cos((2**2)*np.pi*(x+y)))*(1+0.5*tf.sin((2**2)*np.pi*(y-3*x)))
    elif mesh_num==3:
        a_eps = lambda x, y: (1 + 0.5 * tf.cos(2 * np.pi * (x + y))) * (1 + 0.5 * tf.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * tf.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 3) * np.pi * (y - 3 * x)))
    elif mesh_num == 4:
        a_eps = lambda x, y: (1 + 0.5 * tf.cos(2 * np.pi * (x + y))) * (1 + 0.5 * tf.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * tf.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 4) * np.pi * (y - 3 * x)))
    elif mesh_num == 5:
        a_eps = lambda x, y: (1 + 0.5 * tf.cos(2 * np.pi * (x + y))) * (1 + 0.5 * tf.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * tf.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * tf.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 5) * np.pi * (y - 3 * x)))
    elif mesh_num == 6:
        a_eps = lambda x, y: (1 + 0.5 * tf.cos(2 * np.pi * (x + y))) * (1 + 0.5 * tf.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * tf.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * tf.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 5) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * tf.cos((2 ** 6) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 6) * np.pi * (y - 3 * x)))
    elif mesh_num == 7:
        a_eps = lambda x, y: (1 + 0.5 * tf.cos(2 * np.pi * (x + y))) * (1 + 0.5 * tf.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * tf.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * tf.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 5) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * tf.cos((2 ** 6) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 6) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * tf.cos((2 ** 7) * np.pi * (x + y))) * (1 + 0.5 * tf.sin((2 ** 7) * np.pi * (y - 3 * x)))
    return a_eps


# 例五
def true_solution2E5(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E5(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E5(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E5(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+tf.sin(2*np.pi*x/e1))/(1.1+tf.sin(2*np.pi*y/e1)) +
                              (1.1+tf.sin(2*np.pi*y/e2))/(1.1+tf.cos(2*np.pi*x/e2)) +
                              (1.1+tf.cos(2*np.pi*x/e3))/(1.1+tf.sin(2*np.pi*y/e3)) +
                              (1.1+tf.sin(2*np.pi*y/e4))/(1.1+tf.cos(2*np.pi*x/e4)) +
                              (1.1+tf.cos(2*np.pi*x/e5))/(1.1+tf.sin(2*np.pi*y/e5)) +
                              tf.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例六
def true_solution2E6(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = matData2pLaplace.loadMatlabIdata(file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E6(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*tf.ones_like(x)
    return f_side


def boundary2E6(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E6(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+tf.sin(2*np.pi*x/e1))/(1.1+tf.sin(2*np.pi*y/e1)) +
                              (1.1+tf.sin(2*np.pi*y/e2))/(1.1+tf.cos(2*np.pi*x/e2)) +
                              (1.1+tf.cos(2*np.pi*x/e3))/(1.1+tf.sin(2*np.pi*y/e3)) +
                              (1.1+tf.sin(2*np.pi*y/e4))/(1.1+tf.cos(2*np.pi*x/e4)) +
                              (1.1+tf.cos(2*np.pi*x/e5))/(1.1+tf.sin(2*np.pi*y/e5)) +
                              tf.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例七
def true_solution2E7(input_dim=None, output_dim=None, eps=0.1):
    utrue = lambda x, y: 0.5*tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.025*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)
    return utrue


def force_side2E7(input_dim=None, output_dim=None):
    f_side = lambda x, y: 5*((np.pi)**2)*(0.5*tf.sin(np.pi*x)*tf.cos(np.pi*y)+0.25*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y))*\
                          (0.25*tf.cos(5*np.pi*x)*tf.sin(10*np.pi*y)+0.5*tf.cos(15*np.pi*x)*tf.sin(20*np.pi*y))+ \
                          5*((np.pi)**2)*(0.5*tf.cos(np.pi*x)*tf.sin(np.pi*y)+0.25*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y))*\
                          (0.125*tf.sin(5*np.pi*x)*tf.cos(10*np.pi*y)+0.125*3*tf.sin(15*np.pi*x)*tf.cos(20*np.pi*y))+\
                          ((np.pi)**2)*(tf.sin(np.pi*x)*tf.sin(np.pi*y)+5*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y))*\
                          (0.125*tf.cos(5*np.pi*x)*tf.cos(10*np.pi*y)+0.125*tf.cos(15*np.pi*x)*tf.cos(20*np.pi*y)+0.5)

    return f_side


def boundary2E7(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0, eps=0.1):
    ux_left = lambda x, y: tf.zeros_like(x)
    ux_right = lambda x, y: tf.zeros_like(x)
    uy_bottom = lambda x, y: tf.zeros_like(x)
    uy_top = lambda x, y: tf.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E7(input_dim=None, output_dim=None, eps=0.1):
    a_eps = lambda x, y: 0.5 + 0.125*tf.cos(5*np.pi*x)*tf.cos(10*np.pi*y) + 0.125*tf.cos(15*np.pi*x)*tf.cos(20*np.pi*y)
    return a_eps


def get_infos2pLaplace_2D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale2D_1':
        f = force_side2E1(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E1/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E1(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E1(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E1(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_2':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E2(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E2/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E2(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E2(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E2(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_3':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E3(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E3/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E3(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E3(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E3(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_4':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E4(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E4/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E4(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E4(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E4(input_dim, out_dim, mesh_num=mesh_number)
    elif equa_name == 'multi_scale2D_5':
        intervalL = 0
        intervalR = 1.0
        f = force_side2E5(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E5/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E5(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E5(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E5(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_6':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E6(input_dim, out_dim)  # f是一个向量
        u_true_filepath = 'dataMat2pLaplace/E6/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E6(input_dim, out_dim, q=mesh_number, file_name=u_true_filepath)
        u_left, u_right, u_bottom, u_top = boundary2E6(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E6(input_dim, out_dim)

    return u_true, f, A_eps, u_left, u_right, u_bottom, u_top


def get_infos2pLaplace_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale3D_2':
        fside = lambda x, y, z: 3.0*((np.pi)**2)*(1.0+tf.cos(np.pi*x)*tf.cos(3*np.pi*y)*tf.cos(5*np.pi*z))*\
                                (tf.sin(np.pi*x) * tf.sin(np.pi*y) * tf.sin(np.pi*z))+\
        ((np.pi)**2)*(tf.sin(np.pi*x)*tf.cos(3*np.pi*y)*tf.cos(5*np.pi*z)*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z))+\
        3.0*((np.pi)** 2)*(tf.cos(np.pi*x)*tf.sin(3*np.pi*y)*tf.cos(5*np.pi*z)*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z))+ \
        5.0*((np.pi)**2)*(tf.cos(np.pi*x)*tf.cos(3*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(np.pi * x)*tf.sin(np.pi*y)*tf.cos(np.pi*z))
        u_true = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0 + tf.cos(np.pi * x) * tf.cos(3 * np.pi * y) * tf.cos(5 * np.pi * z)
        u_00 = lambda x, y, z: tf.sin(np.pi * intervalL) * tf.sin(np.pi * y) * tf.sin(np.pi * z)
        u_01 = lambda x, y, z: tf.sin(np.pi * intervalR) * tf.sin(np.pi * y) * tf.sin(np.pi * z)
        u_10 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * z)
        u_11 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * z)
        u_20 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_3':
        fside = lambda x, y, z: (63/4)*((np.pi)**2)*(1.0+tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z))*\
                                (tf.sin(np.pi*x) * tf.sin(5*np.pi*y) * tf.sin(10*np.pi*z))+\
        0.125*((np.pi)**2)*tf.sin(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(10*np.pi*z)+\
        (25/4)*((np.pi)** 2)*tf.cos(np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.sin(np.pi*x)*tf.cos(5*np.pi*y)*tf.sin(10*np.pi*z)+ \
        25.0*((np.pi)**2)*tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(np.pi * x)*tf.sin(5*np.pi*y)*tf.cos(10*np.pi*z)
        u_true = lambda x, y, z: 0.5*tf.sin(np.pi * x) * tf.sin(5*np.pi * y) * tf.sin(10*np.pi * z)
        A_eps = lambda x, y, z: 0.25*(1.0 + tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z))
        u_00 = lambda x, y, z: 0.5*tf.sin(np.pi * intervalL) * tf.sin(5*np.pi * y) * tf.sin(np.pi * z)
        u_01 = lambda x, y, z: 0.5*tf.sin(np.pi * intervalR) * tf.sin(5*np.pi * y) * tf.sin(np.pi * z)
        u_10 = lambda x, y, z: 0.5*tf.sin(np.pi * x) * tf.sin(5*np.pi * intervalL) * tf.sin(np.pi * z)
        u_11 = lambda x, y, z: 0.5*tf.sin(np.pi * x) * tf.sin(5*np.pi * intervalR) * tf.sin(np.pi * z)
        u_20 = lambda x, y, z: 0.5*tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10*np.pi * intervalL)
        u_21 = lambda x, y, z: 0.5*tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10*np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_4':
        fside = lambda x, y, z: tf.ones_like(x)
        u_true = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (2.0 + tf.cos(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(40 * np.pi * z))

        u_00 = lambda x, y, z: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*z)
        u_01 = lambda x, y, z: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*z)

        u_10 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*intervalL)*tf.sin(40*np.pi*z)
        u_11 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*intervalR)*tf.sin(40*np.pi*z)

        u_20 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_5':
        fside = lambda x, y, z: tf.ones_like(x)
        u_true = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + tf.cos(10.0*np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z))
        u_00 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_01 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_10 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_11 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_20 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_21 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + \
                                 0.05 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_6':
        fside = lambda x, y, z: tf.ones_like(x)
        u_true = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        A_eps = lambda x, y, z: 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+tf.cos(20.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z))
        u_00 = lambda x, y, z: tf.sin(np.pi * intervalL) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            20 * np.pi * intervalL) * tf.sin(20 * np.pi * y) * tf.sin(20 * np.pi * z)
        u_01 = lambda x, y, z: tf.sin(np.pi * intervalR) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            20 * np.pi * intervalR) * tf.sin(20 * np.pi * y) * tf.sin(20 * np.pi * z)
        u_10 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            20 * np.pi * x) * tf.sin(20 * np.pi * intervalL) * tf.sin(20 * np.pi * z)
        u_11 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            20 * np.pi * x) * tf.sin(20 * np.pi * intervalR) * tf.sin(20 * np.pi * z)
        u_20 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalL) + 0.05 * tf.sin(
            20 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(20 * np.pi * intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR) + 0.05 * tf.sin(
            20 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(20 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_7':
        fside = lambda x, y, z: tf.ones_like(x)
        u_true = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(25.0*np.pi*y)*tf.sin(50.0*np.pi*z)
        A_eps = lambda x, y, z: 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+tf.cos(10.0*np.pi*x)*tf.cos(25.0*np.pi*y)*tf.cos(50.0*np.pi*z))
        u_00 = lambda x, y, z: tf.sin(np.pi * intervalL) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            10 * np.pi * intervalL) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_01 = lambda x, y, z: tf.sin(np.pi * intervalR) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            10 * np.pi * intervalR) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_10 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            10 * np.pi * x) * tf.sin(20 * np.pi * intervalL) * tf.sin(10 * np.pi * z)
        u_11 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * z) + 0.05 * tf.sin(
            10 * np.pi * x) * tf.sin(20 * np.pi * intervalR) * tf.sin(10 * np.pi * z)
        u_20 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalL) + 0.05 * tf.sin(
            10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR) + 0.05 * tf.sin(
            10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21


def get_force2pLaplace3D(x=None, y=None, z=None, equa_name='multi_scale3D_5'):
    if equa_name == 'multi_scale3D_4':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*z)
        Aeps = 0.5 * (2.0 + tf.cos(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(40 * np.pi * z))

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(40*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+0.05*20*np.pi*tf.sin(10*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(40*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+0.05*40*np.pi*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(40*np.pi*z)

        uxx = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) - \
              0.05 * 100 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(40 * np.pi * z)
        uyy = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) - \
              0.05 * 400 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(40 * np.pi * z)
        uzz = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) - \
              0.05 * 1600 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(40 * np.pi * z)

        Aepsx = -0.5 * 10 * np.pi * tf.sin(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(40 * np.pi * z)
        Aepsy = -0.5 * 20 * np.pi * tf.cos(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.cos(40 * np.pi * z)
        Aepsz = -0.5 * 40 * np.pi * tf.cos(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.sin(40 * np.pi * z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz))
        return fside
    elif equa_name == 'multi_scale3D_5':
        u_true = tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.05 * tf.sin(10 * np.pi * x) * tf.sin(
            10 * np.pi * y) * tf.sin(10 * np.pi * z)
        Aeps = 0.5 * (1.0 + tf.cos(10.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z))
        ux = np.pi * tf.cos(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) + 0.5 * np.pi * tf.cos(
            10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        uy = np.pi * tf.sin(np.pi * x) * tf.cos(np.pi * y) * tf.sin(np.pi * z) + 0.5 * np.pi * tf.sin(
            10 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        uz = np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.cos(np.pi * z) + 0.5 * np.pi * tf.sin(
            10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(10 * np.pi * z)

        uxx = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(
            np.pi * z) - 5.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        uyy = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(
            np.pi * z) - 5.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)
        uzz = -1.0 * np.pi * np.pi * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(
            np.pi * z) - 5.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z)

        Aepsx = -0.5 * 10.0 * np.pi * tf.sin(10.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z)
        Aepsy = -0.5 * 10.0 * np.pi * tf.cos(10.0 * np.pi * x) * tf.sin(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z)
        Aepsz = -0.5 * 10.0 * np.pi * tf.cos(10.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.sin(10.0 * np.pi * z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz))
        return fside
    elif equa_name == 'multi_scale3D_6':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        Aeps = 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+tf.cos(20.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z))

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+1.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+1.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+1.0*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z) - 5.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z) - 5.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z)
        Aepsz = -0.25*np.pi*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z) - 5.0*np.pi*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz))
        return fside
    elif equa_name == 'multi_scale3D_7':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(50*np.pi*z)
        Aeps = 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+tf.cos(10.0*np.pi*x)*tf.cos(25.0*np.pi*y)*tf.cos(50.0*np.pi*z))

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(50*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+0.05*25*np.pi*tf.sin(10*np.pi*x)*tf.cos(25*np.pi*y)*tf.sin(50*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+0.05*50*np.pi*tf.sin(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.cos(50*np.pi*z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-0.05*100*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(50*np.pi*z)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-0.05*625*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(50*np.pi*z)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-0.05*2500*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(50*np.pi*z)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z) - 0.25*10*np.pi*tf.sin(10*np.pi*x)*tf.cos(25*np.pi*y)*tf.cos(50*np.pi*z)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z) - 0.25*25*np.pi*tf.cos(10*np.pi*x)*tf.sin(25*np.pi*y)*tf.cos(50*np.pi*z)
        Aepsz = -0.25*np.pi*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z) - 0.25*50*np.pi*tf.cos(10*np.pi*x)*tf.cos(25*np.pi*y)*tf.sin(50*np.pi*z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz))
        return fside


def get_infos2pLaplace_4D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale4D_2':
        fside = lambda x, y, z, s: tf.ones_like(s)
        u_true = lambda x, y, z, s: tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 0.25*(1.0+tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z)*tf.cos(5.0*np.pi*s))
        u_00 = lambda x, y, z, s: tf.sin(5*np.pi * intervalL) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(5*np.pi * intervalR) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_10 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalL) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalR) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_20 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalL) * tf.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalR) * tf.sin(5*np.pi * s)
        u_30 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31
    elif equa_name == 'multi_scale4D_3':
        fside = lambda x, y, z, s: tf.ones_like(s)
        u_true = lambda x, y, z, s: tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 0.25*(1.0+tf.cos(5.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z)*tf.cos(5.0*np.pi*s))
        u_00 = lambda x, y, z, s: tf.sin(5*np.pi * intervalL) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(5*np.pi * intervalR) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_10 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalL) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalR) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_20 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalL) * tf.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalR) * tf.sin(5*np.pi * s)
        u_30 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31
    elif equa_name == 'multi_scale4D_4':
        fside = lambda x, y, z, s: tf.ones_like(s)
        u_true = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s) + \
                                    0.25*tf.sin(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 1.0
        u_00 = lambda x, y, z, s: tf.sin(5*np.pi * intervalL) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(5*np.pi * intervalR) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_10 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalL) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalR) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_20 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalL) * tf.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalR) * tf.sin(5*np.pi * s)
        u_30 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31
    elif equa_name == 'multi_scale4D_5':
        fside = lambda x, y, z, s: tf.ones_like(s)
        u_true = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                    0.05*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)*tf.cos(np.pi*s)+ \
                                    tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z)*tf.cos(5.0*np.pi*s))

        u_00 = lambda x, y, z, s: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * intervalL) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * intervalR) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)

        u_10 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalL) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalR) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)

        u_20 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalL) * tf.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalR) * tf.sin(5*np.pi * s)

        u_30 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalL)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5*np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)+\
                                  tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31
    elif equa_name == 'multi_scale4D_6':
        fside = lambda x, y, z, s: tf.ones_like(s)
        u_true = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
            0.05*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)
        A_eps = lambda x, y, z, s:  0.25*(1.0 + tf.cos(10.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z)*tf.cos(10.0*np.pi*s))

        u_00 = lambda x, y, z, s: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * intervalL) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * intervalR) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)

        u_10 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalL) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * intervalR) * tf.sin(10*np.pi * z) * tf.sin(5*np.pi * s)

        u_20 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalL) * tf.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10*np.pi * y) * tf.sin(10*np.pi * intervalR) * tf.sin(5*np.pi * s)

        u_30 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalL)+\
                                  tf.sin(5*np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5*np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)+\
                                  tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31


def get_force2pLaplace_4D(x=None, y=None, z=None, s=None, equa_name=None):
    if equa_name == 'multi_scale4D_2':
        A = 0.25*(1.0 + tf.cos(5.0*np.pi * x)*tf.cos(10.0*np.pi*y) * tf.cos(10.0 *np.pi*z) * tf.cos(5.0*np.pi * s))
        Ax = -0.25 *5.0*np.pi*tf.sin(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.cos(5.0*np.pi*s)
        Ay = -0.25*10.0*np.pi*tf.cos(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.cos(5.0* np.pi * s)
        Az = -0.25*10.0*np.pi*tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.sin(10.0*np.pi*z) * tf.cos(5.0* np.pi * s)
        As = -0.25 *5.0*np.pi*tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.sin(5.0* np.pi * s)
        U = tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Ux = 5*np.pi * tf.cos(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s)
        Uy = 10*np.pi* tf.sin(5 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s)
        Uz = 10*np.pi* tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(10 * np.pi * z) * tf.sin(5 * np.pi * s)
        Us = 5*np.pi * tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.cos(5 * np.pi * s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) + 250.0*A*np.pi*np.pi*U
        return fside
    elif equa_name == 'multi_scale4D_4':
        A = 1.0
        Ax = 1.0
        Ay = 1.0
        Az = 1.0
        As = 1.0
        U = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s) + \
            0.25*tf.sin(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s) + \
            0.25*5.0*np.pi*tf.cos(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Uy = tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s) + \
            0.25*5.0*np.pi*tf.sin(5.0*np.pi*x)*tf.cos(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s) + \
            0.25*5.0*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.cos(5.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s) + \
            0.25*5.0*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.cos(5.0*np.pi*s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) + \
                4.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s) + \
                0.25*20*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(5.0*np.pi*y)*tf.sin(5.0*np.pi*z)*tf.cos(5.0*np.pi*s)
        return fside
    elif equa_name == 'multi_scale4D_5':
        A = 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)*tf.cos(np.pi*s)+\
                                    tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z)*tf.cos(5.0*np.pi*s))

        Ax = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)*tf.cos(np.pi*s)-\
             0.25 *5.0*np.pi*tf.sin(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.cos(5.0*np.pi*s)

        Ay = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.cos(np.pi*s)-\
             0.25*10.0*np.pi*tf.cos(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.cos(5.0* np.pi * s)

        Az = -0.25*np.pi*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)-\
             0.25*10.0*np.pi*tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.sin(10.0*np.pi*z) * tf.cos(5.0* np.pi * s)

        As = -0.25*np.pi*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)-\
             0.25 *5.0*np.pi*tf.cos(5.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z) * tf.sin(5.0* np.pi * s)

        U = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
            0.05*tf.sin(10.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)

        Ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*5.0*np.pi*tf.cos(5 * np.pi * x)*tf.sin(10 * np.pi*y)*tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s)

        Uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*10.0*np.pi* tf.sin(5 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s)

        Uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*10.0*np.pi* tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(10 * np.pi * z) * tf.sin(5 * np.pi * s)

        Us = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)+\
             0.05*5.0*np.pi * tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.cos(5 * np.pi * s)

        Uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*25.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*100.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*100.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)
        Uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*25.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)*tf.sin(5.0*np.pi*s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) -1.0*A*(Uxx+Uyy+Uzz+Uss)
        return fside
    elif equa_name == 'multi_scale4D_6':
        A = 0.25*(1.0 + tf.cos(10.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z)*tf.cos(10.0*np.pi*s))

        Ax = -0.25*10.0*np.pi*tf.sin(10.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z) * tf.cos(10.0*np.pi*s)

        Ay = -0.25*20.0*np.pi*tf.cos(10.0*np.pi*x)*tf.sin(20.0*np.pi*y)*tf.cos(20.0*np.pi*z) * tf.cos(10.0* np.pi * s)

        Az = -0.25*20.0*np.pi*tf.cos(10.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.sin(20.0*np.pi*z) * tf.cos(10.0* np.pi * s)

        As = -0.25*10.0*np.pi*tf.cos(10.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z) * tf.sin(10.0* np.pi * s)

        U = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
            0.05*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)

        Ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*5.0*np.pi*tf.cos(5 * np.pi * x)*tf.sin(10 * np.pi*y)*tf.sin(15 * np.pi * z) * tf.sin(20*np.pi * s)

        Uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*10.0*np.pi* tf.sin(5 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(15 * np.pi*z)*tf.sin(20*np.pi*s)

        Uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)+\
             0.05*15.0*np.pi* tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(15 * np.pi* z)*tf.sin(20*np.pi*s)

        Us = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)+\
             0.05*20.0*np.pi * tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(15 * np.pi*z)*tf.cos(20*np.pi*s)

        Uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*25.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)
        Uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*100.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)
        Uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*125.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)
        Uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)+\
                -0.05*400.0*np.pi*np.pi*tf.sin(5.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.sin(15.0*np.pi*z)*tf.sin(20.0*np.pi*s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) -1.0*A*(Uxx+Uyy+Uzz+Uss)
        return fside


def get_infos2pLaplace_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale5D_1':
        fside = lambda x, y, z, s, t: 5.0 * ((np.pi) ** 2) * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z)* tf.sin(np.pi * s)* tf.sin(np.pi * t)
        u_true = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z)* tf.sin(np.pi * s)* tf.sin(np.pi * t)
        A_eps = lambda x, y, z, s, t: 1.0
        u_00 = lambda x, y, z, s, t: tf.sin(np.pi * intervalL) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi * intervalR) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * s) * tf.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * s) * tf.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    elif equa_name == 'multi_scale5D_2':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * t)
        fside = lambda x, y, z, s, t: 5.0*((np.pi)**2)*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                      *(1.0+tf.cos(np.pi*x)*tf.cos(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.cos(np.pi*t))\
                                      +((np.pi)**2)*tf.sin(np.pi*x)*tf.cos(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.cos(np.pi * x) * tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                      +(2.0*(np.pi)**2)*tf.cos(np.pi*x)*tf.sin(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                      +(3.0*(np.pi)** 2)*tf.cos(np.pi*x)*tf.cos(2*np.pi*y)*tf.sin(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                      +(2.0*(np.pi)**2)*tf.cos(np.pi*x)*tf.cos(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.sin(2*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t)\
                                      +((np.pi)**2)*tf.cos(np.pi*x)*tf.cos(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.sin(np.pi*t) \
                                      *tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t)
        A_eps = lambda x, y, z, s, t: 1.0 + tf.cos(np.pi*x)*tf.cos(2*np.pi*y)*tf.cos(3*np.pi*z)*tf.cos(2*np.pi*s)*tf.cos(np.pi*t)
        u_00 = lambda x, y, z, s, t: tf.sin(np.pi * intervalL) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi * intervalR) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalL) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR) * tf.sin(
            np.pi * s) * tf.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * intervalL) * tf.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * intervalR) * tf.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(
            np.pi * s) * tf.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    elif equa_name == 'multi_scale5D_3':
        u_true = lambda x, y, z, s, t: 0.5*tf.sin(np.pi * x) * tf.sin(5*np.pi * y) * tf.sin(10*np.pi * z)* tf.sin(5*np.pi * s)* tf.sin(np.pi * t)
        fside = lambda x, y, z, s, t: 19*((np.pi)**2)*tf.sin(np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(np.pi*t)\
                                      *(1.0+tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(np.pi*t))\
                                      +0.125*((np.pi)**2)*tf.sin(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.cos(np.pi * x) * tf.sin(5*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(np.pi*t)\
                                      +6.25*((np.pi)**2)*tf.cos(np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.cos(5*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(np.pi*t)\
                                      +25*((np.pi)** 2)*tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.sin(5*np.pi*y)*tf.cos(10*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(np.pi*t)\
                                      +6.25*((np.pi)**2)*tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.sin(10*np.pi*s)*tf.cos(np.pi*t)\
                                      *tf.sin(np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(10*np.pi*z)*tf.cos(5*np.pi*s)*tf.sin(np.pi*t)\
                                      +0.125*((np.pi)**2)*tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.sin(np.pi*t) \
                                      *tf.sin(np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(5*np.pi*s)*tf.cos(np.pi*t)
        A_eps = lambda x, y, z, s, t: 0.25*(1.0 + tf.cos(np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(np.pi*t))
        u_00 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * intervalL) * tf.sin(5 * np.pi * y) * \
                                     tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * intervalR) * tf.sin(5 * np.pi * y) * \
                                     tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalL) * tf.sin(10 * np.pi * z) \
                                     * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalR) * tf.sin(10 * np.pi * z) \
                                     * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * intervalL) \
                                     * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * intervalR) \
                                     * tf.sin(5 * np.pi * s) * tf.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z) * \
                                     tf.sin(5 * np.pi * intervalL) * tf.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z) * \
                                     tf.sin(5 * np.pi * intervalR) * tf.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z) * \
                                     tf.sin(5 * np.pi * s) * tf.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z) * \
                                     tf.sin(np.pi * s) * tf.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    elif equa_name == 'multi_scale5D_4':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                       + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*\
                                       tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: tf.ones_like(x)

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalL)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalR)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * intervalL)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    elif equa_name == 'multi_scale5D_5':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)
        A_eps = lambda x, y, z, s, t: tf.ones_like(x)

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                     0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    elif equa_name == 'multi_scale5D_6':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)
        A_eps = lambda x, y, z, s, t: tf.ones_like(x)

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                     0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+ \
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_7':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                       + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*\
                                       tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: 0.5 + 0.5*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalL)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalR)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * intervalL)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_8':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.5*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) +\
             0.05*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             0.01*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: 1.0

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalL)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalR)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * intervalL)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_9':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.1*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) +\
             0.01*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: 1.0

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalL)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalR)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * intervalL)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*\
                                     tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41


def get_forceSide2pLaplace5D(x=None, y=None, z=None, s=None, t=None, equa_name='multi_scale5D_5'):
    if equa_name == 'multi_scale5D_4':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        Aeps = 1.0

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.sin(10*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.cos(10*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uzz =-1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        Aepsx = 1.0
        Aepsy = 1.0
        Aepsz = 1.0
        Aepss = 1.0
        Aepst = 1.0

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside
    elif equa_name == 'multi_scale5D_5':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        Aeps = 1.0

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             2.5*np.pi*tf.cos(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             2.5*np.pi*tf.sin(5*np.pi*x)*tf.cos(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             2.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.cos(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             2.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.cos(5*np.pi*s)*tf.sin(5*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             2.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.cos(5*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             12.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             12.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uzz =-1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             12.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             12.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             12.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        Aepsx = 1.0
        Aepsy = 1.0
        Aepsz = 1.0
        Aepss = 1.0
        Aepst = 1.0

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside
    elif equa_name == 'multi_scale5D_6':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        Aeps = 1.0

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.cos(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(5*np.pi*x)*tf.cos(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.cos(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.cos(5*np.pi*s)*tf.sin(5*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             0.5*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.cos(5*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             2.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             2.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uzz =-1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             2.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             2.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             2.5*np.pi*np.pi*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)

        Aepsx = 1.0
        Aepsy = 1.0
        Aepsz = 1.0
        Aepss = 1.0
        Aepst = 1.0

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside
    elif equa_name == 'multi_scale5D_7':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        Aeps = 0.5 + 0.5*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.sin(10*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.cos(10*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uzz =-1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)

        Aepsx = -0.5*10*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)
        Aepsy = -0.5*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)
        Aepsz = -0.5*10*np.pi*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)
        Aepss = -0.5*10*np.pi*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.sin(10*np.pi*s)*tf.cos(10*np.pi*t)
        Aepst = -0.5*10*np.pi*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.sin(10*np.pi*t)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside
    elif equa_name == 'multi_scale5D_8':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.5*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) +\
             0.05*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             0.01*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        Aeps = 1.0

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*25.0*np.pi*tf.cos(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+\
             0.05*50*np.pi*tf.cos(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             np.pi*tf.cos(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*25*np.pi*tf.sin(25*np.pi*x)*tf.cos(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+ \
             0.05*50*np.pi*tf.sin(50*np.pi*x)*tf.cos(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             np.pi**tf.sin(100*np.pi*x)*tf.cos(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.cos(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+\
             0.05*50*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.cos(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.cos(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             0.5*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.cos(25*np.pi*s)*tf.sin(25*np.pi*t) + \
             0.05*50*np.pitf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.cos(50*np.pi*s)*tf.sin(50*np.pi*t) + \
             np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.cos(100*np.pi*s)*tf.sin(100*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             0.5*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.cos(25*np.pi*t)+ \
             0.05*50*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.cos(50*np.pi*t) + \
             np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.cos(100*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             312.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             125*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) - \
             100*np.pi*np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             312.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             125*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) - \
             100*np.pi*np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             312.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             125*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) - \
             100*np.pi*np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             312.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             125*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) - \
             100*np.pi*np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             312.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             125*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t) - \
             100*np.pi*np.pi*tf.sin(100*np.pi*x)*tf.sin(100*np.pi*y)*tf.sin(100*np.pi*z)*tf.sin(100*np.pi*s)*tf.sin(100*np.pi*t)

        Aepsx = 1.0
        Aepsy = 1.0
        Aepsz = 1.0
        Aepss = 1.0
        Aepst = 1.0

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside
    elif equa_name == 'multi_scale5D_9':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.1*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) +\
             0.01*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        Aeps = 1.0

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.1*25.0*np.pi*tf.cos(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+\
             0.01*50*np.pi*tf.cos(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.1*25*np.pi*tf.sin(25*np.pi*x)*tf.cos(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+ \
             0.01*50*np.pi*tf.sin(50*np.pi*x)*tf.cos(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             0.1*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.cos(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t)+\
             0.01*50*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.cos(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             0.1*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.cos(25*np.pi*s)*tf.sin(25*np.pi*t) + \
             0.01*50*np.pitf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.cos(50*np.pi*s)*tf.sin(50*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             0.1*25*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.cos(25*np.pi*t)+ \
             0.01*50*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.cos(50*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             62.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             62.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             62.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             62.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) -\
             62.5*np.pi*np.pi*tf.sin(25*np.pi*x)*tf.sin(25*np.pi*y)*tf.sin(25*np.pi*z)*tf.sin(25*np.pi*s)*tf.sin(25*np.pi*t) -\
             25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)*tf.sin(50*np.pi*s)*tf.sin(50*np.pi*t)

        Aepsx = 1.0
        Aepsy = 1.0
        Aepsz = 1.0
        Aepss = 1.0
        Aepst = 1.0

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside


def get_infos2pLaplace_10D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale10D_1':
        fside = lambda x, y, z: 10.0 * ((np.pi)**2) * (tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z))
        u_true = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0
        return u_true, fside, A_eps