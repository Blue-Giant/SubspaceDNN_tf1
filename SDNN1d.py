"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import DNN_data
import time
import DNN_base
import DNN_tools
import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName2normal=None, actName2scale=None):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    DNN_tools.log_string('The order to p-laplace: %s\n' % (R_dic['order2laplace']), log_fileout)
    DNN_tools.log_string('The epsilon to p-laplace: %s\n' % (R_dic['epsilon']), log_fileout)

    DNN_tools.log_string('Network model of solving normal-part: %s\n' % str(R_dic['model2normal']), log_fileout)
    DNN_tools.log_string('Network model of solving scale-part: %s\n' % str(R_dic['model2scale']), log_fileout)
    DNN_tools.log_string('Activate function for normal-part network: %s\n' % str(actName2normal), log_fileout)
    DNN_tools.log_string('Activate function for scale-part network: %s\n' % str(actName2scale), log_fileout)
    DNN_tools.log_string('hidden layer to normal:%s\n' % str(R_dic['hidden2normal']), log_fileout)
    DNN_tools.log_string('hidden layer to scale :%s\n' % str(R_dic['hidden2scale']), log_fileout)
    DNN_tools.log_string('The frequency to scale-part network: %s\n' % (R_dic['freqs']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)

    if (R_dic['train_opt']) == 0:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_opt']) == 1:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd + loss_U2U', log_fileout)
    elif (R_dic['train_opt']) == 2:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)
    elif (R_dic['train_opt']) == 4:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_U2U', log_fileout)

    if R_dic['variational_loss'] == 1 or R_dic['variational_loss'] == 2:
        DNN_tools.log_string('Loss function: variational loss ' + str(R_dic['variational_loss']) +'\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    if R_dic['variational_loss'] == 1:
        if R_dic['wavelet'] == 1:
            DNN_tools.log_string('Option of loss for coarse and fine is: L2 wavelet. \n', log_fileout)
        elif R_dic['wavelet'] == 2:
            DNN_tools.log_string('Option of loss for coarse and fine is: Energy minimization. \n', log_fileout)
        else:
            DNN_tools.log_string('Option of loss for coarse and fine is: L2 wavelet + Energy minimization. \n',
                                 log_fileout)

    if R_dic['variational_loss'] == 2:
        if R_dic['wavelet'] == 1:
            DNN_tools.log_string('Option of loss for coarse and fine is: L2 wavelet. \n', log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                             log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout_NN = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout_NN, actName2normal=R['act_name2NN1'], actName2scale=R['act_name2NN2'])

    # laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    bd_penalty_init = R['init_boundary_penalty']         # Regularization parameter for boundary conditions
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    init_penalty2powU = R['balance2solus']
    hidden2normal = R['hidden2normal']
    hidden2scale = R['hidden2scale']
    wb_regular = R['regular_weight_biases']         # Regularization parameter for weights and biases

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']
    alpha = R['contrib2scale']
    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']

    region_l = 0.0
    region_r = 1.0
    if R['PDE_type'] == 'general_laplace':
        # -laplace u = f
        region_l = 0.0
        region_r = 1.0
        f, u_true, u_left, u_right = General_Laplace.get_infos2Laplace_1D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'p_laplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2laplace']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        u_true, f, A_eps, u_left, u_right = MS_LaplaceEqs.get_infos2pLaplace_1D(
            in_dim=input_dim, out_dim=out_dim, xleft=region_l, xright=region_r, index2p=p_index, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + K(x)u_eps(x) =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2laplace']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        A_eps, kappa, u_true, u_left, u_right, f = MS_BoltzmannEqs.get_infos2Boltzmann_1D(
            in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, index2p=p_index, eps=epsilon,
            eqs_name=R['equa_name'])

    flag_normal = 'WB_NN2normal'
    flag_scale = 'WB_NN2scale'
    # Weights, Biases = PDE_DNN_base.Initial_DNN2different_hidden(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = laplace_DNN1d_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag1)
    # Weights, Biases = laplace_DNN1d_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag1)
    if R['model2normal'] == 'PDE_DNN_Cos_C_Sin_Base' or R['model2normal'] == 'DNN_adaptCosSin_Base':
        W2NN_Normal, B2NN_Normal = DNN_base.initialize_NN_random_normal2_CS(input_dim, out_dim, hidden2normal, flag_normal)
    else:
        W2NN_Normal, B2NN_Normal = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden2normal, flag_normal)

    if R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base' or R['model2scale'] == 'DNN_adaptCosSin_Base':
        W2NN_freqs, B2NN_freqs = DNN_base.initialize_NN_random_normal2_CS(input_dim, out_dim, hidden2scale, flag_scale)
    else:
        W2NN_freqs, B2NN_freqs = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden2scale, flag_scale)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 1 列
            X_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 1 列
            X_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            penalty2powU = tf.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if R['model2normal'] == 'PDE_DNN':
                U_NN_Normal = DNN_base.PDE_DNN(X_it, W2NN_Normal, B2NN_Normal, hidden2normal, activate_name=act_func1)
                ULeft_NN_Normal = DNN_base.PDE_DNN(X_left_bd, W2NN_Normal, B2NN_Normal, hidden2normal, activate_name=act_func1)
                URight_NN_Normal = DNN_base.PDE_DNN(X_right_bd, W2NN_Normal, B2NN_Normal, hidden2normal, activate_name=act_func1)
            elif R['model2normal'] == 'PDE_DNN_Cos_C_Sin_Base':
                freq = [1]
                U_NN_Normal = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_it, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)
                ULeft_NN_Normal = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_left_bd, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)
                URight_NN_Normal = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_right_bd, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)
            elif R['model2normal'] == 'DNN_adaptCosSin_Base':
                freq = [1]
                U_NN_Normal = DNN_base.DNN_adaptCosSin_Base(X_it, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)
                ULeft_NN_Normal = DNN_base.DNN_adaptCosSin_Base(X_left_bd, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)
                URight_NN_Normal = DNN_base.DNN_adaptCosSin_Base(X_right_bd, W2NN_Normal, B2NN_Normal, hidden2normal, freq, activate_name=act_func1)

            freqs = R['freqs']
            if R['model2scale'] == 'PDE_DNN_scale':
                U_NN_freqs = DNN_base.PDE_DNN_scale(X_it, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_freqs = DNN_base.PDE_DNN_scale(X_left_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_freqs = DNN_base.PDE_DNN_scale(X_right_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_adapt_scale':
                U_NN_freqs = DNN_base.PDE_DNN_adapt_scale(X_it, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_freqs = DNN_base.PDE_DNN_adapt_scale(X_left_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_freqs = DNN_base.PDE_DNN_adapt_scale(X_right_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_FourierBase':
                U_NN_freqs = DNN_base.PDE_DNN_FourierBase(X_it, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_freqs = DNN_base.PDE_DNN_FourierBase(X_left_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_freqs = DNN_base.PDE_DNN_FourierBase(X_right_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base':
                U_NN_freqs = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_it, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_freqs = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_left_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_freqs = DNN_base.PDE_DNN_Cos_C_Sin_Base(X_right_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'DNN_adaptCosSin_Base':
                U_NN_freqs = DNN_base.DNN_adaptCosSin_Base(X_it, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_freqs = DNN_base.DNN_adaptCosSin_Base(X_left_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_freqs = DNN_base.DNN_adaptCosSin_Base(X_right_bd, W2NN_freqs, B2NN_freqs, hidden2scale, freqs, activate_name=act_func2)

            U_NN = U_NN_Normal + alpha*U_NN_freqs

            # 变分形式的loss of interior，训练得到的 U_NN1 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            dU_NN_Normal = tf.gradients(U_NN_Normal, X_it)[0]    # * 行 1 列
            dU_NN_freqs = tf.gradients(U_NN_freqs, X_it)[0]      # * 行 1 列
            if R['variational_loss'] == 0:
                if R['PDE_type'] == 'general_laplace':
                    ddUNN_Normal = tf.gradients(dU_NN_Normal, X_it)[0]                        # * 行 1 列
                    ddUNN_freqs = tf.gradients(dU_NN_freqs, X_it)[0]                          # * 行 1 列
                    diff2ddUNN = ddUNN_Normal-ddUNN_freqs-tf.reshape(f(X_it), shape=[-1, 1])
                    loss_it_NN = tf.square(diff2ddUNN)
                elif R['PDE_type'] == 'p_laplace':
                    # a_eps = A_eps(X_it)                                                     # * 行 1 列
                    a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    dAdUNN_Normal = tf.gradients(a_eps*dU_NN_Normal, X_it)[0]                 # * 行 1 列
                    dAdUNN_freqs = tf.gradients(a_eps*dU_NN_freqs, X_it)[0]                   # * 行 1 列
                    diff2dAdUNN = -dAdUNN_Normal - dAdUNN_freqs - tf.reshape(f(X_it), shape=[-1, 1])
                    loss_it_NN = tf.square(diff2dAdUNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    dAdUNN_Normal = tf.gradients(a_eps*dU_NN_Normal, X_it)[0]                    # * 行 1 列
                    dAdUNN_freqs = tf.gradients(a_eps*dU_NN_freqs, X_it)[0]                      # * 行 1 列
                    diff2dAdUNN = -dAdUNN_Normal - dAdUNN_freqs + Kappa*U_NN - tf.reshape(f(X_it), shape=[-1, 1])
                    loss_it_NN = tf.square(diff2dAdUNN)

                Loss_it2NN = tf.reduce_mean(loss_it_NN)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0 Uc 和 Uf 是两个列向量 形状为(*,1)
                    # norm2UdU = tf.square(tf.multiply(U_NN_Normal, U_NN_freqs))
                    norm2UdU = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_freqs)), axis=-1)
                    # norm2UdU = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, U_NN_freqs)), axis=-1)
                    UNN_dot_UNN = tf.reduce_mean(tf.reshape(norm2UdU, shape=[-1, 1]))
                elif R['wavelet'] == 2:
                    # |a(x)*(grad Uc)*(grad Uf)|^2-->0 a(x) 是 (*,1)的；(grad Uc)*(grad Uf)是向量相乘(*,2)·(*,2)
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_freqs)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    # norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
                else:  # |Uc*Uf|^2-->0 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    U_dot_U = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_freqs)), axis=-1)
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_freqs)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
            elif R['variational_loss'] == 1:
                dUNN = tf.add(dU_NN_Normal, alpha*dU_NN_freqs)
                if R['PDE_type'] == 'general_laplace':
                    laplace_norm2NN = tf.reduce_sum(tf.square(dUNN), axis=-1)
                    loss_it_NN = (1.0 / 2) * tf.reshape(laplace_norm2NN, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                elif R['PDE_type'] == 'p_laplace':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    dUNN_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(dUNN), axis=-1), shape=[-1, 1]))
                    laplace_p_pow2NN = a_eps * tf.pow(dUNN_norm, p_index)
                    loss_it_NN = (1.0 / p_index) * laplace_p_pow2NN - \
                                 tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    dUNN_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(dUNN), axis=-1), shape=[-1, 1]))
                    divAdUNN = a_eps * tf.pow(dUNN_norm, p_index)
                    loss_it_NN = (1.0 / p_index) * (divAdUNN + Kappa*U_NN*U_NN) - \
                                 tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)

                Loss_it2NN = tf.reduce_mean(loss_it_NN)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0 Uc 和 Uf 是两个列向量 形状为(*,1)
                    # norm2UdU = tf.square(tf.multiply(U_NN_Normal, U_NN_freqs))
                    norm2UdU = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_freqs)), axis=-1)
                    # norm2UdU = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, U_NN_freqs)), axis=-1)
                    UNN_dot_UNN = tf.reduce_mean(tf.reshape(norm2UdU, shape=[-1, 1]))
                elif R['wavelet'] == 2:
                    # |a(x)*(grad Uc)*(grad Uf)|^2-->0 a(x) 是 (*,1)的；(grad Uc)*(grad Uf)是向量相乘(*,2)·(*,2)
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_freqs)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    # norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
                else:  # |Uc*Uf|^2-->0 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    U_dot_U = tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_freqs)), axis=-1)
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_freqs)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
            elif R['variational_loss'] == 2:
                dU_NN = tf.add(dU_NN_Normal, alpha*dU_NN_freqs)
                if R['PDE_type'] == 'general_laplace':
                    laplace_norm2NN = tf.reduce_sum(tf.square(dU_NN), axis=-1)
                    loss_it_NN = (1.0 / 2) * tf.reshape(laplace_norm2NN, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                elif R['PDE_type'] == 'p_laplace':
                    # a_eps = A_eps(X_it)                          # * 行 1 列
                    a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    laplace_p_pow2NN = tf.reduce_sum(a_eps*tf.pow(tf.abs(dU_NN), p_index), axis=-1)
                    loss_it_NN = (1.0 / p_index) * tf.reshape(laplace_p_pow2NN, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    divAdUNN = tf.reduce_sum(a_eps*tf.pow(tf.abs(dU_NN), p_index), axis=-1)
                    loss_it_NN = (1.0 / p_index) * (tf.reshape(divAdUNN, shape=[-1, 1]) + Kappa*U_NN*U_NN) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                Loss_it2NN = tf.reduce_mean(loss_it_NN)*(region_r-region_l)
                if R['wavelet'] == 1:
                    norm2UdU = tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_freqs))
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU, axis=0)
                else:
                    UNN_dot_UNN = tf.constant(0.0)

            Loss2UNN_dot_UNN = penalty2powU * UNN_dot_UNN

            U_left = tf.reshape(u_left(X_left_bd), shape=[-1, 1])
            U_right = tf.reshape(u_right(X_right_bd), shape=[-1, 1])
            loss_bd_Normal = tf.square(ULeft_NN_Normal - U_left) + tf.square(URight_NN_Normal - U_right)
            loss_bd_Freqs = tf.square(alpha*ULeft_NN_freqs) + tf.square(alpha*URight_NN_freqs)
            Loss_bd2NN = tf.reduce_mean(loss_bd_Normal) + tf.reduce_mean(loss_bd_Freqs)

            Loss_bd2NNs = bd_penalty * Loss_bd2NN

            if R['regular_weight_model'] == 'L1':
                regular_WB_Normal = DNN_base.regular_weights_biases_L1(W2NN_Normal, B2NN_Normal)    # 正则化权重和偏置 L1正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L1(W2NN_freqs, B2NN_freqs)  # 正则化权重和偏置 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB_Normal = DNN_base.regular_weights_biases_L2(W2NN_Normal, B2NN_Normal)    # 正则化权重和偏置 L2正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L2(W2NN_freqs, B2NN_freqs)  # 正则化权重和偏置 L2正则化
            else:
                regular_WB_Normal = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB_Scale = tf.constant(0.0)

            penalty_Weigth_Bias = wb_regular * (regular_WB_Normal + regular_WB_Scale)

            if R['train_opt'] == 3:
                Loss2NN = Loss_it2NN + Loss_bd2NNs + penalty_Weigth_Bias
            else:
                # Loss2NN = Loss_it2NN + Loss_bd2NNs + Loss2UNN_dot_UNN + penalty_Weigth_Bias
                Loss2scale = tf.square(alpha * tf.reduce_mean(U_NN_freqs))
                Loss2NN = Loss_it2NN + Loss2scale + Loss_bd2NNs + Loss2UNN_dot_UNN + penalty_Weigth_Bias

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['variational_loss'] == 0 or R['variational_loss'] == 1:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_opt'] == 2:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_opt'] == 3:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_opt'] == 4:
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op3, train_op4)
                else:
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            elif R['variational_loss'] == 2:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NN, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_opt'] == 2:
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2)
                else:
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            U_true = u_true(X_it)
            train_mse_NN = tf.reduce_mean(tf.square(U_true - U_NN))
            train_rel_NN = train_mse_NN / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    lossIt_all2NN, lossBD_all2NN, loss_all2NN, UDU_NN, train_mse_all2NN, train_rel_all2NN = [], [], [], [], [], []
    test_mse_all2NN, test_rel_all2NN = [], []
    test_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])
    saveData.save_testData_or_solus2mat(test_x_bach, dataName='testx', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            elif R['activate_penalty2bd_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = 5*bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 1 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 0.5 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 0.1 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 0.05 * bd_penalty_init
                else:
                    temp_penalty_bd = 0.02 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            if R['activate_powSolus_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_powU = init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_powU = 10* init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_powU = 50*init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_powU = 100*init_penalty2powU
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_powU = 200*init_penalty2powU
                else:
                    temp_penalty_powU = 500*init_penalty2powU
            elif R['activate_powSolus_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_powU = 5 * init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_powU = 1 * init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_powU = 0.5 * init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_powU = 0.1 * init_penalty2powU
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_powU = 0.05 * init_penalty2powU
                else:
                    temp_penalty_powU = 0.02 * init_penalty2powU
            else:
                temp_penalty_powU = init_penalty2powU

            p_WB = 0.0
            _, loss_it_nn, loss_bd_nn, loss_nn, udu_nn, train_mse_nn, train_rel_nn = sess.run(
                [train_Loss2NN, Loss_it2NN, Loss_bd2NN, Loss2NN, UNN_dot_UNN, train_mse_NN, train_rel_NN],
                feed_dict={X_it: x_it_batch, X_left_bd: xl_bd_batch, X_right_bd: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd, penalty2powU: temp_penalty_powU})
            lossIt_all2NN.append(loss_it_nn)
            lossBD_all2NN.append(loss_bd_nn)
            loss_all2NN.append(loss_nn)
            UDU_NN.append(udu_nn)
            train_mse_all2NN.append(train_mse_nn)
            train_rel_all2NN.append(train_rel_nn)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, p_WB, loss_it_nn, loss_bd_nn, loss_nn,
                    udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout_NN)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, utest_nn, unn_normal, unn_scale = sess.run(
                    [U_true, U_NN, U_NN_Normal, alpha*U_NN_freqs], feed_dict={X_it: test_x_bach, train_opt: train_option})
                test_mse2nn = np.mean(np.square(u_true2test - utest_nn))
                test_mse_all2NN.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all2NN.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout_NN)

        # -----------------------  save training results to mat files, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN, lossBD_all2NN, loss_all2NN, actName=act_func1,
                                             outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all2NN, train_rel_all2NN, actName=act_func1, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(lossIt_all2NN, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(lossBD_all2NN, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all2NN, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plotTrain_MSE_REL_1act_func(train_mse_all2NN, train_rel_all2NN, actName=act_func2, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_nn, dataName=act_func1, outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(unn_normal, dataName='normal', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(unn_scale, dataName='scale', outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all2NN, test_rel_all2NN, actName=act_func2, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all2NN, test_rel_all2NN, test_epoch, actName=act_func2, seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 3
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    # store_file = 'Laplace1D'
    # store_file = 'pLaplace1D'
    store_file = 'Boltzmann1D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # R['PDE_type'] = 'general_laplace'
    # R['equa_name'] = 'PDE1'
    # R['equa_name'] = 'PDE2'
    # R['equa_name'] = 'PDE3'
    # R['equa_name'] = 'PDE4'
    # R['equa_name'] = 'PDE5'
    # R['equa_name'] = 'PDE6'
    # R['equa_name'] = 'PDE7'

    R['PDE_type'] = 'p_laplace'
    R['equa_name'] = 'multi_scale'

    R['PDE_type'] = 'Possion_Boltzmann'
    # R['equa_name'] = 'Boltzmann1'
    R['equa_name'] = 'Boltzmann2'

    if R['PDE_type'] == 'general_laplace':
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
    elif R['PDE_type'] == 'p_laplace' or R['PDE_type'] == 'Possion_Boltzmann':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)              # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order

    R['input_dim'] = 1                         # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                        # 输出维数
    # R['variational_loss'] = 1                  # PDE变分
    R['variational_loss'] = 0                  # L2 loss
    # R['wavelet'] = 0                         # 0: L2 wavelet+energy    1: wavelet    2:energy
    R['wavelet'] = 1                         # 0: L2 wavelet+energy    1: wavelet    2:energy
    # R['wavelet'] = 2                           # 0: L2 wavelet+energy    1: wavelet    2:energy

    # ---------------------------- Setup of DNN -------------------------------
    R['batch_size2interior'] = 3000            # 内部训练数据的批大小
    R['batch_size2boundary'] = 500             # 边界训练数据大小

    R['weight_biases_model'] = 'general_model'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000     # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001   # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025  # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100                           # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['balance2solus'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['balance2solus'] = 10000.0
    else:
        R['balance2solus'] = 20.0
        # R['balance2solus'] = 15.0
        # R['balance2solus'] = 10.0

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器
    R['train_opt'] = 0
    # R['train_opt'] = 1
    # R['train_opt'] = 3
    # R['train_opt'] = 4

    R['model2normal'] = 'PDE_DNN'  # 使用的网络模型
    # R['model2normal'] = 'PDE_DNN_scale'
    # R['model2normal'] = 'PDE_DNN_adapt_scale'
    # R['model2normal'] = 'PDE_DNN_FourierBase'
    # R['model2normal'] = 'PDE_DNN_Cos_C_Sin_Base'
    # R['model2normal'] = 'DNN_adaptCosSin_Base'

    # R['model2scale'] = 'PDE_DNN'                         # 使用的网络模型
    # R['model2scale'] = 'PDE_DNN_BN'
    # R['model2scale'] = 'PDE_DNN_scale'
    # R['model2scale'] = 'PDE_DNN_adapt_scale'
    # R['model2scale'] = 'PDE_DNN_FourierBase'
    R['model2scale'] = 'PDE_DNN_Cos_C_Sin_Base'
    # R['model2scale'] = 'DNN_adaptCosSin_Base'

    # normal 和 scale 网络的总参数数目:12520 + 29360 = 41880
    if R['model2normal'] == 'PDE_DNN_Cos_C_Sin_Base' or R['model2normal'] == 'DNN_adaptCosSin_Base':
        R['hidden2normal'] = (50, 80, 60, 60, 40)  # 1*100+100*80+80*60+60*60+60*40+40*1 = 18940个参数
    else:
        R['hidden2normal'] = (100, 80, 60, 60, 40)          # 1*100+100*80+80*60+60*60+60*40+40*1 = 18940个参数
        # R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)

    if R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base' or R['model2scale'] == 'DNN_adaptCosSin_Base':
        if R['order2laplace'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (100, 60, 60, 50, 40)        # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
            else:
                R['hidden2scale'] = (125, 60, 60, 60, 50)        # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数
        elif R['order2laplace'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (100, 80, 80, 60, 40)        # 1*200+200*80+80*80+80*60+60*40+40*1=29840 个参数
            else:
                R['hidden2scale'] = (125, 80, 80, 60, 40)        # 1*250+250*80+80*80+80*60+60*40+40*1=33890 个参数
        elif R['order2laplace'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (100, 120, 80, 80, 60)       # 1*200+200*120+120*80+80*80+80*60+60*1=45060 个参数
            else:
                R['hidden2scale'] = (125, 120, 80, 80, 60)       # 1*250+250*120+120*80+80*80+80*60+60*1=51110 个参数
    else:
        if R['order2laplace'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (200, 60, 60, 50, 40)        # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
            else:
                R['hidden2scale'] = (250, 60, 60, 60, 50)        # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数
        elif R['order2laplace'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (200, 80, 80, 60, 40)        # 1*200+200*80+80*80+80*60+60*40+40*1=29840 个参数
            else:
                R['hidden2scale'] = (250, 80, 80, 60, 40)       # 1*250+250*80+80*80+80*60+60*40+40*1=33890 个参数
        elif R['order2laplace'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden2scale'] = (200, 120, 80, 80, 60)       # 1*200+200*120+120*80+80*80+80*60+60*1=45060 个参数
            else:
                R['hidden2scale'] = (250, 120, 80, 80, 60)       # 1*250+250*120+120*80+80*80+80*60+60*1=51110 个参数
        else:
            R['hidden2scale'] = (250, 120, 80, 80, 60)               # 1*250+250*120+120*80+80*80+80*60+60*1=51110 个参数
            # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
            # R['hidden2scale'] = (500, 400, 300, 200, 100)
            # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)
            # R['hidden2scale'] = (500, 400, 300, 200, 200, 100)

    # 激活函数的选择
    # R['act_name2NN1'] = 'relu'
    R['act_name2NN1'] = 'tanh'
    # R['act_name2NN1'] = 'srelu'
    # R['act_name2NN1'] = 'sin'
    # R['act_name2NN1'] = 's2relu'

    # R['act_name2NN2'] = 'relu'
    # R['act_name2NN2']' = leaky_relu'
    # R['act_name2NN2'] = 'srelu'
    R['act_name2NN2'] = 's2relu'
    # R['act_name2NN2'] = 'powsin_srelu'
    # R['act_name2NN2'] = 'slrelu'
    # R['act_name2NN2'] = 'gauss'
    # R['act_name2NN2'] = 'metican'
    # R['act_name2NN2'] = 'modify_mexican'
    # R['act_name2NN2'] = 'elu'
    # R['act_name2NN2'] = 'selu'
    # R['act_name2NN2'] = 'phi'

    if R['variational_loss'] == 0:
        R['act_name2NN2'] = 'tanh'

    R['plot_ongoing'] = 0
    R['subfig_type'] = 0
    R['freqs'] = np.arange(11, 101)
    # freqs = np.arange(20, 110, 10)
    # freqs = np.arange(15, 115, 10)
    # R['freqs'] = np.repeat(freqs, 10, 0)

    R['contrib2scale'] = 0.01
    # R['contrib2scale'] = 0.1

    solve_Multiscale_PDE(R)

