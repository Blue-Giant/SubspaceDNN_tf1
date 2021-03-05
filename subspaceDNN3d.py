"""
@author: LXA
 Data: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base
import DNN_tools
import MS_Laplace_eqs
import general_laplace_eqs
import matData2HighDim_PDE
import DNN_data
import saveData
import plotData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName2normal=None, actName2scale=None):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    DNN_tools.log_string('The order to p-laplace: %s\n' % (R_dic['order2laplace']), log_fileout)

    DNN_tools.log_string('Network model of solving normal-part: %s\n' % str(R_dic['model2normal']), log_fileout)
    DNN_tools.log_string('Network model of solving scale-part: %s\n' % str(R_dic['model2scale']), log_fileout)
    DNN_tools.log_string('Activate function for normal-part network: %s\n' % str(actName2normal), log_fileout)
    DNN_tools.log_string('Activate function for scale-part network: %s\n' % str(actName2scale), log_fileout)

    DNN_tools.log_string('The frequency for scale-part network: %s\n' % (R_dic['freqs']), log_fileout)

    DNN_tools.log_string('hidden layer to normal:%s\n' % str(R_dic['hidden2normal']), log_fileout)
    DNN_tools.log_string('hidden layer to scale :%s\n' % str(R_dic['hidden2scale']), log_fileout)

    if R_dic['PDE_type'] == 'p_laplace2multi_scale_implicit' or R_dic['PDE_type'] == 'p_laplace2multi_scale_explicit':
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)  # 替换上两行

    if R_dic['PDE_type'] == 'p_laplace2multi_scale_implicit':
        DNN_tools.log_string('The mesh_number: %f\n' % (R_dic['mesh_number']), log_fileout)  # 替换上两行

    if R_dic['variational_loss'] == 1 or R_dic['variational_loss'] == 2:
        DNN_tools.log_string('Loss function: variational loss with ' + str(R_dic['variational_loss']) + '\n',
                             log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    if (R_dic['train_opt']) == 0:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_opt']) == 1:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd + loss_U2U', log_fileout)
    elif (R_dic['train_opt']) == 2:
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)

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

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)

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
    if R['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R['activate_penalty2bd_increase'] == 2:
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

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    init_penalty2powU = R['balance2solus']
    hidden2normal = R['hidden2normal']
    hidden2scale = R['hidden2scale']
    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    alpha = R['contrib2scale']

    # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
    #       d      ****         d         ****
    #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
    #       dx     ****         dx        ****
    # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
    p = R['order2laplace']
    epsilon = R['epsilon']
    mesh_number = R['mesh_number']

    region_lb = 0.0
    region_rt = 1.0
    u_true, f, A_eps = MS_Laplace_eqs.get_laplace_multi_scale_infos3D(
        input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], region_lb=0.0, region_rt=1.0,
        laplace_name=R['equa_name'])

    flag2Normal = 'WB2Normal'
    flag2Scale = 'WB2Scale'
    # Weights, Biases = PDE_DNN_base.Initial_DNN2different_hidden(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag1)
    # Weights, Biases = DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag1)
    Ws_Normal, B_Normal = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden2normal, flag2Normal)
    if R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base':
        Ws_Scale, B_Scale = DNN_base.initialize_NN_random_normal2_CS(input_dim, out_dim, hidden2scale, flag2Scale)
    else:
        Ws_Scale, B_Scale = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden2scale, flag2Scale)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZ_it = tf.placeholder(tf.float32, name='XYZ_it', shape=[None, input_dim])
            XYZ_bottom_bd = tf.placeholder(tf.float32, name='bottom_bd', shape=[None, input_dim])
            XYZ_top_bd = tf.placeholder(tf.float32, name='top_bd', shape=[None, input_dim])
            XYZ_left_bd = tf.placeholder(tf.float32, name='left_bd', shape=[None, input_dim])
            XYZ_right_bd = tf.placeholder(tf.float32, name='right_bd', shape=[None, input_dim])
            XYZ_front_bd = tf.placeholder(tf.float32, name='front_bd', shape=[None, input_dim])
            XYZ_behind_bd = tf.placeholder(tf.float32, name='behind_bd', shape=[None, input_dim])
            bd_penalty = tf.placeholder_with_default(input=1e2, shape=[], name='bd_p')
            penalty2powU = tf.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            if R['model2normal'] == 'PDE_DNN':
                U_NN_Normal = DNN_base.PDE_DNN(XYZ_it, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                ULeft_NN_Normal = DNN_base.PDE_DNN(XYZ_left_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                URight_NN_Normal = DNN_base.PDE_DNN(XYZ_right_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                UBottom_NN_Normal = DNN_base.PDE_DNN(XYZ_bottom_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                UTop_NN_Normal = DNN_base.PDE_DNN(XYZ_top_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                UFront_NN_Normal = DNN_base.PDE_DNN(XYZ_front_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                UBehind_NN_Normal = DNN_base.PDE_DNN(XYZ_behind_bd, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)

            freqs = R['freqs']
            if R['model2scale'] == 'PDE_DNN_scale':
                U_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_left_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_right_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBottom_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_bottom_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UTop_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_top_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UFront_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_front_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBehind_NN_Scale = DNN_base.PDE_DNN_scale(XYZ_behind_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_adapt_scale':
                U_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_left_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_right_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBottom_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_bottom_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UTop_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_top_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UFront_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_front_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBehind_NN_Scale = DNN_base.PDE_DNN_adapt_scale(XYZ_behind_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_FourierBase':
                U_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_left_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_right_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBottom_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_bottom_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UTop_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_top_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UFront_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_front_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBehind_NN_Scale = DNN_base.PDE_DNN_FourierBase(XYZ_behind_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base':
                U_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                ULeft_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_left_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                URight_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_right_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBottom_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_bottom_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UTop_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_top_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UFront_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_front_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                UBehind_NN_Scale = DNN_base.PDE_DNN_Cos_C_Sin_Base(XYZ_behind_bd, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)

            X_it = tf.reshape(XYZ_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZ_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZ_it[:, 2], shape=[-1, 1])
            U_NN = U_NN_Normal + alpha*U_NN_Scale

            dU_NN_Normal = tf.gradients(U_NN_Normal, XYZ_it)[0]  # * 行 3 列
            dU_NN_Scale = tf.gradients(U_NN_Scale, XYZ_it)[0]  # * 行 3 列

            if R['variational_loss'] == 1:
                # 0.5*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),            grad (Uc+Uf) = grad Uc + grad Uf
                # 0.5*a(x)*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),       grad (Uc+Uf) = grad Uc + grad Uf
                dU_NN = tf.add(dU_NN_Normal, alpha*dU_NN_Scale)
                norm2dU_NN = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_laplace':
                    laplace_pow_Normal = tf.square(norm2dU_NN)
                    loss_it_variational_NN = (1.0 / 2) *laplace_pow_Normal - tf.multiply(f(X_it, Y_it, Z_it), U_NN)
                else:
                    a_eps = A_eps(X_it, Y_it, Z_it)                          # * 行 1 列

                    laplace_p_pow_Normal = a_eps*tf.pow(norm2dU_NN, p)
                    loss_it_variational_NN = (1.0 / p) * laplace_p_pow_Normal - tf.multiply(f(X_it, Y_it, Z_it), U_NN)

                Loss_it2NN = tf.reduce_mean(loss_it_variational_NN) * np.power(region_rt - region_lb, p)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0
                    norm2UdU = tf.reshape(tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_Scale)), axis=-1),
                                          shape=[-1, 1])
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU, axis=0)
                elif R['wavelet'] == 2:
                    # |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    # norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU, axis=0)
                else:
                    # |Uc*Uf|^2 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    # |Uc*Uf|^2 + |(grad Uc)*(grad Uf)|^2-->0
                    U_dot_U = tf.reduce_mean(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_Scale)), axis=0)
                    dU_dot_dU = tf.multiply(dU_NN_Normal, alpha*dU_NN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    # norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU, axis=0) + U_dot_U
            elif R['variational_loss'] == 2:
                # 0.5*|grad Uc|^p + 0.5*|grad Uf|^p - f(x)*(Uc+Uf)
                # 0.5*a(x)*|grad Uc|^p + 0.5*a(x)*|grad Uf|^p - f(x)*(Uc+Uf)
                norm2dU_NN_Normal = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN_Normal), axis=-1)),
                                               shape=[-1, 1])  # 按行求和
                norm2dU_NN_freqs = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN_Scale), axis=-1)),
                                              shape=[-1, 1])  # 按行求和

                if R['PDE_type'] == 'general_laplace':
                    laplace_NN = tf.square(norm2dU_NN_Normal) + tf.square(norm2dU_NN_freqs)
                    loss_it_variational2NN = (1.0 / 2) * laplace_NN - tf.multiply(f(X_it, Y_it,Z_it), U_NN)
                else:
                    a_eps = A_eps(X_it, Y_it, Z_it)  # * 行 1 列
                    laplace_p_NN = a_eps * tf.pow(norm2dU_NN_Normal, p) + a_eps * tf.pow(norm2dU_NN_freqs, p)
                    loss_it_variational2NN = (1.0 / p) * laplace_p_NN - tf.multiply(f(X_it, Y_it,Z_it), U_NN)
                Loss_it2NN = tf.reduce_mean(loss_it_variational2NN) * (region_rt - region_lb) * (region_rt - region_lb)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0
                    norm2UdU = tf.reshape(tf.reduce_sum(tf.square(tf.multiply(U_NN_Normal, alpha*U_NN_Scale)), axis=-1),
                                          shape=[-1, 1])
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU, axis=0)
                else:
                    UNN_dot_UNN = tf.constant(0.0)

            Loss2UNN_dot_UNN = penalty2powU * UNN_dot_UNN

            loss_bd_square2Normal = tf.square(ULeft_NN_Normal) + tf.square(URight_NN_Normal) + \
                                    tf.square(UBottom_NN_Normal) + tf.square(UTop_NN_Normal) + \
                                    tf.square(UFront_NN_Normal) + tf.square(UBehind_NN_Normal)
            loss_bd_square2Scale = tf.square(alpha*ULeft_NN_Scale) + tf.square(alpha*URight_NN_Scale) + \
                                    tf.square(alpha*UBottom_NN_Scale) + tf.square(alpha*UTop_NN_Scale) + \
                                    tf.square(alpha*UFront_NN_Scale) + tf.square(alpha*UBehind_NN_Scale)
            Loss_bd2NN = tf.reduce_mean(loss_bd_square2Normal) + tf.reduce_mean(loss_bd_square2Scale)

            Loss_bd2NNs = bd_penalty * Loss_bd2NN

            if R['regular_weight_model'] == 'L1':
                regular_WB_Normal = DNN_base.regular_weights_biases_L1(Ws_Normal, B_Normal)    # 正则化权重和偏置 L1正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L1(Ws_Scale, B_Scale)
            elif R['regular_weight_model'] == 'L2':
                regular_WB_Normal = DNN_base.regular_weights_biases_L2(Ws_Normal, B_Normal)    # 正则化权重和偏置 L2正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L2(Ws_Scale, B_Scale)
            else:
                regular_WB_Normal = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB_Scale = tf.constant(0.0)

            # 要优化的loss function
            penalty_Weigth_Bias = wb_regular * (regular_WB_Normal + regular_WB_Scale)

            Loss2NN = Loss_it2NN + Loss_bd2NNs + Loss2UNN_dot_UNN + penalty_Weigth_Bias

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['variational_loss'] == 1:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_opt'] == 2:
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            elif R['variational_loss'] == 2:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_opt'] == 2:
                    train_sin_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_sin_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_loss2NN = tf.group(train_sin_op1, train_sin_op2)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale':
                # 训练上的真解值和训练结果的误差
                U_true = u_true(X_it, Y_it, Z_it)

                train_mse2NN = tf.reduce_mean(tf.square(U_true - U_NN))
                train_rel2NN = train_mse2NN / tf.reduce_mean(tf.square(U_true))
            else:
                train_mse2NN = tf.constant(0.0)
                train_rel2NN = tf.constant(0.0)

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    lossIt_all2NN, lossBD_all2NN, loss_all2NN, train_mse_all2NN, train_rel_all2NN = [], [], [], [], []
    UDU_NN = []
    test_mse_all2NN, test_rel_all2NN = [], []
    test_epoch = []

    # 画网格解图
    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        # test_bach_size = 900
        # size2test = 30
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        # test_bach_size = 1000000
        # size2test = 1000
        test_xyz_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'data2mat'
        test_xyz_bach = matData2HighDim_PDE.get_data2Biharmonic(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            xyz_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyz_bottom_batch, xyz_top_batch, xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch = \
                DNN_data.rand_bd_3D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
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
                    temp_penalty_powU = 10 * init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_powU = 50 * init_penalty2powU
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_powU = 100 * init_penalty2powU
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_powU = 200 * init_penalty2powU
                else:
                    temp_penalty_powU = 500 * init_penalty2powU
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

            _, loss_it_nn, loss_bd_nn, loss_nn, udu_nn, train_mse_nn, train_rel_nn = sess.run(
                [train_loss2NN, Loss_it2NN, Loss_bd2NN, Loss2NN, UNN_dot_UNN, train_mse2NN, train_rel2NN],
                feed_dict={XYZ_it: xyz_it_batch, XYZ_left_bd: xyz_left_batch, XYZ_right_bd: xyz_right_batch,
                           XYZ_bottom_bd: xyz_bottom_batch, XYZ_top_bd: xyz_top_batch, XYZ_front_bd: xyz_front_batch,
                           XYZ_behind_bd: xyz_behind_batch, in_learning_rate: tmp_lr,
                           bd_penalty: temp_penalty_bd, penalty2powU: temp_penalty_powU, train_opt: train_option})
            lossIt_all2NN.append(loss_it_nn)
            lossBD_all2NN.append(loss_bd_nn)
            loss_all2NN.append(loss_nn)
            UDU_NN.append(udu_nn)
            train_mse_all2NN.append(train_mse_nn)
            train_rel_all2NN.append(train_rel_nn)

            if i_epoch % 1000 == 0:
                p_WB = 0.0
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, p_WB, loss_it_nn, loss_bd_nn, loss_nn,
                    udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout_NN)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale':
                    u_true2test, utest_nn, utest_normal, utest_freqs = sess.run(
                        [U_true, U_NN, U_NN_Normal, U_NN_Scale], feed_dict={XYZ_it: test_xyz_bach, train_opt: train_option})
                else:
                    u_true2test = u_true
                    utest_nn, utest_normal, utest_freqs = sess.run(
                        [U_NN, U_NN_Normal, U_NN_Scale], feed_dict={XYZ_it: test_xyz_bach, train_opt: train_option})

                point_ERR2NN = np.square(u_true2test - utest_nn)
                test_mse2nn = np.mean(point_ERR2NN)
                test_mse_all2NN.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all2NN.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout_NN)

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN, lossBD_all2NN, loss_all2NN, actName=act_func1,
                                             outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all2NN, train_rel_all2NN, actName=act_func1,
                                        outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(lossIt_all2NN, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(lossBD_all2NN, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all2NN, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(UDU_NN, lossType='udu', seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plotTrain_MSE_REL_1act_func(train_mse_all2NN, train_rel_all2NN, actName=act_func1, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------- save test data to mat file and plot the testing results into figures -----------------------
        if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale':
            saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])

        saveData.save_testData_or_solus2mat(utest_nn, dataName='test', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_normal, dataName='normal', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_freqs, dataName='scale', outPath=R['FolderName'])

        if R['hot_power'] == 1:
            # ----------------------------------------------------------------------------------------------------------
            #                                      绘制解的热力图(真解和DNN解)
            # ----------------------------------------------------------------------------------------------------------
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                            seedNo=R['seed'], outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(utest_nn, size_vec2mat=size2test, actName=act_func1,
                                            seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all2NN, test_rel_all2NN, actName=act_func1, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all2NN, test_rel_all2NN, test_epoch, actName=act_func1, seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_ERR2NN, actName=act_func1, outPath=R['FolderName'])

        plotData.plot_Hot_point_wise_err(point_ERR2NN, size_vec2mat=size2test, actName=act_func1,
                                         seedNo=R['seed'], outPath=R['FolderName'])


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

    # 文件保存路径设置
    store_file = 'laplace3d'
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

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ---------------------------- Setup of multi-scale problem-------------------------------
    R['input_dim'] = 3                # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1               # 输出维数

    # R['PDE_type'] = 'general_laplace'
    # R['equa_name'] = 'PDE1'
    # R['equa_name'] = 'PDE2'
    # R['equa_name'] = 'PDE3'
    # R['equa_name'] = 'PDE4'
    # R['equa_name'] = 'PDE5'
    # R['equa_name'] = 'PDE6'
    # R['equa_name'] = 'PDE7'

    R['PDE_type'] = 'p_laplace2multi_scale'
    # R['equa_name'] = 'multi_scale3D_1'
    # R['equa_name'] = 'multi_scale3D_2'
    R['equa_name'] = 'multi_scale3D_3'

    if R['PDE_type'] == 'general_laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 5000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000
    elif R['PDE_type'] == 'p_laplace2multi_scale':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 5000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000

    # ---------------------------- Setup of DNN -------------------------------
    R['variational_loss'] = 1           # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['variational_loss'] = 2         # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    R['wavelet'] = 1                    # 0: L2 wavelet+energy    1: L2 wavelet     2:energy
    # R['wavelet'] = 2                  # 0: L2 wavelet+energy    1: L2 wavelet     2:energy

    R['hot_power'] = 1
    R['freqs'] = np.arange(10, 101)
    # R['freqs'] = np.arange(6, 105)
    # R['freqs'] = np.concatenate((np.arange(2, 100), [100]), axis=0)
    R['testData_model'] = 'loadData'

    R['weight_biases_model'] = 'general_model'
    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'

    R['regular_weight_biases'] = 0.000  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                   # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100                     # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['balance2solus'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['balance2solus'] = 10000.0
    else:
        R['balance2solus'] = 20.0

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_opt'] = 0
    # R['train_opt'] = 1
    # R['train_opt'] = 3
    # R['train_opt'] = 4

    R['model2normal'] = 'PDE_DNN'  # 使用的网络模型
    # R['model2normal'] = 'PDE_DNN_sin'
    # R['model2normal'] = 'PDE_DNN_BN'
    # R['model2normal'] = 'PDE_DNN_scale'
    # R['model2normal'] = 'PDE_DNN_adapt_scale'
    # R['model2normal'] = 'PDE_DNN_FourierBase'

    # R['model2scale'] = 'PDE_DNN'                         # 使用的网络模型
    # R['model2scale'] = 'PDE_DNN_BN'
    # R['model2scale'] = 'PDE_DNN_scale'
    # R['model2scale'] = 'PDE_DNN_adapt_scale'
    # R['model2scale'] = 'PDE_DNN_FourierBase'
    R['model2scale'] = 'PDE_DNN_Cos_C_Sin_Base'

    # normal 和 scale 网络的总参数数目:41380 + 85780 = 127160
    # R['hidden2normal'] = (50, 10, 8, 8, 6)
    R['hidden2normal'] = (250, 200, 200, 150, 100, 100)   # 250+250*200+200*200+200*150+150*100+100*100+100=145350
    # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
    # R['hidden2normal'] = (500, 400, 300, 200, 100)
    # R['hidden2normal'] = (500, 400, 300, 300, 200, 100)
    # R['hidden2normal'] = (500, 400, 300, 200, 200, 100)
    # R['hidden2normal'] = (500, 400, 300, 300, 200, 100, 100)
    # R['hidden2normal'] = (500, 300, 200, 200, 100, 100, 50)
    # R['hidden2normal'] = (1000, 800, 600, 400, 200)
    # R['hidden2normal'] = (1000, 500, 400, 300, 300, 200, 100, 100)
    # R['hidden2normal'] = (2000, 1500, 1000, 500, 250)

    if R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base':
        R['hidden2scale'] = (250, 300, 300, 200, 200, 100)  # 1*500+500*300+300*300+300*200+200*150+150*150+150 = 360600
    else:
        R['hidden2scale'] = (500, 300, 300, 200, 200, 100)  # 1*500+500*300+300*300+300*200+200*150+150*150+150 = 360600
        # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 200, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100, 100)
        # R['hidden2scale'] = (500, 300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (1000, 800, 600, 400, 200)
        # R['hidden2scale'] = (1000, 500, 400, 300, 300, 200, 100, 100)
        # R['hidden2scale'] = (2000, 1500, 1000, 500, 250)

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
    # R['act_name2NN2'] = 'elu'
    # R['act_name2NN2'] = 'selu'
    # R['act_name2NN2'] = 'phi'
    R['contrib2scale'] = 0.01
    solve_Multiscale_PDE(R)

