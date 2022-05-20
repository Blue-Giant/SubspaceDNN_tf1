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
import time
import DNN_base
import DNN_tools
import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import Load_data2Mat
import DNN_Print_Log
import DNN_data
import saveData
import plotData


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name = '%s%s.txt' % ('log2', 'train')
    log_file = open(os.path.join(log_out_path, outfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.dictionary2file(R, log_file, actName2normal=R['actName2Normal'], actName2scale=R['actName2Scale'])

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                     # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    init_penalty2powU = R['init_penalty2orthogonal']
    hidden2normal = R['hidden2normal']
    hidden2scale = R['hidden2scale']
    act2Normal = R['actName2Normal']
    act2Scale = R['actName2Scale']

    if R['contrib_scale2orthogonal'] == 'with_contrib':
        using_scale2orthogonal = R['contrib2scale']
    else:
        using_scale2orthogonal = 1.0

    if R['opt2loss_bd'] != 'unified_boundary' and R['contrib_scale2boundary'] == 'with_contrib':
        using_scale2boundary = R['contrib2scale']
    else:
        using_scale2boundary = 1.0

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    alpha = R['contrib2scale']

    # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
    #       d      ****         d         ****
    #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
    #       dx     ****         dx        ****
    # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
    p_index = R['order2pLaplace_operator']
    epsilon = R['epsilon']
    mesh_number = R['mesh_number']

    if R['PDE_type'] == 'Possion_Boltzmann':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'Boltzmann1':
            region_lb = -1.0
            region_rt = 1.0
        else:
            region_lb = 0.0
            region_rt = 1.0
        A_eps, kappa, f, u_true, u00, u01, u10, u11, u20, u21 = MS_BoltzmannEqs.get_infos2Boltzmann_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=region_lb,
            intervalR=region_rt, equa_name=R['equa_name'])
    else:
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, A_eps, u00, u01, u10, u11, u20, u21 = MS_LaplaceEqs.get_infos2pLaplace_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])

    flag2Normal = 'WB2Normal'
    flag2Scale = 'WB2Scale'
    if R['model2Normal'] == 'Fourier_DNN':
        Ws_Normal, Bs_Normal = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2normal, flag2Normal)
    else:
        Ws_Normal, Bs_Normal = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2normal, flag2Normal)
    if R['model2Scale'] == 'Fourier_DNN':
        Ws_Scale, Bs_Scale = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2scale, flag2Scale)
    else:
        Ws_Scale, Bs_Scale = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2scale, flag2Scale)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZ_it = tf.compat.v1.placeholder(tf.float32, name='XYZ_it', shape=[None, input_dim])
            XYZ_bottom_bd = tf.compat.v1.placeholder(tf.float32, name='bottom_bd', shape=[None, input_dim])
            XYZ_top_bd = tf.compat.v1.placeholder(tf.float32, name='top_bd', shape=[None, input_dim])
            XYZ_left_bd = tf.compat.v1.placeholder(tf.float32, name='left_bd', shape=[None, input_dim])
            XYZ_right_bd = tf.compat.v1.placeholder(tf.float32, name='right_bd', shape=[None, input_dim])
            XYZ_front_bd = tf.compat.v1.placeholder(tf.float32, name='front_bd', shape=[None, input_dim])
            XYZ_behind_bd = tf.compat.v1.placeholder(tf.float32, name='behind_bd', shape=[None, input_dim])
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e2, shape=[], name='bd_p')
            UdotU_penalty = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if R['model2Normal'] == 'DNN':
                UNN_Normal = DNN_base.DNN(XYZ_it, Ws_Normal, Bs_Normal, hidden2normal, activateIn_name=act2Normal,
                                          activate_name=act2Normal)
                UNN_Left2Normal = DNN_base.DNN(XYZ_left_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                               activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Right2Normal = DNN_base.DNN(XYZ_right_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Bottom2Normal = DNN_base.DNN(XYZ_bottom_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                 activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Top2Normal = DNN_base.DNN(XYZ_top_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                              activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Front2Normal = DNN_base.DNN(XYZ_front_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Behind2Normal = DNN_base.DNN(XYZ_behind_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                 activateIn_name=act2Normal, activate_name=act2Normal)
            elif R['model2Normal'] == 'Fourier_DNN':
                freq2Normal = R['freq2Normal']
                UNN_Normal = DNN_base.DNN_FourierBase(XYZ_it, Ws_Normal, Bs_Normal, hidden2normal, freq2Normal,
                                                      activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Left2Normal = DNN_base.DNN_FourierBase(XYZ_left_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                           freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Right2Normal = DNN_base.DNN_FourierBase(XYZ_right_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                            freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Bottom2Normal = DNN_base.DNN_FourierBase(XYZ_bottom_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                             freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Top2Normal = DNN_base.DNN_FourierBase(XYZ_top_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                          freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Front2Normal = DNN_base.DNN_FourierBase(XYZ_front_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                            freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Behind2Normal = DNN_base.DNN_FourierBase(XYZ_behind_bd, Ws_Normal, Bs_Normal, hidden2normal,
                                                             freq2Normal, activate_name=act2Normal, repeat_Highfreq=False)

            freqs = R['freq2Scale']
            if R['model2Scale'] == 'DNN_scale':
                UNN_Scale = DNN_base.DNN_scale(XYZ_it, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                               activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Left2Scale = DNN_base.DNN_scale(XYZ_left_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                    activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Right2Scale = DNN_base.DNN_scale(XYZ_right_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Bottom2Scale = DNN_base.DNN_scale(XYZ_bottom_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                      activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Top2Scale = DNN_base.DNN_scale(XYZ_top_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                   activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Front2Scale = DNN_base.DNN_scale(XYZ_front_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Behind2Scale = DNN_base.DNN_scale(XYZ_behind_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                      activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale'] == 'Fourier_DNN':
                UNN_Scale = DNN_base.DNN_FourierBase(XYZ_it, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                     activate_name=act2Scale)
                UNN_Left2Scale = DNN_base.DNN_FourierBase(XYZ_left_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                          activate_name=act2Scale)
                UNN_Right2Scale = DNN_base.DNN_FourierBase(XYZ_right_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                           activate_name=act2Scale)
                UNN_Bottom2Scale = DNN_base.DNN_FourierBase(XYZ_bottom_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                            activate_name=act2Scale)
                UNN_Top2Scale = DNN_base.DNN_FourierBase(XYZ_top_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                         activate_name=act2Scale)
                UNN_Front2Scale = DNN_base.DNN_FourierBase(XYZ_front_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                           activate_name=act2Scale)
                UNN_Behind2Scale = DNN_base.DNN_FourierBase(XYZ_behind_bd, Ws_Scale, Bs_Scale, hidden2scale, freqs,
                                                            activate_name=act2Scale)

            X_it = tf.reshape(XYZ_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZ_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZ_it[:, 2], shape=[-1, 1])
            U_NN = UNN_Normal + alpha*UNN_Scale

            dUNN_Normal = tf.gradients(UNN_Normal, XYZ_it)[0]  # * 行 3 列
            dUNN_Scale = tf.gradients(UNN_Scale, XYZ_it)[0]  # * 行 3 列

            if R['loss_type'] == 'variational_loss':
                # 0.5*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),            grad (Uc+Uf) = grad Uc + grad Uf
                # 0.5*a(x)*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),       grad (Uc+Uf) = grad Uc + grad Uf
                dU_NN = tf.add(dUNN_Normal, alpha*dUNN_Scale)
                norm2dU_NN = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    dUNN_2Norm = tf.square(norm2dU_NN)
                    loss_it_variational = (1.0 / 2) *dUNN_2Norm - tf.multiply(f(X_it, Y_it, Z_it), U_NN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it)
                    Kappa = kappa(X_it, Y_it, Z_it)       # * 行 1 列
                    AdUNN_pNorm = a_eps * tf.pow(norm2dU_NN, p_index)
                    # AdUNN_pNorm = tf.multiply(a_eps, tf.pow(norm2dU_NN, p_index))
                    if R['equa_name'] == 'Boltzmann5' or R['equa_name'] == 'Boltzmann6' or \
                            R['equa_name'] == 'Boltzmann7' or R['equa_name'] == 'Boltzmann4':
                        fxyz = MS_BoltzmannEqs.get_force2Boltzmann3D(x=X_it, y=Y_it, z=Z_it, equa_name=R['equa_name'])
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa * U_NN * U_NN) - \
                                              tf.multiply(fxyz, U_NN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa*U_NN*U_NN) - \
                                             tf.multiply(f(X_it, Y_it, Z_it), U_NN)
                else:
                    a_eps = A_eps(X_it, Y_it, Z_it)                          # * 行 1 列
                    AdUNN_pNorm = a_eps * tf.pow(norm2dU_NN, p_index)
                    # AdUNN_pNorm = tf.multiply(a_eps, tf.pow(norm2dU_NN, p_index))
                    if R['equa_name'] == 'multi_scale3D_5' or R['equa_name'] == 'multi_scale3D_6' or \
                            R['equa_name'] == 'multi_scale3D_7':
                        fxyz = MS_LaplaceEqs.get_force2pLaplace3D(x=X_it, y=Y_it, z=Z_it, equa_name=R['equa_name'])
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - tf.multiply(fxyz, U_NN)
                    else:
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - tf.multiply(f(X_it, Y_it, Z_it), U_NN)

                # Loss_it2NN = tf.reduce_mean(loss_it_variational)*np.power(region_rt - region_lb, input_dim)
                Loss_it2NN = tf.reduce_mean(loss_it_variational)
            elif R['loss_type'] == 'variational_loss2':
                # 0.5*|grad Uc|^p + 0.5*|grad Uf|^p - f(x)*(Uc+Uf)
                # 0.5*a(x)*|grad Uc|^p + 0.5*a(x)*|grad Uf|^p - f(x)*(Uc+Uf)
                norm2dUNN_Normal = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN_Normal), axis=-1)), shape=[-1, 1])
                norm2dU_NN_Scale = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN_Scale), axis=-1)), shape=[-1, 1])

                if R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it)
                    Kappa = kappa(X_it, Y_it, Z_it)
                    ApNorm2dU_NN = a_eps * tf.pow(norm2dUNN_Normal, p_index) + \
                                   a_eps * tf.pow(alpha * norm2dU_NN_Scale, p_index)
                    if R['equa_name'] == 'Boltzmann5' or R['equa_name'] == 'Boltzmann6'or \
                        R['equa_name'] == 'Boltzmann7' or R['equa_name'] == 'Boltzmann4':
                        fxyz = MS_BoltzmannEqs.get_force2Boltzmann3D(x=X_it, y=Y_it, z=Z_it, equa_name=R['equa_name'])
                        loss_it_variational = (1.0 / p_index) * (ApNorm2dU_NN + Kappa * U_NN * U_NN) - \
                                              tf.multiply(fxyz, U_NN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (ApNorm2dU_NN + Kappa*U_NN*U_NN) - \
                                              tf.multiply(f(X_it, Y_it, Z_it), U_NN)
                else:
                    a_eps = A_eps(X_it, Y_it, Z_it)  # * 行 1 列
                    ApNorm2dU_NN = a_eps * tf.pow(norm2dUNN_Normal, p_index) + \
                                   a_eps * tf.pow(alpha * norm2dU_NN_Scale, p_index)
                    loss_it_variational = (1.0 / p_index) * ApNorm2dU_NN - tf.multiply(f(X_it, Y_it, Z_it), U_NN)
                # Loss_it2NN = tf.reduce_mean(loss_it_variational) * np.power(region_rt - region_lb, input_dim)
                Loss_it2NN = tf.reduce_mean(loss_it_variational)

            if R['opt2loss_udotu'] == 'with_orthogonal':
                if R['opt2orthogonal'] == 0:  # L2
                    norm2UdU = tf.reduce_mean(tf.multiply(UNN_Normal, using_scale2orthogonal * UNN_Scale))
                    UNN_dot_UNN = tf.square(norm2UdU)
                elif R['opt2orthogonal'] == 1:
                    # |Uc*Uf|^2-->0
                    norm2UdU = tf.square(tf.multiply(UNN_Normal, using_scale2orthogonal * UNN_Scale))
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU)
                elif R['opt2orthogonal'] == 2:
                    # |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    dU_dot_dU = tf.multiply(dUNN_Normal, using_scale2orthogonal * dUNN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
                else:
                    # |Uc*Uf|^2 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    # |Uc*Uf|^2 + |(grad Uc)*(grad Uf)|^2-->0
                    U_dot_U = tf.reshape(tf.square(tf.multiply(UNN_Normal, using_scale2orthogonal * UNN_Scale)),
                                         shape=[-1, 1])
                    dU_dot_dU = tf.multiply(dUNN_Normal, using_scale2orthogonal * dUNN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
                Loss2UNN_dot_UNN = UdotU_penalty * UNN_dot_UNN
            else:
                Loss2UNN_dot_UNN = tf.constant(0.0)

            if R['opt2loss_bd'] == 'unified_boundary':
                UNN_left = UNN_Left2Normal + using_scale2boundary * UNN_Left2Scale
                UNN_right = UNN_Right2Scale + using_scale2boundary * UNN_Right2Scale
                UNN_bottom = UNN_Bottom2Normal + using_scale2boundary * UNN_Bottom2Scale
                UNN_top = UNN_Top2Normal + using_scale2boundary * UNN_Top2Scale
                UNN_front = UNN_Front2Normal + using_scale2boundary * UNN_Front2Scale
                UNN_behind = UNN_Behind2Normal + using_scale2boundary * UNN_Behind2Scale

                Loss_bd2NN = tf.square(UNN_left) + tf.square(UNN_right) + \
                             tf.square(UNN_bottom) + tf.square(UNN_top) + \
                             tf.square(UNN_front) + tf.square(UNN_behind)
                Loss_bds = bd_penalty * tf.reduce_mean(Loss_bd2NN)
            else:
                loss_bd_square2Normal = tf.square(UNN_Left2Normal) + tf.square(UNN_Right2Normal) + \
                                        tf.square(UNN_Bottom2Normal) + tf.square(UNN_Top2Normal) + \
                                        tf.square(UNN_Front2Normal) + tf.square(UNN_Behind2Normal)
                loss_bd_square2Scale = tf.square(using_scale2boundary*UNN_Left2Scale) + \
                                       tf.square(using_scale2boundary*UNN_Right2Scale) + \
                                       tf.square(using_scale2boundary*UNN_Bottom2Scale) + \
                                       tf.square(using_scale2boundary*UNN_Top2Scale) + \
                                       tf.square(using_scale2boundary*UNN_Front2Scale) + \
                                       tf.square(using_scale2boundary*UNN_Behind2Scale)
                Loss_bd2Normal = tf.reduce_mean(loss_bd_square2Normal)
                Loss_bd2Scale = tf.reduce_mean(loss_bd_square2Scale)
                Loss_bds = bd_penalty * (Loss_bd2Normal + Loss_bd2Scale)

            if R['regular_wb_model'] == 'L1':
                regular_WBs_Normal = DNN_base.regular_weights_biases_L1(Ws_Normal, Bs_Normal)    # 正则化权重和偏置 L1正则化
                regular_WBs_Scale = DNN_base.regular_weights_biases_L1(Ws_Scale, Bs_Scale)
            elif R['regular_wb_model'] == 'L2':
                regular_WBs_Normal = DNN_base.regular_weights_biases_L2(Ws_Normal, Bs_Normal)    # 正则化权重和偏置 L2正则化
                regular_WBs_Scale = DNN_base.regular_weights_biases_L2(Ws_Scale, Bs_Scale)
            else:
                regular_WBs_Normal = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WBs_Scale = tf.constant(0.0)

            # 要优化的loss function
            PWB = penalty2WB * (regular_WBs_Normal + regular_WBs_Scale)

            Loss2NN = Loss_it2NN + Loss_bds + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['loss_type'] == 'variational_loss':
                if R['train_model'] == 'training_group4':
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 'training_group3':
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            elif R['loss_type'] == 'variational_loss2':
                if R['train_model'] == 'training_group3':
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_model'] == 'training_group2':
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            if R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'pLaplace':
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
        test_xyz_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyz_bach = Load_data2Mat.get_randomData2mat(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
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
                [train_loss2NN, Loss_it2NN, Loss_bds, Loss2NN, UNN_dot_UNN, train_mse2NN, train_rel2NN],
                feed_dict={XYZ_it: xyz_it_batch, XYZ_left_bd: xyz_left_batch, XYZ_right_bd: xyz_right_batch,
                           XYZ_bottom_bd: xyz_bottom_batch, XYZ_top_bd: xyz_top_batch, XYZ_front_bd: xyz_front_batch,
                           XYZ_behind_bd: xyz_behind_batch, in_learning_rate: tmp_lr,
                           bd_penalty: temp_penalty_bd, UdotU_penalty: temp_penalty_powU})
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
                    udu_nn, train_mse_nn, train_rel_nn, log_out=log_file)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                if R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'pLaplace':
                    u_true2test, utest_nn, utest_normal, utest_freqs = sess.run(
                        [U_true, U_NN, UNN_Normal, alpha*UNN_Scale], feed_dict={XYZ_it: test_xyz_bach})
                else:
                    u_true2test = u_true
                    utest_nn, utest_normal, utest_freqs = sess.run(
                        [U_NN, UNN_Normal, alpha*UNN_Scale], feed_dict={XYZ_it: test_xyz_bach})

                point_ERR2NN = np.square(u_true2test - utest_nn)
                test_mse2nn = np.mean(point_ERR2NN)
                test_mse_all2NN.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all2NN.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_file)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN, lossBD_all2NN, loss_all2NN, actName=act2Normal,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all2NN, train_rel_all2NN, actName=act2Normal,
                                    outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(lossIt_all2NN, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(lossBD_all2NN, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all2NN, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(UDU_NN, lossType='udu', seedNo=R['seed'], outPath=R['FolderName'])

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all2NN, train_rel_all2NN, actName=act2Normal, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------- save test data to mat file and plot the testing results into figures -----------------------
    if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'pLaplace':
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
        plotData.plot_Hot_solution2test(utest_nn, size_vec2mat=size2test, actName=act2Normal,
                                        seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all2NN, test_rel_all2NN, actName=act2Normal, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all2NN, test_rel_all2NN, test_epoch, actName=act2Normal, seedNo=R['seed'],
                              outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_ERR2NN, actName=act2Normal, outPath=R['FolderName'])

    plotData.plot_Hot_point_wise_err(point_ERR2NN, size_vec2mat=size2test, actName=act2Normal,
                                     seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"              # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"               # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"   # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 文件保存路径设置
    # store_file = 'Laplace3D'
    # store_file = 'pLaplace3D'
    store_file = 'Boltzmann3D'
    # store_file = 'Convection3D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                                # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)             # 路径连接
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

    if store_file == 'Laplace3D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace3D':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale3D_1'
        # R['equa_name'] = 'multi_scale3D_2'
        # R['equa_name'] = 'multi_scale3D_3'
        # R['equa_name'] = 'multi_scale3D_5'
        # R['equa_name'] = 'multi_scale3D_6'
        R['equa_name'] = 'multi_scale3D_7'
    elif store_file == 'Boltzmann3D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'
        # R['equa_name'] = 'Boltzmann2'
        # R['equa_name'] = 'Boltzmann3'
        R['equa_name'] = 'Boltzmann4'
        # R['equa_name'] = 'Boltzmann5'
        # R['equa_name'] = 'Boltzmann6'
        # R['equa_name'] = 'Boltzmann7'

    if R['PDE_type'] == 'general_laplace':
        R['mesh_number'] = 2
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000
    elif R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
        R['mesh_number'] = 2
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000

    # ---------------------------- Setup of DNN -------------------------------
    # R['loss_type'] = 'L2_loss'                 # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    R['loss_type'] = 'variational_loss'          # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['loss_type'] = 'variational_loss2'       # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的

    R['opt2orthogonal'] = 0                      # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 1                    # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 2                    # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy

    R['hot_power'] = 1
    R['testData_model'] = 'loadData'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'

    R['penalty2weight_biases'] = 0.000          # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001        # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025       # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100            # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['init_penalty2orthogonal'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['init_penalty2orthogonal'] = 10000.0
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'training_union'                   # 训练模式, 一个 loss 联结训练
    # R['train_model'] = 'training_group1'                # 训练模式, 多个 loss 组团训练
    # R['train_model'] = 'training_group2'
    # R['train_model'] = 'training_group3'
    # R['train_model'] = 'training_group4'

    # R['model2Normal'] = 'DNN'                           # 使用的网络模型
    # R['model2Normal'] = 'DNN_scale'
    # R['model2Normal'] = 'DNN_adapt_scale'
    R['model2Normal'] = 'Fourier_DNN'

    # R['model2Scale'] = 'DNN'                            # 使用的网络模型
    # R['model2Scale'] = 'DNN_scale'
    # R['model2Scale'] = 'DNN_adapt_scale'
    R['model2Scale'] = 'Fourier_DNN'

    # 单纯的 MscaleDNN 网络 FourierBase(250,400,400,200,200,150)  250+500*400+400*400+400*200+200*200+200*150+150 = 510400
    # 单纯的 MscaleDNN 网络 GeneralBase(500,400,400,200,200,150) 500+500*400+400*400+400*200+200*200+200*150+150 = 510650
    # FourierBase normal 和 FourierBase scale 网络的总参数数目:143220 + 365400 = 508870
    # GeneralBase normal 和 FourierBase scale 网络的总参数数目:143290 + 365650 = 508940
    if R['model2Normal'] == 'Fourier_DNN':
        R['hidden2normal'] = (70, 200, 200, 150, 150, 150)  # 70+140*200+200*200+200*150+150*150+150*150+150=143220
    else:
        R['hidden2normal'] = (140, 200, 200, 150, 150, 150)   # 140+140*200+200*200+200*150+150*150+150*150+150=143290
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)
        # R['hidden2normal'] = (500, 400, 300, 300, 200, 100)

    if R['model2Scale'] == 'Fourier_DNN':
        R['hidden2scale'] = (250, 300, 290, 200, 200, 150)  # 1*250+500*300+300*290+290*200+200*200+200*150+150 = 365400
    else:
        R['hidden2scale'] = (500, 300, 280, 200, 200, 150)  # 1*500+500*300+300*290+290*200+200*150+150*150+150 = 365650
        # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)

    # R['freq2Normal'] = np.concatenate(([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5], np.arange(5, 31)), axis=0)
    R['freq2Normal'] = np.arange(1, 41)*0.5
    if R['model2Scale'] == 'Fourier_DNN':
        R['freq2Scale'] = np.arange(21, 121)
        # R['freq2Scale'] = np.arange(16, 101)
        # R['freq2Scale'] = np.arange(21, 101)
        # R['freq2Scale'] = np.arange(6, 105)
        # R['freq2Scale'] = np.arange(1, 101)
    else:
        R['freq2Scale'] = np.arange(21, 121)
        # R['freq2Scale'] = np.arange(21, 101)
        # R['freq2Scale'] = np.arange(6, 105)
        # R['freq2Scale'] = np.arange(1, 101)

    # 激活函数的选择
    # R['act_in2Normal'] = 'relu'
    R['act_in2Normal'] = 'tanh'

    # R['actName2Normal'] = 'relu'
    R['actName2Normal'] = 'tanh'
    # R['actName2Normal'] = 'srelu'
    # R['actName2Normal'] = 'sin'
    # R['actName2Normal'] = 's2relu'

    R['act_out2Normal'] = 'linear'

    # R['act_in2Scale'] = 'relu'
    R['act_in2Scale'] = 'tanh'

    # R['actName2Scale'] = 'relu'
    # R['actName2Scale']' = leaky_relu'
    # R['actName2Scale'] = 'srelu'
    R['actName2Scale'] = 's2relu'
    # R['actName2Scale'] = 'tanh'
    # R['actName2Scale'] = 'elu'
    # R['actName2Scale'] = 'phi'

    R['act_out2Scale'] = 'linear'
    
    if R['model2Normal'] == 'Fourier_DNN' and R['actName2Normal'] == 'tanh':
        R['sFourier2Normal'] = 1.0
    elif R['model2Normal'] == 'Fourier_DNN' and R['actName2Normal'] == 's2relu':
        R['sFourier2Normal'] = 0.5

    if R['model2Scale'] == 'Fourier_DNN' and R['actName2Scale'] == 'tanh':
        R['sFourier2Scale'] = 1.0
    elif R['model2Scale'] == 'Fourier_DNN' and R['actName2Scale'] == 's2relu':
        R['sFourier2Scale'] = 0.5
    
    if R['loss_type'] == 'variational_loss2':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss3':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        # R['contrib2scale'] = 0.05
        R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss4':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.025
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005

    R['opt2loss_udotu'] = 'with_orthogonal'
    # R['opt2loss_udotu'] = 'without_orthogonal'

    # R['opt2loss_bd'] = 'unified_boundary'
    R['opt2loss_bd'] = 'individual_boundary'

    R['contrib_scale2orthogonal'] = 'with_contrib'
    # R['contrib_scale2orthogonal'] = 'without_contrib'

    R['contrib_scale2boundary'] = 'with_contrib'
    # R['contrib_scale2boundary'] = 'without_contrib'

    solve_Multiscale_PDE(R)

