"""
@author: Xi'an Li
 Create Date: 2020.10.31
 The last modification on 2021.10.15
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
import DNN_data
import General_Laplace
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import matData2Laplace
import matData2Boltzmann
import saveData
import plotData
import DNN_Print_Log


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']          # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):    # 判断路径是否已经存在
        os.mkdir(log_out_path)              # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.dictionary_out2file(R, log_fileout, actName2normal=R['actName2Normal'],
                                      actName2scale=R['actName2Scale'])

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']           # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    init_UdotU_penalty = R['init_penalty2orthogonal']
    hidden2normal = R['hidden2normal']
    hidden2scale = R['hidden2scale']
    act2Normal = R['actName2Normal']
    act2Scale = R['actName2Scale']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    alpha = R['contrib2scale']

    if R['contrib_scale2orthogonal'] == 'with_contrib':
        using_scale2orthogonal = R['contrib2scale']
    else:
        using_scale2orthogonal = 1.0

    if R['opt2loss_bd'] != 'unified_boundary' and R['contrib_scale2boundary'] == 'with_contrib':
        using_scale2boundary = R['contrib2scale']
    else:
        using_scale2boundary = 1.0

    # p laplace 问题需要的额外设置, 先预设一下
    p_index = 2
    epsilon = 0.1
    mesh_number = 2

    # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = General_Laplace.get_infos2Laplace_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_implicit':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.get_infos2pLaplace_2D(
                input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
                equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_explicit':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        if R['equa_name'] == 'multi_scale2D_7':
            region_lb = -1.0
            region_rt = 1.0
            f = MS_LaplaceEqs.force_side2E7(input_dim, out_dim)                       # f是一个向量
            u_true = MS_LaplaceEqs.true_solution2E7(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.boundary2E7(
                input_dim, out_dim, region_lb, region_rt, eps=epsilon)
            A_eps = MS_LaplaceEqs.elliptic_coef2E7(input_dim, out_dim, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  + K *u_eps =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        # region_lb = -1.0
        region_lb = 0.0
        region_rt = 1.0
        A_eps, kappa, u_true, u_left, u_right, u_top, u_bottom, f = MS_BoltzmannEqs.get_infos2Boltzmann_2D(
             intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Convection_diffusion':
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_2D(
            equa_name=R['equa_name'], eps=epsilon, region_lb=0.0, region_rt=1.0)

    flag_Normal = 'WB_Normal'
    flag_Scale = 'WB_Scale'
    if R['model2Normal'] == 'Fourier_DNN':
        W2NN_Normal, B2NN_Normal = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2normal, flag_Normal)
    else:
        W2NN_Normal, B2NN_Normal = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2normal, flag_Normal)
    if R['model2Scale'] == 'Fourier_DNN':
        W2NN_Scale, B2NN_Scale = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2scale, flag_Scale)
    else:
        W2NN_Scale, B2NN_Scale = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2scale, flag_Scale)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_it = tf.compat.v1.placeholder(tf.float32, name='X_it', shape=[None, input_dim])
            XY_left = tf.compat.v1.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 2 列
            XY_right = tf.compat.v1.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 2 列
            XY_bottom = tf.compat.v1.placeholder(tf.float32, name='Y_bottom_bd', shape=[None, input_dim])  # * 行 2 列
            XY_top = tf.compat.v1.placeholder(tf.float32, name='Y_top_bd', shape=[None, input_dim])        # * 行 2 列
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            UdotU_penalty = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if R['model2Normal'] == 'DNN':
                UNN_Normal = DNN_base.DNN(XY_it, W2NN_Normal, B2NN_Normal, hidden2normal, activateIn_name=act2Normal, 
                                          activate_name=act2Normal)
                UNN_Left_Normal = DNN_base.DNN(XY_left, W2NN_Normal, B2NN_Normal, hidden2normal,
                                               activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Right_Normal = DNN_base.DNN(XY_right, W2NN_Normal, B2NN_Normal, hidden2normal,
                                                activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Bottom_Normal = DNN_base.DNN(XY_bottom, W2NN_Normal, B2NN_Normal, hidden2normal,
                                                 activateIn_name=act2Normal, activate_name=act2Normal)
                UNN_Top_Normal = DNN_base.DNN(XY_top, W2NN_Normal, B2NN_Normal, hidden2normal,
                                              activateIn_name=act2Normal, activate_name=act2Normal)
            elif R['model2Normal'] == 'Fourier_DNN':
                freq2Normal = R['freq2Normal']
                UNN_Normal = DNN_base.DNN_FourierBase(XY_it, W2NN_Normal, B2NN_Normal, hidden2normal, freq2Normal,
                                                      activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Left_Normal = DNN_base.DNN_FourierBase(XY_left, W2NN_Normal, B2NN_Normal, hidden2normal, freq2Normal,
                                                           activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Right_Normal = DNN_base.DNN_FourierBase(XY_right, W2NN_Normal, B2NN_Normal, hidden2normal, freq2Normal,
                                                            activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Bottom_Normal = DNN_base.DNN_FourierBase(XY_bottom, W2NN_Normal, B2NN_Normal, hidden2normal, freq2Normal,
                                                             activate_name=act2Normal, repeat_Highfreq=False)
                UNN_Top_Normal = DNN_base.DNN_FourierBase(XY_top, W2NN_Normal, B2NN_Normal, hidden2normal, freq2Normal,
                                                          activate_name=act2Normal, repeat_Highfreq=False)

            freq2Scale = R['freq2Scale']
            if R['model2Scale'] == 'Scale_DNN':
                UNN_Scale= DNN_base.DNN_scale(XY_it, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                              activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Left2Scale= DNN_base.DNN_scale(XY_left, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                   activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Right2Scale= DNN_base.DNN_scale(XY_right, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                    activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Bottom2Scale= DNN_base.DNN_scale(XY_bottom, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Top2Scale= DNN_base.DNN_scale(XY_top, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                  activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale'] == 'Adapt_Scale_DNN':
                UNN_Scale = DNN_base.DNN_adapt_scale(XY_it, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Left2Scale = DNN_base.DNN_adapt_scale(XY_left, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                          activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Right2Scale = DNN_base.DNN_adapt_scale(XY_right, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                           activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Bottom2Scale = DNN_base.DNN_adapt_scale(XY_bottom, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                            activateIn_name=act2Scale, activate_name=act2Scale)
                UNN_Top2Scale = DNN_base.DNN_adapt_scale(XY_top, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                         activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale'] == 'Fourier_DNN':
                UNN_Scale = DNN_base.DNN_FourierBase(XY_it, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                     activate_name=act2Scale)
                UNN_Left2Scale = DNN_base.DNN_FourierBase(XY_left, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                          activate_name=act2Scale)
                UNN_Right2Scale = DNN_base.DNN_FourierBase(XY_right, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                           activate_name=act2Scale)
                UNN_Bottom2Scale = DNN_base.DNN_FourierBase(XY_bottom, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                            activate_name=act2Scale)
                UNN_Top2Scale = DNN_base.DNN_FourierBase(XY_top, W2NN_Scale, B2NN_Scale, hidden2scale, freq2Scale,
                                                         activate_name=act2Scale)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])

            UNN = UNN_Normal + alpha*UNN_Scale

            dUNN_Normal = tf.gradients(UNN_Normal, XY_it)[0]  # * 行 2 列
            dUNN_Scale = tf.gradients(UNN_Scale, XY_it)[0]    # * 行 2 列
            if R['loss_type'] == 'L2_loss':
                if R['PDE_type'] == 'Convection_diffusion':
                    a_eps = A_eps(X_it, Y_it)  # * 行 1 列
                    bx = Bx(X_it, Y_it)
                    by = By(X_it, Y_it)
                    dUNNx2Normal = tf.gather(dUNN_Normal, [0], axis=-1)
                    dUNNy2Normal = tf.gather(dUNN_Normal, [1], axis=-1)

                    dUNNxxy2Normal = tf.gradients(dUNNx2Normal, XY_it)[0]
                    dUNNyxy2Normal = tf.gradients(dUNNy2Normal, XY_it)[0]
                    dUNNxx2Normal = tf.gather(dUNNxxy2Normal, [0], axis=-1)
                    dUNNyy2Normal = tf.gather(dUNNyxy2Normal, [1], axis=-1)

                    ddUNN2Normal = tf.add(dUNNxx2Normal, dUNNyy2Normal)

                    dUNNx2freqs = tf.gather(dUNN_Scale, [0], axis=-1)
                    dUNNy2freqs = tf.gather(dUNN_Scale, [1], axis=-1)

                    dUNNxxy2freqs = tf.gradients(dUNNx2freqs, XY_it)[0]
                    dUNNyxy2freqs = tf.gradients(dUNNy2freqs, XY_it)[0]
                    dUNNxx2freqs = tf.gather(dUNNxxy2freqs, [0], axis=-1)
                    dUNNyy2freqs = tf.gather(dUNNyxy2freqs, [1], axis=-1)

                    ddUNN2freqs = tf.add(dUNNxx2freqs, dUNNyy2freqs)

                    ddUNN = tf.add(ddUNN2Normal, alpha*ddUNN2freqs)

                    bdUNN = bx * tf.add(dUNNx2Normal, dUNNx2freqs) + by * tf.add(dUNNy2Normal, dUNNy2freqs)
                    loss2it = -1.0*a_eps * ddUNN + bdUNN - f(X_it, Y_it)
                Loss_it2NN = tf.reduce_mean(tf.square(loss2it))
            elif R['loss_type'] == 'variational_loss':
                # 0.5*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),            grad (Uc+Uf) = grad Uc + grad Uf
                # 0.5*a(x)*|grad (Uc+Uf)|^p - f(x)*(Uc+Uf),       grad (Uc+Uf) = grad Uc + grad Uf
                # dUNN = tf.add(dUNN_Normal, alpha*dUNN_Scale)
                dUNN = dUNN_Normal + alpha * dUNN_Scale
                norm2dUNN = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和

                if R['PDE_type'] == 'general_Laplace':
                    dUNN_2Norm = tf.square(norm2dUNN)
                    loss_it_variational = (1.0 / p_index) * dUNN_2Norm - \
                                          tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it)
                    Kappa = kappa(X_it, Y_it)
                    AdUNN_pNorm = a_eps*tf.pow(norm2dUNN, p_index)
                    # AdUNN_pNorm = tf.multiply(a_eps, tf.pow(norm2dUNN, p_index))
                    KU2 = Kappa*UNN*UNN
                    # KU2 = tf.multiply(Kappa, tf.square(UNN))
                    if R['equa_name'] == 'Boltzmann3' or R['equa_name'] == 'Boltzmann4'\
                            or R['equa_name'] == 'Boltzmann5'or R['equa_name'] == 'Boltzmann6':
                        fxy = MS_BoltzmannEqs.get_foreside2Boltzmann2D(x=X_it, y=Y_it)
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa*UNN*UNN) - \
                                          tf.multiply(tf.reshape(fxy, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + KU2) - \
                                             tf.multiply(f(X_it, Y_it), UNN)
                elif R['PDE_type'] == 'Convection_diffusion':
                    a_eps = A_eps(X_it, Y_it)                            # * 行 1 列
                    bx = Bx(X_it, Y_it)                                  # * 行 1 列
                    by = By(X_it, Y_it)                                  # * 行 1 列
                    dUNN_Normalx = tf.gather(dUNN_Normal, [0], axis=-1)  # * 行 1 列
                    dUNN_Normaly = tf.gather(dUNN_Normal, [1], axis=-1)  # * 行 1 列
                    dUNN_Scalex = tf.gather(dUNN_Scale, [0], axis=-1)    # * 行 1 列
                    dUNN_Scaley = tf.gather(dUNN_Scale, [1], axis=-1)    # * 行 1 列
                    dUNNx = tf.add(dUNN_Normalx, dUNN_Scalex)
                    dUNNy = tf.add(dUNN_Normaly, dUNN_Scaley)
                    bdUNN = bx * dUNNx + by * dUNNy                      # * 行 1 列
                    dAdU_pNorm = a_eps * tf.pow(norm2dUNN, p_index)      # * 行 1 列
                    # loss_it_variational = (dAdU_pNorm + bdUNN*UNN) - tf.multiply(f(X_it, Y_it), UNN)
                    # loss_it_variational = 0.5*(dAdU_pNorm + bdUNN * UNN) - tf.multiply(f(X_it, Y_it), UNN)
                    loss_it_variational = tf.square((dAdU_pNorm + bdUNN * UNN) - tf.multiply(f(X_it, Y_it), UNN))
                else:
                    a_eps = A_eps(X_it, Y_it)                                                   # * 行 1 列
                    # AdUNN_pNorm = tf.multiply(a_eps, tf.pow(norm2dUNN, p_index))
                    AdUNN_pNorm = a_eps*tf.pow(norm2dUNN, p_index)
                    loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - \
                                          tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                Loss_it2NN = tf.reduce_mean(loss_it_variational)

            if R['opt2loss_udotu'] == 'with_orthogonal':
                if R['opt2orthogonal'] == 0:
                    # |Uc*Uf|^2-->0
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
                    U_dot_U = tf.reshape(tf.square(tf.multiply(UNN_Normal, using_scale2orthogonal * UNN_Scale)), shape=[-1, 1])
                    dU_dot_dU = tf.multiply(dUNN_Normal, using_scale2orthogonal * dUNN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
                Loss2UNN_dot_UNN = UdotU_penalty * UNN_dot_UNN
            else:
                Loss2UNN_dot_UNN = tf.constant(0.0)

            U_left = u_left(tf.reshape(XY_left[:, 0], shape=[-1, 1]), tf.reshape(XY_left[:, 1], shape=[-1, 1]))
            U_right = u_right(tf.reshape(XY_right[:, 0], shape=[-1, 1]), tf.reshape(XY_right[:, 1], shape=[-1, 1]))
            U_bottom = u_bottom(tf.reshape(XY_bottom[:, 0], shape=[-1, 1]), tf.reshape(XY_bottom[:, 1], shape=[-1, 1]))
            U_top = u_top(tf.reshape(XY_top[:, 0], shape=[-1, 1]), tf.reshape(XY_top[:, 1], shape=[-1, 1]))

            if R['opt2loss_bd'] == 'unified_boundary':
                UNN_left = UNN_Left_Normal + alpha*UNN_Left2Scale
                UNN_right = UNN_Right_Normal + alpha*UNN_Right2Scale
                UNN_bottom = UNN_Bottom_Normal + alpha*UNN_Bottom2Scale
                UNN_top = UNN_Top_Normal + alpha * UNN_Top2Scale
                Loss_bd2NN = tf.square(UNN_left - U_left) + tf.square(UNN_right - U_right) + \
                             tf.square(UNN_bottom - U_bottom) + tf.square(UNN_top - U_top)
                Loss_bd2NNs = bd_penalty * tf.reduce_mean(Loss_bd2NN)
            else:
                loss_bd_square2Normal = tf.square(UNN_Left_Normal - U_left) + tf.square(UNN_Right_Normal - U_right) + \
                                    tf.square(UNN_Bottom_Normal - U_bottom) + tf.square(UNN_Top_Normal - U_top)
                loss_bd_square2freqs = tf.square(using_scale2boundary * UNN_Left2Scale) + \
                                       tf.square(using_scale2boundary * UNN_Right2Scale) + \
                                       tf.square(using_scale2boundary * UNN_Bottom2Scale) + \
                                       tf.square(using_scale2boundary * UNN_Top2Scale)

                Loss_bd2Normal = tf.reduce_mean(loss_bd_square2Normal)
                Loss_bd2Scale = tf.reduce_mean(loss_bd_square2freqs)
                Loss_bd2NNs = bd_penalty * (Loss_bd2Normal + Loss_bd2Scale)

            if R['regular_wb_model'] == 'L1':
                regularSum2WBNormal = DNN_base.regular_weights_biases_L1(W2NN_Normal, B2NN_Normal)   # 正则化权重和偏置 L1正则化
                regularSum2WBScale = DNN_base.regular_weights_biases_L1(W2NN_Scale, B2NN_Scale)
            elif R['regular_wb_model'] == 'L2':
                regularSum2WBNormal = DNN_base.regular_weights_biases_L2(W2NN_Normal, B2NN_Normal)   # 正则化权重和偏置 L2正则化
                regularSum2WBScale = DNN_base.regular_weights_biases_L2(W2NN_Scale, B2NN_Scale)
            else:
                regularSum2WBNormal = tf.constant(0.0)                                               # 无正则化权重参数
                regularSum2WBScale = tf.constant(0.0)

            PWB = penalty2WB * (regularSum2WBNormal + regularSum2WBScale)
            Loss2NN = Loss_it2NN + Loss_bd2NNs + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 1:
                train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                train_loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
            elif R['train_model'] == 2:
                train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                train_loss2NN = tf.group(train_op1, train_op2, train_op3)
            else:
                train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' \
                    or R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'Convection_diffusion':
                Utrue = u_true(X_it, Y_it)

                train_mse2NN = tf.reduce_mean(tf.square(Utrue - UNN))
                train_rel2NN = train_mse2NN / tf.reduce_mean(tf.square(Utrue))
            else:
                train_mse2NN = tf.constant(0.0)
                train_rel2NN = tf.constant(0.0)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, loss_udu_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        test_bach_size = 900
        size2test = 30
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        test_xy_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    else:
        if R['PDE_type'] == 'pLaplace_implicit':
            test_xy_bach = matData2Laplace.get_meshData2Laplace(equation_name=R['equa_name'], mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
        elif R['PDE_type'] == 'Possion_Boltzmann':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
        elif R['PDE_type'] == 'Convection_diffusion':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
        else:
            test_xy_bach = matData2Laplace.get_randData2Laplace(dim=input_dim, data_path='dataMat_highDim')
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = \
                DNN_data.rand_bd_2D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
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
            else:
                temp_penalty_bd = bd_penalty_init

            if R['activate_powSolus_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_powU = init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_powU = 10 * init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_powU = 50*init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_powU = 100*init_UdotU_penalty
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_powU = 200*init_UdotU_penalty
                else:
                    temp_penalty_powU = 500*init_UdotU_penalty
            else:
                temp_penalty_powU = init_UdotU_penalty

            _, loss_it_nn, loss_bd_nn, loss_nn, udu_nn, train_mse_nn, train_rel_nn, pwb = sess.run(
                [train_loss2NN, Loss_it2NN, Loss_bd2NNs, Loss2NN, UNN_dot_UNN, train_mse2NN, train_rel2NN, PWB],
                feed_dict={XY_it: xy_it_batch, XY_left: xl_bd_batch, XY_right: xr_bd_batch,
                           XY_bottom: yb_bd_batch, XY_top: yt_bd_batch, in_learning_rate: tmp_lr,
                           bd_penalty: temp_penalty_bd, UdotU_penalty: temp_penalty_powU})
            loss_it_all.append(loss_it_nn)
            loss_bd_all.append(loss_bd_nn)
            loss_all.append(loss_nn)
            loss_udu_all.append(udu_nn)
            train_mse_all.append(train_mse_nn)
            train_rel_all.append(train_rel_nn)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, pwb, loss_it_nn, loss_bd_nn,
                    loss_nn, udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' or \
                        R['PDE_type'] == 'Possion_Boltzmann'or R['PDE_type'] == 'Convection_diffusion':
                    u_true2test, utest_nn, utest_normal, utest_freqs = sess.run(
                        [Utrue, UNN, UNN_Normal, alpha*UNN_Scale], feed_dict={XY_it: test_xy_bach})
                else:
                    u_true2test = u_true
                    utest_nn, utest_normal, utest_freqs = sess.run(
                        [UNN, UNN_Normal, alpha*UNN_Scale], feed_dict={XY_it: test_xy_bach})
                point_ERR2NN = np.square(u_true2test - utest_nn)
                test_mse2nn = np.mean(point_ERR2NN)
                test_mse_all.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout)

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act2Normal,
                                             outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act2Normal, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_udu_all, lossType='udu', seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act2Normal, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------- save test data to mat file and plot the testing results into figures -----------------------
        if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'pLaplace_explicit':
            saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])

        saveData.save_testData_or_solus2mat(utest_nn, dataName='test', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_normal, dataName='normal', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_freqs, dataName='scale', outPath=R['FolderName'])

        if R['hot_power'] == 0:
            plotData.plot_scatter_solutions2test(u_true2test, utest_nn, test_xy_bach, actName1='Utrue',
                                                 actName2=act2Normal, seedNo=R['seed'], outPath=R['FolderName'])
        elif R['hot_power'] == 1:
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                            outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(utest_nn, size_vec2mat=size2test, actName=act2Normal, seedNo=R['seed'],
                                            outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act2Normal, outPath=R['FolderName'])
        saveData.save_test_point_wise_err2mat(point_ERR2NN, actName=act2Normal, outPath=R['FolderName'])

        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act2Normal, seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        plotData.plot_Hot_point_wise_err(point_ERR2NN, size_vec2mat=size2test, actName=act2Normal,
                                         seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
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
    # store_file = 'Laplace2D'
    store_file = 'pLaplace2D'
    # store_file = 'Boltzmann2D'
    # store_file = 'Convection2D'
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
    R['input_dim'] = 2                  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                 # 输出维数

    if store_file == 'Laplace2D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace2D':
        R['PDE_type'] = 'pLaplace_implicit'
        # R['equa_name'] = 'multi_scale2D_1'      # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'multi_scale2D_2'      # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'multi_scale2D_3'      # p=2 区域为 [-1,1]X[-1,1]
        R['equa_name'] = 'multi_scale2D_4'      # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'multi_scale2D_5'      # p=3 区域为 [0,1]X[0,1] 和例三的系数一样
        # R['equa_name'] = 'multi_scale2D_6'      # p=3 区域为 [-1,1]X[-1,1] 和例三的系数一样
    elif store_file == 'Boltzmann2D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'           # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'Boltzmann2'             # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'Boltzmann3'
        # R['equa_name'] = 'Boltzmann4'
        R['equa_name'] = 'Boltzmann5'
        # R['equa_name'] = 'Boltzmann6'
    elif store_file == 'Convection2D':
        R['PDE_type'] = 'Convection_diffusion'
        # R['equa_name'] = 'Convection1'
        R['equa_name'] = 'Convection2'

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 3000            # 内部训练数据的批大小
        R['batch_size2boundary'] = 500
    elif R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'Possion_Boltzmann'\
            or R['PDE_type'] == 'Convection_diffusion':
        # 频率设置
        mesh_number = input('please input mesh_number =')  # 由终端输入的会记录为字符串形式
        R['mesh_number'] = int(mesh_number)  # 字符串转为浮点

        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2pLaplace_operator'] = order

        R['batch_size2interior'] = 3000  # 内部训练数据的批大小
        if R['mesh_number'] == 2:
            R['batch_size2boundary'] = 25  # 边界训练数据的批大小
        elif R['mesh_number'] == 3:
            R['batch_size2boundary'] = 100  # 边界训练数据的批大小
        elif R['mesh_number'] == 4:
            R['batch_size2boundary'] = 200  # 边界训练数据的批大小
        elif R['mesh_number'] == 5:
            R['batch_size2boundary'] = 300  # 边界训练数据的批大小
        elif R['mesh_number'] == 6:
            R['batch_size2boundary'] = 500  # 边界训练数据的批大小
        elif R['mesh_number'] == 7:
            R['batch_size2boundary'] = 500  # 边界训练数据的批大小
    elif R['PDE_type'] == 'pLaplace_explicit':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)              # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2pLaplace_operator'] = order
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 3000            # 内部训练数据的批大小
        R['batch_size2boundary'] = 500             # 边界训练数据的批大小
        
    R['testData_model'] = 'loadData'
    # ---------------------------- Setup of DNN -------------------------------
    # R['loss_type'] = 'L2_loss'                 # PDE-L2loss 1: grad U = grad Uc + grad Uf;
    R['loss_type'] = 'variational_loss'        # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式不是分开的
    # R['loss_type'] = 'variational_loss2'       # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['loss_type'] = 'variational_loss3'       # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['loss_type'] = 'variational_loss4'      # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的

    R['opt2orthogonal'] = 0                    # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 1                  # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 2                  # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy

    R['hot_power'] = 1

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'

    R['penalty2weight_biases'] = 0.000                   # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                 # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0001                # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100                    # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['init_penalty2orthogonal'] = 5.0
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器
    R['train_model'] = 'training_union'  # 训练模式, 一个 loss 联结训练
    # R['train_model'] = 'training_group1'          # 训练模式, 多个 loss 组团训练
    # R['train_model'] = 'training_group2'
    # R['train_model'] = 'training_group3'
    # R['train_model'] = 'training_group4'

    # R['model2Normal'] = 'DNN'
    # R['model2Normal'] = 'Scale_DNN'
    # R['model2Normal'] = 'Adapt_Scale_DNN'
    R['model2Normal'] = 'Fourier_DNN'

    # R['model2Scale'] = 'DNN'
    # R['model2Scale'] = 'Scale_DNN'
    # R['model2Scale'] = 'Adapt_Scale_DNN'
    R['model2Scale'] = 'Fourier_DNN'

    # 单纯的 MscaleDNN 网络 FourierBase(125, 200, 200, 100, 100, 80) 125+250*200+200*200+200*100+100*100+100*80+80=128205
    # 单纯的 MscaleDNN 网络 GeneralBase(250, 200, 200, 100, 100, 80) 250+250*200+200*200+200*100+100*100+100*80+80=128330
    # FourierBase normal 和 FourierBase scale 网络的总参数数目:35710 + 91700 = 127410
    # GeneralBase normal 和 FourierBase scale 网络的总参数数目:35760 + 91820 = 127580
    if R['model2Normal'] == 'Fourier_DNN':
        R['hidden2normal'] = (50, 100, 80, 80, 80, 60)  # 1*50+100*100+100*80+80*80+80*80+80*60+60*1=35710
    else:
        R['hidden2normal'] = (100, 100, 80, 80, 80, 60)  # 1*100+100*100+100*100+100*80+80*80+80*60+60*1=35760
        # R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)

    if R['model2Scale'] == 'Fourier_DNN':
        R['hidden2scale'] = (120, 150, 150, 100, 100, 80)  # 1*120+240*150+150*150+150*100+100*100+100*80+80*1 = 91700
    else:
        R['hidden2scale'] = (240, 150, 150, 100, 100, 80)  # 1*240+240*150+150*150+150*100+100*100+100*80+80*1 = 91820
        # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)

    # R['freq2Normal'] = np.arange(10, 100)
    # R['freq2Normal'] = np.concatenate(([1, 1, 1, 1, 1], np.arange(1, 26)), axis=0)
    # R['freq2Normal'] = np.concatenate(([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], np.arange(1, 21)), axis=0)
    # R['freq2Normal'] = np.concatenate(([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], np.arange(1, 26)), axis=0)
    # R['freq2Normal'] = np.concatenate(([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], np.arange(1, 31)), axis=0)
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
    # R['actName2Normal'] = 'relu'
    R['actName2Normal'] = 'tanh'
    # R['actName2Normal'] = 'srelu'
    # R['actName2Normal'] = 'sin'
    # R['actName2Normal'] = 's2relu'

    if R['model2Normal'] == 'Fourier_DNN':
        # R['actHidden_name2Normal'] = 's2relu'
        R['actName2Normal'] = 'tanh'

    # R['actName2Scale'] = 'relu'
    # R['actName2Scale']' = leaky_relu'
    # R['actName2Scale'] = 'srelu'
    R['actName2Scale'] = 's2relu'
    # R['actName2Scale'] = 'tanh'
    # R['actName2Scale'] = 'elu'
    # R['actName2Scale'] = 'phi'

    if R['loss_type'] == 'L2_loss':
        R['actHidden_name2Scale'] = 'tanh'

    if R['loss_type'] == 'variational_loss2':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss3':
        # R['contrib2scale'] = 0.1
        # R['contrib2scale'] = 0.05
        R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss4':
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    else:
        R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.025
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005

    R['opt2loss_udotu'] = 'with_orthogonal'
    # R['opt2loss_udotu'] = 'without_orthogonal'

    # R['contrib_scale2orthogonal'] = 'with_contrib'
    R['contrib_scale2orthogonal'] = 'without_contrib'

    # R['opt2loss_bd'] = 'unified_boundary'
    R['opt2loss_bd'] = 'individual_boundary'

    # R['contrib_scale2boundary'] = 'with_contrib'
    R['contrib_scale2boundary'] = 'without_contrib'
    
    solve_Multiscale_PDE(R)

