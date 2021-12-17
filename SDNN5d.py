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
import DNN_Print_Log
import General_Laplace
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import Load_data2Mat
import DNN_data
import saveData
import plotData


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout_NN = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.dictionary2file(R, log_fileout_NN, actName2normal=R['act2normal'], actName2scale=R['act2scale'])

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']           # Regularization parameter for boundary conditions
    init_penalty2powU = R['balance2solus']
    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden2normal = R['hidden2normal']
    hidden2scale = R['hidden2scale']
    act_func1 = R['act2normal']
    act_func2 = R['act2scale']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    alpha = R['contrib2scale']

    # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = General_Laplace.get_infos2Laplace_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2laplace']
        mesh_number = R['mesh_number']
        u_true, f, A_eps, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = MS_LaplaceEqs.get_infos2pLaplace_5D(
                input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
                equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  + K *u_eps =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2laplace']
        u_true, f, A_eps, kappa, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = MS_BoltzmannEqs.get_infos2Boltzmann_5D(
             intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Convection_diffusion':
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2laplace']
        mesh_number = R['mesh_number']
        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_5D(
            input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None)

    flag2Normal = 'WB2normal'
    flag2Scale = 'WB2scale'
    if R['model2normal'] == 'DNN_FourierBase':
        Ws_Normal, B_Normal = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2normal, flag2Normal)
    else:
        Ws_Normal, B_Normal = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2normal, flag2Normal)
    if R['model2scale'] == 'DNN_FourierBase':
        Ws_Scale, B_Scale = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2scale, flag2Scale)
    else:
        Ws_Scale, B_Scale = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2scale, flag2Scale)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZST_it = tf.compat.v1.placeholder(tf.float32, name='XYZST_it', shape=[None, input_dim])
            XYZST00 = tf.compat.v1.placeholder(tf.float32, name='XYZST00', shape=[None, input_dim])
            XYZST01 = tf.compat.v1.placeholder(tf.float32, name='XYZST01', shape=[None, input_dim])
            XYZST10 = tf.compat.v1.placeholder(tf.float32, name='XYZST10', shape=[None, input_dim])
            XYZST11 = tf.compat.v1.placeholder(tf.float32, name='XYZST11', shape=[None, input_dim])
            XYZST20 = tf.compat.v1.placeholder(tf.float32, name='XYZST20', shape=[None, input_dim])
            XYZST21 = tf.compat.v1.placeholder(tf.float32, name='XYZST21', shape=[None, input_dim])
            XYZST30 = tf.compat.v1.placeholder(tf.float32, name='XYZST30', shape=[None, input_dim])
            XYZST31 = tf.compat.v1.placeholder(tf.float32, name='XYZST31', shape=[None, input_dim])
            XYZST40 = tf.compat.v1.placeholder(tf.float32, name='XYZST40', shape=[None, input_dim])
            XYZST41 = tf.compat.v1.placeholder(tf.float32, name='XYZST41', shape=[None, input_dim])
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e2, shape=[], name='bd_p')
            penalty2powU = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.compat.v1.placeholder_with_default(input=True, shape=[], name='train_opt')
            if R['model2normal'] == 'DNN':
                UNN_Normal = DNN_base.DNN(XYZST_it, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U00_NN_Normal = DNN_base.DNN(XYZST00, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U01_NN_Normal = DNN_base.DNN(XYZST01, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U10_NN_Normal = DNN_base.DNN(XYZST10, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U11_NN_Normal = DNN_base.DNN(XYZST11, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U20_NN_Normal = DNN_base.DNN(XYZST20, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U21_NN_Normal = DNN_base.DNN(XYZST21, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U30_NN_Normal = DNN_base.DNN(XYZST30, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U31_NN_Normal = DNN_base.DNN(XYZST31, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U40_NN_Normal = DNN_base.DNN(XYZST40, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
                U41_NN_Normal = DNN_base.DNN(XYZST41, Ws_Normal, B_Normal, hidden2normal, activate_name=act_func1)
            elif R['model2normal'] == 'DNN_FourierBase':
                freq2Normal = R['freq2Normal']
                UNN_Normal = DNN_base.DNN_FourierBase(XYZST_it, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                      activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U00_NN_Normal = DNN_base.DNN_FourierBase(XYZST00, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U01_NN_Normal = DNN_base.DNN_FourierBase(XYZST01, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U10_NN_Normal = DNN_base.DNN_FourierBase(XYZST10, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U11_NN_Normal = DNN_base.DNN_FourierBase(XYZST11, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U20_NN_Normal = DNN_base.DNN_FourierBase(XYZST20, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U21_NN_Normal = DNN_base.DNN_FourierBase(XYZST21, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U30_NN_Normal = DNN_base.DNN_FourierBase(XYZST30, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U31_NN_Normal = DNN_base.DNN_FourierBase(XYZST31, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U40_NN_Normal = DNN_base.DNN_FourierBase(XYZST40, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])
                U41_NN_Normal = DNN_base.DNN_FourierBase(XYZST41, Ws_Normal, B_Normal, hidden2normal, freq2Normal,
                                                         activate_name=act_func1, repeat_Highfreq=R['repeat_high_freq'])

            freqs = R['freq2Scale']
            if R['model2scale'] == 'DNN_scale':
                UNN_Scale = DNN_base.DNN_scale(XYZST_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U00_NN_Scale = DNN_base.DNN_scale(XYZST00, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U01_NN_Scale = DNN_base.DNN_scale(XYZST01, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U10_NN_Scale = DNN_base.DNN_scale(XYZST10, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U11_NN_Scale = DNN_base.DNN_scale(XYZST11, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U20_NN_Scale = DNN_base.DNN_scale(XYZST20, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U21_NN_Scale = DNN_base.DNN_scale(XYZST21, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U30_NN_Scale = DNN_base.DNN_scale(XYZST30, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U31_NN_Scale = DNN_base.DNN_scale(XYZST31, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U40_NN_Scale = DNN_base.DNN_scale(XYZST40, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U41_NN_Scale = DNN_base.DNN_scale(XYZST41, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'DNN_adapt_scale':
                UNN_Scale = DNN_base.DNN_adapt_scale(XYZST_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U00_NN_Scale = DNN_base.DNN_adapt_scale(XYZST00, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U01_NN_Scale = DNN_base.DNN_adapt_scale(XYZST01, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U10_NN_Scale = DNN_base.DNN_adapt_scale(XYZST10, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U11_NN_Scale = DNN_base.DNN_adapt_scale(XYZST11, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U20_NN_Scale = DNN_base.DNN_adapt_scale(XYZST20, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U21_NN_Scale = DNN_base.DNN_adapt_scale(XYZST21, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U30_NN_Scale = DNN_base.DNN_adapt_scale(XYZST30, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U31_NN_Scale = DNN_base.DNN_adapt_scale(XYZST31, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U40_NN_Scale = DNN_base.DNN_adapt_scale(XYZST40, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U41_NN_Scale = DNN_base.DNN_adapt_scale(XYZST41, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'DNN_FourierBase':
                UNN_Scale = DNN_base.DNN_FourierBase(XYZST_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U00_NN_Scale = DNN_base.DNN_FourierBase(XYZST00, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U01_NN_Scale = DNN_base.DNN_FourierBase(XYZST01, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U10_NN_Scale = DNN_base.DNN_FourierBase(XYZST10, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U11_NN_Scale = DNN_base.DNN_FourierBase(XYZST11, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U20_NN_Scale = DNN_base.DNN_FourierBase(XYZST20, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U21_NN_Scale = DNN_base.DNN_FourierBase(XYZST21, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U30_NN_Scale = DNN_base.DNN_FourierBase(XYZST30, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U31_NN_Scale = DNN_base.DNN_FourierBase(XYZST31, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U40_NN_Scale = DNN_base.DNN_FourierBase(XYZST40, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U41_NN_Scale = DNN_base.DNN_FourierBase(XYZST41, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
            elif R['model2scale'] == 'DNN_Sin+Cos_Base':
                UNN_Scale = DNN_base.DNN_SinAddCos(XYZST_it, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U00_NN_Scale = DNN_base.DNN_SinAddCos(XYZST00, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U01_NN_Scale = DNN_base.DNN_SinAddCos(XYZST01, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U10_NN_Scale = DNN_base.DNN_SinAddCos(XYZST10, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U11_NN_Scale = DNN_base.DNN_SinAddCos(XYZST11, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U20_NN_Scale = DNN_base.DNN_SinAddCos(XYZST20, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U21_NN_Scale = DNN_base.DNN_SinAddCos(XYZST21, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U30_NN_Scale = DNN_base.DNN_SinAddCos(XYZST30, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U31_NN_Scale = DNN_base.DNN_SinAddCos(XYZST31, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U40_NN_Scale = DNN_base.DNN_SinAddCos(XYZST40, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)
                U41_NN_Scale = DNN_base.DNN_SinAddCos(XYZST41, Ws_Scale, B_Scale, hidden2scale, freqs, activate_name=act_func2)

            X_it = tf.reshape(XYZST_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZST_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZST_it[:, 2], shape=[-1, 1])
            S_it = tf.reshape(XYZST_it[:, 3], shape=[-1, 1])
            T_it = tf.reshape(XYZST_it[:, 4], shape=[-1, 1])

            UNN = UNN_Normal + alpha * UNN_Scale
            dUNN_Normal = tf.gradients(UNN_Normal, XYZST_it)[0]  # * 行 2 列
            dUNN_Scale = tf.gradients(UNN_Scale, XYZST_it)[0]  # * 行 2 列

            if R['variational_loss'] == 1:
                dUNN = tf.add(dUNN_Normal, alpha * dUNN_Scale)
                norm2dUNN = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    laplace_pow_Normal = tf.square(norm2dUNN)
                    loss_it_variational2NN = (1.0 / 2) *laplace_pow_Normal - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    AdUNN_pNorm = a_eps*tf.pow(norm2dUNN, p_index)
                    if R['equa_name'] == 'multi_scale5D_5' or R['equa_name'] == 'multi_scale5D_8' or \
                            R['equa_name'] == 'multi_scale5D_9':
                        fxyzst = MS_LaplaceEqs.get_forceSide2pLaplace5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it)
                        loss_it_variational2NN = (1.0 / p_index) * AdUNN_pNorm - \
                                              tf.multiply(tf.reshape(fxyzst, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational2NN = (1.0 / p_index) * AdUNN_pNorm - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    Kappa = kappa(X_it, Y_it, Z_it, S_it, T_it)  # * 行 1 列
                    AdUNN_pNorm = a_eps * tf.pow(norm2dUNN, p_index)
                    if R['equa_name'] == 'multi_scale5D_4' or R['equa_name'] == 'multi_scale5D_5' or \
                            R['equa_name'] == 'multi_scale5D_6' or R['equa_name'] == 'multi_scale5D_7':
                        fxyzst = MS_BoltzmannEqs.get_forceSide2Boltzmann_5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it,
                                                                            equa_name=R['equa_name'])
                        loss_it_variational2NN = (1.0 / p_index) * (AdUNN_pNorm + Kappa * UNN * UNN) - \
                                             tf.multiply(fxyzst, UNN)
                    else:
                        loss_it_variational2NN = (1.0 / p_index) * (AdUNN_pNorm + Kappa * UNN * UNN) - \
                                             tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0
                    norm2UdU = tf.reshape(tf.square(tf.multiply(UNN_Normal, alpha*UNN_Scale)), shape=[-1, 1])
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU)
                elif R['wavelet'] == 2:
                    # |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    dU_dot_dU = tf.multiply(dUNN_Normal, alpha*dUNN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    # norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
                else:
                    # |Uc*Uf|^2 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                    # |Uc*Uf|^2 + |(grad Uc)*(grad Uf)|^2-->0
                    U_dot_U = tf.reduce_mean(tf.square(tf.multiply(UNN_Normal, UNN_Scale)), axis=0)
                    dU_dot_dU = tf.multiply(dUNN_Normal, dUNN_Scale)
                    sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                    # norm2AdUdU = tf.square(tf.multiply(a_eps, sum2dUdU))
                    norm2AdUdU = tf.square(sum2dUdU)
                    UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + U_dot_U
            elif R['variational_loss'] == 2:
                # 0.5*|grad Uc|^p + 0.5*|grad Uf|^p - f(x)*(Uc+Uf)
                # 0.5*a(x)*|grad Uc|^p + 0.5*a(x)*|grad Uf|^p - f(x)*(Uc+Uf)
                norm2dUNN_Normal = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN_Normal), axis=-1)), shape=[-1, 1])
                norm2dUNN_Scale = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN_Scale), axis=-1)), shape=[-1, 1])
                if R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)
                    Kappa = kappa(X_it, Y_it, Z_it, S_it, T_it)
                    ApNorm2dUNN = a_eps * tf.pow(norm2dUNN_Normal, p_index) + \
                                  a_eps * tf.pow(alpha*norm2dUNN_Scale, p_index)
                    if R['equa_name'] == 'multi_scale5D_4' or R['equa_name'] == 'multi_scale5D_5' or \
                            R['equa_name'] == 'multi_scale5D_6' or R['equa_name'] == 'multi_scale5D_7':
                        fxyzst = MS_BoltzmannEqs.get_forceSide2Boltzmann_5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it,
                                                                            equa_name=R['equa_name'])
                        loss_it_variational2NN = (1.0 / p_index) * (ApNorm2dUNN + Kappa * UNN * UNN) - \
                                                 tf.multiply(fxyzst, UNN)
                    else:
                        loss_it_variational2NN = (1.0 / p_index) * (ApNorm2dUNN + Kappa * UNN * UNN) - \
                                             tf.multiply(f(X_it, Y_it, Z_it, S_it), UNN)
                else:
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)  # * 行 1 列
                    ApNorm2dUNN = a_eps * tf.pow(norm2dUNN_Normal, p_index) + \
                                  a_eps * tf.pow(alpha * norm2dUNN_Scale, p_index)
                    if R['equa_name'] == 'multi_scale5D_4' or R['equa_name'] == 'multi_scale5D_8' or \
                            R['equa_name'] == 'multi_scale5D_9':
                        fxyzst = MS_LaplaceEqs.get_forceSide2pLaplace5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it)
                        loss_it_variational2NN = (1.0 / p_index) * ApNorm2dUNN - \
                                              tf.multiply(tf.reshape(fxyzst, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational2NN = (1.0 / p_index) * ApNorm2dUNN - \
                                                 tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)

                if R['wavelet'] == 1:
                    # |Uc*Uf|^2-->0
                    norm2UdU = tf.reshape(tf.square(tf.multiply(UNN_Normal, UNN_Scale)), shape=[-1, 1])
                    UNN_dot_UNN = tf.reduce_mean(norm2UdU)
                else:
                    UNN_dot_UNN = tf.constant(0.0)

            Loss_it2NN = tf.reduce_mean(loss_it_variational2NN)

            Loss2UNN_dot_UNN = penalty2powU * UNN_dot_UNN

            loss_bd_square2Normal = tf.square(U00_NN_Normal) + tf.square(U01_NN_Normal) + tf.square(U10_NN_Normal) + \
                                   tf.square(U11_NN_Normal) + tf.square(U20_NN_Normal) + tf.square(U21_NN_Normal) + \
                                   tf.square(U30_NN_Normal) + tf.square(U31_NN_Normal) + tf.square(U40_NN_Normal) + \
                                    tf.square(U41_NN_Normal)
            loss_bd_square2Scale = tf.square(U00_NN_Scale) + tf.square(U01_NN_Scale) + tf.square(U10_NN_Scale) + \
                                   tf.square(U11_NN_Scale) + tf.square(U20_NN_Scale) + tf.square(U21_NN_Scale) + \
                                   tf.square(U30_NN_Scale) + tf.square(U31_NN_Scale) + tf.square(U40_NN_Scale) + \
                                   tf.square(U41_NN_Scale)
            Loss_bd2Normal = bd_penalty*tf.reduce_mean(loss_bd_square2Normal)
            Loss_bd2Scaale = bd_penalty*tf.reduce_mean(loss_bd_square2Scale)
            Loss_bds = Loss_bd2Normal + Loss_bd2Scaale

            if R['regular_weight_model'] == 'L1':
                regular_WB_Normal = DNN_base.regular_weights_biases_L1(Ws_Normal, B_Normal)    # 正则化权重和偏置 L1正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L1(Ws_Scale, B_Scale)
            elif R['regular_weight_model'] == 'L2':
                regular_WB_Normal = DNN_base.regular_weights_biases_L2(Ws_Normal, B_Normal)    # 正则化权重和偏置 L2正则化
                regular_WB_Scale = DNN_base.regular_weights_biases_L2(Ws_Scale, B_Scale)
            else:
                regular_WB_Normal = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB_Scale = tf.constant(0.0)

            PWB = wb_regular * (regular_WB_Normal + regular_WB_Scale)

            # 要优化的loss function
            Loss2NN = Loss_it2NN + Loss_bds + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['variational_loss'] == 1 or R['variational_loss'] == 0:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_opt'] == 2:
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            elif R['variational_loss'] == 2:
                if R['train_opt'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_opt'] == 2:
                    train_sin_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_sin_op2 = my_optimizer.minimize(Loss_bds, global_step=global_steps)
                    train_loss2NN = tf.group(train_sin_op1, train_sin_op2)
                else:
                    train_loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
                # 训练上的真解值和训练结果的误差
                U_true = u_true(X_it, Y_it, Z_it, S_it, T_it)

                train_mse2NN = tf.reduce_mean(tf.square(U_true - UNN))
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
        # 画网格热力解图 ---- 生成测试数据，用于测试训练后的网络
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
        test_xyzst_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyzst_bach = Load_data2Mat.get_randomData2mat(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = learning_rate
        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            xyzst_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyzst00_batch, xyzst01_batch, xyzst10_batch, xyzst11_batch, xyzst20_batch, xyzst21_batch, xyzst30_batch,\
            xyzst31_batch, xyzst40_batch, xyzst41_batch = DNN_data.rand_bd_5D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
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

            _, loss_it_nn, loss_bd_nn, loss_nn, udu_nn, train_mse_nn, train_rel_nn, pwb = sess.run(
                [train_loss2NN, Loss_it2NN, Loss_bds, Loss2NN, UNN_dot_UNN, train_mse2NN, train_rel2NN, PWB],
                feed_dict={XYZST_it: xyzst_it_batch, XYZST00: xyzst00_batch, XYZST01: xyzst01_batch,
                           XYZST10: xyzst10_batch, XYZST11: xyzst11_batch, XYZST20: xyzst20_batch,
                           XYZST21: xyzst21_batch, XYZST30: xyzst30_batch, XYZST31: xyzst31_batch,
                           XYZST40: xyzst40_batch, XYZST41: xyzst41_batch,
                           bd_penalty: temp_penalty_bd,  penalty2powU: temp_penalty_powU, train_opt: train_option})
            lossIt_all2NN.append(loss_it_nn)
            lossBD_all2NN.append(loss_bd_nn)
            loss_all2NN.append(loss_nn)
            UDU_NN.append(udu_nn)
            train_mse_all2NN.append(train_mse_nn)
            train_rel_all2NN.append(train_rel_nn)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, pwb, loss_it_nn, loss_bd_nn,
                    loss_nn, udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout_NN)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, utest_nn, utest_normal, utest_freqs = sess.run(
                    [U_true, UNN, UNN_Normal, alpha*UNN_Scale], feed_dict={XYZST_it: test_xyzst_bach, train_opt: train_option})

                point_ERR2NN = np.square(u_true2test - utest_nn)
                test_mse2nn = np.mean(point_ERR2NN)
                test_mse_all2NN.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all2NN.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout_NN)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN, lossBD_all2NN, loss_all2NN, actName=act_func1,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all2NN, train_rel_all2NN, actName=act_func1, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(lossIt_all2NN, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(lossBD_all2NN, lossType='loss_bd', seedNo=R['seed'],
                                      outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all2NN, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(UDU_NN, lossType='udu', seedNo=R['seed'], outPath=R['FolderName'])

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all2NN, train_rel_all2NN, actName=act_func1,
                                         seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

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

    saveData.save_testMSE_REL2mat(test_mse_all2NN, test_rel_all2NN, actName=act_func1,
                                  outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all2NN, test_rel_all2NN, test_epoch, actName=act_func1,
                              seedNo=R['seed'],
                              outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_ERR2NN, actName=act_func1, outPath=R['FolderName'])

    plotData.plot_Hot_point_wise_err(point_ERR2NN, size_vec2mat=size2test, actName=act_func1,
                                     seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
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
    # store_file = 'Laplace5D'
    store_file = 'pLaplace5D'
    # store_file = 'Boltzmann5D'
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
    R['input_dim'] = 5                # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1               # 输出维数

    if store_file == 'Laplace5D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'.
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace5D':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale5D_1'  # general laplace
        # R['equa_name'] = 'multi_scale5D_2'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_3'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_4'    # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_5'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_6'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_7'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_8'  # multi-scale laplace
        R['equa_name'] = 'multi_scale5D_9'  # multi-scale laplace
    elif store_file == 'Boltzmann5D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'multi_scale5D_4'
        # R['equa_name'] = 'multi_scale5D_5'
        # R['equa_name'] = 'multi_scale5D_6'
        R['equa_name'] = 'multi_scale5D_7'

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 12500  # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000  # 内部训练数据的批大小
        # R['batch_size2boundary'] = 1500
        R['batch_size2boundary'] = 2000
    elif R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 12500  # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000  # 内部训练数据的批大小
        # R['batch_size2boundary'] = 1500
        R['batch_size2boundary'] = 2000

    # ---------------------------- Setup of DNN -------------------------------
    R['testData_model'] = 'loadData'

    R['variational_loss'] = 1           # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['variational_loss'] = 2         # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    R['wavelet'] = 1                    # 0: L2 wavelet+energy    1: L2 wavelet     2:energy
    # R['wavelet'] = 2                  # 0: L2 wavelet+energy    1: L2 wavelet     2:energy

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 2e-4  # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_opt'] = 0
    # R['train_opt'] = 1
    # R['train_opt'] = 3

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000      # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                   # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100       # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['balance2solus'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['balance2solus'] = 10000.0
    else:
        R['balance2solus'] = 20.0
        # R['balance2solus'] = 0.0

    # R['model2normal'] = 'DNN'  # 使用的网络模型
    # R['model2normal'] = 'DNN_scale'
    # R['model2normal'] = 'DNN_adapt_scale'
    R['model2normal'] = 'DNN_FourierBase'

    # R['model2scale'] = 'DNN'                         # 使用的网络模型
    # R['model2scale'] = 'DNN_scale'
    # R['model2scale'] = 'DNN_adapt_scale'
    # R['model2scale'] = 'DNN_Sin+Cos_Base'
    R['model2scale'] = 'DNN_FourierBase'

    # 单纯的 MscaleDNN 网络 FourierBase(250,400,400,300,300,200) 250+500*400+400*400+400*300+300*300+300*200+200=630450
    # 单纯的 MscaleDNN 网络 GeneralBase(500,400,400,300,300,200) 500+500*400+400*400+400*300+300*300+300*200+200=630700
    # FourierBase normal 和 FourierBase scale 网络的总参数数目:200730 + 422950 = 623680
    # GeneralBase normal 和 FourierBase scale 网络的总参数数目:200810 + 423200 = 624101
    if R['model2normal'] == 'DNN_FourierBase':
        R['hidden2normal'] = (80, 300, 200, 200, 150, 150)  # 80+160*300+300*200+200*200+200*150+150*150+150=200730
    else:
        R['hidden2normal'] = (160, 300, 200, 200, 150, 150)  # 160+160*300+300*200+200*200+200*150+150*150+150=200810
        # R['hidden2normal'] = (250, 300, 250, 200, 200, 100)  # 260350
        # R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)
        # R['hidden2normal'] = (500, 400, 300, 300, 200, 100)

    if R['model2scale'] == 'DNN_FourierBase':
        # R['hidden2scale'] = (250, 400, 200, 150, 150, 100)
        # R['hidden2scale'] = (250, 400, 300, 200, 200, 100)
        # R['hidden2scale'] = (250, 400, 350, 200, 200, 150)
        R['hidden2scale'] = (250, 360, 250, 250, 200, 200)  # 250+500*360+360*250+250*250+250*200+200*200+200 =422950
        # R['hidden2scale'] = (350, 300, 300, 250, 250, 150)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)
    else:
        # R['hidden2scale'] = (12, 10, 8, 8, 6)
        # R['hidden2scale'] = (100, 80, 60, 60, 40, 40, 20)
        # R['hidden2scale'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2scale'] = (400, 300, 300, 250, 250, 150)
        # R['hidden2scale'] = (500, 400, 200, 150, 150, 100)
        R['hidden2scale'] = (500, 360, 250, 250, 200, 200)  # 250+500*360+360*250+250*250+250*200+200*200+200 =423200
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 200, 200, 100)

    R['freq2Scale'] = np.arange(11, 101)
    # R['freqs'] = np.arange(5, 101)
    # R['freqs'] = np.arange(1, 101)
    # R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
    R['freq2Normal'] = np.concatenate(([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5], np.arange(5, 31)), axis=0)

    # 激活函数的选择
    # R['act2normal'] = 'relu'
    R['act2normal'] = 'tanh'
    # R['act2normal'] = 'srelu'
    # R['act2normal'] = 'sin'
    # R['act2normal'] = 's2relu'

    # R['act2scale'] = 'relu'
    # R['act2scale']' = leaky_relu'
    # R['act2scale'] = 'srelu'
    R['act2scale'] = 's2relu'
    # R['act2scale'] = 'tanh'
    # R['act2scale'] = 'elu'
    # R['act2scale'] = 'phi'

    R['hot_power'] = 1

    if R['PDE_type'] == 'Convection_diffusion':
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.1
        # R['contrib2scale'] = 0.005
    else:
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 1.0

    # R['repeat_high_freq'] = True
    R['repeat_high_freq'] = False

    solve_Multiscale_PDE(R)

