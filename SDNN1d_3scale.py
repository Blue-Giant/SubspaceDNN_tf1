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
import DNN_data
import time
import DNN_base
import DNN_tools
import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import plotData
import saveData
import DNN_Print_Log


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.log_dictionary_3Scale(R, log_fileout, actName2normal=R['actHidden_name2Normal'],
                                        actName2scale=R['actHidden_name2Scale'])

    # laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    init_bd_penalty = R['init_boundary_penalty']         # Regularization parameter for boundary conditions
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    init_UdotU_penalty = R['init_penalty2orthogonal']
    hidden2normal = R['hidden2normal']
    hidden2scale1 = R['hidden2scale1']
    hidden2scale2 = R['hidden2scale2']
    penalty2WB = R['penalty2weight_biases']         # Regularization parameter for weights and biases

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']
    act2Normal = R['actHidden_name2Normal']
    act2Scale = R['actHidden_name2Scale']

    alpha1 = R['contrib2scale1']
    alpha2 = R['contrib2scale2']
    
    if R['contrib_scale2orthogonal'] == 'with_contrib':
        using_scale1_orthogonal = R['contrib2scale1']
        using_scale2_orthogonal = R['contrib2scale2']
    else:
        using_scale1_orthogonal = 1.0
        using_scale2_orthogonal = 1.0

    if R['opt2loss_bd'] == 'unified_boundary' and R['contrib_scale2boundary'] == 'with_contrib':
        using_scale1_boundary = R['contrib2scale1']
        using_scale2_boundary = R['contrib2scale2']
    elif R['opt2loss_bd'] == 'individual_boundary' and R['contrib_scale2boundary'] == 'with_contrib':
        using_scale1_boundary = R['contrib2scale1']
        using_scale2_boundary = R['contrib2scale2']
    else:
        using_scale1_boundary = 1.0
        using_scale2_boundary = 1.0

    region_l = 0.0
    region_r = 1.0
    if R['PDE_type'] == 'pLaplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon1 = R['epsilon1']
        epsilon2 = R['epsilon2']
        region_l = 0.0
        region_r = 1.0
        u_true, f, A_eps, u_left, u_right = MS_LaplaceEqs.get_infos2pLaplace_1D_2(
            in_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, index2p=p_index, eps1=epsilon1,
            eps2=epsilon2)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + K(x)u_eps(x) =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        A_eps, kappa, u_true, u_left, u_right, f = MS_BoltzmannEqs.get_infos2Boltzmann_1D(
            in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, index2p=p_index, eps=epsilon,
            eqs_name=R['equa_name'])

    flag_Normal = 'WB_Normal'
    flag_Scale1 = 'WB_scale1'
    flag_Scale2 = 'WB_scale2'
    if R['model2Normal'] == 'Fourier_DNN':
        Ws_Normal, Bs_Normal = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2normal, flag_Normal)
    else:
        Ws_Normal, Bs_Normal = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2normal, flag_Normal)

    if R['model2Scale1'] == 'Fourier_DNN':
        Ws_Scale1, B2NN_scale1 = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2scale1, flag_Scale1)
    else:
        Ws_Scale1, B2NN_scale1 = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2scale1, flag_Scale1)

    if R['model2Scale2'] == 'Fourier_DNN':
        Ws_Scale2, Bs_Scale2 = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden2scale2, flag_Scale2)
    else:
        Ws_Scale2, Bs_Scale2 = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden2scale2, flag_Scale2)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            X_it = tf.compat.v1.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 1 列
            X_left = tf.compat.v1.placeholder(tf.float32, name='X_left', shape=[None, input_dim])      # * 行 1 列
            X_right = tf.compat.v1.placeholder(tf.float32, name='X_right', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            UdotU_penalty = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if R['model2Normal'] == 'DNN':
                UNN_Normal = DNN_base.DNN(X_it, Ws_Normal, Bs_Normal, hidden2normal, activateIn_name=act2Normal,
                                          activate_name=act2Normal)
                UNNLeft_Normal = DNN_base.DNN(X_left, Ws_Normal, Bs_Normal, hidden2normal, activateIn_name=act2Normal,
                                              activate_name=act2Normal)
                UNNRight_Normal = DNN_base.DNN(X_right, Ws_Normal, Bs_Normal, hidden2normal, activateIn_name=act2Normal,
                                               activate_name=act2Normal)
            elif R['model2Normal'] == 'Fourier_DNN':
                freq = R['freq2Normal']
                UNN_Normal = DNN_base.DNN_FourierBase(X_it, Ws_Normal, Bs_Normal, hidden2normal, freq,
                                                      activate_name=act2Normal, repeat_Highfreq=False)
                UNNLeft_Normal = DNN_base.DNN_FourierBase(X_left, Ws_Normal, Bs_Normal, hidden2normal, freq,
                                                          activate_name=act2Normal, repeat_Highfreq=False)
                UNNRight_Normal = DNN_base.DNN_FourierBase(X_right, Ws_Normal, Bs_Normal, hidden2normal, freq,
                                                           activate_name=act2Normal, repeat_Highfreq=False)

            freqs1 = R['freq2Scale1']
            if R['model2Scale1'] == 'Scale_DNN':
                UNN_scale1 = DNN_base.DNN_scale(X_it, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                activateIn_name=act2Scale, activate_name=act2Scale)
                UNNLeft_scale1 = DNN_base.DNN_scale(X_left, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                    activateIn_name=act2Scale, activate_name=act2Scale)
                UNNRight_scale1 = DNN_base.DNN_scale(X_right, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale1'] == 'Adapt_Scale_DNN':
                UNN_scale1 = DNN_base.DNN_adapt_scale(X_it, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                      activateIn_name=act2Scale, activate_name=act2Scale)
                UNNLeft_scale1 = DNN_base.DNN_adapt_scale(X_left, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                          activateIn_name=act2Scale, activate_name=act2Scale)
                UNNRight_scale1 = DNN_base.DNN_adapt_scale(X_right, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                           activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale1'] == 'Fourier_DNN':
                UNN_scale1 = DNN_base.DNN_FourierBase(X_it, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                      activate_name=act2Scale)
                UNNLeft_scale1 = DNN_base.DNN_FourierBase(X_left, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                          activate_name=act2Scale)
                UNNRight_scale1 = DNN_base.DNN_FourierBase(X_right, Ws_Scale1, B2NN_scale1, hidden2scale1, freqs1,
                                                           activate_name=act2Scale)

            freqs2 = R['freq2Scale2']
            if R['model2Scale2'] == 'Scale_DNN':
                UNN_scale2 = DNN_base.DNN_scale(X_it, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                activateIn_name=act2Scale, activate_name=act2Scale)
                UNNLeft_scale2 = DNN_base.DNN_scale(X_left, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                    activateIn_name=act2Scale, activate_name=act2Scale)
                UNNRight_scale2 = DNN_base.DNN_scale(X_right, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                     activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale2'] == 'Adapt_Scale_DNN':
                UNN_scale2 = DNN_base.DNN_adapt_scale(X_it, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                      activateIn_name=act2Scale, activate_name=act2Scale)
                UNNLeft_scale2 = DNN_base.DNN_adapt_scale(X_left, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                          activateIn_name=act2Scale, activate_name=act2Scale)
                UNNRight_scale2 = DNN_base.DNN_adapt_scale(X_right, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                           activateIn_name=act2Scale, activate_name=act2Scale)
            elif R['model2Scale2'] == 'Fourier_DNN':
                UNN_scale2 = DNN_base.DNN_FourierBase(X_it, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                      activate_name=act2Scale)
                UNNLeft_scale2 = DNN_base.DNN_FourierBase(X_left, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                          activate_name=act2Scale)
                UNNRight_scale2 = DNN_base.DNN_FourierBase(X_right, Ws_Scale2, Bs_Scale2, hidden2scale2, freqs2,
                                                           activate_name=act2Scale)

            # alpha2 = 0.01 * alpha
            UNN = UNN_Normal + R['contrib2scale1']*UNN_scale1 + R['contrib2scale2']*UNN_scale2

            # 变分形式的loss of interior，训练得到的 U_NN1 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            dUNN_Normal = tf.gradients(UNN_Normal, X_it)[0]    # * 行 1 列
            dUNN_scale1= tf.gradients(UNN_scale1, X_it)[0]      # * 行 1 列
            dUNN_scale2 = tf.gradients(UNN_scale2, X_it)[0]  # * 行 1 列
            if R['loss_type'] == 'variational_loss':
                dUNN = tf.add(tf.add(dUNN_Normal, R['contrib2scale1']*dUNN_scale1), R['contrib2scale2']*dUNN_scale2)
                if R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    dUNN_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(dUNN), axis=-1), shape=[-1, 1]))
                    AdUNN_pnorm = a_eps * tf.pow(dUNN_norm, p_index)
                    fx = MS_LaplaceEqs.force_sice_3scale2(X_it, eps1=R['epsilon1'], eps2=R['epsilon2'])
                    loss_it_NN = (1.0 / p_index) * AdUNN_pnorm - tf.multiply(tf.reshape(fx, shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    dUNN_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(dUNN), axis=-1), shape=[-1, 1]))
                    divAdUNN = a_eps * tf.pow(dUNN_norm, p_index)
                    loss_it_NN = (1.0 / p_index) * (divAdUNN + Kappa*UNN*UNN) - \
                                 tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), UNN)

                Loss_it2NN = tf.reduce_mean(loss_it_NN)
            elif R['loss_type'] == 'variational_loss2':
                if R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    dUNN_N_norm = tf.reshape(tf.abs(dUNN_Normal), shape=[-1, 1])
                    dUNN_S1_norm = tf.reshape(tf.abs(dUNN_scale1), shape=[-1, 1])
                    dUNN_S2_norm = tf.reshape(tf.abs(dUNN_scale2), shape=[-1, 1])
                    AdUNN_pnorm = a_eps * tf.pow(dUNN_N_norm, p_index) + \
                                  a_eps * R['contrib2scale1']*R['contrib2scale1']*tf.pow(dUNN_S1_norm, p_index) + \
                                  a_eps * R['contrib2scale2']*R['contrib2scale2']*tf.pow(dUNN_S2_norm, p_index)
                    fx = MS_LaplaceEqs.force_sice_3scale2(X_it, eps1=R['epsilon1'], eps2=R['epsilon2'])
                    loss_it_NN = (1.0 / p_index) * AdUNN_pnorm - tf.multiply(tf.reshape(fx, shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)                          # * 行 1 列
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    dUNN_N_norm = tf.reshape(tf.abs(dUNN_Normal), shape=[-1, 1])
                    dUNN_S1_norm = tf.reshape(tf.abs(dUNN_scale1), shape=[-1, 1])
                    dUNN_S2_norm = tf.reshape(tf.abs(dUNN_scale2), shape=[-1, 1])
                    AdUNN_pnorm = a_eps * tf.pow(dUNN_N_norm, p_index) + \
                                  a_eps * R['contrib2scale1'] * R['contrib2scale1'] * tf.pow(dUNN_S1_norm, p_index) + \
                                  a_eps * R['contrib2scale2'] * R['contrib2scale2'] * tf.pow(dUNN_S2_norm, p_index)
                    loss_it_NN = (1.0 / p_index) * (AdUNN_pnorm + Kappa*UNN*UNN) - \
                                 tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), UNN)

                Loss_it2NN = tf.reduce_mean(loss_it_NN)

            if R['opt2orthogonal'] == 1:
                # # |Uc*Uf|^2-->0 Uc 和 Uf 是两个列向量 形状为(*,1)
                norm2UdU01 = tf.square(tf.multiply(UNN_Normal, using_scale1_orthogonal * UNN_scale1))
                norm2UdU02 = tf.square(tf.multiply(UNN_Normal, using_scale2_orthogonal * UNN_scale2))
                norm2UdU12 = tf.square(tf.multiply(using_scale1_orthogonal * UNN_scale1, using_scale2_orthogonal * UNN_scale2))
                UNN_dot_UNN = tf.reduce_mean(norm2UdU01) + tf.reduce_mean(norm2UdU02) + tf.reduce_mean(norm2UdU12)

            if R['opt2loss_udotu'] == 'with_orthogonal':
                Loss2UNN_dot_UNN = UdotU_penalty * UNN_dot_UNN
            else:
                Loss2UNN_dot_UNN = tf.constant(0.0)

            U_left = tf.reshape(u_left(X_left), shape=[-1, 1])
            U_right = tf.reshape(u_right(X_right), shape=[-1, 1])
            if R['opt2loss_bd'] == 'unified_boundary':
                UNN_left = UNNLeft_Normal + using_scale1_boundary*UNNLeft_scale1 + using_scale2_boundary*UNNLeft_scale2
                UNN_right = UNNRight_Normal + using_scale1_boundary * UNNRight_scale1 + using_scale2_boundary * UNNRight_scale2
                Loss_bd2NN = tf.square(UNN_left - U_left) + tf.square(UNN_right - U_right)
                Loss_bd2NNs = tf.reduce_mean(Loss_bd2NN)
            else:
                loss_bd_Normal = tf.square(UNNLeft_Normal - U_left) + tf.square(UNNRight_Normal - U_right)
                loss_bd_scale1 = tf.square(using_scale1_boundary*UNNLeft_scale1) + \
                                 tf.square(using_scale1_boundary*UNNRight_scale1)
                loss_bd_scale2 = tf.square(using_scale2_boundary*UNNLeft_scale2) + \
                                 tf.square(using_scale2_boundary*UNNRight_scale2)
                Loss_bd2NNs = tf.reduce_mean(loss_bd_Normal) + tf.reduce_mean(loss_bd_scale1) + tf.reduce_mean(loss_bd_scale2)

            if R['regular_wb_model'] == 'L1':
                regular_WB_Normal = DNN_base.regular_weights_biases_L1(Ws_Normal, Bs_Normal)    # 正则化权重和偏置 L1正则化
                regular_WB_Scale1 = DNN_base.regular_weights_biases_L1(Ws_Scale1, B2NN_scale1)  # 正则化权重和偏置 L1正则化
                regular_WB_Scale2 = DNN_base.regular_weights_biases_L1(Ws_Scale2, Bs_Scale2)
            elif R['regular_wb_model'] == 'L2':
                regular_WB_Normal = DNN_base.regular_weights_biases_L2(Ws_Normal, Bs_Normal)    # 正则化权重和偏置 L2正则化
                regular_WB_Scale1 = DNN_base.regular_weights_biases_L2(Ws_Scale1, B2NN_scale1)  # 正则化权重和偏置 L2正则化
                regular_WB_Scale2 = DNN_base.regular_weights_biases_L2(Ws_Scale2, Bs_Scale2)
            else:
                regular_WB_Normal = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB_Scale1 = tf.constant(0.0)
                regular_WB_Scale2 = tf.constant(0.0)

            PWB = penalty2WB * (regular_WB_Normal + regular_WB_Scale1 + regular_WB_Scale2)

            Loss2NN = Loss_it2NN + bd_penalty*Loss_bd2NNs + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['loss_type'] == 'variational_loss' or R['loss_type'] == 'variational_loss2':
                if R['train_model'] == 1:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 2:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_model'] == 3:
                    train_op1 = my_optimizer.minimize(Loss_it2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 4:
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op3, train_op4)
                else:
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            U_true = u_true(X_it)
            train_mse_NN = tf.reduce_mean(tf.square(U_true - UNN))
            train_rel_NN = train_mse_NN / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    loss_it_all, loss_bd_all, loss_all, loss_udu_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])
    saveData.save_testData_or_solus2mat(test_x_bach, dataName='testx', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * init_bd_penalty
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * init_bd_penalty
                else:
                    temp_penalty_bd = 500 * init_bd_penalty
            else:
                temp_penalty_bd = init_bd_penalty

            if R['activate_powSolus_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_powU = init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_powU = 10* init_UdotU_penalty
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
                [train_Loss2NN, Loss_it2NN, Loss_bd2NNs, Loss2NN, UNN_dot_UNN, train_mse_NN, train_rel_NN, PWB],
                feed_dict={X_it: x_it_batch, X_left: xl_bd_batch, X_right: xr_bd_batch, in_learning_rate: tmp_lr,
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
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, pwb, loss_it_nn, loss_bd_nn, loss_nn,
                    udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                u_true2test, utest_nn, unn_normal, unn_scale1, unn_scale2 = sess.run(
                    [U_true, UNN, UNN_Normal, alpha1*UNN_scale1, alpha2*UNN_scale2], feed_dict={X_it: test_x_bach})
                test_mse2nn = np.mean(np.square(u_true2test - utest_nn))
                test_mse_all.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout)

            if (i_epoch != 0) and (i_epoch != 100000) and (i_epoch % 1000 == 0) and i_epoch >= int(0.5*R['max_epoch']):
                pathOut = '%s/%s' % (R['FolderName'], int(i_epoch / 1000))
                if not os.path.exists(pathOut):       # 判断路径是否已经存在
                    os.mkdir(pathOut)                 # 无 log_out_path 路径，创建一个 log_out_path 路径

                saveData.save_testData_or_solus2mat(utest_nn, dataName=act2Normal, outPath=pathOut)
                saveData.save_testData_or_solus2mat(unn_normal, dataName='normal', outPath=pathOut)
                saveData.save_testData_or_solus2mat(unn_scale1, dataName='scale1', outPath=pathOut)
                saveData.save_testData_or_solus2mat(unn_scale2, dataName='scale2', outPath=pathOut)

                saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act2Scale, outPath=pathOut)
                plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act2Scale, seedNo=R['seed'],
                                          outPath=pathOut, yaxis_scale=True)
    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act2Normal,
                                         outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act2Normal, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act2Scale, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(utest_nn, dataName=act2Normal, outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_normal, dataName='normal', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_scale1, dataName='scale1', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_scale2, dataName='scale2', outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act2Scale, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act2Scale, seedNo=R['seed'],
                              outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
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

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    store_file = 'pLaplace1D'
    # store_file = 'Boltzmann1D'
    # store_file = 'Convection1D'
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

    if store_file == 'Laplace1D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace1D':
        R['PDE_type'] = 'pLaplace'
        R['equa_name'] = '3scale2'
        # R['equa_name'] = '3scale3'
    elif store_file == 'Boltzmann1D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'
        R['equa_name'] = 'Boltzmann2'

    if R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
        # 频率设置
        epsilon1 = input('please input epsilon1 =')  # 由终端输入的会记录为字符串形式
        R['epsilon1'] = float(epsilon1)              # 字符串转为浮点

        epsilon2 = input('please input epsilon2 =')  # 由终端输入的会记录为字符串形式
        R['epsilon2'] = float(epsilon2)  # 字符串转为浮点

        # 问题幂次
        order2pLaplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2pLaplace)
        R['order2pLaplace_operator'] = order

    R['input_dim'] = 1                         # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                        # 输出维数
    R['loss_type'] = 'variational_loss'        # PDE变分
    # R['loss_type'] = 'variational_loss2'  # PDE变分
    # R['loss_type'] = 'L2_loss'                # L2 loss
    # R['opt2orthogonal'] = 0                   # 0: L2 orthogonal+energy    1: orthogonal    2:energy
    R['opt2orthogonal'] = 1                     # 0: L2 orthogonal+energy    1: orthogonal    2:energy
    # R['opt2orthogonal'] = 2                   # 0: L2 orthogonal+energy    1: orthogonal    2:energy

    # ---------------------------- Setup of DNN -------------------------------
    R['batch_size2interior'] = 3000             # 内部训练数据的批大小
    R['batch_size2boundary'] = 500              # 边界训练数据大小

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000     # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001   # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025  # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100                           # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['init_penalty2orthogonal'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['init_penalty2orthogonal'] = 10000.0
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 15.0
        # R['init_penalty2orthogonal'] = 10.0

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器
    R['train_model'] = 'training_union'  # 训练模式, 一个 loss 联结训练
    # R['train_model'] = 1
    # R['train_model'] = 3
    # R['train_model'] = 4

    # R['model2Normal'] = 'DNN'  # 使用的网络模型
    # R['model2Normal'] = 'Scale_DNN'
    # R['model2Normal'] = 'Adapt_Scale_DNN'
    R['model2Normal'] = 'Fourier_DNN'
    # R['model2Normal'] = 'Sin+Cos_DNN'

    # R['model2Scale1'] = 'DNN'                         # 使用的网络模型
    # R['model2Scale1'] = 'Scale_DNN'
    # R['model2Scale1'] = 'Adapt_Scale_DNN'
    # R['model2Scale1'] = 'Sin+Cos_DNN'
    R['model2Scale1'] = 'Fourier_DNN'

    # R['model2Scale2'] = 'DNN'                         # 使用的网络模型
    # R['model2Scale2'] = 'Scale_DNN'
    # R['model2Scale2'] = 'Adapt_Scale_DNN'
    # R['model2Scale2'] = 'Sin+Cos_DNN'
    R['model2Scale2'] = 'Fourier_DNN'

    # normal 和 scale 网络的总参数数目:12520 + 29360 = 41880
    if R['model2Normal'] == 'Fourier_DNN':
        R['hidden2normal'] = (50, 80, 60, 60, 40)  # 1*50+100*80+80*60+60*60+60*40+40*1 = 18890个参数
    else:
        R['hidden2normal'] = (100, 80, 60, 60, 40)          # 1*100+100*80+80*60+60*60+60*40+40*1 = 18940个参数
        # R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)

    if R['model2Scale1'] == 'Fourier_DNN':
        if R['epsilon1'] == 0.1:
            R['hidden2scale1'] = (125, 80, 60, 60, 40)        # 1*125+250*80+80*60+60*60+60*40+40*1=30925 个参数
        else:
            R['hidden2scale1'] = (225, 200, 150, 150, 100)    # 1*225+450*200+200*150+150*150+150*100+100*1=157825 个参数
    else:
        if R['epsilon1'] == 0.1:
            R['hidden2scale1'] = (160, 100, 80, 80, 60)        # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
        else:
            R['hidden2scale1'] = (250, 200, 150, 150, 100)       # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数

    if R['model2Scale2'] == 'Fourier_DNN':
        if R['epsilon2'] == 0.1:
            R['hidden2scale2'] = (125, 80, 60, 60, 40)       # 1*125+250*80+80*60+60*60+60*40+40*1=30925 个参数
        else:
            R['hidden2scale2'] = (225, 200, 150, 150, 100)    # 1*225+450*200+200*150+150*150+150*100+100*1=157825 个参数
    else:
        if R['epsilon2'] == 0.1:
            R['hidden2scale2'] = (160, 100, 80, 80, 60)        # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
        else:
            R['hidden2scale2'] = (250, 200, 150, 150, 100)       # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数

    # 激活函数的选择
    # R['actHidden_name2Normal'] = 'relu'
    R['actHidden_name2Normal'] = 'tanh'
    # R['actHidden_name2Normal'] = 'srelu'
    # R['actHidden_name2Normal'] = 'sin'
    # R['actHidden_name2Normal'] = 's2relu'

    # R['actHidden_name2Scale'] = 'relu'
    # R['actHidden_name2Scale']' = leaky_relu'
    # R['actHidden_name2Scale'] = 'srelu'
    R['actHidden_name2Scale'] = 's2relu'
    # R['actHidden_name2Scale'] = 'tanh'
    # R['actHidden_name2Scale'] = 'elu'
    # R['actHidden_name2Scale'] = 'phi'

    R['plot_ongoing'] = 0
    R['subfig_type'] = 0
    # R['freq2Normal'] = np.concatenate(
    # ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 21)), axis=0)
    # R['freq2Normal'] = np.concatenate(
    #    ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 26)), axis=0)
    # R['freq2Normal'] = np.concatenate(
    #    ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 31)), axis=0)
    # R['freq2Normal'] = np.arange(1, 41)*0.5
    # R['freq2Normal'] = np.arange(1, 21) * 0.5
    R['freq2Normal'] = np.arange(1, 31) * 0.5
    R['freq2Scale1'] = np.arange(30, 71)
    # R['freq2Scale2'] = np.arange(100, 200)
    R['freq2Scale2'] = np.arange(250, 360)
    # R['freq2Scale2'] = np.arange(300, 400)
    # R['freq2Scale2'] = np.arange(60, 111)

    if R['loss_type'] == 'variational_loss':
        # R['contrib2scale1'] = 0.05
        # R['contrib2scale1'] = 0.06
        # R['contrib2scale1'] = 0.07
        # R['contrib2scale1'] = 0.075
        R['contrib2scale1'] = 0.1
        # R['contrib2scale1'] = 0.125
        # R['contrib2scale1'] = 0.15

        R['contrib2scale2'] = 0.01
        # R['contrib2scale2'] = 0.0075
        # R['contrib2scale2'] = 0.008
    elif R['loss_type'] == 'variational_loss2':
        # R['contrib2scale'] = 0.01
        R['contrib2scale1'] = 0.25
        # R['contrib2scale1'] = 0.06
        # R['contrib2scale1'] = 0.07
        # R['contrib2scale1'] = 0.075
        R['contrib2scale2'] = 0.075
        # R['contrib2scale2'] = 0.0075
        # R['contrib2scale2'] = 0.008

    R['opt2loss_udotu'] = 'with_orthogonal'
    # R['opt2loss_udotu'] = 'without_orthogonal'

    R['contrib_scale2orthogonal'] = 'with_contrib'
    # R['contrib_scale2orthogonal'] = 'without_contrib'

    # R['opt2loss_bd'] = 'unified_boundary'
    R['opt2loss_bd'] = 'individual_boundary'

    # R['contrib_scale2boundary'] = 'with_contrib'
    R['contrib_scale2boundary'] = 'without_contrib'

    if R['opt2loss_bd'] == 'unified_boundary':
        R['contrib_scale2boundary'] = 'with_contrib'

    solve_Multiscale_PDE(R)

