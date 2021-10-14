#  日志记数
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')  # 将字符串写到文件log_fileout中去，末尾加换行
    log_out.flush()                # 清空缓存区
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName2normal=None, actName2scale=None):
    log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    if R_dic['PDE_type'] == 'pLaplace' or R_dic['PDE_type'] == 'pLaplace_implicit' or \
            R_dic['PDE_type'] == 'Possion_Boltzmann':
        log_string('The order to multiscale: %s\n' % (R_dic['order2pLaplace_operator']), log_fileout)
        log_string('The epsilon to multiscale: %s\n' % (R_dic['epsilon']), log_fileout)

    log_string('Network model for Normal-part: %s\n' % str(R_dic['model2Normal']), log_fileout)
    log_string('Network model for Scale-part: %s\n' % str(R_dic['model2Scale']), log_fileout)
    log_string('Hidden layers for Normal-part:%s\n' % str(R_dic['hidden2normal']), log_fileout)
    log_string('Hidden layers for Scale-part:%s\n' % str(R_dic['hidden2scale']), log_fileout)

    if R_dic['model2Normal'] == 'Fourier_DNN':
        log_string('Input activate function for Normal-part network: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        log_string('Input activate function for Normal-part network: %s\n' % str(actName2normal), log_fileout)
    log_string('Hidden activate function for Normal-part network: %s\n' % str(actName2normal), log_fileout)

    if R_dic['model2Scale'] == 'Fourier_DNN':
        log_string('Input activate function for Scale-part network: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        log_string('Input activate function for Scale-part network: %s\n' % str(actName2scale), log_fileout)
    log_string('Activate function for Scale-part network: %s\n' % str(actName2scale), log_fileout)

    log_string('The contribution factor of Scale-part network: %s\n' % str(R_dic['contrib2scale']), log_fileout)

    if R_dic['model2Normal'] == 'Fourier_DNN':
        log_string('The frequency to Normal-part network: %s\n' % (R_dic['freq2Normal']), log_fileout)
        log_string('Repeating low frequency component for Normal-part!!\n', log_fileout)
    log_string('The frequency to Scale-part network: %s\n' % (R_dic['freq2Scale']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        log_string('optimizer:%s with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss' or R_dic['loss_type'] == 'variational_loss2':
        log_string('Loss function: ' + str(R_dic['loss_type']) +'\n', log_fileout)
    else:
        log_string('Loss function: L2 loss\n', log_fileout)

    if R_dic['opt2loss_udotu'] == 'with_orthogonal':
        log_string('With the orthogonality for coarse and fine. \n', log_fileout)
        log_string(str(R_dic['contrib_scale2orthogonal']) + ' scale for the orthogonality of coarse and fine. \n', log_fileout)
        if R_dic['loss_type'] == 'variational_loss' or R_dic['loss_type'] == 'L2_loss':
            if R_dic['opt2orthogonal'] == 1:
                log_string('The loss of product for coarse and fine: L2-orthogonal. \n', log_fileout)
            elif R_dic['opt2orthogonal'] == 2:
                log_string('The loss of product for coarse and fine: Energy-orthogonal. \n', log_fileout)
            else:
                log_string('The loss of product for coarse and fine: L2-orthogonal + Energy-orthogonal.\n', log_fileout)
        elif R_dic['loss_type'] == 'variational_loss2':
            if R_dic['opt2orthogonal'] == 1:
                log_string('The loss of product for coarse and fine: L2-orthogonal. \n', log_fileout)
    else:
        log_string('Without the orthogonality for coarse and fine. \n', log_fileout)

    if R_dic['opt2loss_bd'] == 'unified_boundary':
        log_string('With the unified_boundary for coarse and fine. \n', log_fileout)
    else:
        log_string('With the individual_boundary for coarse and fine. \n', log_fileout)
        log_string(str(R_dic['contrib_scale2boundary']) + ' scale for individual_boundary. \n', log_fileout)

    if (R_dic['train_model']) == 'training_union':
        log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_opt']) == 'training_group4':
        log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd + loss_UdotU', log_fileout)
    elif (R_dic['train_opt']) == 'training_group3':
        log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)
    elif (R_dic['train_opt']) == 'training_group2':
        log_string('The model for training loss: %s\n' % 'total loss + loss_UdotU', log_fileout)

    log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)

    if R_dic['activate_stop'] != 0:
        log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


# 记录字典中的一些设置
def log_dictionary_3Scale(R_dic, log_fileout, actName2normal=None, actName2scale=None):
    log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    if R_dic['PDE_type'] == 'pLaplace' or R_dic['PDE_type'] == 'Possion_Boltzmann':
        log_string('The order to multiscale: %s\n' % (R_dic['order2pLaplace_operator']), log_fileout)
        log_string('The epsilon to 3scale: %s\n' % (R_dic['epsilon1']), log_fileout)
        log_string('The epsilon to 3scale: %s\n' % (R_dic['epsilon2']), log_fileout)

    log_string('Network model for Normal-part: %s\n' % str(R_dic['model2Normal']), log_fileout)
    log_string('Network model for Scale1-part: %s\n' % str(R_dic['model2Scale1']), log_fileout)
    log_string('Network model for Scale2-part: %s\n' % str(R_dic['model2Scale2']), log_fileout)
    log_string('Hidden layers for Normal-part:%s\n' % str(R_dic['hidden2normal']), log_fileout)
    log_string('Hidden layers for Scale1-part:%s\n' % str(R_dic['hidden2scale1']), log_fileout)
    log_string('Hidden layers for Scale1-part:%s\n' % str(R_dic['hidden2scale2']), log_fileout)

    if R_dic['model2Normal'] == 'Fourier_DNN':
        log_string('Input activate function for Normal-part network: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        log_string('Input activate function for Normal-part network: %s\n' % str(actName2normal), log_fileout)
    log_string('Hidden activate function for Normal-part network: %s\n' % str(actName2normal), log_fileout)

    if R_dic['model2Scale1'] == 'Fourier_DNN':
        log_string('Input activate function for Scale1-part network: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        log_string('Input activate function for Scale1-part network: %s\n' % str(actName2scale), log_fileout)
    log_string('Activate function for Scale1-part network: %s\n' % str(actName2scale), log_fileout)

    if R_dic['model2Scale2'] == 'Fourier_DNN':
        log_string('Input activate function for Scale2-part network: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        log_string('Input activate function for Scale2-part network: %s\n' % str(actName2scale), log_fileout)
    log_string('Activate function for Scale2-part network: %s\n' % str(actName2scale), log_fileout)

    log_string('The contribution factor of Scale1-part network: %s\n' % str(R_dic['contrib2scale1']), log_fileout)
    log_string('The contribution factor of Scale2-part network: %s\n' % str(R_dic['contrib2scale2']), log_fileout)

    if R_dic['model2Normal'] == 'Fourier_DNN':
        log_string('The frequency to Normal-part network: %s\n' % (R_dic['freq2Normal']), log_fileout)
        log_string('Repeating low frequency component for Normal-part!!\n', log_fileout)
    log_string('The frequency to Scale-part network: %s\n' % (R_dic['freq2Scale1']), log_fileout)
    log_string('The frequency to Scale-part network: %s\n' % (R_dic['freq2Scale2']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        log_string('optimizer:%s with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss' or R_dic['loss_type'] == 'variational_loss2':
        log_string('Loss function: ' + str(R_dic['loss_type']) + '\n', log_fileout)
    else:
        log_string('Loss function: L2 loss\n', log_fileout)

    if R_dic['opt2loss_udotu'] == 'with_orthogonal':
        log_string('With the orthogonality for coarse and fine. \n', log_fileout)
        log_string(str(R_dic['contrib_scale2orthogonal']) + ' scale for the orthogonality of coarse and fine. \n', log_fileout)
        if R_dic['loss_type'] == 'variational_loss' or R_dic['loss_type'] == 'L2_loss':
            if R_dic['opt2orthogonal'] == 1:
                log_string('The loss of product for coarse and fine: L2-orthogonal. \n', log_fileout)
            elif R_dic['opt2orthogonal'] == 2:
                log_string('The loss of product for coarse and fine: Energy-orthogonal. \n', log_fileout)
            else:
                log_string('The loss of product for coarse and fine: L2-orthogonal + Energy-orthogonal.\n', log_fileout)
        elif R_dic['loss_type'] == 'variational_loss2':
            if R_dic['opt2orthogonal'] == 1:
                log_string('The loss of product for coarse and fine: L2-orthogonal. \n', log_fileout)
    else:
        log_string('Without the orthogonality for coarse and fine. \n', log_fileout)

    if R_dic['opt2loss_bd'] == 'unified_boundary':
        log_string('With the unified_boundary for coarse and fine. \n', log_fileout)
    else:
        log_string('With the individual_boundary for coarse and fine. \n', log_fileout)
        log_string(str(R_dic['contrib_scale2boundary']) + ' scale for individual_boundary. \n', log_fileout)

    if (R_dic['train_model']) == 'training_union':
        log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_opt']) == 'training_group4':
        log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd + loss_UdotU', log_fileout)
    elif (R_dic['train_opt']) == 'training_group3':
        log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)
    elif (R_dic['train_opt']) == 'training_group2':
        log_string('The model for training loss: %s\n' % 'total loss + loss_UdotU', log_fileout)

    log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)

    if R_dic['activate_stop'] != 0:
        log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)
