import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import Laplace1d_split

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

# ------------------------------------------- 文件保存路径设置 ----------------------------------------
store_file = 'laplace1d'
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
R['activate_stop'] = int(1)
R['max_epoch'] = 200000
if 0 != R['activate_stop']:
    # R['max_epoch'] = int(60000)
    R['max_epoch'] = int(100000)

# R['equa_name'] = 'general_laplace'
# R['equa_name'] = 'PDE1'
# R['equa_name'] = 'PDE2'
# R['equa_name'] = 'PDE3'
# R['equa_name'] = 'PDE4'
# R['equa_name'] = 'PDE5'
# R['equa_name'] = 'PDE6'
# R['equa_name'] = 'PDE7'

R['PDE_type'] = 'p_laplace'
R['equa_name'] = 'multi_scale'

if R['equa_name'] == 'general_laplace':
    R['epsilon'] = 0.1
    R['order2laplace'] = 2
elif R['PDE_type'] == 'p_laplace':
    # 频率设置
    R['epsilon'] = float(0.01)

    # 问题幂次
    order = float(8)
    R['order2laplace'] = order

R['input_dim'] = 1                         # 输入维数，即问题的维数(几元问题)
R['output_dim'] = 1                        # 输出维数
R['variational_loss'] = 1                  # PDE变分
# R['wavelet'] = 0                           # 0:: L2 wavelet+energy 1: wavelet 2:energy
R['wavelet'] = 1                           # 0:: L2 wavelet+energy 1: wavelet 2:energy
# R['wavelet'] = 2                           # 0:: L2 wavelet+energy 1: wavelet 2:energy

# ---------------------------- Setup of DNN -------------------------------
R['batch_size2interior'] = 3000            # 内部训练数据的批大小
R['batch_size2boundary'] = 500             # 边界训练数据大小

R['weight_biases_model'] = 'general_model'
# R['weight_biases_model'] = 'phase_shift_model'

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

R['learning_rate'] = 2e-4                             # 学习率
R['learning_rate_decay'] = 5e-5                       # 学习率 decay
R['optimizer_name'] = 'Adam'                          # 优化器
R['train_group'] = 1

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

# normal 和 scale 网络的总参数数目:12520 + 29360 = 41880
R['hidden2normal'] = (100, 80, 60, 60, 40)                   # 1*100+100*80+80*60+60*60+60*40+40*1 = 18940个参数
# R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
# R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
# R['hidden2normal'] = (500, 400, 300, 200, 100)

if R['model2scale'] == 'PDE_DNN_Cos_C_Sin_Base':
    if R['order2laplace'] == 2:
        if R['epsilon'] == 0.1:
            R['hidden2scale'] = (100, 60, 60, 50, 40)        # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
        else:
            R['hidden2scale'] = (125, 60, 60, 60, 50)        # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数
    elif R['order2laplace'] == 5:
        if R['epsilon'] == 0.1:
            R['hidden2scale'] = (100, 80, 80, 60, 40)        # 1*200+200*80+80*80+80*60+60*40+40*1=29840 个参数
        else:
            R['hidden2scale'] = (125, 80, 80, 60, 40)       # 1*250+250*80+80*80+80*60+60*40+40*1=33890 个参数
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
            R['hidden2scale'] = (250, 80, 80, 60, 40)        # 1*250+250*80+80*80+80*60+60*40+40*1=33890 个参数
    elif R['order2laplace'] == 8:
        if R['epsilon'] == 0.1:
            R['hidden2scale'] = (200, 120, 80, 80, 60)       # 1*200+200*120+120*80+80*80+80*60+60*1=45060 个参数
        else:
            R['hidden2scale'] = (250, 120, 80, 80, 60)       # 1*250+250*120+120*80+80*80+80*60+60*1=51110 个参数
    else:
        R['hidden2scale'] = (250, 120, 80, 80, 60)           # 1*250+250*120+120*80+80*80+80*60+60*1=51110 个参数
        # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 200, 200, 100)


# 激活函数的选择
# R['act_name2NN1'] = 'relu'
# R['act_name2NN1'] = 'tanh'
# R['act_name2NN1'] = 'srelu'
# R['act_name2NN1'] = 'sin'
R['act_name2NN1'] = 's2relu'

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

R['plot_ongoing'] = 0
R['subfig_type'] = 0
R['freqs'] = np.arange(10, 100)

Laplace1d_split.solve_Multiscale_PDE(R)