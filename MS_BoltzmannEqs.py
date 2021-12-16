import numpy as np
import tensorflow as tf


def get_infos2Boltzmann_1D(in_dim=1, out_dim=1, region_a=0.0, region_b=1.0, index2p=2, eps=0.01, eqs_name=None):
    if eqs_name == 'Boltzmann1':
        llam = 20
        mu = 50
        f = lambda x: (llam*llam+mu*mu)*tf.sin(x)
        Aeps = lambda x: 1.0*tf.ones_like(x)
        kappa = lambda x: llam*llam*tf.ones_like(x)
        utrue = lambda x: -1.0*(np.sin(mu)/np.sinh(llam))*tf.sinh(llam*x) + tf.sin(mu*x)
        ul = lambda x: tf.zeros_like(x)
        ur = lambda x: tf.zeros_like(x)
        return Aeps, kappa, utrue, ul, ur, f
    elif eqs_name == 'Boltzmann2':
        kappa = lambda x: tf.ones_like(x)
        Aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

        utrue = lambda x: x - x*x + (eps / (4*np.pi)) * tf.sin(np.pi * 2 * x / eps)

        ul = lambda x: tf.zeros_like(x)

        ur = lambda x: tf.zeros_like(x)

        if index2p == 1:
            f = lambda x: 2.0/(2 + tf.cos(2 * np.pi * x / eps)) + (4*np.pi*x/eps)*tf.sin(np.pi * 2 * x / eps)/\
                          ((2 + tf.cos(2 * np.pi * x / eps))*(2 + tf.cos(2 * np.pi * x / eps))) + x - tf.square(x) \
                          + (eps / (4*np.pi)) * tf.sin(np.pi * 2 * x / eps)
        elif index2p == 2:
            f = lambda x: 2.0/(2 + tf.cos(2 * np.pi * x / eps)) + \
                          (4*np.pi*x/eps)*tf.sin(np.pi*2*x/eps)/((2 + tf.cos(2*np.pi*x/eps))*(2+tf.cos(2*np.pi*x/eps))) + \
                          (x-x*x+(eps/(4*np.pi))*tf.sin(np.pi*2*x/eps))*(x-x*x+(eps/(4*np.pi))*tf.sin(np.pi*2*x/eps))

        return Aeps, kappa, utrue, ul, ur, f


def get_force_side2Boltzmann_1D(x, index2p=2, eps=0.01, eqs_name=None):
    utrue = x - x * x + (eps / (4 * np.pi)) * tf.sin(np.pi * 2 * x / eps)
    ux = 1-2*x + 0.5*tf.cos(np.pi * 2 * x / eps)
    uxx = - 2.0 - (np.pi/eps)*tf.sin(np.pi * 2 * x / eps)
    fenmu2Aeps = 2 + tf.cos(2 * np.pi * x / eps)
    Aeps = 1.0 / fenmu2Aeps
    Aepsx = (2.0*np.pi/eps)*tf.sin(np.pi*2*x/eps)/(fenmu2Aeps*fenmu2Aeps)
    if index2p == 1:
        f = -1.0 * (Aepsx * ux + Aeps * uxx) + utrue
    elif index2p == 2:
        f = -1.0 * (Aepsx * ux + Aeps * uxx) + utrue*utrue
    return f


def get_infos2Boltzmann_2D(equa_name=None, intervalL=0.1, intervalR=1.0):
    if equa_name == 'Boltzmann1':
        lam = 2
        mu = 30
        f = lambda x, y: (lam*lam+mu*mu)*(tf.sin(mu*x) + tf.sin(mu*y))
        A_eps = lambda x, y: 1.0*tf.ones_like(x)
        kappa = lambda x, y: lam*lam*tf.ones_like(x)
        u = lambda x, y: -1.0*(np.sin(mu)/np.sinh(lam))*tf.sinh(lam*x) + tf.sin(mu*x) -1.0*(np.sin(mu)/np.sinh(lam))*tf.sinh(lam*y) + tf.sin(mu*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)
    elif equa_name == 'Boltzmann2':
        f = lambda x, y: 5 * ((np.pi) ** 2) * (0.5 * tf.sin(np.pi * x) * tf.cos(np.pi * y) + 0.25 * tf.sin(10 * np.pi * x) * tf.cos(10 * np.pi * y)) * \
                        (0.25 * tf.cos(5 * np.pi * x) * tf.sin(10 * np.pi * y) + 0.5 * tf.cos(15 * np.pi * x) * tf.sin(20 * np.pi * y)) + \
                        5 * ((np.pi) ** 2) * (0.5 * tf.cos(np.pi * x) * tf.sin(np.pi * y) + 0.25 * tf.cos(10 * np.pi * x) * tf.sin(10 * np.pi * y)) * \
                        (0.125 * tf.sin(5 * np.pi * x) * tf.cos(10 * np.pi * y) + 0.125 * 3 * tf.sin(15 * np.pi * x) * tf.cos(20 * np.pi * y)) + \
                        ((np.pi) ** 2) * (tf.sin(np.pi * x) * tf.sin(np.pi * y) + 5 * tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y)) * \
                        (0.125 * tf.cos(5 * np.pi * x) * tf.cos(10 * np.pi * y) + 0.125 * tf.cos(15 * np.pi * x) * tf.cos(20 * np.pi * y) + 0.5) + \
                         0.5 *np.pi*np.pi* tf.sin(np.pi * x) * tf.sin(np.pi * y) + 0.025 *np.pi*np.pi* tf.sin(10 * np.pi * x) * tf.sin(10 * np.pi * y)

        A_eps = lambda x, y: 0.5 + 0.125*tf.cos(5*np.pi*x)*tf.cos(10*np.pi*y) + 0.125*tf.cos(15*np.pi*x)*tf.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*tf.ones_like(x)
        u = lambda x, y: 0.5*tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.025*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)
    elif equa_name == 'Boltzmann3':
        f = lambda x, y: tf.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y) + 0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*tf.ones_like(x)
        u = lambda x, y: tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)
    elif equa_name == 'Boltzmann4':
        f = lambda x, y: tf.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y) + 0.25*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y) + 0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*tf.ones_like(x)
        u = lambda x, y: tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)+0.01*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)
    elif equa_name == 'Boltzmann5':
        f = lambda x, y: tf.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y) + 0.25*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y) + 0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*tf.ones_like(x)
        u = lambda x, y: tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)
    elif equa_name == 'Boltzmann6':
        f = lambda x, y: tf.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y) + 0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*tf.ones_like(x)
        u = lambda x, y: tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)+0.01*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        ux_left = lambda x, y: tf.zeros_like(x)
        ux_right = lambda x, y: tf.zeros_like(x)
        uy_bottom = lambda x, y: tf.zeros_like(x)
        uy_top = lambda x, y: tf.zeros_like(x)

    return A_eps, kappa, u, ux_left, ux_right, uy_top, uy_bottom, f


def get_foreside2Boltzmann2D(x=None, y=None, equa_name='Boltzmann3'):
    if equa_name == 'Boltzmann3':
        u = tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        Aeps = 0.5+0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y)+0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)

        ux = 1.0*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)+1.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)
        uy = 1.0*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)+1.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-20*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-20*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)-0.25*20*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)-0.25*20*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann4':
        u = tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)+0.01*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        Aeps = 0.5+0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y)+0.25*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)+0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)

        ux = 1.0*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)+0.5*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)+0.2*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)
        uy = 1.0*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)+0.5*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)+0.2*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)-4.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)-4.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)-0.25*10*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)-0.25*20*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)-0.25*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)-0.25*20*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann5':
        u = tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        Aeps = 0.5+0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y)+0.25*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)+0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)

        ux = 1.0*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)+1.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)
        uy = 1.0*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)+1.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-20*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-20*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)-0.25*10*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)-0.25*20*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)-0.25*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)-0.25*20*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann6':
        u = tf.sin(np.pi*x)*tf.sin(np.pi*y)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)+0.01*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        Aeps = 0.5+0.25*tf.cos(np.pi*x)*tf.cos(np.pi*y) + 0.25*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)

        ux = 1.0*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)+0.5*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)+0.2*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)
        uy = 1.0*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)+0.5*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)+0.2*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)-4.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)-4.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)-0.25*20*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)-0.25*20*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u

    return fside


def get_infos2Boltzmann_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann0':
        # mu1= 2*np.pi
        # mu2 = 4*np.pi
        # mu3 = 8*np.pi
        # mu1 = np.pi
        # mu2 = 5 * np.pi
        # mu3 = 10 * np.pi
        mu1 = np.pi
        mu2 = 10 * np.pi
        mu3 = 20 * np.pi
        fside = lambda x, y, z: (mu1*mu1+mu2*mu2+mu3*mu3+x*x+2*y*y+3*z*z)*tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        A_eps = lambda x, y, z: 1.0*tf.ones_like(x)
        kappa = lambda x, y, z: x*x+2*y*y+3*z*z
        utrue = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_00 = lambda x, y, z: tf.sin(mu1*intervalL)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_01 = lambda x, y, z: tf.sin(mu1*intervalR)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_10 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalL)*tf.sin(mu3*z)
        u_11 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalR)*tf.sin(mu3*z)
        u_20 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalL)
        u_21 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalR)

        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann1':
        fside = lambda x, y, z: tf.ones_like(x)
        utrue = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.25 * (1.0 + tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z))
        kappa = lambda x, y, z: tf.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: tf.sin(np.pi * intervalL) * tf.sin(5 * np.pi * y) * tf.sin(10*np.pi * z)
        u_01 = lambda x, y, z: tf.sin(np.pi * intervalR) * tf.sin(5 * np.pi * y) * tf.sin(10*np.pi * z)
        u_10 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalL) * tf.sin(10*np.pi * z)
        u_11 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalR) * tf.sin(10*np.pi * z)
        u_20 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann2':
        fside = lambda x, y, z: (63 / 4) * ((np.pi) ** 2) * (
                    1.0 + tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z)) * \
                                (tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)) + \
                                0.125 * ((np.pi) ** 2) * tf.sin(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(
            20 * np.pi * z) * tf.cos(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z) + \
                                (25 / 4) * ((np.pi) ** 2) * tf.cos(np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(
            20 * np.pi * z) * tf.sin(np.pi * x) * tf.cos(5 * np.pi * y) * tf.sin(10 * np.pi * z) + \
                                25.0 * ((np.pi) ** 2) * tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(
            20 * np.pi * z) * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.cos(10 * np.pi * z) + \
                                0.5 * (np.pi * np.pi) * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        utrue = lambda x, y, z: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.25 * (1.0 + tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z))
        kappa = lambda x, y, z: tf.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: 0.5 * tf.sin(np.pi * intervalL) * tf.sin(5 * np.pi * y) * tf.sin(10*np.pi * z)
        u_01 = lambda x, y, z: 0.5 * tf.sin(np.pi * intervalR) * tf.sin(5 * np.pi * y) * tf.sin(10*np.pi * z)
        u_10 = lambda x, y, z: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalL) * tf.sin(10*np.pi * z)
        u_11 = lambda x, y, z: 0.5 * tf.sin(np.pi * x) * tf.sin(5 * np.pi * intervalR) * tf.sin(10*np.pi * z)
        u_20 = lambda x, y, z: 0.5 * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: 0.5 * tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(10 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann3':
        fside = lambda x, y, z: tf.ones_like(x)
        utrue = lambda x, y, z: tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + tf.cos(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(10 * np.pi * z))
        kappa = lambda x, y, z: tf.ones_like(x) * (np.pi) * (np.pi)
        u_00 = lambda x, y, z: tf.sin(10 * np.pi * intervalL) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_01 = lambda x, y, z: tf.sin(10 * np.pi * intervalR) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        u_10 = lambda x, y, z: tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * intervalL) * tf.sin(10 * np.pi * z)
        u_11 = lambda x, y, z: tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * intervalR) * tf.sin(10 * np.pi * z)
        u_20 = lambda x, y, z: tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann4':
        fside = lambda x, y, z: tf.ones_like(x)
        u_true = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (2.0 + tf.cos(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(30 * np.pi * z))
        kappa = lambda x, y, z: tf.ones_like(x) * (np.pi) * (np.pi)

        u_00 = lambda x, y, z: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*intervalL)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        u_01 = lambda x, y, z: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*intervalR)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)

        u_10 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*intervalL)*tf.sin(30*np.pi*z)
        u_11 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*intervalR)*tf.sin(30*np.pi*z)

        u_20 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*intervalR)
        return A_eps, kappa, fside, u_true, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann5':
        fside = lambda x, y, z: tf.ones_like(x)
        utrue = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z) + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + tf.cos(10.0*np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z))
        kappa = lambda x, y, z: tf.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z) + 0.05*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        u_01 = lambda x, y, z: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z) + 0.05*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        u_10 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z) + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalL)*tf.sin(10*np.pi*z)
        u_11 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z) + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*intervalR)*tf.sin(10*np.pi*z)
        u_20 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL) + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR) + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann6':
        fside = lambda x, y, z: tf.ones_like(x)
        utrue = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        A_eps = lambda x, y, z: 0.25*(2.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+tf.cos(20.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z))
        kappa = lambda x, y, z: tf.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z) + 0.05*tf.sin(20*np.pi*intervalL)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        u_01 = lambda x, y, z: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z) + 0.05*tf.sin(20*np.pi*intervalR)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
        u_10 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z) + 0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*intervalL)*tf.sin(20*np.pi*z)
        u_11 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z) + 0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*intervalR)*tf.sin(20*np.pi*z)
        u_20 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL) + 0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR) + 0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'Boltzmann8':
        fside = lambda x, y, z: tf.ones_like(x)
        utrue = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)
        A_eps = lambda x, y, z: 0.25 * (2.0 + tf.cos(np.pi*x) * tf.cos(np.pi*y) * tf.cos(np.pi*z) + tf.cos(20.0*np.pi*x) * tf.cos(20.0*np.pi*y) * tf.cos(20.0*np.pi*z))
        kappa = lambda x, y, z: tf.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)
        u_01 = lambda x, y, z: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)
        u_10 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)
        u_11 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)
        u_20 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21


def get_force2Boltzmann3D(x=None, y=None, z=None, equa_name=None):
    if equa_name == 'Boltzmann1':
        utrue = tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        Aeps = 0.25 * (1.0 + tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z))
        ux = 1.0*np.pi*tf.cos(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        uy = 5.0*np.pi*tf.sin(np.pi * x) * tf.cos(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        uz = 10.0*np.pi*tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.cos(10 * np.pi * z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        uyy = -25.0*np.pi*np.pi*tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)
        uzz = -100.0*np.pi*np.pi*tf.sin(np.pi * x) * tf.sin(5 * np.pi * y) * tf.sin(10 * np.pi * z)

        Aepsx = -0.25*np.pi*tf.sin(np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(20 * np.pi * z)
        Aepsy = -0.25*10*np.pi*tf.cos(np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(20 * np.pi * z)
        Aepsz = -0.25*20*np.pi*tf.cos(np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(20 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*utrue
    elif equa_name == 'Boltzmann3':
        utrue = tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        Aeps = 0.5 * (1.0 + tf.cos(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(10 * np.pi * z))
        ux = 10.0 * np.pi * tf.cos(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        uy = 20.0 * np.pi * tf.sin(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        uz = 10.0 * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.cos(10 * np.pi * z)

        uxx = - 100.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        uyy = - 400.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)
        uzz = - 100.0 * np.pi * np.pi * tf.sin(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.sin(10 * np.pi * z)

        Aepsx = -0.5 * 10 * np.pi * tf.sin(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(10 * np.pi * z)
        Aepsy = -0.5 * 20 * np.pi * tf.cos(10 * np.pi * x) * tf.sin(20 * np.pi * y) * tf.cos(10 * np.pi * z)
        Aepsz = -0.5 * 10 * np.pi * tf.cos(10 * np.pi * x) * tf.cos(20 * np.pi * y) * tf.sin(10 * np.pi * z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * utrue
    elif equa_name == 'Boltzmann4':
        utrue = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        Aeps = 0.5 * (2.0 + tf.cos(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(30 * np.pi * z))
        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*10*np.pi*tf.cos(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+0.05*20*np.pi*tf.sin(10*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(30*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+0.05*30*np.pi*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(30*np.pi*z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              0.05*100*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              0.05*400*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              0.05*900*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(30*np.pi*z)

        Aepsx = -0.5*10*np.pi*tf.sin(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.cos(30 * np.pi * z)
        Aepsy = -0.5*20*np.pi*tf.cos(10*np.pi * x) * tf.sin(20 * np.pi * y) * tf.cos(30 * np.pi * z)
        Aepsz = -0.5*30*np.pi*tf.cos(10*np.pi * x) * tf.cos(20 * np.pi * y) * tf.sin(30 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*utrue
    elif equa_name == 'Boltzmann5':
        utrue = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi* z)
        Aeps = 0.5 * (1.0 + tf.cos(10.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi * z))

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.5*np.pi*tf.cos(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+0.5*np.pi*tf.sin(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.sin(10*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+0.5*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.cos(10*np.pi*z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-5.0*np.pi*np.pi*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)

        Aepsx = -0.5*10.0*np.pi*tf.sin(10.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.cos(10.0*np.pi*z)
        Aepsy = -0.5*10.0*np.pi*tf.cos(10.0*np.pi*x)*tf.sin(10.0*np.pi*y)*tf.cos(10.0*np.pi*z)
        Aepsz = -0.5*10.0*np.pi*tf.cos(10.0*np.pi*x)*tf.cos(10.0*np.pi*y)*tf.sin(10.0*np.pi*z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * utrue
        return fside
    elif equa_name == 'Boltzmann6':
        utrue = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)
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

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * utrue
        return fside
    elif equa_name == 'Boltzmann8':
        utrue = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+\
                0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z) + \
                0.01*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)
        Aeps = 0.25*(3.0+tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z)+
                     tf.cos(20.0*np.pi*x)*tf.cos(20.0*np.pi*y)*tf.cos(20.0*np.pi*z)+
                     tf.cos(50.0*np.pi*x)*tf.cos(50.0*np.pi*y)*tf.cos(50.0*np.pi*z))

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)+\
             1.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z) + \
             0.5*np.pi*tf.cos(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)+\
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z) + \
             0.5*np.pi*tf.sin(50*np.pi*x)*tf.cos(50*np.pi*y)*tf.sin(50*np.pi*z)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)+\
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z) + \
             0.5*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.cos(50*np.pi*z)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z) - \
              25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z) - \
              25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)
        uzz = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)-\
              20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z) - \
              25.0*np.pi*np.pi*tf.sin(50*np.pi*x)*tf.sin(50*np.pi*y)*tf.sin(50*np.pi*z)
        Aepsx = -0.25*np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.cos(np.pi*z) - \
                0.25*20.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z) - \
                0.25*50*np.pi*tf.sin(50.0*np.pi*x)*tf.cos(50.0*np.pi*y)*tf.cos(50.0*np.pi*z)
        Aepsy = -0.25*np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z) - \
                0.25*20.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z) - \
                50*np.pi*tf.cos(50.0*np.pi*x)*tf.sin(50.0*np.pi*y)*tf.cos(50.0*np.pi*z)
        Aepsz = -0.25*np.pi*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z) - \
                0.25*20.0*np.pi*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z)-\
                0.25*50*np.pi*tf.cos(50.0*np.pi*x)*tf.cos(50.0*np.pi*y)*tf.sin(50.0*np.pi*z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * utrue
        return fside

    return fside


def get_infos2Boltzmann_4D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann1':
        # mu1= 2*np.pi
        # mu2 = 4*np.pi
        # mu3 = 8*np.pi
        # mu1 = np.pi
        # mu2 = 5 * np.pi
        # mu3 = 10 * np.pi
        mu1 = np.pi
        mu2 = 10 * np.pi
        mu3 = 20 * np.pi
        fside = lambda x, y, z: (mu1*mu1+mu2*mu2+mu3*mu3+x*x+2*y*y+3*z*z)*tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        A_eps = lambda x, y, z: 1.0*tf.ones_like(x)
        kappa = lambda x, y, z: x*x+2*y*y+3*z*z
        utrue = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_00 = lambda x, y, z: tf.sin(mu1*intervalL)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_01 = lambda x, y, z: tf.sin(mu1*intervalR)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_10 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalL)*tf.sin(mu3*z)
        u_11 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalR)*tf.sin(mu3*z)
        u_20 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalL)
        u_21 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalR)

        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann2':
        fside = lambda x, y, z, s: tf.ones_like(x)
        u_true = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(
            5 * np.pi * s)
        A_eps = lambda x, y, z, s: 0.25 * (
                    1.0 + tf.cos(5 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.cos(10 * np.pi * z) * tf.cos(
                5 * np.pi * s))
        u_00 = lambda x, y, z, s: tf.sin(5 * np.pi * intervalL) * tf.sin(10 * np.pi * y) * tf.sin(
            10 * np.pi * z) * tf.sin(5 * np.pi * s)
        u_01 = lambda x, y, z, s: tf.sin(5 * np.pi * intervalR) * tf.sin(10 * np.pi * y) * tf.sin(
            10 * np.pi * z) * tf.sin(5 * np.pi * s)
        u_10 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * intervalL) * tf.sin(
            10 * np.pi * z) * tf.sin(5 * np.pi * s)
        u_11 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * intervalR) * tf.sin(
            10 * np.pi * z) * tf.sin(5 * np.pi * s)
        u_20 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(
            10 * np.pi * intervalL) * tf.sin(5 * np.pi * s)
        u_21 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(
            10 * np.pi * intervalR) * tf.sin(5 * np.pi * s)
        u_30 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(
            5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(
            5 * np.pi * intervalR)

        kappa = lambda x, y, z, s: tf.ones_like(x)*(np.pi)*(np.pi)
        return A_eps, kappa, fside, u_true, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31


def get_force2Boltzmann_4D(x=None, y=None, z=None, s=None, equa_name=None):
    if equa_name == 'Boltzmann2':
        A = 0.25 * (1.0 + tf.cos(5.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(10.0 * np.pi * z) * tf.cos(
            5.0 * np.pi * s))
        Ax = -0.25 * 5.0 * np.pi * tf.sin(5.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(
            10.0 * np.pi * z) * tf.cos(5.0 * np.pi * s)
        Ay = -0.25 * 10.0 * np.pi * tf.cos(5.0 * np.pi * x) * tf.sin(10.0 * np.pi * y) * tf.cos(
            10.0 * np.pi * z) * tf.cos(5.0 * np.pi * s)
        Az = -0.25 * 10.0 * np.pi * tf.cos(5.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.sin(
            10.0 * np.pi * z) * tf.cos(5.0 * np.pi * s)
        As = -0.25 * 5.0 * np.pi * tf.cos(5.0 * np.pi * x) * tf.cos(10.0 * np.pi * y) * tf.cos(
            10.0 * np.pi * z) * tf.sin(5.0 * np.pi * s)
        U = tf.sin(5.0 * np.pi * x) * tf.sin(10.0 * np.pi * y) * tf.sin(10.0 * np.pi * z) * tf.sin(5.0 * np.pi * s)
        Ux = 5 * np.pi * tf.cos(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(5 * np.pi * s)
        Uy = 10 * np.pi * tf.sin(5 * np.pi * x) * tf.cos(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.sin(
            5 * np.pi * s)
        Uz = 10 * np.pi * tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.cos(10 * np.pi * z) * tf.sin(
            5 * np.pi * s)
        Us = 5 * np.pi * tf.sin(5 * np.pi * x) * tf.sin(10 * np.pi * y) * tf.sin(10 * np.pi * z) * tf.cos(5 * np.pi * s)

        Kappa = np.pi*np.pi

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) + 250.0*A*np.pi*np.pi*U + Kappa*U
        return fside


def get_infos2Boltzmann_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale5D_1':
        fside = lambda x, y, z, s, t: 5.0*((np.pi)**2)*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)
        A_eps = lambda x, y, z, s, t: tf.ones_like(x)
        Kappa = lambda x, y, z, s, t: np.pi*np.pi*tf.ones_like(x)
        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL) * tf.sin(np.pi * s) * tf.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * s) * tf.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * intervalL) * tf.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * intervalR) * tf.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi * x) * tf.sin(np.pi * y) * tf.sin(np.pi * z) * tf.sin(np.pi * s) * tf.sin(np.pi * intervalR)
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
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
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)
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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
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
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)
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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_4':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                       + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*\
                                       tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: tf.ones_like(x)

        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)

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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_5':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
                                       0.5*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)
        A_eps = lambda x, y, z, s, t: tf.ones_like(x)
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)

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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_6':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
                                       0.1*tf.sin(5*np.pi*x)*tf.sin(5*np.pi*y)*tf.sin(5*np.pi*z)*tf.sin(5*np.pi*s)*tf.sin(5*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)
        A_eps = lambda x, y, z, s, t: tf.ones_like(x)
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)

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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_7':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                       + 0.05*tf.sin(10*np.pi*x)*tf.sin(10*np.pi*y)*tf.sin(10*np.pi*z)*\
                                       tf.sin(10*np.pi*s)*tf.sin(10*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: 0.5 + 0.5*tf.cos(10*np.pi*x)*tf.cos(10*np.pi*y)*tf.cos(10*np.pi*z)*tf.cos(10*np.pi*s)*tf.cos(10*np.pi*t)
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)

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
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41
    elif equa_name == 'multi_scale5D_7':
        u_true = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)\
                                       + 0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*\
                                       tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        fside = lambda x, y, z, s, t: tf.ones_like(x)

        A_eps = lambda x, y, z, s, t: 0.5 + 0.5*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.cos(20*np.pi*t)
        Kappa = lambda x, y, z, s, t: np.pi * np.pi * tf.ones_like(x)

        u_00 = lambda x, y, z, s, t: tf.sin(np.pi*intervalL)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*intervalL)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        u_01 = lambda x, y, z, s, t: tf.sin(np.pi*intervalR)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*intervalR)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)

        u_10 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalL)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*intervalL)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        u_11 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*intervalR)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)

        u_20 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalL)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*intervalL)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        u_21 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*intervalR)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*intervalR)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)

        u_30 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * intervalL)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*intervalL)*tf.sin(20*np.pi*t)
        u_31 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*intervalR)*\
                                     tf.sin(np.pi*t)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*intervalR)*tf.sin(20*np.pi*t)

        u_40 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi * s)*\
                                     tf.sin(np.pi*intervalL)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*\
                                     tf.sin(np.pi*intervalR)+0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*\
                                     tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*intervalR)
        return u_true, fside, A_eps, Kappa, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41


def get_forceSide2Boltzmann_5D(x=None, y=None, z=None, s=None, t=None, equa_name='multi_scale5D_5'):
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

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt) + np.pi*np.pi*u_true
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

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt) + np.pi*np.pi*u_true
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

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt) + np.pi*np.pi*u_true
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

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt) + np.pi*np.pi*u_true
        return fside
    elif equa_name == 'multi_scale5D_8':
        u_true = tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t)+\
             0.05*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        Aeps = 0.5 + 0.5*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.cos(20*np.pi*t)

        ux = np.pi*tf.cos(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             1.0*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        uy = np.pi*tf.sin(np.pi*x)*tf.cos(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        uz = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.cos(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) + \
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        us = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.cos(np.pi*s)*tf.sin(np.pi*t) + \
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.cos(10*np.pi*s)*tf.sin(20*np.pi*t)
        ut = np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.cos(np.pi*t) + \
             1.0*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.cos(20*np.pi*t)

        uxx = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        uyy = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        uzz =-1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        uss = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)
        utt = -1.0*np.pi*np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(np.pi*z)*tf.sin(np.pi*s)*tf.sin(np.pi*t) - \
             20.0*np.pi*np.pi*tf.sin(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.sin(20*np.pi*t)

        Aepsx = -0.5*20*np.pi*tf.sin(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.cos(20*np.pi*t)
        Aepsy = -0.5*20*np.pi*tf.cos(20*np.pi*x)*tf.sin(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.cos(20*np.pi*t)
        Aepsz = -0.5*20*np.pi*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.sin(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.cos(20*np.pi*t)
        Aepss = -0.5*20*np.pi*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.sin(20*np.pi*s)*tf.cos(20*np.pi*t)
        Aepst = -0.5*20*np.pi*tf.cos(20*np.pi*x)*tf.cos(20*np.pi*y)*tf.cos(20*np.pi*z)*tf.cos(20*np.pi*s)*tf.sin(20*np.pi*t)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt) + np.pi*np.pi*u_true
        return fside