import numpy as np
import scipy.interpolate as itpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d

# Generate grid coordinates 
def genCoords(N, xmin, xmax, ymin, ymax):
    '''
    Use this to generate meshgrid for interpolation.

    Args
        N - number of points
        xmin, xmax - min and max of x coordinate
        ymin, ymax - min and max of y coordinate

    Returns
        x_mesh, y_mesh from meshgrid
    '''
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    x_mesh, y_mesh = np.meshgrid(x, y)

    return x_mesh, y_mesh

def findSource(x, y, amp, x_mesh, y_mesh, method='multiquadric'):
    '''
    Args
        x, y - coordinates of receiver
        amp - sound amplitude
        x_mesh, y_mesh - gridded region to compute interpolation over
        method - scipy.interpolate.rbf basis function

    Returns
        Tuple containing (
                interpolated amplitudes, 
                gradx, 
                grady, 
                max amplitude,
                max amplitude location in (x, y),
                max gradx,
                max gradx location in (x, y),
                max grady,
                max grady location in (x, y)
             )        

        We only really need the max amplitude location in (x, y).
        Unpack the tuple. 
    '''
    # Create RBF interpolator instance
    rbfi = itpl.Rbf(x, y, amp, function=method, smooth=0)

    # Do interpolation
    itpl_amp = rbfi(x_mesh, y_mesh)

    # Compute gradient
    gx, gy = np.gradient(itpl_amp)

    # Get location of maximum amp
    maxamp = np.amax(itpl_amp)
    listloc_maxamp = np.unravel_index(np.argmax(itpl_amp), itpl_amp.shape)
    locx_maxamp = x_mesh[listloc_maxamp[0]][listloc_maxamp[1]]
    locy_maxamp = y_mesh[listloc_maxamp[0]][listloc_maxamp[1]]
    locmax = np.array([locx_maxamp, locy_maxamp])
    
    # Get location of maximum gradient
    # x direction
    maxgradx = np.amax(gx)

    # gives index of max point in the 2d array
    listloc_maxgradx = np.unravel_index(np.argmax(gx), gx.shape)

    # get x, y coords
    locx_maxgradx = x_mesh[listloc_maxgradx[0]][listloc_maxgradx[1]]
    locy_maxgradx = y_mesh[listloc_maxgradx[0]][listloc_maxgradx[1]]
    locmaxgradx = np.array([locx_maxgradx, locy_maxgradx])

    # y direction 
    maxgrady = np.amax(gy)
    
    # gives index of max point in the 2d array
    listloc_maxgrady = np.unravel_index(np.argmax(gy), gy.shape)
    
    # get x, y coords
    locx_maxgrady = x_mesh[listloc_maxgrady]
    locy_maxgrady = y_mesh[listloc_maxgrady]
    locmaxgrady = np.array([locx_maxgrady, locy_maxgrady])

    return itpl_amp, gx, gy, maxamp, locmax, \
        maxgradx, locmaxgradx, maxgrady, locmaxgrady


'''
============================================================
TEST
============================================================
============================================================
TEST FUNCTIONS
============================================================
'''
# Easy test function 1
def funcEasy(x, y):
    return np.cos(np.pi * x) * np.sin(np.pi * y)

# Hard test function 
def funcHard(x, y):
    return x * y / (np.power(x, 2) + np.power(y, 2))

# Hard test function 
def funcHard2(x, y):
    return x * np.power(y, 2) / (np.power(x, 3) + np.power(y, 2))

# Function that models sound amplitude
# Approximation
def funcSound(x, y, P0):
    '''
    beta = (10 dB) { log [P0 / (4Pi I0)] - 2 log (R) }
    log = log base 10

    beta: loudness of sound, measured in dB
    P0: power produced by source
    I0: threshold of hearing (=10E-12 W/m2)
    '''
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    I0 = 10e-12
    return 10 * (np.log10(P0 / (4 * np.pi * I0)) - 2 * np.log10(r))

# Function that models attenuated sound amplitude
# Approximation
def funcSoundAtt(x, y, P0, a):
    '''
    beta = (10 dB) { log [P0 / (4Pi I0)] - 2 log (R) - a * R}
    log = log base 10

    beta: loudness of sound, measured in dB
    P0: power produced by source
    I0: threshold of hearing (=10E-12 W/m2)
    '''
    r = np.sqrt(np.power(x,2) + np.power(y, 2))
    I0 = 10e-12
    return 10 * (np.log10(P0 / (4 * np.pi * I0)) - 2 * np.log10(r) + a * r)

def funcSin(x, y):
    r = np.sqrt(np.power(x,2) + np.power(y, 2))
    return np.sin(10*r) / r

'''
============================================================
'''


def getErr(func, locmax):
    '''
    True max locations:
    funcEasy : [-1. -0.49989998]
    funcHard : [-1. -1.]
    funcHard2 : [-0.11111111  0.03703704]
    funcSound : [-0.001001 -0.001001]
    funcSoundAtt : [-0.001001 -0.001001]
    funcSin : [-0.001001 -0.001001]
    '''
    locmaxTrue = np.zeros(2)

    if func.__name__ == 'funcEasy':
        locmaxTrue = np.array([-1., -0.49989998])
    
    elif func.__name__ == 'funcHard':
        locmaxTrue = np.array([-1., -1.])

    elif func.__name__ == 'funcHard2':
        locmaxTrue = np.array([-0.11111111, 0.03703704])

    elif func.__name__ == 'funcSound':
        locmaxTrue = np.array([0., 0.])

    elif func.__name__ == 'funcSoundAtt':
        locmaxTrue = np.array([0., 0.])

    elif func.__name__ == 'funcSin':
        locmaxTrue = np.array([0., 0.])

    err_ = locmaxTrue - locmax
    '''
    print(f'loc max True = {locmaxTrue}')
    print(f'loc max = {locmax}')
    print(err_)
    '''
    err = np.linalg.norm(err_)
    return err

def test(N, func, xmin, xmax, ymin, ymax, method, *args):
    # Generate scattered datapoints
    x_itpl, y_itpl = np.random.uniform(-1, 1, (2, N))
    
    # Generate grid points
    x_mesh, y_mesh = genCoords(100, xmin, xmax, ymin, ymax)
    x_mesh_test = x_mesh.astype(int)[:, 0]
    y_mesh_test = y_mesh.astype(int)[0, :]
    
    #print(f'x_mesh_test = {x_mesh_test}')
    #print(f'x_mesh_test size = {x_mesh_test.shape}')

    amp = func(x_itpl, y_itpl, *args)
    amp_og = func(x_mesh, y_mesh, *args)
 
    # Get location of actual maximum amp
    maxamp_og = np.amax(amp_og)
    listloc_maxamp_og = np.unravel_index(np.argmax(amp_og), amp_og.shape)
    locx_maxamp_og = x_mesh[listloc_maxamp_og[0]][listloc_maxamp_og[1]]
    locy_maxamp_og = y_mesh[listloc_maxamp_og[0]][listloc_maxamp_og[1]]
    locmax_og = np.array([locx_maxamp_og, locy_maxamp_og])
    
    out_amp = findSource(x_itpl, y_itpl, amp, x_mesh, y_mesh, method)
    
    itpl_amp, gx_amp, gy_amp, maxamp, locmax, maxgradx, locmaxgradx, maxgrady, locmaxgrady = out_amp

    err = getErr(func, locmax)

    print('===================================================')
    print(f'function: {func.__name__}')
    print(f'method: {method}')
    print(f'N = {N}')
    print(f'\nmax gradx = {maxgradx}')
    print(f'max grady = {maxgrady}')

    print(f'\nlocation of max gradx = {locmaxgradx}')
    print(f'location of max grady = {locmaxgrady}')

    print(f'\nmax amp = {maxamp}')
    print(f'location of max amp = {locmax}')

    print(f'\nactual max amp = {maxamp_og}')
    print(f'actual location of max amp = {locmax_og}')

    print(f'\nError: {err}')


    print('\n\n')

    # Plot
    # Interpolation plot
    fig1 = plt.figure(1, figsize=(11., 5.))
    ax1 = plt.subplot("131")
    plt.contourf(x_mesh, y_mesh, itpl_amp, cmap=plt.cm.gray)
    plt.colorbar()
    ax1.scatter(x_itpl, y_itpl, marker='.', color='orange')
    ax1.scatter(locmax[0], locmax[1], color='black')
    ax1.scatter(locmaxgradx[0], locmaxgradx[1], marker='>')
    ax1.scatter(locmaxgrady[0], locmaxgrady[1], marker='^', color='green')

    # Gradient in x direction
    ax1 = plt.subplot("132")
    ax1.set_title('gradient in x direction')
    ax1.imshow(gx_amp, extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.gray)
    ax1.autoscale(False)
    ax1.scatter(locmax[0], locmax[1], color='black')
    ax1.scatter(locmaxgradx[0], locmaxgradx[1], marker='>')

    # Gradient in y direction
    ax1 = plt.subplot("133")
    ax1.set_title('gradient in y direction')
    ax1.imshow(gy_amp, extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.gray)
    ax1.autoscale(False)
    ax1.scatter(locmax[0], locmax[1], color='black')
    ax1.scatter(locmaxgrady[0], locmaxgrady[1], marker='^')

    fig1.tight_layout()
    
    # Surface plot
    fig2 = plt.figure(2, figsize=(12, 5))
    ax2 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax2.set_title('Interpolated')
    ax2.plot_surface(x_mesh, y_mesh, itpl_amp)

    ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('Original')
    ax2.plot_surface(x_mesh, y_mesh, amp_og)

    fig1.savefig(func.__name__ + '_' + method + '_' + str(N) + '_interp')
    fig2.savefig(func.__name__ + '_' + method + '_' + str(N))

    plt.close(fig1)
    plt.close(fig2)
