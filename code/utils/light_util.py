import numpy as np
import torch

def add_SHlight(constant_factor, normal_images, sh_coeff):
    '''
        sh_coeff: [bz, 9, 1]
    '''
    N = normal_images
    # sh = torch.stack([
    #         N[:,0]*0.+1., N[:,0], N[:,1], \
    #         N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
    #         N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
    #         ], 
    #         1) # [bz, 9, h, w]
    sh = torch.stack([
    N[:,0]*0.+1., N[:,1], N[:,2], \
    N[:,0], N[:,0]*N[:,1], N[:,1]*N[:,2], 
    3*(N[:,2]**2) - 1, N[:,0]*N[:,2], N[:,0]**2 - N[:,1]**2
    ], 
    1) # [bz, 9, h, w] 
    sh = sh*constant_factor[None,:,None,None]
    shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 1, h, w]  
    return shading

def add_SHlight_att(constant_factor, normal_images, sh_coeff):
    '''
        sh_coeff: [bz, 9, 1]
    '''
    N = normal_images
    # sh = torch.stack([
    #         N[:,0]*0.+1., N[:,0], N[:,1], \
    #         N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
    #         N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
    #         ], 
    #         1) # [bz, 9, h, w]
    sh = torch.stack([
    N[:,0]*0.+1., N[:,1] *2.0/3.0, N[:,2]*2.0/3.0, \
    N[:,0]*2.0/3.0, N[:,0]*N[:,1]/4.0, N[:,1]*N[:,2]/4.0, 
    (3*(N[:,2]**2) - 1)/4.0, N[:,0]*N[:,2]/4.0, (N[:,0]**2 - N[:,1]**2)/4.0
    ], 
    1) # [bz, 9, h, w] 
    sh = sh*constant_factor[None,:,None,None]
    shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 1, h, w]  
    return shading

def add_sample_SHlight(constant_factor, normal_images, sh_coeff):
    '''
        normal_images: [bz, 3]
        constant_factor: [9]
        sh_coeff: [9]
    '''
    N = normal_images
    sh = torch.stack([
    N[:,0]*0.+1., N[:,1], N[:,2], \
    N[:,0], N[:,0]*N[:,1], N[:,1]*N[:,2], 
    3*(N[:,2]**2) - 1, N[:,0]*N[:,2], N[:,0]**2 - N[:,1]**2
    ], 
    1) # [bz, 9] 
    sh = sh*constant_factor[None,:] # [bz, 9] 
    shading = torch.sum(sh_coeff[None,:,None]*sh[:,:,None], 1) # [bz, 9, 1]  
    return shading # [bz, 1]  

def SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis

def SH_basis_noatt(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)
    return sh_basis

def get_shading(normal, SH):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis_noatt(normal)
    # sh_basis = SH_basis_noatt(normal)
    shading = np.matmul(sh_basis, SH)
    return shading

def draw_shading(sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256

    x = np.linspace(-1, 1, img_size)
    y = np.linspace(1, -1, img_size)
    x, y = np.meshgrid(x, y)

    mag = np.sqrt(x**2 + y**2)
    valid = mag <=1
    z = np.sqrt(1 - (x*valid)**2 - (y*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh.detach().cpu().numpy())
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    return shading

def normal_shading(sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(1, -1, img_size)
    x, y = np.meshgrid(x, y)

    mag = np.sqrt(x**2 + y**2)
    valid = mag <=1
    z = np.sqrt(1 - (x*valid)**2 - (y*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh.detach().cpu().numpy())
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    return normal, shading

def normal_shading_env(sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(1, -1, img_size)
    x, y = np.meshgrid(x, y)

    mag = np.sqrt(x**2 + y**2)
    valid = mag <=1
    z = np.sqrt(1 - (x*valid)**2 - (y*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh.detach().cpu().numpy())
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256, 3))
    shading = shading * valid.reshape(256, 256 ,1)

    return normal, shading

def normal_shading_sh(sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid

    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh.detach().cpu().numpy())
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    return normal, shading

def normal_shading_sh_env(sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid

    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh.detach().cpu().numpy())
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256, 3))
    shading = shading * valid.reshape(256, 256 ,1)

    return normal, shading

def normalize(x):
    return x/np.linalg.norm(x, axis=1)[:, np.newaxis]