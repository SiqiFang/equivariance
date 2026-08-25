'''
Rotation and reflection equivariance using moment kernels.

We support scalar fields, vector fields, convolution maps between them, batch norm, and nonlinearity.


'''
import torch
import numpy as np
import os as _os
import warnings
from itertools import permutations,product

# Optional spherical (corner-zeroed) kernel support. If EQ_KERNEL_RADIUS is set,
# every OEConv kernel is masked to taps with radius D <= r_max (a radial mask,
# constant within each octahedral shell -> equivariance preserved; it only
# reweights radial shells to zero). Unset (default) = full cube support, identical
# to prior behavior. For ks=5 (taps at radius up to sqrt(12)~3.46): r_max=2.0 gives
# a clean inscribed-sphere support; r_max=2.9 drops just the (2,2,2) corners.
def _kernel_radius():
    v = _os.environ.get("EQ_KERNEL_RADIUS", "")
    return float(v) if v else None


class ScalarToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels == 0:
            self.forward = forward_empty
            return
        
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3.0*in_channels)) # TODO: use the right normalizatoin
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))       
    def forward_empty(self,x):
        ''' 
        Return an array that's the same size as the input but with 0 channels
        This can be used to concatenate with other arguments
        Note this requires a batch dimension
        TODO: compute the correct size with respect to padding and kernel size
        I'm not sure if this is a good approach.
        '''
        return torch.zeros((x.shape[0],0,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        
    def forward(self,x):
        # note this works with size 1 images, as long as padding is 0
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size        
        c = self.weights[...,self.inds]
        self.c = c
                        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias)
    
    
        
        
class ScalarToScalar3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # build a 3D grid of positions
        r = (kernel_size - 1)//2
        x = torch.arange(-r, r+1)
        X = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), -1)  # D x H x W x 3
        R = torch.sqrt(torch.sum(X**2, -1))                           # D x H x W
        rs, inds = torch.unique(R, return_inverse=True)
        self.register_buffer('inds', inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels, in_channels, len(rs)) / np.sqrt(3.0 * in_channels))
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels) / np.sqrt(3.0))

    def forward(self, x):
        # construct kernel: out x in x D x H x W
        c = self.weights[..., self.inds]
        p = self.padding
        tmp = torch.nn.functional.pad(x, (p, p, p, p, p, p), mode=self.padding_mode)
        return torch.nn.functional.conv3d(tmp, c, self.bias)


class ScalarToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, out channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[:,None]
        Xhat = Xhat.repeat((out_channels,1,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        
        inds = inds - 1 # we will not use r=0.  the filter will get assigned a different number, but then multiplied by 0
        inds[inds==-1] = 0        
        self.register_buffer('inds',inds) # don't need a parameter for r=0, but this makes
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # TODO: use the right normalizatoin
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):        
        # kernel size 1 needs to be a special case because self.inds is empty, the result is just 0
        # no padding allowed
        # note we assume square
        return torch.zeros(x.shape[0],self.out_channels*2,1,1,dtype=x.dtype,device=x.device)
    
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size          
        
        c = torch.repeat_interleave(self.weights,2,0)[...,self.inds]*self.Xhat
        self.c = c
        
        
        # for somme reason the output is not zero mean, has to do with padding
        # here's a better way
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)
        return torch.nn.functional.conv2d(tmp,c)
        
        

class ScalarToVector90(torch.nn.Module):
    '''This module adds an additional basis function with a 90 degree rotation'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, out channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[:,None]
        X90hat = Xhat.flip(0)*torch.tensor([-1.0,1.0])[:,None,None,None]
        Xhat = Xhat.repeat((out_channels,1,1,1))
        X90hat = X90hat.repeat((out_channels,1,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)        
        self.register_buffer('X90hat',X90hat)
        
        inds = inds - 1 # we will not use r=0.  the filter will get assigned a different number, but then multiplied by 0
        inds[inds==-1] = 0        
        self.register_buffer('inds',inds) # don't need a parameter for r=0, but this makes
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # 
        self.weights90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # TODO: use the right normalizatoin
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):        
        # kernel size 1 needs to be a special case because self.inds is empty, the result is just 0
        # no padding allowed
        # note we assume square
        return torch.zeros(x.shape[0],self.out_channels*2,1,1,dtype=x.dtype,device=x.device)
    
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size          
        
        c = torch.repeat_interleave(self.weights,2,0)[...,self.inds]*self.Xhat
        c90 = torch.repeat_interleave(self.weights90,2,0)[...,self.inds]*self.X90hat
        self.c = c + c90
        
        
        # for somme reason the output is not zero mean, has to do with padding
        # here's a better way
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)
        return torch.nn.functional.conv2d(tmp,c)
        

        
def rotate_vector_and_image(x):
    with torch.no_grad():
        tmp = x.rot90(1,(-1,-2))
        tmp2 = tmp.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2

def rotate_vector(x):
    with torch.no_grad():  
        tmp = x.clone()
        tmp2 = x.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2




class VectorToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[None]
        Xhat = Xhat.repeat((1,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        inds = inds - 1
        inds[inds<0] = 0
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
        else:
            self.bias = None
            
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):
        # size 1 is a special case because there are no parameters, just return 0 + bias
        # self.ind is empty
        return torch.zeros(x.shape[0],self.out_channels,1,1,dtype=x.dtype,device=x.device) + self.bias[...,None,None]
    
    def forwardg1(self,x):
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        c = torch.repeat_interleave(self.weights[...,self.inds],2,1)*self.Xhat                
        self.c = c
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
        

class VectorToScalar90(torch.nn.Module):
    ''' In this version we include the extra basis rotated by 90 degrees'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[None]
        X90hat = Xhat.flip(1)*torch.tensor([-1.0,1.0])[None,:,None,None]
        Xhat = Xhat.repeat((1,in_channels,1,1))
        X90hat = X90hat.repeat((1,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        inds = inds - 1
        inds[inds<0] = 0
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('X90hat',X90hat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        self.weights90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
        else:
            self.bias = None
            
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):
        # size 1 is a special case because there are no parameters, just return 0 + bias
        # self.ind is empty
        return torch.zeros(x.shape[0],self.out_channels,1,1,dtype=x.dtype,device=x.device) + self.bias[...,None,None]
    
    def forwardg1(self,x):
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        c = torch.repeat_interleave(self.weights[...,self.inds],2,1)*self.Xhat + torch.repeat_interleave(self.weights90[...,self.inds],2,1)*self.X90hat
        self.c = c
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
                
        
        
class VectorToVector(torch.nn.Module):
    '''Question, should I separate these into two types and interleave them somehow rather than combining them'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None]        
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        # special case if kernel is size 1
        # print(x.shape)
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    def forwarde1(self,x):
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity[...,self.indsidentity],2,0),2,1)*self.identity
        self.cidentity = cidentity
        return torch.nn.functional.conv2d(x,cidentity)
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        cxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx,2,0),2,1)[...,self.indsxx]*self.XhatXhat
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity,2,0),2,1)[...,self.indsidentity]*self.identity
        c = cxx + cidentity
        self.c = c
        self.cxx = cxx
        self.cidentity = cidentity
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is vector

    
class VectorToVector90(torch.nn.Module):
    '''Question, should I separate these into two types and interleave them somehow rather than combining them
    This one uses the extra basis functions rotated by 90 degrees.
    '''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)        
        X90hat = Xhat.flip(0)*torch.tensor([-1.0,1.0])[...,None,None]
        # now there are 4
        XhatXhat = Xhat[None,:]*Xhat[:,None]
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        X90hatXhat = X90hat[None,:]*Xhat[:,None]
        X90hatXhat = X90hatXhat.repeat((out_channels,in_channels,1,1))
        XhatX90hat = Xhat[None,:]*X90hat[:,None]
        XhatX90hat = XhatX90hat.repeat((out_channels,in_channels,1,1))
        X90hatX90hat = X90hat[None,:]*X90hat[:,None]
        X90hatX90hat = X90hatX90hat.repeat((out_channels,in_channels,1,1))
        
        
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('X90hatXhat',X90hatXhat)
        self.register_buffer('XhatX90hat',XhatX90hat)
        self.register_buffer('X90hatX90hat',X90hatX90hat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsx90x = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxx90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsx90x90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        # special case if kernel is size 1
        # print(x.shape)
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    def forwarde1(self,x):
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity[...,self.indsidentity],2,0),2,1)*self.identity
        self.cidentity = cidentity
        return torch.nn.functional.conv2d(x,cidentity)
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        cxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx,2,0),2,1)[...,self.indsxx]*self.XhatXhat
        cx90x = torch.repeat_interleave(torch.repeat_interleave(self.weightsx90x,2,0),2,1)[...,self.indsxx]*self.X90hatXhat
        cxx90 = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx90,2,0),2,1)[...,self.indsxx]*self.XhatX90hat
        cx90x90 = torch.repeat_interleave(torch.repeat_interleave(self.weightsx90x90,2,0),2,1)[...,self.indsxx]*self.X90hatX90hat
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity,2,0),2,1)[...,self.indsidentity]*self.identity
        c = cxx + cx90x + cxx90 + cx90x90 + cidentity
        self.c = c
        self.cxx = cxx # don't really need this, but may want to look at it later
        self.cidentity = cidentity
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is vector
    
class ScalarVectorToScalarVector(torch.nn.Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        
        self.in_scalars = in_scalars
        self.in_vectors = in_vectors
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        
        if in_scalars > 0 and out_scalars > 0:
            self.ss = ScalarToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_scalars > 0 and out_vectors > 0:
            self.sv = ScalarToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
        if in_vectors > 0 and out_scalars > 0:
            self.vs = VectorToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_vectors > 0 and out_vectors > 0:
            self.vv = VectorToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
            
        # it seems there are 16 total possibilities for forward functions given missing data
        # perhaps we could handle these cases above?
        
        
    def forward(self,x):
        # TODO implement this without if statements
        outs = torch.zeros((x.shape[0],self.out_scalars,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        outv = torch.zeros((x.shape[0],self.out_vectors*2,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        #print(outs.shape,outv.shape)
        if self.in_scalars > 0 and self.out_scalars > 0:
            outs = outs + self.ss( x[:,:self.in_scalars])
        if self.in_scalars > 0 and self.out_vectors > 0:
            outv = outv + self.sv( x[:,:self.in_scalars])
        if self.in_vectors > 0 and self.out_scalars > 0:
            outs = outs + self.vs(x[:,self.in_scalars:])
        if self.in_vectors > 0 and self.out_vectors > 0:
            outv = outv + self.vv(x[:,self.in_scalars:])        
        #print(outs.shape,outv.shape)
        
        return torch.cat(  ( outs, outv )  , dim=-3)
    
    
class Downsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,x):
        # downsample on the last two (2D) or last three (3D) spatial dimensions by a factor of 2
        # if it is even, we average
        # if it is odd we skip
        if not x.shape[-1]%2: # if even
            x = (x[...,0::2] + x[...,1::2])*0.5
        else:
            x = x[...,0::2]

        if not x.shape[-2]%2: # if even
            x = (x[...,0::2,:] + x[...,1::2,:])*0.5
        else:
            x = x[...,0::2,:]

        if x.ndim == 5: # 3D volume: also downsample depth dimension
            if not x.shape[-3]%2: # if even
                x = (x[...,0::2,:,:] + x[...,1::2,:,:])*0.5
            else:
                x = x[...,0::2,:,:]

        return x
            
    
class Upsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,roweven=True,coleven=True,deptheven=True):

        if coleven:
            x = torch.repeat_interleave(x,2,dim=-1)
        else:
            # if odd we insert zeros
            x = (torch.repeat_interleave(x,2,dim=-1) * (1-torch.arange(2*x.shape[-1])%2))[...,:-1]

        if roweven:
            x = torch.repeat_interleave(x,2,dim=-2)
        else:
            x = (torch.repeat_interleave(x,2,dim=-2) * (1-torch.arange(2*x.shape[-2])%2)[...,None])[...,:-1,:]

        if x.ndim == 5: # 3D volume: also upsample depth dimension
            if deptheven:
                x = torch.repeat_interleave(x,2,dim=-3)
            else:
                x = (torch.repeat_interleave(x,2,dim=-3) * (1-torch.arange(2*x.shape[-3])%2)[...,None,None])[...,:-1,:,:]

        return x

    
    
# as before, the sigmoid is causing a problem that leads to nans
# perhaps because the sqrt has an infinite slope at x=0?
class VectorSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #return torch.relu(x)
        #return torch.abs(x)
        # the vector has some multiple of 2 chanels
        
        x2 = x**2
        l2 = x2[:,0::2] + x2[:,1::2] + 1e-6
        #l2r = torch.repeat_interleave(l2,2,dim=1)
        #return x * l2r / (l2r + 1.0)
        #return x / torch.sqrt((l2r + 1.0))
        #return x*torch.relu(l2r-1)/l2r
        l = torch.sqrt(l2)
        # now I have the length of each vector
        lr = torch.repeat_interleave(l,2,dim=1)
        # now it is repeated
        return x*torch.relu((lr-1.0))/lr
        
        #return torch.relu(x)
class VectorSigmoidLog(torch.nn.Module):
    '''This one is just relu on the log magnitude'''
    def __init__(self,ep=1e-6):
        super().__init__()
        self.ep = ep
    def forward(self,x):
        # first get magnitude
        x2 = x**2
        l2 = x2[:,0::2] + x2[:,1::2] + self.ep
        logl2 = torch.log(l2)
        newlogl2 = torch.relu(logl2)
        factor = ( (newlogl2 - logl2)*0.5 ).exp()
        return x*factor.repeat_interleave(2,dim=1)
        
        
        
class ScalarSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #l = torch.sqrt(x**2 + 1e-5)
        #return x*torch.relu((l-1.0))/l
        return torch.relu(x)
        
class ScalarVectorSigmoid(torch.nn.Module):
    def __init__(self,n_scalars):
        super().__init__()
        self.n_scalars = n_scalars
        self.s = ScalarSigmoid()
        self.v = VectorSigmoid()
    def forward(self,x):
        return torch.cat((self.s(x[:,:self.n_scalars]), self.v(x[:,self.n_scalars:])),-3)
        
        
class ScalarBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):
        return self.b(x)
        
        
class ScalarBatchnorm3D(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.b = torch.nn.BatchNorm3d(n)
    def forward(self, x):
        return self.b(x)


class VectorBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):                
        magnitude2 = x[:,0::2]**2 + x[:,1::2]**2 + 1e-6
        logmagnitude2 = torch.log(magnitude2)
        #scaledlogmagnitude2 = self.b(logmagnitude2)
        # let's think about this normalization
        # do I really need the 0.5 below?
        
        #return x * torch.repeat_interleave((  (scaledlogmagnitude2 - logmagnitude2)*0.5 ).exp(),2,dim=1)

        logmagnitude = 0.5*torch.log(magnitude2)
        scaledlogmagnitude = self.b(logmagnitude)
        return x * torch.repeat_interleave((  (scaledlogmagnitude - logmagnitude) ).exp(),2,dim=1)
        
class ScalarVectorBatchnorm(torch.nn.Module):
    def __init__(self,nscalar,nvector):
        super().__init__()
        self.nscalar = nscalar
        self.nvector = nvector
        self.bs = ScalarBatchnorm(nscalar)
        self.bv = VectorBatchnorm(nvector)
    def forward(self,x):
        return torch.cat( (self.bs(x[:,:self.nscalar]),self.bv(x[:,self.nscalar:])) , 1)
    
    
    
class ScalarToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        super().__init__()
        # what's the main idea here?
        # for the identity, we can do a regular conv, then multiply by identity
        # for the xx we have to actually do the bigger convolution
        # since we're doing the bigger convolution, we might as well just do it
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None] # 2x2xkxk
        XhatXhat = XhatXhat.reshape(4,1,kernel_size,kernel_size) # 4x1xkxk
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).reshape(4,1).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
    def forward(self,x):
        # note
        # the input is going to have in_channels
        # the output is going to have out_channels*4              
        cxx = torch.repeat_interleave(self.weightsxx,4,0)[...,self.indsxx] * self.XhatXhat
        cidentity = torch.repeat_interleave(self.weightsidentity,4,0)[...,self.indsidentity]*self.identity
        c = cxx + cidentity        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is matrix
                
        
        
class MatrixToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        super().__init__()
        # what's the main idea here?
        # for the identity, we can do a regular conv, then multiply by identity
        # for the xx we have to actually do the bigger convolution
        # since we're doing the bigger convolution, we might as well just do it
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None]      
        XhatXhat = XhatXhat.reshape(1,4,kernel_size,kernel_size)
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).reshape(1,4).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))       
        
    def forward(self,x):                   
        cxx = torch.repeat_interleave(self.weightsxx,4,1)[...,self.indsxx]*self.XhatXhat
        cidentity = torch.repeat_interleave(self.weightsidentity,4,1)[...,self.indsidentity]*self.identity
        c = cxx + cidentity        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
    
class MatrixToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxx = inds.clone()-1
        indsxxx[indsxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # now we have these guys
        XXX = Xhat[:,None,None]*Xhat[None,:,None]*Xhat[None,None,:]
        # or
        XDD = Xhat[:,None,None] * identity[None,:,:]
        # or
        DXD = Xhat[None,:,None]*identity[:,None,:]
        # or
        DDX = identity[:,:,None]*Xhat[None,None,:]
        # now reshape them and tile them
        XXX = XXX.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDD = XDD.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXD = DXD.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDX = DDX.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
                                                           
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXX',XXX)
        self.register_buffer('XDD',XDD)
        self.register_buffer('DXD',DXD)
        self.register_buffer('DDX',DDX)
        self.register_buffer('indsxxx',indsxxx)
                
        self.weightsxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        
        
    def forward(self,x):                           
        cxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxx,4,1),2,0)[...,self.indsxxx]*self.XXX
        cddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddx,4,1),2,0)[...,self.indsxxx]*self.DDX
        cdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxd,4,1),2,0)[...,self.indsxxx]*self.DXD
        cxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdd,4,1),2,0)[...,self.indsxxx]*self.XDD
        
        
        c = cxxx + cddx + cdxd + cxdd        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     

class VectorToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxx = inds.clone()-1
        indsxxx[indsxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # now we have these guys
        XXX = Xhat[:,None,None]*Xhat[None,:,None]*Xhat[None,None,:]
        # or
        XDD = Xhat[:,None,None] * identity[None,:,:]
        # or
        DXD = Xhat[None,:,None]*identity[:,None,:]
        # or
        DDX = identity[:,:,None]*Xhat[None,None,:]
        # now reshape them and tile them
        XXX = XXX.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDD = XDD.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXD = DXD.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDX = DDX.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXX',XXX)
        self.register_buffer('XDD',XDD)
        self.register_buffer('DXD',DXD)
        self.register_buffer('DDX',DDX)
        self.register_buffer('indsxxx',indsxxx)
        
        
        self.weightsxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        
        
    def forward(self,x):                           
        cxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxx,2,1),4,0)[...,self.indsxxx]*self.XXX
        cddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddx,2,1),4,0)[...,self.indsxxx]*self.DDX
        cdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxd,2,1),4,0)[...,self.indsxxx]*self.DXD
        cxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdd,2,1),4,0)[...,self.indsxxx]*self.XDD
        
        
        c = cxxx + cddx + cdxd + cxdd        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     

    
class MatrixToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxxx = inds.clone()-1
        indsxxxx[indsxxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # first all Xs (1)
        XXXX = Xhat[:,None,None,None]*Xhat[None,:,None,None]*Xhat[None,None,:,None]*Xhat[None,None,None,:]
        # now with one identity (6)         
        XXDD = Xhat[:,None,None,None]*Xhat[None,:,None,None]*identity[None,None,:,:]
        # or
        XDXD = Xhat[:,None,None,None]*identity[None,:,None,:]*Xhat[None,None,:,None]
        # or
        XDDX = Xhat[:,None,None,None]*identity[None,:,:,None]*Xhat[None,None,None,:]
        # or
        DXXD = identity[:,None,None,:]*Xhat[None,:,None,None]*Xhat[None,None,:,None]
        # or
        DXDX = identity[:,None,:,None]*Xhat[None,:,None,None]*Xhat[None,None,None,:]
        # or
        DDXX = identity[:,:,None,None]*Xhat[None,None,:,None]*Xhat[None,None,None,:]
        # now with two identities (2)
        DDDD0 = identity[:,:,None,None]*identity[None,None,:,:]
        DDDD1 = identity[:,None,:,None]*identity[None,:,None,:]
        DDDD2 = identity[:,None,None,:]*identity[None,:,:,None]
        
        
        # now reshape them and tile them        
        XXXX = XXXX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XXDD = XXDD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDXD = XDXD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDDX = XDDX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXXD = DXXD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXDX = DXDX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDXX = DDXX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDDD0 = DDDD0.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        DDDD1 = DDDD1.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        DDDD2 = DDDD2.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        

        
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXXX',XXXX)
        self.register_buffer('XXDD',XXDD)        
        self.register_buffer('XDXD',XDXD)
        self.register_buffer('XDDX',XDDX)
        self.register_buffer('DXXD',DXXD)
        self.register_buffer('DXDX',DXDX)
        self.register_buffer('DDXX',DDXX)
        self.register_buffer('DDDD0',DDDD0)
        self.register_buffer('DDDD1',DDDD1)
        self.register_buffer('DDDD2',DDDD2)
        
        self.register_buffer('indsxxxx',indsxxxx)
        
        self.register_buffer('indsidentity',indsidentity)
        
        
        self.weightsxxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxdx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdddd0 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.weightsdddd1 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.weightsdddd2 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        
        
    def forward(self,x):                           
        cxxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxxx,4,1),4,0)[...,self.indsxxxx]*self.XXXX
        cxxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxdd,4,1),4,0)[...,self.indsxxxx]*self.XXDD
        cxdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdxd,4,1),4,0)[...,self.indsxxxx]*self.XDXD
        cxddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxddx,4,1),4,0)[...,self.indsxxxx]*self.XDDX
        cdxxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxxd,4,1),4,0)[...,self.indsxxxx]*self.DXXD
        cdxdx = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxdx,4,1),4,0)[...,self.indsxxxx]*self.DXDX
        cddxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddxx,4,1),4,0)[...,self.indsxxxx]*self.DDXX
        cdddd0 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd0,4,1),4,0)[...,self.indsidentity]*self.DDDD0
        cdddd1 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd1,4,1),4,0)[...,self.indsidentity]*self.DDDD1
        cdddd2 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd2,4,1),4,0)[...,self.indsidentity]*self.DDDD2
        
        
        c = cxxxx + cxxdd + cxdxd + cxddx + cdxxd + cdxdx + cddxx + cdddd0 + cdddd1 + cdddd2 
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     
    
class MatrixSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
                        
        x2 = x**2
        l2 = x2[:,0::4] + x2[:,1::4] + x2[:,2::4] + x2[:,3::4] + 1e-6        
        l = torch.sqrt(l2)
        # now I have the length of each vector
        lr = torch.repeat_interleave(l,4,dim=1)
        # now it is repeated
        return x*torch.relu((lr-1.0))/lr
                
class MatrixBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):                
        magnitude2 = x[:,0::4]**2 + x[:,1::4]**2 + x[:,2::4]**2 + x[:,3::4]**2 + 1e-6
        logmagnitude2 = torch.log(magnitude2)
        scaledlogmagnitude2 = self.b(logmagnitude2)
        
        return x * torch.repeat_interleave((  (scaledlogmagnitude2 - logmagnitude2)*0.5 ).exp(),4,dim=1)
        
        
        
        
        
# finally some tools for general tensors and dimension


def _radial_interp_matrix(xspace, dunique, nspace, mode, sigma_factor=1.0):
    """Build the (len(dunique) x nspace) matrix that maps `nspace` knot
    weights to per-unique-radius kernel coefficients.

    All three modes return the same shape, so they're drop-in interchangeable
    in the OEConv kernel construction. They differ only in smoothness:
      - 'linear':   piecewise-linear between knots (C^0, kinked).
      - 'cubic':    natural cubic spline (C^2 smooth, S''(endpoints)=0).
      - 'gaussian': Gaussian basis at fixed centers (xspace) with fixed width
                    sigma = sigma_factor * knot_spacing. Only the amplitudes
                    (the per-knot weights) are learnable.
    """
    dtype = xspace.dtype

    if mode == 'linear':
        dxspace = xspace[1] - xspace[0]
        interp = torch.zeros(len(dunique), nspace, dtype=dtype)
        for i in range(len(dunique)):
            ind0 = (dunique[i] / dxspace).floor().int()
            ind1 = ind0 + 1
            p = dunique[i] / dxspace - ind0
            interp[i, ind0] = 1 - p
            if ind1 < nspace:
                interp[i, ind1] = p
        return interp

    if mode == 'cubic':
        # Natural cubic spline: solve tridiagonal system for the second
        # derivatives M_i at each knot, then express S(r) as a linear
        # combination of the knot values y. Endpoints have M=0 (natural BC).
        n = nspace
        h = xspace[1:] - xspace[:-1]                         # (n-1,)
        A = torch.zeros(n, n, dtype=dtype)
        B = torch.zeros(n, n, dtype=dtype)
        A[0, 0] = 1.0
        A[n - 1, n - 1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i]     = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            B[i, i - 1] = 6.0 / h[i - 1]
            B[i, i]     = -6.0 / h[i - 1] - 6.0 / h[i]
            B[i, i + 1] = 6.0 / h[i]
        T = torch.linalg.solve(A, B)                         # M = T @ y

        W = torch.zeros(len(dunique), n, dtype=dtype)
        for q in range(len(dunique)):
            r = dunique[q]
            i = int(torch.clamp(
                torch.searchsorted(xspace, r, right=True) - 1, 0, n - 2))
            h_i = h[i]
            a = (xspace[i + 1] - r) / h_i
            b = (r - xspace[i]) / h_i
            # S(r) = a y_i + b y_{i+1} + h^2 [(a^3-a)/6 M_i + (b^3-b)/6 M_{i+1}]
            W[q, i]     += a
            W[q, i + 1] += b
            c_i  = (a ** 3 - a) * h_i ** 2 / 6.0
            c_i1 = (b ** 3 - b) * h_i ** 2 / 6.0
            W[q] += c_i  * T[i]
            W[q] += c_i1 * T[i + 1]
        return W

    if mode == 'gaussian':
        # Plain Gaussian basis. Not partition-of-unity (sum may differ from 1
        # at any r); the model learns amplitudes that absorb this. Centers at
        # xspace, widths fixed.
        h = xspace[1] - xspace[0]
        sigma = sigma_factor * h
        diff = dunique[:, None] - xspace[None, :]
        return torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))

    raise ValueError(
        f"interp_mode must be 'linear'/'cubic'/'gaussian', got {mode!r}")


class OEConv(torch.nn.Module):
    ''' Class for orthogonally equivariant convolutions
    
    For notation, I will include specific 2d and 3d versions
    
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 dimension, in_rank, out_rank,
                 bias=None, interp_mode='linear', sigma_factor=1.0, **kwargs):
        ''' Default bias behavior will be true if out_rank=1 and zero otherwise
        interp_mode: how the per-knot radial weights are mapped onto the kernel
            grid. 'linear' (default, original behavior), 'cubic' (natural
            spline), or 'gaussian' (fixed-center, fixed-width Gaussian basis).
        sigma_factor: only used when interp_mode=='gaussian'; sigma of each
            basis function = sigma_factor * knot_spacing.
        Any other kwargs are passed onto conv function
        '''
        super().__init__()
        self.interp_mode = interp_mode
        self.sigma_factor = sigma_factor
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not kernel_size%2:
            raise Exception(f'kernel_size must be odd, but you input {kernel_size}')
        self.kernel_size = kernel_size
        
        
        if type(dimension) is not int or dimension < 1:
            raise Exception(f'dimension must be a positive integer but you input {dimension}')
        self.dimension = dimension
        if self.dimension == 2:
            self.conv = torch.nn.functional.conv2d
        elif self.dimension == 3:
            self.conv = torch.nn.functional.conv3d
        else:
            raise Exception('Only dimensions 2 or 3 supported')
        
        if type(in_rank) is not int or in_rank < 0:
            raise Exception(f'in_rank must be a nonnegative integer but you input {in_rank}')
        self.in_rank = in_rank
        
        if type(out_rank) is not int or out_rank < 0:
            raise Exception(f'out_rank must be a nonnegative integer but you input {out_rank}')
        self.out_rank = out_rank
        self.rank = self.in_rank + self.out_rank
        
        # how many points to sample in space.  This could be a variable bot for now I fix it
        #self.nspace = int(np.ceil((self.kernel_size-1)/2*np.sqrt(self.dimension)))
        self.nspace = (self.kernel_size-1)//2+1
        if self.kernel_size == 3:
            # if kernel size is 3 the above is just not enough
            self.nspace = 3
        #self.nspace = max(1,self.nspace) # for 1x1 convolution we need at least 1 sample
        
        # build the space domain for the kernels
        x = torch.arange(self.kernel_size) - (self.kernel_size-1)/2
        X = torch.stack(torch.meshgrid((x,)*self.dimension,indexing='ij')) # keep space components last
        D = torch.sum(X**2,0)**0.5
        dunique,dinds = torch.unique(D,return_inverse=True)
        Xhat = X/torch.sum(X**2,0,keepdims=True)**0.5
        Xhat[torch.isnan(Xhat)] = 0
        # optional spherical support: radial mask over the tap grid (see _kernel_radius).
        # constant within each shell of D -> preserves octahedral/steerable equivariance.
        rmax = _kernel_radius()
        cmask = torch.ones_like(D) if rmax is None else (D <= rmax + 1e-6).to(D.dtype)
        # non-persistent: not in state_dict, so pre-existing checkpoints still load and
        # the mask is rebuilt from EQ_KERNEL_RADIUS at construction (set it to match at eval).
        self.register_buffer('corner_mask', cmask, persistent=False)
        self.corner_zeroed = rmax is not None
        # copy buffers
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('dunique',dunique)
        self.register_buffer('dinds',dinds)
        self.register_buffer('eye',torch.eye(self.dimension))
        
        if self.kernel_size > 1:
            xspace = torch.linspace(0, torch.max(dunique), self.nspace).to(dunique.dtype)
            interp = _radial_interp_matrix(
                xspace, dunique, self.nspace,
                mode=self.interp_mode, sigma_factor=self.sigma_factor,
            )
        else:
            interp = torch.ones((1, 1))
        self.register_buffer('interp', interp)
        
        # get the signatures
        self.signatures = self.get_all_signatures(self.rank)
        self.nsignatures = len(self.signatures)
        T = []
        for scount,s in enumerate(self.signatures):
            #print(s)    

            # these are all the indices
            # how am I going to do the assignments?
            # I could initialize to ones and then do the products
            # we need to build a slice here

            # find all the indices not in the tuple
            
            if self.rank > 0:
                tmp = torch.ones_like(Xhat)
                xinds = list((set(range(self.rank)) - set(np.array(s).ravel().tolist())))              
                for i in xinds:        
                    sl = [None]*self.rank
                    sl[i] = slice(None)                     
                    factor = self.Xhat[tuple(sl)]                                
                    tmp = tmp * factor
                    

                # now we go through the indices that are in the tuple
                for i in s:                                
                    sl = [None]*(self.rank+self.dimension) # plus d for the d space dimensions
                    sl[i[0]] = slice(None)  
                    sl[i[1]] = slice(None)  

                    tmp = tmp * self.eye[tuple(sl)]   
            else:
                tmp = torch.ones_like(Xhat[0])            
            T.append(tmp)
        T = torch.stack(T,-self.dimension-1)
        
        self.register_buffer('T',T)
        # get information for permutations
        self.sl = (slice(None),slice(None),) + (None,)*self.rank
        mydims = torch.arange(self.rank+2+self.dimension)                
        mylist = [mydims[0],]
        mylist.extend(  mydims[2:2+self.out_rank]   )
        mylist.append(mydims[1])
        mylist.extend( mydims[2+self.out_rank:2+self.out_rank+self.in_rank] )
        mylist.extend(mydims[-self.dimension:])
        myperm = [p.item() for p in mylist]        
        self.myperm = myperm
        self.reshape =  (self.out_channels*self.dimension**self.out_rank,
                         self.in_channels*self.dimension**self.in_rank) + (self.kernel_size,)*self.dimension
        
        
        # what if in rank is zero? special case above
        
        
        
        # weights        
        # how
        bound = 1/(self.kernel_size**self.dimension * self.in_channels)**0.5
        bound = 1/(self.kernel_size**self.dimension * self.in_channels * self.nsignatures)**0.5 
        # I think I need the nsignatures here because its what I'm summing over
        #self.weights = torch.nn.Parameter(torch.randn(self.out_channels,self.in_channels,self.nsignatures,self.nspace))
        self.weights = torch.nn.Parameter(
            ( torch.rand(self.out_channels,self.in_channels,self.nsignatures,self.nspace) -0.5 )*2*bound 
        )
        # bias, currently only scalar
        # TODO if we are even dimension we can have a bias proportional to identity
        # for odd dimension
        # for higher order I won't implement it yet
        
        
        if (bias is None or bias) and (self.out_rank == 0 or self.out_rank == 2):
            # for scalars or tensor order 2 we have the same number of components
            #self.bias = torch.nn.Parameter(torch.randn(self.out_channels))
            self.bias = torch.nn.Parameter((torch.rand(self.out_channels)-0.5)*2*bound)
            if self.out_rank == 0:
                self.get_bias = self.get_bias_0
            elif self.out_rank == 2:
                self.get_bias = self.get_bias_2
        else:
            self.bias = None  
            self.get_bias = self.get_bias_other
        self.kwargs = kwargs
        
        
    # bias in a few different forms
    def get_bias_0(self):
        return self.bias
    def get_bias_2(self):
        #print(self.eye)
        #print(self.bias)
        return (self.eye[None]*self.bias[...,None,None]).reshape(-1) 
    # for higher order even there are different ways to take products of identity
    def get_bias_other(self):
        return None
        
        
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        '''Load checkpoints written before signatures were canonicalised.

        Those runs stored a signature axis with duplicated entries (11 instead of
        10 at rank 4; see get_all_signatures). The duplicates are exact copies of
        an existing basis tensor, so a layer's output depends only on the SUM of
        the weights sitting on identical basis tensors:
            w_a . T + w_b . T = (w_a + w_b) . T
        Folding them is therefore exact, not an approximation.

        The mapping is recovered by matching the checkpoint's own T buffer against
        this module's T, column by column. Matching on tensor content rather than
        on signature bookkeeping is essential: the legacy ordering depended on the
        training interpreter's set-iteration order, so it cannot be reconstructed
        from a different Python version.
        '''
        w_key, T_key = prefix + 'weights', prefix + 'T'
        if w_key in state_dict and T_key in state_dict:
            T_old = state_dict[T_key]
            sig_axis = self.rank            # T is (d,)*rank + (nsig,) + (k,)*dim
            n_old = T_old.shape[sig_axis]
            n_new = self.nsignatures
            if n_old != n_new:
                T_new = self.T.to(T_old.dtype)
                # old column -> new column, by exact basis-tensor match
                mapping = {}
                for k in range(n_old):
                    col_old = T_old.select(sig_axis, k)
                    for j in range(n_new):
                        if torch.allclose(col_old, T_new.select(sig_axis, j)):
                            mapping[k] = j
                            break
                if len(mapping) == n_old and set(mapping.values()) == set(range(n_new)):
                    w_old = state_dict[w_key]          # (out, in, nsig, nspace)
                    w_new = torch.zeros(
                        w_old.shape[:2] + (n_new,) + w_old.shape[3:],
                        dtype=w_old.dtype, device=w_old.device)
                    for k, j in mapping.items():
                        w_new[:, :, j] += w_old[:, :, k]
                    state_dict[w_key] = w_new
                    state_dict[T_key] = self.T.to(T_old.dtype)
                    warnings.warn(
                        f'{prefix}: folded a legacy {n_old}-signature checkpoint into '
                        f'{n_new} canonical signatures. This is an exact rewrite '
                        f'(duplicate basis tensors summed), not an approximation.')
                else:
                    error_msgs.append(
                        f'{prefix}: checkpoint has {n_old} signatures and this module '
                        f'has {n_new}, but the basis tensors could not be matched '
                        f'one-to-one, so no safe conversion exists.')
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def get_all_signatures(self,rank):
        '''
        Get all sets of pairs for a given total rank
        These pairs will be set with identity.
        '''

        # first get all the permutations
        permutations_ = list(permutations(range(rank)))
        # even length
        signatures = []
        for i in range(rank//2+1):
            if i == 0:
                signatures.append(tuple())
                continue
            # otherwise
            theseperms = [p[:2*i] for p in permutations_]
            # group them in sets of 2

            for p in theseperms:
                thesesets = []
                for j in range(i):
                    thesesets.append( tuple(set((p[2*j:2*j+2])) ) )
                # Canonical form: sort within each pair, then sort the pairs.
                # A signature is an unordered set of unordered pairs, but
                # tuple(set(...)) freezes whatever order that set object happened
                # to iterate in, which is hash-table dependent. The later
                # set(signatures) then compares tuples, so ((0,1),(2,3)) and
                # ((2,3),(0,1)) both survived as separate basis elements -- giving
                # 11 signatures at rank 4 where the theorem gives 10, with the
                # extra one an exact duplicate. Worse, tuple hashing changed in
                # Python 3.8, so the count varied by interpreter version and
                # checkpoints were not portable across environments.
                signatures.append( tuple(sorted(tuple(sorted(p_)) for p_ in thesesets)) )

        # sort them by length and by axis
        signatures = sorted(tuple(set(signatures)))
        signatures = sorted(signatures,key=len)


        return signatures
    def make_kernel(self):
        fx = (self.interp@self.weights[...,None])[...,0][...,self.dinds]        
        #print(fx.shape)
        fxsl = fx[self.sl]     
        #print(fxsl.shape)
        #print(self.T.shape)
        fxslT = fxsl*self.T
        #print(fxslT.shape)
        K = torch.sum( (fxslT) , -self.dimension-1)    
        #print(K.shape)
        Kp = K.permute(*self.myperm)
        #print(self.myperm)
        kernel = Kp.reshape(*self.reshape)
        if self.corner_zeroed:                      # radial (spherical-support) mask
            kernel = kernel * self.corner_mask
        self.kernel = kernel # save it as a member
        return kernel
        
        
        
    def forward(self,x):
        # TODO, only call make kernel in train mode
        # TODO implement padding modes using the pad function
        # in eval mode the kernel is not changing and I can use the old one
        self.make_kernel()
        #print(kernel.shape)
        return self.conv(x,self.kernel,self.get_bias(),**self.kwargs)
        
        

class OEConvBlock(torch.nn.Module):
    '''Concatenate signals'''
    def __init__(self, in_channels, out_channels, kernel_size, dimension,
                 bias=None, interp_mode='linear', sigma_factor=1.0, **kwargs):
        '''
        In channels and out channels are a list now

        I want to call make_kernel on all of them, concatenate them into a big kernel,
        then feed in a concatenated vector

        interp_mode/sigma_factor: forwarded to each OEConv (see OEConv docstring).
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        blocks = {}
        # NOTE they don't all need a bias, even for the scalar outputs, only one of them needs a bias
        # note for matrix outputs, we should add the identity bias
        for i in range(len(in_channels)):
            for j in range(len(out_channels)):
                blocks[str((i,j))] = OEConv(in_channels[i], out_channels[j], kernel_size, dimension, i, j,
                                            bias=((j==0) or (j==2))*(i==0)*(bias!=False),
                                            interp_mode=interp_mode, sigma_factor=sigma_factor)
                blocks[str((i,j))].weights.data /= (len(in_channels))**0.5
                if blocks[str((i,j))].bias is not None:
                    blocks[str((i,j))].bias.data /= (len(in_channels))**0.5
        self.blocks = torch.nn.ModuleDict(blocks)
        # note
        # the weights and the bias should be normalized by one over the sqrt of number of input channels
        # now here the number of input channels is more
        # I think we just need to normalize by 1/sqrt(len(in_channels)) assuming they are normalized properly
        # further more, a map from a vector with 2 channels to a scalar with 1 channel, they get summed over
        # but, they get summed over times a unit length vector, so I think this doesn't change anything
        
        
        # we need to think about how to do the bias here
        self.kwargs = kwargs
        
        
        
        self.dimension = dimension
        if self.dimension == 2:
            self.conv = torch.nn.functional.conv2d
        elif self.dimension == 3:
            self.conv = torch.nn.functional.conv3d
        else:
            raise Exception('only 2d or 3d supported')
            
        # I'm not sure this is the right way to do it
        self.bias = self.blocks[str((0,0))].bias
        if self.bias is None:
            self.get_bias = self.get_bias_none
        
        # this block of zeros is for if I'm putting bias on the scalar part
        if len(self.out_channels) > 1:
            #print(torch.sum( 2**torch.arange(1,len(out_channels)) * torch.tensor(self.out_channels[1:]))   )
            zeros = torch.zeros(     torch.sum( self.dimension**torch.arange(1,len(out_channels)) * torch.tensor(self.out_channels[1:]))   )
            self.get_bias = self.get_bias_0
            #print(zeros)
        else:
            zeros = torch.zeros(0)
            self.get_bias = self.get_bias_0
        self.register_buffer('zeros',zeros)
        
        # what if I want to add in bias for matrices proportional to identity
        if len(self.out_channels) > 2: # scalars, vectors, and matrices
            # number of components for vectors            
            zeros1 = torch.zeros(     torch.sum( self.dimension**torch.arange(1,2) * torch.tensor(self.out_channels[1:2]))   )
            self.register_buffer('zeros1',zeros1)
            # remaining            
            if len(self.out_channels) > 3:            
                # this has not been validated
                zeros2 = torch.zeros(torch.sum( self.dimension**torch.arange(3,len(out_channels)) * torch.tensor(self.out_channels[3:])))
            else:
                zeros2 = torch.zeros(0)
            self.register_buffer('zeros2',zeros2)
            self.get_bias = self.get_bias_0_2
        
        
        
        
        
        self.padding_kwargs = {}
        if 'padding_mode' in kwargs:
            self.padding_kwargs['mode'] = kwargs.pop('padding_mode')
        if 'padding' in kwargs:
            self.padding = kwargs.pop('padding')
        else:
            self.padding = 0
    
    def get_bias_none(self):
        return None
    def get_bias_0(self):
        bias = torch.cat(
                (
                    self.blocks[str((0,0))].bias,
                    self.zeros # could this be procomputed?
                )
            )
        return bias
    def get_bias_0_2(self):
        # index is in then out
        bias = torch.cat(
                (
                    self.blocks[str((0,0))].bias,
                    self.zeros1,
                    self.blocks[str((0,2))].get_bias(),
                    self.zeros2
                )
            )
        return bias
    def forward(self,x):
        kernel = []
        
        for j in range(len(self.out_channels)):
            kernel_ = []
            for i in range(len(self.in_channels)):
                kernel_.append( self.blocks[str((i,j))].make_kernel() ) 
            kernel_ = torch.cat(kernel_,1)
            kernel.append(kernel_)
        kernel = torch.cat(kernel,0)
        
        # when out channel is a scalar we have a bias, otherwise 0   
        # TODO, still haven't worked out details of bias
        
        #if self.bias is not None:
        #    bias = torch.cat(
        #        (
        #            self.bias,
        #            self.zeros # could this be procomputed?
        #        )
        #    )
        #else:
        #    bias = None
        bias = self.get_bias()
        #print(bias.shape)
        
        
        # TODO padding ahead of time
        tmp = torch.nn.functional.pad(x,(self.padding,)*(2*self.dimension),**self.padding_kwargs)
        return self.conv(tmp,kernel,bias=bias,**self.kwargs)    
    # below these tests were for speed.  The above appraoch seems to be significantly faster
    def test(self,x):        
        # do it by hand
        xs,xv = self.blocks[str((0,0))](x), self.blocks[str((0,1))](x)
        
        return torch.cat((xs,xv),-3)
    def test1(self,x):        
        # do it by hand
        nscalar = x.shape[1]//3
        xs,xv = (
            self.blocks[str((0,0))](x[:,:nscalar])+self.blocks[str((1,0))](x[:,nscalar:]), 
            self.blocks[str((0,1))](x[:,:nscalar])+self.blocks[str((1,1))](x[:,nscalar:])
        )
        
        return torch.cat((xs,xv),-3)
    
def get_tensor_inds(channels,dimension):
    '''
    Parameters
    ----------
    Channels is a list of channels for each tensor order, starting with scalars.
    '''
    
    indices = []
    count = 0
    for i in range(len(channels)): # scalars vectors etc.
        for j in range(channels[i]):
            indices.extend(  [count]*(dimension**i)  )
            count += 1
    #print(indices)
    indices = torch.tensor(indices)
    return indices
    
    
class OESigmoidBlock(torch.nn.Module):
    def __init__(self,channels,dimension,epsilon=1e-5,subtract_one=True):
        super().__init__()
        # channels is a list of scalar, vector, etc
        self.channels = channels
        self.dimension = dimension
        self.epsilon = epsilon        
        self.channelstot = np.sum(channels*self.dimension**np.arange(len(channels)))
        self.channelssum = np.sum(channels)
                        
        self.register_buffer('indices',get_tensor_inds(channels,dimension))
        self.sl  = (...,self.indices,) + (slice(None),)*dimension        
        self.mag2sl = (...,slice(0,self.channelssum,)) + (slice(None),)*self.dimension
        
        self.subtract_one = subtract_one
        if self.subtract_one:
            self.normalization = self.normalization_subtract_one
        else:
            self.normalization = self.normalization_no_subtract_one
        
    def normalization_subtract_one(self,logmag,newlogmag):
        return (newlogmag.exp()-1)/logmag.exp()
    def normalization_no_subtract_one(self,logmag,newlogmag):
        return (newlogmag-logmag).exp()
            
        
    def forward(self,x):
        
        # square it
        x2 = x**2
        # now get the magnitude
        mag2 = torch.zeros_like(x[self.mag2sl])
        mag2.index_add_(-self.dimension-1,self.indices,x2)
        mag2 += self.epsilon
        logmag = 0.5*torch.log(mag2)
        # now a new logmag
        newlogmag = torch.relu(logmag) # minus 1 somewhere??
        # now the logmag is always positive
        # so when I exponentiate it the mag is always bigger than 1
        # so if I subtract 1 from it I get things zeroed out
        #magfactor = torch.exp(newlogmag - logmag)# - 1
        #magfactor = (newlogmag.exp()-1)/logmag.exp()
        #magfactor = (newlogmag.exp()-1)/mag2**0.5
        # magfactor = (newlogmag-logmax).exp() # with no minus one there is less of a nonlinearity, less sparsity
        # now I need to map magfactor back up to the original size
        # this is like a repeat        
        magfactor = self.normalization(newlogmag,logmag)
        return x*magfactor[self.sl]    
    
class OESigmoidBlock_(torch.nn.Module):
    def __init__(self,channels,dimension,epsilon=1e-5,**kwargs):
        super().__init__()
        # channels is a list of scalar, vector, etc
        self.channels = channels
        self.dimension = dimension
        self.epsilon = epsilon        
        self.channelstot = np.sum(channels*2**np.arange(len(channels)))
        self.channelssum = np.sum(channels)
                        
        self.register_buffer('indices',get_tensor_inds(channels,dimension))
        self.sl  = (...,self.indices,) + (slice(None),)*dimension        
        self.mag2sl = (...,slice(0,self.channelssum,)) + (slice(None),)*self.dimension
        
        
            
        
    def forward(self,x):
        
        # square it
        x2 = x**2
        # now get the magnitude
        mag2 = torch.zeros_like(x[self.mag2sl])
        mag2.index_add_(-self.dimension-1,self.indices,x2)
        # essentially the magnitude gets cubed, very simple but it blows up
        return x*mag2[self.sl]        
    
    
class OESigmoidBlock_(torch.nn.Module):
    def __init__(self,*params,**kwargs):
        super().__init__()
        pass
    def forward(self,x):
        return x
class OEBatchnormBlock(torch.nn.Module):
    def __init__(self,channels,dimension,epsilon=1e-5,**kwargs):
        ''' Note there is code duplication here. TODO merge them with sigmoid
        
        Note if a variable is a standard normal
        And I take exp 
        Then its std becomes
        ( (np.exp(1)-1)*np.exp(1) )**0.5
        This is a lognormal with mu=0 and var=1
        
        Essentially, after normalization we should divide by 2.
        Question, should I scale in the log domain? This would be equivalent to applying a power
        
        ( (np.exp(sigma2)-1)*np.exp(sigma2) )**0.5 = 1
        exp(2sigma2) - exp(sigma2) = 1
        I could solve for exp(sigma2) which would be a quadratic
        then take the log
        just
        x^2 - x - 1 = 0
        [1 pm sqrt( 1 + 4  )]/2
        [1 pm sqrt(5)]/2
        sigma2 = np.log((1 + np.sqrt(5))/2)
        sigma = sqrt( np.log((1 + np.sqrt(5))/2) )
        '''
        super().__init__()
        # channels is a list of scalar, vector, etc
        self.channels = channels
        self.dimension = dimension
        self.epsilon = epsilon
        
        # first I will square it
        # then I will sum it with a single matrix multiplication
        # 
        self.channelstot = np.sum(channels*self.dimension**np.arange(len(channels)))
        self.channelssum = np.sum(channels)
                        
        self.register_buffer('indices',get_tensor_inds(channels,dimension))
        self.sl = (...,self.indices,) + (slice(None),)*dimension        
        self.mag2sl = (...,slice(0,self.channelssum,)) + (slice(None),)*self.dimension
        
        if dimension == 2:
            self.b = torch.nn.BatchNorm2d(self.channelssum,**kwargs)
        elif dimension == 3:
            self.b = torch.nn.BatchNorm3d(self.channelssum,**kwargs)
        else:
            raise Exception('currently only supported for 2d and 3d')
            
        # this is a factor to multiply by after exp
        self.factor = 1.0/((np.e-1)*np.e)**0.5
        # this is the factor to multiply before
        self.factor = np.log((1 + np.sqrt(5))/2)**0.5
        
    def forward(self,x):
        
        # square it
        x2 = x**2
        # now get the magnitude
        mag2 = torch.zeros_like(x[self.mag2sl]) #        
        #print(mag2.shape)
        
        mag2.index_add_(-self.dimension-1,self.indices,x2)
        mag2 += self.epsilon
        logmag = 0.5*torch.log(mag2)        
        # now a new logmag
        newlogmag = self.b(logmag)
        # this give sigma1
        # below we get sigma such that the variance will be 1 after exp
        newlogmag = newlogmag*self.factor/2 # i think the /2 might be a good idea to help stabilize
        
                
        magfactor = torch.exp(newlogmag - logmag) 
        # now I need to map magfactor back up to the original size
        # this is like a repeat        
        
        return x*magfactor[self.sl]#*self.factor
class OEBatchnormBlock_(torch.nn.Module):
    def __init__(self,*params,**kwargs):
        super().__init__()
    def forward(self,x):
        return x
    

def rotate2D(x,channels,theta):
    '''
    Parameters
    ----------
    x : tensor
        An image
    TODO
    ----
    Implement a function that can rotate an image with an arbitrary numer of channels
    '''
    R = torch.tensor([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)],
    ],dtype=x.dtype,device=x.device)
    Ri = torch.linalg.inv(R)
    nx = x.shape[-2:]
    #print(nx)
    points = [torch.arange(n,dtype=x.dtype,device=x.device) - (n-1)/2 for n in nx]
    X = torch.stack(torch.meshgrid(points,indexing='ij'),-1)
    
    Xs = (Ri@X[...,None])[...,0]
    Xs = Xs - torch.stack([p[0] for p in points])
    Xs = Xs / torch.stack([p[-1]-p[0] for p in points]) # between 0 and 1
    Xs = Xs*2-1
    #print(Xs)
    xr0 = torch.nn.functional.grid_sample(x,Xs[None].flip(-1),align_corners=True)
    xr = torch.zeros_like(xr0)
    # first loop is over the channel types
    for i in range(len(channels)):
        # second loop is over the channels of this type
        for j in range(channels[i]):
            # third loop is over the output components in this channel
            for k in product( *((0,1),)*i  ):
                print(k)
            


            pass
    
    return xr0

# these functions below will rotate one image
def rotate_scalar(x,theta):
    R = torch.tensor([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)],
    ],dtype=x.dtype,device=x.device)
    Ri = torch.linalg.inv(R)
    nx = x.shape[-2:]
    #print(nx)
    points = [torch.arange(n,dtype=x.dtype,device=x.device) - (n-1)/2 for n in nx]
    X = torch.stack(torch.meshgrid(points,indexing='ij'),-1)
    
    Xs = (Ri@X[...,None])[...,0]
    Xs = Xs - torch.stack([p[0] for p in points])
    Xs = Xs / torch.stack([p[-1]-p[0] for p in points]) # between 0 and 1
    Xs = Xs*2-1
    #print(Xs)
    xr0 = torch.nn.functional.grid_sample(x,Xs[None].flip(-1),align_corners=True)
    return xr0


def rotate_vector(x,theta):
    R = torch.tensor([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)],
    ],dtype=x.dtype,device=x.device)
    Ri = torch.linalg.inv(R)
    nx = x.shape[-2:]
    #print(nx)
    points = [torch.arange(n,dtype=x.dtype,device=x.device) - (n-1)/2 for n in nx]
    X = torch.stack(torch.meshgrid(points,indexing='ij'),-1)
    
    Xs = (Ri@X[...,None])[...,0]
    Xs = Xs - torch.stack([p[0] for p in points])
    Xs = Xs / torch.stack([p[-1]-p[0] for p in points]) # between 0 and 1
    Xs = Xs*2-1
    #print(Xs)
    xr0 = torch.nn.functional.grid_sample(x,Xs[None].flip(-1),align_corners=True)
    # the channels is the second index, but we want to move it to the last for matmul
    xr0p = xr0.transpose(-3,-1)
    xr = (R@xr0p[...,None])[...,0]
    xr = xr.transpose(-3,-1)
    return xr

def rotate_matrix(x,theta):
    R = torch.tensor([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)],
    ],dtype=x.dtype,device=x.device)
    Ri = torch.linalg.inv(R)
    nx = x.shape[-2:]
    #print(nx)
    points = [torch.arange(n,dtype=x.dtype,device=x.device) - (n-1)/2 for n in nx]
    X = torch.stack(torch.meshgrid(points,indexing='ij'),-1)
    
    Xs = (Ri@X[...,None])[...,0]
    Xs = Xs - torch.stack([p[0] for p in points])
    Xs = Xs / torch.stack([p[-1]-p[0] for p in points]) # between 0 and 1
    Xs = Xs*2-1
    #print(Xs)
    xr0 = torch.nn.functional.grid_sample(x,Xs[None].flip(-1),align_corners=True)
    # the channels is the second index, but we want to move it to the last for matmul
    xr0p = xr0.transpose(-3,-1)
    xr0pm = xr0p.reshape(-1,2,2)
    xr = R@xr0pm@R.T
    xr = xr.reshape(xr0p.shape)
    xr = xr.transpose(-3,-1)
    return xr


