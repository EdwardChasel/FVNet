# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:23:17 2022

@author: yimen
"""


import numpy as np
import torch
import torch.nn as nn
#import math
import torch.nn.functional as  F
#import MyLibForSteerCNN as ML
import scipy.io as sio    
import math
from PIL import Image

def Getini(sizeP, inNum, outNum, expand):
    
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0,0),0),4),0)
    y  = Y0[:,1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y,0),0),3),0)

    orlW = np.zeros([outNum,inNum,expand,sizeP,sizeP,1,1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(Image.fromarray(((np.random.randn(3,3))*2.4495/np.sqrt((inNum)*sizeP*sizeP))).resize((sizeP,sizeP)))
                orlW[i,j,k,:,:,0,0] = temp
             
    v = np.pi/sizeP*(sizeP-1)
    k = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP,1])
    l = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP])

    tempA =  np.sum(np.cos(k*v*X0)*orlW,4)/sizeP
    tempB = -np.sum(np.sin(k*v*X0)*orlW,4)/sizeP
    A     =  np.sum(np.cos(l*v*y)*tempA+np.sin(l*v*y)*tempB,3)/sizeP
    B     =  np.sum(np.cos(l*v*y)*tempB-np.sin(l*v*y)*tempA,3)/sizeP #是否少了一个负号？
    A     = np.reshape(A, [outNum,inNum,expand,sizeP*sizeP])
    B     = np.reshape(B, [outNum,inNum,expand,sizeP*sizeP]) 
    iniW  = np.concatenate((A,B), axis = 3)
    return torch.FloatTensor(iniW)

def Getini_3D(sizeP, inNum, outNum, expand):
    
    inX, inY,inZ, Mask = MaskC_3D(sizeP)#(P,P,P)
    X0 = np.expand_dims(inX,3)#(P,P,P,1)
    Y0 = np.expand_dims(inY,3)#(P,P,P,1)
    Z0 = np.expand_dims(inZ,3)#(P,P,P,1)
    x = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0,0),0),5),6),0)#(1,1,1,P,P,P,1,1,1)
    y = Y0[:,:,1]#(P,P,1)
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y,0),0),4),5),0)#(1,1,1,P,P,1,1,1)
    z = Z0[:,:,1][:,1]#(P,1)
    z = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(z,0),0),3),4),0)#(1,1,1,P,1,1,1)
    
    orlW = np.zeros([outNum,inNum,expand,sizeP,sizeP,sizeP,1,1,1])#(outNum,inNum,expand,P,P,P,1,1,1)
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                #temp = np.array(Image.fromarray(((np.random.randn(3,3,3))*2.4495/np.sqrt((inNum)*sizeP*sizeP*sizeP))).resize((sizeP,sizeP,sizeP)))#(P,P,P)
                temp = (np.random.randn(sizeP,sizeP,sizeP))*2.4495/np.sqrt((inNum)*sizeP*sizeP*sizeP)
                orlW[i,j,k,:,:,:,0,0,0] = temp
    
    v = np.pi/sizeP*(sizeP-1)
    
    k = np.reshape((np.arange(sizeP)),[1,1,1,1,1,1,sizeP,1,1])
    l = np.reshape((np.arange(sizeP)),[1,1,1,1,1,1,sizeP,1])
    s = np.reshape((np.arange(sizeP)),[1,1,1,1,1,1,sizeP])
    
    tempA  = np.sum(np.cos(k*v*x)*orlW,5)/sizeP
    tempB  = np.sum(np.sin(k*v*x)*orlW,5)/sizeP
    
    tempA1 = np.sum(np.cos(l*v*y)*tempA,4)/sizeP
    tempA2 = np.sum(np.sin(l*v*y)*tempB,4)/sizeP
    tempA3 = np.sum(np.cos(l*v*y)*tempB,4)/sizeP
    tempA4 = np.sum(np.sin(l*v*y)*tempA,4)/sizeP
    
    A      = np.sum(np.cos(s*v*z)*tempA1-np.cos(s*v*z)*tempA2-np.sin(s*v*z)*tempA3-np.sin(s*v*z)*tempA4,3)/sizeP
    B      = -np.sum(np.sin(s*v*z)*tempA1+np.cos(s*v*z)*tempA4+np.cos(s*v*z)*tempA3-np.sin(s*v*z)*tempA2,3)/sizeP
    A     = np.reshape(A, [outNum,inNum,expand,sizeP*sizeP*sizeP])
    B     = np.reshape(B, [outNum,inNum,expand,sizeP*sizeP*sizeP])
    iniW  = np.concatenate((A,B), axis = 3)
    return torch.FloatTensor(iniW)

class Fconv(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True):
       
        super(Fconv, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        BasisC, BasisS = GetBasis(sizeP,tranNum,inP)        
        self.register_buffer("Basis", torch.cat([BasisC, BasisS], 3))#.cuda() (P,P,tranNum,2*P*P)
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini(inP, inNum, outNum, self.expand)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
                    
        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
               
        for i in range(expand):
            ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
                
#        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})  
        bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])

        output = F.conv2d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + bias

class Fconv_3D(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, stride=1, ifIni=0, bias=True):
       
        super(Fconv_3D, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.kernel_size = (sizeP, sizeP, sizeP)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        BasisC, BasisS = GetBasis_3D(sizeP,tranNum,inP)  
        Basis = torch.cat([BasisC, BasisS], 4)
        self.register_buffer("Basis", Basis)#.cuda() (P,P,P,tranNum,2*P*P)
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini_3D(inP, inNum, outNum, self.expand)#(outNum,inNum,expand,2*sizeP*sizeP*sizeP)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
                    
        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1,1)

    def forward(self, input):
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            
            tempW = torch.einsum('ijosk,mnak->msnaijo', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP, self.sizeP ])
                    
    #        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})  
            _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
        else:
            _filter = self.filter
            _bias   = self.bias

        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=1,
                        groups=1)
        return output + _bias
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijosk,mnak->msnaijo', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ,self.sizeP ])
            _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)

        return super(Fconv_3D, self).train(mode)
        
def Getini_3D_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)#(outNum,inNum,expand,nNum)

class Fconv_3D_PCA(nn.Module):
    #Tip: 小核用Fconv_3D!!! P >= 3
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None,stride=1, ifIni=0, bias=True):
       
        super(Fconv_3D_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.kernel_size = (sizeP, sizeP, sizeP)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        Basis, Rank, weight = GetBasis_3D_PCA(sizeP,tranNum,inP)        
        self.register_buffer("Basis", Basis)#.cuda())
        #sio.savemat('Basis.mat', {'Basis': Basis.cpu().detach().numpy()})#基底？？？
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(4))*0.03
        iniw = Getini_3D_reg(Basis.size(4), inNum, outNum, self.expand, weight)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        
        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1,1)

    def forward(self, input):
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijosk,mnak->msnaijo', self.Basis, self.weights)
            # tempW = tempW.permute(0,2,3,4,1,5,6)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
                   
            # for i in range(expand):
            #     ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            #     tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP, self.sizeP ])
            _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
        else:
            _filter = self.filter
            _bias   = self.bias

        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=1,
                        groups=1)
        return output + _bias
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijosk,mnak->msnaijo', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ,self.sizeP ])
            _bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)

        return super(Fconv_3D_PCA, self).train(mode)
    
    
class PointwiseAvgPoolAntialiased(nn.Module):
    
    def __init__(self, sizeF, stride, padding=None ):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF-1)/2/3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        
        
        if padding is None:
            padding = int((sizeF-1)//2)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)#(P,P)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)#(P,P,2)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())#(P,P)
        # [[-2., -1., -2.],
        # [-1., -0., -1.],
        # [-2., -1., -2.]])
        _filter = torch.exp(r / (2 * variance))#(P,P)
        _filter /= torch.sum(_filter)#(P,P)
        _filter = _filter.view(1, 1, sizeF, sizeF)#(1,1,P,P)
        self.filter = nn.Parameter(_filter, requires_grad=True)
    
    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])        
        return output


class PointwiseAvgPoolAntialiased_3D(nn.Module):
    
    def __init__(self, sizeF, stride, padding=None ):
        super(PointwiseAvgPoolAntialiased_3D, self).__init__()
        sigma = (sizeF-1)/2/3
        self.kernel_size = (sizeF, sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        
        
        if padding is None:
            padding = int((sizeF-1)//2)
        if isinstance(padding, int):
            self.padding = (padding, padding,padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF*sizeF).view(sizeF, sizeF, sizeF)
        grid_y = grid_x.permute(2,1,0)
        grid = torch.stack([grid_x, grid_y], dim=-1)#(P,P,2)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=True)
    
    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1,1))
        output = F.conv3d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])        
        return output

class F_BN(nn.Module):
    def __init__(self,channels, tranNum=8):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_BN_3D(nn.Module):
    def __init__(self,channels, tranNum=8):
        super(F_BN_3D, self).__init__()
        self.BN = nn.BatchNorm3d(channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3), X.size(4)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3), X.size(4)])

class F_Dropout(nn.Module):
    def __init__(self,zero_prob = 0.5,  tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)
    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_Dropout_3D(nn.Module):
    def __init__(self,zero_prob = 0.5,  tranNum=8, inplace=False):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout_3D, self).__init__()
        self.Dropout = nn.Dropout3d(zero_prob,inplace=inplace)
    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3), X.size(4)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3), X.size(4)])

def MaskC(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        
        Mask = np.ones([SizeP,SizeP])
#        Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
# X:SizeP*SizeP Y:SizeP*SizeP Mask:SizeP*SizeP
        return X, Y, Mask
    
def MaskC_3D(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y,Z  = np.meshgrid(x,x,x)
        C    =X**2+Y**2+Z**2
        
        Mask = np.ones([SizeP,SizeP,SizeP])
#        Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
# X:(P,P,P) Y:(P,P,P) Z:(P,P,P) Mask:(P,P,P)
        return X, Y, Z, Mask
    
def GetBasis(sizeP,tranNum=8,inP = None):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)#(P,P,1)
    Y0 = np.expand_dims(inY,2)#(P,P,1)
    Mask = np.expand_dims(Mask,2)#(P,P,1)
    theta = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)#(1,1,tranNum)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0#(P,P,tranNm)
    Y = np.cos(theta)*Y0+np.sin(theta)*X0#(P,P,tranNm)
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)#(P,P,tranNm,1,1)
    Y = np.expand_dims(np.expand_dims(Y,3),4)#(P,P,tranNm,1,1)
    v = np.pi/inP*(inP-1)#float
    p = inP/2#float
    
    k = np.reshape(np.arange(inP),[1,1,1,inP,1])#(1,1,1,P,1)
    l = np.reshape(np.arange(inP),[1,1,1,1,inP])#(1,1,1,1,P)

    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)#(P,P,tranNum,P,P)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)#(P,P,tranNum,P,P)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)#(P,P,tranNum,P*P)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)#(P,P,tranNum,P*P)
    return torch.FloatTensor(BasisC), torch.FloatTensor(BasisS)       


def GetBasis_3D(sizeP,tranNum=8,inP = None):
    if inP==None:
        inP = sizeP
    inX, inY,inZ, Mask = MaskC_3D(sizeP)#(P,P,P)
    X0 = np.expand_dims(inX,3)#(P,P,P,1)
    Y0 = np.expand_dims(inY,3)#(P,P,P,1)
    Z0 = np.expand_dims(inZ,3)#(P,P,P,1)
    Mask = np.expand_dims(Mask,3)#(P,P,P,1)
    #角度旋转，3个自由度
    alpha = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    alpha = np.expand_dims(np.expand_dims(np.expand_dims(alpha,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
    
    beta = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    beta = np.expand_dims(np.expand_dims(np.expand_dims(beta,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
    
    gamma = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    gamma = np.expand_dims(np.expand_dims(np.expand_dims(gamma,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
#    theta = torch.FloatTensor(theta)
    #坐标旋转
    X = (np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma))*X0+\
        (np.sin(alpha)*np.cos(beta)*np.cos(gamma)+np.cos(alpha)*np.sin(gamma))*Y0-np.sin(beta)*np.cos(gamma)*Z0#(P,P,P,tranNum)
    Y = (-np.cos(alpha)*np.cos(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma))*X0+\
        (-np.sin(alpha)*np.cos(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma))*Y0+np.sin(beta)*np.sin(gamma)*Z0#(P,P,P,tranNum)
    Z = np.cos(alpha)*np.sin(beta)*X0+np.sin(alpha)*np.sin(beta)*Y0+np.cos(beta)*Z0#(P,P,P,tranNum)
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(np.expand_dims(X,4),5),6)#(P,P,P,tranNm,1,1,1)
    Y = np.expand_dims(np.expand_dims(np.expand_dims(Y,4),5),6)#(P,P,P,tranNm,1,1,1)
    Z = np.expand_dims(np.expand_dims(np.expand_dims(Z,4),5),6)#(P,P,P,tranNm,1,1,1)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,1,inP,1,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,1,inP,1])
    z = np.reshape(np.arange(inP),[1,1,1,1,1,1,inP])

    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)#(P,P,P,tranNum,P,P,P)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)#(P,P,P,tranNum,P*P*P)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)
    return torch.FloatTensor(BasisC), torch.FloatTensor(BasisS)    

def GetBasis_PCA(sizeP, tranNum=8, inP=None):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,inP,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,inP])
    
    
    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
    Trod = 1
    Ind = var<Trod
    Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR

    return torch.FloatTensor(BasisR), Rank, Weight
    
def GetBasis_3D_PCA(sizeP, tranNum=8, inP=None):
    if inP==None:
        inP = sizeP
    inX, inY,inZ, Mask = MaskC_3D(sizeP)#(P,P,P)
    X0 = np.expand_dims(inX,3)#(P,P,P,1)
    Y0 = np.expand_dims(inY,3)#(P,P,P,1)
    Z0 = np.expand_dims(inZ,3)#(P,P,P,1)
    Mask = np.expand_dims(Mask,3)#(P,P,P,1)
    #角度旋转，3个自由度
    alpha = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    alpha = np.expand_dims(np.expand_dims(np.expand_dims(alpha,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
    
    beta = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    beta = np.expand_dims(np.expand_dims(np.expand_dims(beta,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
    
    gamma = np.arange(tranNum)/tranNum*2*np.pi#(tranNum,)
    gamma = np.expand_dims(np.expand_dims(np.expand_dims(gamma,axis=0),axis=0),axis=0)#(1,1,1,tranNum)
#    theta = torch.FloatTensor(theta)
    #坐标旋转
    X = (np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma))*X0+\
        (np.sin(alpha)*np.cos(beta)*np.cos(gamma)+np.cos(alpha)*np.sin(gamma))*Y0-np.sin(beta)*np.cos(gamma)*Z0#(P,P,P,tranNum)
    Y = (-np.cos(alpha)*np.cos(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma))*X0+\
        (-np.sin(alpha)*np.cos(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma))*Y0+np.sin(beta)*np.sin(gamma)*Z0#(P,P,P,tranNum)
    Z = np.cos(alpha)*np.sin(beta)*X0+np.sin(alpha)*np.sin(beta)*Y0+np.cos(beta)*Z0#(P,P,P,tranNum)
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(np.expand_dims(X,4),5),6)#(P,P,P,tranNm,1,1,1)
    Y = np.expand_dims(np.expand_dims(np.expand_dims(Y,4),5),6)#(P,P,P,tranNm,1,1,1)
    Z = np.expand_dims(np.expand_dims(np.expand_dims(Z,4),5),6)#(P,P,P,tranNm,1,1,1)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,1,inP,1,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,1,inP,1])
    z = np.reshape(np.arange(inP),[1,1,1,1,1,1,inP])

    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)#(P,P,P,tranNum,P,P,P)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y+(z-inP*(z>p))*v*Z)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)#(P,P,P,tranNum,P*P*P)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, sizeP, tranNum, inP*inP*inP])*np.expand_dims(Mask,4)
#!!!!!PCA!!!!!#
    BasisC = np.reshape(BasisC,[sizeP*sizeP*sizeP*tranNum, inP*inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*sizeP*tranNum, inP*inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP*sizeP, axis = 0),axis = 0)
    Trod = 1
    Ind = var<Trod
    Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    # BasisR = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0),0)*BasisR #(P,P,P,tranNum,P*P*P)

    return torch.FloatTensor(BasisR), Rank, Weight

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype).cuda()
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask

class MaskModule(nn.Module):

    def __init__(self, S: int, margin: float = 0.):

        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)


    def forward(self, input):

        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out

def build_mask_3D(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, s, dtype=dtype).cuda()
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            for z in range(s):
                r = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                if r > t:
                    mask[..., x, y, z] = math.exp((t - r)/sig**2)
                else:
                    mask[..., x, y, z] = 1.
    return mask

class MaskModule_3D(nn.Module):

    def __init__(self, S: int, margin: float = 0.):

        super(MaskModule_3D, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask_3D(S, margin=margin), requires_grad=False)
        
    def forward(self, input):
        
        assert input.shape[2:] == self.mask.shape[2:]
        out = input * self.mask
        return out
    
class GroupPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.max(output,2).values
        return output

class GroupPooling_3D(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling_3D, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3), input.size(4)]) 
        output = torch.max(output,2).values
        return output

class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.mean(output,2)
        return output
    
class GroupMeanPooling_3D(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling_3D, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3), input.size(4)]) 
        output = torch.mean(output,2)
        return output

class F_GroupNorm_3D(nn.Module):
    def __init__(self,groups,channels, tranNum=8):
        super(F_GroupNorm_3D, self).__init__()
        self.GN = nn.GroupNorm(num_groups=groups,num_channels = channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.GN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3), X.size(4)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3), X.size(4)])

class F_InstanceNorm_3D(nn.Module):
    def __init__(self,channels, tranNum=8):
        super(F_InstanceNorm_3D, self).__init__()
        self.IN = nn.InstanceNorm3d(channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.IN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3), X.size(4)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3), X.size(4)])

class FConvTranspose_3D(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, stride=2 ,ifIni=0, bias=True):
       
        super(FConvTranspose_3D, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.kernel_size = (sizeP, sizeP, sizeP)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        BasisC, BasisS = GetBasis_3D(sizeP,tranNum,inP)        
        self.register_buffer("Basis", torch.cat([BasisC, BasisS], 4))#.cuda() (P,P,P,tranNum,2*P*P)
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini_3D(inP, inNum, outNum, self.expand)#(outNum,inNum,expand,2*sizeP*sizeP)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
                    
        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1,1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = torch.einsum('ijosk,mnak->namsijo', self.Basis, self.weights)
        # tempW = tempW.permute(3,2,0,4,1,5,6)
        for i in range(expand):
            ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
        _filter = tempW.reshape([int(inNum)*self.expand, outNum*tranNum, self.sizeP, self.sizeP, self.sizeP ])
                
#        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})  
        bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1])

        output = F.conv_transpose3d(input, _filter,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=1,
                        groups=1)
        return output + bias

class Fconv_3D_out(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True,iniScale = 1.0):
       
        super(Fconv_3D_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        BasisC, BasisS = GetBasis_3D(sizeP,tranNum,inP)    
        self.register_buffer("Basis", torch.cat([BasisC, BasisS], 4))#.cuda() (P,P,P,tranNum,2*P*P)

        iniw = Getini_3D(inP, inNum, outNum, 1)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=bias)

    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijosk,mnak->mansijo', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP, self.sizeP ])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias

class Fconv_3D_PCA_out(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True,iniScale = 1.0):
       
        super(Fconv_3D_PCA_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_3D_PCA(sizeP,tranNum,inP)        
        self.register_buffer("Basis", Basis)#.cuda())        

        iniw = Getini_3D_reg(Basis.size(4), inNum, outNum, 1, weight)*iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=bias)

    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijosk,mnak->mansijo', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP, self.sizeP ])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijosk,mnak->mansijo', self.Basis, self.weights)
            
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP, self.sizeP ])
            self.register_buffer("filter", _filter)
        return super(Fconv_3D_PCA_out, self).train(mode)


class Fconv_1X1X1(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth = True, iniScale = 1.0):
       
        super(Fconv_1X1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_3D_reg(1, inNum, outNum, self.expand)*iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1,1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(5).unsqueeze(1).repeat([1,tranNum,1,1,1,1,1])
        
        Num = tranNum//expand
        tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,...],tempW[:,i*Num:(i+1)*Num,:,:-i,...]], dim = 3) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim = 1)

        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, 1, 1, 1 ]).cuda()
                
        bias = self.c.repeat([1,1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1,1]).cuda()

        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output+bias

class Fconv_1X1X1_out(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, bias=True):
       
        super(Fconv_1X1X1_out, self).__init__()
        

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

        iniw = Getini_3D_reg(1, inNum, outNum, 1)
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        # self.bias = bias

        
        self.c = nn.Parameter(torch.zeros(1,outNum,1,1,1), requires_grad=bias).cuda()

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = 1#self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(5).unsqueeze(1).repeat([1,tranNum,1,1,1,1,1])
        
        Num = tranNum//expand
        tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,...],tempW[:,i*Num:(i+1)*Num,:,:-i,...]], dim = 3) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim = 1)

        _filter = tempW.reshape([outNum, inNum*tranNum, 1, 1, 1 ]).cuda()
                
        bias = self.c

        output = F.conv3d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output+bias
##############test Fconv_3D###############
# SizeP = 3
# p   = None
# tranNum = 2
# conv1 = Fconv_3D(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
# conv2 = Fconv_3D(SizeP, 2, 2, tranNum, stride=2,inP=p).cuda()
# X = torch.randn([1,3,29,29,29]).cuda()
# X = conv1(X)
# print(X.shape)
# X = conv2(X)
# print(X.shape)
    

############test Fconv_3D_PCA############
# SizeP = 3
# p   = 2
# tranNum = 8
# conv3 = Fconv_3D_PCA(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
# conv4 = Fconv_3D_PCA(SizeP, 2, 2, tranNum, stride = 1,inP=p).cuda()
# X = torch.randn([1,3,29,29,29]).cuda()
# X = conv3(X)
# print(X.shape)
# X = conv4(X)
# print(X.shape)
    
#############test PointwiseAvgPoolAntialiased_3D&&F_BN_3D##########
# SizeP = 7
# p   = 3
# tranNum = 8
# conv3 = Fconv_3D_PCA(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
# conv4 = Fconv_3D_PCA(SizeP, 2, 2, tranNum, inP=p).cuda()
# X = torch.randn([1,3,29,29,29]).cuda()
# BN1   = F_BN_3D(2, tranNum).cuda()
# Pool1 = PointwiseAvgPoolAntialiased_3D(2,2).cuda()
# Drop1 = F_Dropout_3D(0.5,tranNum).cuda()
# X = conv3(X)
# print(X.shape)
# X = conv4(X)
# print(X.shape)
# X = BN1(X)
# print(X.shape)
# X = Pool1(X)
# print(X.shape)
# X = Drop1(X)
# print(X.shape)

###############test others#######################
# SizeP = 7
# p   = 3
# tranNum = 8 
# X = torch.randn([1,3,29,29,29]).cuda()
# conv1 = Fconv(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
# conv3 = Fconv_3D_PCA(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
# X = conv3(X)
# Gp = GroupPooling_3D(tranNum)
# Gp = GroupMeanPooling_3D(tranNum)
# X = Gp(X)

# M1 = MaskModule_3D(7)

# Basis1 = torch.randn([1,3,7,7,7]).cuda()
# X = M1(Basis1)