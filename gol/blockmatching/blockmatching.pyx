'''
Created on 08.10.2014

@author: Flo
'''

import numpy as np
import time
import pylab
import scipy.linalg as la
import numpy.random as rnd
from os import listdir
from os.path import isfile, join
import cv2
from math import ceil
import gc
import shelve
import random

import numpy.random as rnd

class PatchCollection(object):
    
    def __init__(self,init_size):
        self.scaling=None
        self.actual_size=0
        self.patches=np.zeros(init_size,dtype=np.float32)
    
    def add_patches(self,patches):
        #print patches.shape
        #print self.patches.shape
        if self.patches.shape[0]<= self.actual_size+patches.shape[0]:
            new_patches = np.zeros(((self.actual_size+patches.shape[0])*2,self.patches.shape[1]),dtype=np.float32)
            new_patches[0:self.patches.shape[0],:]=self.patches
            self.patches=new_patches
            
        self.patches[self.actual_size:self.actual_size+patches.shape[0],:]=patches
        self.actual_size=self.actual_size+patches.shape[0]
    
    def get_patches(self):
        return self.patches[0:self.actual_size,:]
    
    def pack(self):
        self.patches=self.get_patches()
        
    def prepare_for_saving(self):
        self.scale=np.max(self.patches)
        self.patches = ((self.patches/self.scale)*255).astype(np.int8)
        
    def restore_after_load(self):
        self.patches = self.patches.astype(np.float32)/255.0 * self.scale

def stack_groups(patches,matches,dists,offset,cutoff=None):
    patches=patches.astype(np.float64)
    result = np.zeros([matches.shape[0]*(matches.shape[1]+1),patches.shape[1]],dtype=np.float64)
    j=0
    group_idx=np.zeros([matches.shape[0]],dtype=np.int32)
    for i in xrange(0,matches.shape[0]):
        result[j,:]=patches[offset+i,:]
        group_idx[i]=j
        j=j+1
        for k,d in zip(matches[i,:],dists[i,:]):
            if cutoff and d < cutoff:
                result[j,:]=patches[k,:]
                j=j+1
    return result,group_idx


def img2col_iter(imgs,block,step):
    for image in imgs:
        imgcol,_ = im2col_step(image,block,step)
        yield imgcol
        

def select(iter,i):
    for o in iter:
        yield o[i]

def first_n(iter,n):
    for i in xrange(0,n):
        yield iter.next()

def all_files(path):
    return (join(path,f) for f in listdir(path) if isfile(join(path, f)))

def image_iter(image_files,color_space='RGB'):
    for f in image_files:
        img=f
        #print "processing image ",img
        if color_space=='RGB':
            yield cv2.imread(img)/255.0,img
        else:
            if color_space=='gray':
                im = cv2.imread(img)/255.0
                b, g, r = cv2.split(im)  # get b,g,r
                im = (b+g+r)/3.0
                yield im,img

def padimage(image,block):
    if len(image.shape)==3:
        return np.pad(image,((block[0],block[0]*2),(block[1],block[1]*2),(0,0)),mode='reflect')
    else:
        return np.pad(image,((block[0],block[0]*2),(block[1],block[1]*2)),mode='reflect')

def unpadimage(image,block):
    if len(image.shape)==3:
        return image[block[0]:-(block[0]*2),block[1]:-(block[1]*2),:]
    else:
        return image[block[0]:-(block[0]*2),block[1]:-(block[1]*2)]
    

def col2im(imcol,imsize,block,step,fusion='mean'):
    bx,by=block
    image=np.zeros(imsize,dtype=np.float32)
    rj=imsize[1]-block[1]
    nj=int(ceil(float(rj)/step[1]))
    multiplier = float(np.prod(step))/np.prod(block)
    if fusion=='mean':
        for idx in xrange(0,imcol.shape[0]):
            i = (idx/nj)*step[0]
            j = (idx%nj)*step[1]
            if len(image.shape)==3:
                image[i:i+block[0],j:j+block[1],:]=image[i:i+block[0],j:j+block[1],:]+imcol[idx,:].reshape([bx,by,3])
            else:
                image[i:i+block[0],j:j+block[1]]=image[i:i+block[0],j:j+block[1]]+imcol[idx,:].reshape([bx,by])
        image=image*multiplier
    if fusion=='median':
        buffer = np.zeros([imsize[0],imsize[1],imsize[2],int(ceil(1/multiplier))],np.float32)
        imcol8 = imcol#*255).astype(np.int8)
        counter = np.zeros([imsize[0],imsize[1]],np.int16)
        for idx in xrange(0,imcol.shape[0]):
            i = (idx/nj)*step[0]
            j = (idx%nj)*step[1]
            patch = imcol8[idx,:].reshape([bx,by,3])
            for h in xrange(0,block[0]):
                for w in xrange(0,block[1]):
                    c = counter[i+h,j+w]
                    counter[i+h,j+w]=counter[i+h,j+w]+1
                    buffer[i+w,j+h,:,c]=patch[w,h,:]
        image=np.median(buffer, axis=3,overwrite_input=True)#.astype(np.float32)/255.0
    return image

def im2col_step(Im,block,step=(1,1)):
    bx,by=block
    if len(Im.shape)>2:
        channels=Im.shape[-1]
    else:
        channels=1
    sx,sy=step
    Imcol=[]
    indices=[]

    ri=Im.shape[0]-bx
    rj=Im.shape[1]-by

    for i in xrange(0,ri,sx):
        for j in xrange(0,rj,sy):
            if channels>1:
                Imcol.append((Im[i:i+bx,j:j+by,:].reshape(bx*by*channels)))
            else:
                Imcol.append((Im[i:i+bx,j:j+by].reshape(bx*by*channels)))

    return np.asarray(Imcol),indices
      
def im2col(Im,block,style='sliding'):
    bx,by=block
    h=bx*by*3
    w=(Im.shape[0]-bx)*(Im.shape[1]-by)
    Imcol=np.zeros((w,h))
    indices=[]
    start=time.clock()
    for i in xrange(0,Im.shape[0]-bx):
        for j in xrange(0,Im.shape[1]-by):
            Imcol[i*(Im.shape[1]-by)+j,:]=(Im[i:i+bx,j:j+by,:].reshape(bx*by*3))
            indices.append((i,j))
    return Imcol,indices



def cut_off_distance(dists,p=0.5):
    return sorted(dists.ravel())[int(p*dists.size)]


                           
