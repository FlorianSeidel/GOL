
from gol.optimization.CG import CG
from gol.goal.constraints import no_linear_dependencies_constraint, full_rank_constraint
from gol.goal.cost_functions import normalized_l2norm_cost, logsquared_cost
from gol.optimization.ObliqueCG import ObliqueCG

import numpy.random as rnd

__author__ = 'florian'

import cv2
from gol.blockmatching.blockmatching import im2col_step, col2im, PatchCollection, stack_groups, padimage, unpadimage
import numpy as np

rng = np.random
import theano
import theano.tensor as T
from math import sqrt, log, log10

def noise_image(image,stddev):
    if len(image.shape)==3:
        return np.maximum(0.0,np.minimum(1.0,image + rnd.randn(image.shape[0],image.shape[1],image.shape[2])*stddev))
    return np.maximum(0.0,np.minimum(1.0,image + rnd.randn(image.shape[0],image.shape[1])*stddev))

def rgb2gray_op(img):
    return T.mean(img, axis=2)


def rgb2gray_stack_op(imcol,k):
    return (imcol[0:k:3, :] + imcol[1:k:3, :] + imcol[2:k:3, :])/3.0

def normalizing_AOP(Omega, k):
    normalizer = theano.shared(np.eye(k, dtype=np.float32) - (1.0 / k) * np.ones(k, dtype=np.float32))
    return T.dot(Omega, normalizer)


def AOP_to_theano_format(Omega, filter_size):
    no_channels = Omega.shape[0]
    no_img_channels = 1
    return Omega.reshape([no_channels, no_img_channels, filter_size, filter_size])


def filter_image(dictionary,channels):
    dict = dictionary
    shape =  dict.shape
    size = int(sqrt(float(shape[0]/float(channels))))
    block = (size,size)
    rows = []
    j=0
    while j<shape[1]:
        row = np.zeros((block[0],block[1]*26,channels),dtype=np.float32)
        for i in xrange(0,25):
            if j>=shape[1]:
                break
            filter = dict[:,j].reshape(block[0],block[1],channels)
            filter -= np.min(filter)
            filter /= np.max(filter)
            row[:,i*block[1]:(i+1)*block[1],:]=filter
            j+=1
        rows.append(row)

    image = np.zeros((block[0],block[1]*26,channels),dtype=np.float32)
    for row in rows:
        image = np.concatenate([image,row],axis=0)
    return image


def create_aop_learner(data, lifting=2, nu=1E3, kappa=1 * 1e6, mu=8 * 1E4):
    k = data.shape[0]
    init_Omega = np.float32(rng.randn(k * lifting, k))
    for i in xrange(0, k * lifting):
        init_Omega[i, :] = init_Omega[i, :] / sqrt((init_Omega[i, :] ** 2).sum())
    Omega = theano.shared(init_Omega)
    Omega_sym = T.matrix("Omega")
    kappa *= 1.0 / log((k ** 2))
    mu = mu * 2.0 / ((lifting * k) ** 2 - lifting * k)

    Omega_normal = normalizing_AOP(Omega_sym, k)
    no_lin_constraint = mu * no_linear_dependencies_constraint(Omega_sym, k * lifting)
    full_rank_constraint_ = kappa * full_rank_constraint(Omega_sym, k * lifting, k)

    cost_Omega_data = normalized_l2norm_cost(logsquared_cost(T.dot(Omega_normal, data), nu, axis=0),
                                             data.shape[1]) + no_lin_constraint + full_rank_constraint_

    grad_Omega_data = T.grad(cost_Omega_data, Omega_sym)

    cg_Omega_data = ObliqueCG(Omega, cost_Omega_data, grad_Omega_data, Omega_sym, k * lifting, k, t_init=1, rho=0.9,
                              max_iter_line_search=125)
    return cg_Omega_data, Omega, Omega_sym


def create_aop_denoiser(Omega, noisy,nu, alpha):
    Omega_sym = T.matrix('Omega')
    noisy_shared = theano.shared(noisy.astype(np.float32))
    denoised = theano.shared(np.copy(noisy.astype(np.float32)))
    denoised_sym = T.matrix("denoised")
    Omega_normal = normalizing_AOP(Omega_sym, noisy.shape[0])
    cost = (((noisy_shared - denoised_sym) ** 2).sum() / denoised_sym.shape[1]) + \
           alpha * normalized_l2norm_cost(logsquared_cost(T.dot(Omega_normal, denoised_sym), nu, axis=0), denoised_sym.shape[1])
    grad = theano.grad(cost, denoised_sym)
    cg_denoising = CG(denoised, cost, grad, denoised_sym, k=noisy.shape[0], n=noisy.shape[1], t_init=1, rho=0.9,
                      max_iter_line_search=125, other_givens={Omega_sym: Omega})
    return cg_denoising,denoised


def collect_training_data(images, select_patches):
    patches = None
    for image in images:
        if not patches:
            patches = PatchCollection((1,select_patches(image).shape[0]))
        patches.add_patches(select_patches(image).T)
    return patches.get_patches().T


def load_images(image_files,block,step):
    for f in image_files:
        yield Image(cv2.imread(f).astype(np.float32) / 255.0,block,step)


class Image:
    def __init__(self, image, block, step):
        self.image = image
        self.r = None
        self.g = None
        self.b = None
        self.block = block
        self.step = step
        self.image_block = None
        self.r_block = None
        self.g_block = None
        self.b_block = None
        self.split_image()
        self.block_image()
        self.U = None
        self.thin = None
        self.flann = None

    def block_image(self):
        self.image_block = im2col_step(self.image, self.block, self.step)[0].T
        self.r_block = im2col_step(self.r, self.block, self.step)[0].T
        self.g_block = im2col_step(self.g, self.block, self.step)[0].T
        self.b_block = im2col_step(self.b, self.block, self.step)[0].T
        return self

    def split_image(self):
        self.r, self.g, self.b = cv2.split(self.image)  # get b,g,r
        return self

    def bgr2rgb(self):
        self.b, self.g, self.r = cv2.split(self.image)  # get b,g,r
        self.merge_channels()
        self.split_image()
        return self

    def blocks2image(self,fusion='mean'):
        self.image = col2im(self.image_block.T, self.image.shape, self.block, self.step,fusion=fusion)
        return self

    def blocks2channels(self,fusion='mean'):
        self.r = col2im(self.r_block.T, self.r.shape, self.block, self.step,fusion=fusion)
        self.g = col2im(self.g_block.T, self.g.shape, self.block, self.step,fusion=fusion)
        self.b = col2im(self.b_block.T, self.b.shape, self.block, self.step,fusion=fusion)
        return self

    def merge_channels(self):
        self.image = cv2.merge([self.r, self.g, self.b])
        return self

    def sub_image(self, left_upper, size):
        part = self.image[left_upper[0]:left_upper[0] + size[0],
               left_upper[1]:left_upper[1] + size[1]]
        return Image(part, self.block, self.step)


    def noise(self,sigma):
        self.image=noise_image(self.image,sigma)
        self.split_image()
        return self

    def copy(self,step=None):
        if step==None:
            return Image(np.copy(self.image), self.block, self.step)
        return Image(np.copy(self.image),self.block,step)

    def pad(self):
        self.image=padimage(self.image,self.block)
        return self

    def unpad(self):
        self.image=unpadimage(self.image,self.block)
        return self

    def PSNR(self,other_image):
        xd = self.image - other_image.image
        return 10 * log10(1.0 / np.mean(xd ** 2))
