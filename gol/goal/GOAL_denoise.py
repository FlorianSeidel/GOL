'''
Copyright (c) 2015 FlorianSeidel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
__author__ = 'florian'

import theano

from gol.goal.utilities import Image, \
    create_aop_denoiser

import matplotlib.pyplot as plt
import cv2
import shelve
import numpy as np

if __name__ == '__main__':
    block=(8,8)
    step=(2,2)
    workbench = shelve.open('denoising_workbench.db')
    Omega_r_mat,Omega_g_mat,Omega_b_mat = workbench['Omega']
    params = workbench['Omega_param']
    nu = params['nu']
    image_raw = (cv2.imread('test_images/lena.png')/255.0).astype(np.float32)
    image = Image(image_raw,block,step).bgr2rgb()
    original_image = image.copy()
    image.pad().noise(25.0/255.0).split_image().block_image()
    noisy_image = image.copy().unpad()

    Omega_r = theano.shared(Omega_r_mat)
    Omega_g = theano.shared(Omega_g_mat)
    Omega_b = theano.shared(Omega_b_mat)

    aop_denoiser_r, denoised_r = create_aop_denoiser(Omega_r,image.r_block,nu,0.5)
    aop_denoiser_g, denoised_g = create_aop_denoiser(Omega_g,image.g_block,nu,0.5)
    aop_denoiser_b, denoised_b = create_aop_denoiser(Omega_b,image.b_block,nu,0.5)

    def cost_monitor(D):
        print "Cost: ", D['c']

    aop_denoiser_r.optimize(1000,cost_monitor,10)
    aop_denoiser_g.optimize(1000,cost_monitor,10)
    aop_denoiser_b.optimize(1000,cost_monitor,10)

    image.r_block=denoised_r.get_value()
    image.g_block=denoised_g.get_value()
    image.b_block=denoised_b.get_value()

    image.blocks2channels().merge_channels().unpad()
    print "PSNR after AOP step: ",image.PSNR(original_image)

    workbench['aop_result']=image.image

    plt.subplots(1,3)
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.image)
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image.image)
    plt.subplot(1, 3, 3)
    plt.imshow(image.image)
    plt.show()

