__author__ = 'florian'
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
import numpy as np
import sys
from math import log, exp

import cv2

import gol.subspace_tracking.utilities.changedetection_net as cdn
import gol.subspace_tracking.utilities.preprocessing as pp
import gol.subspace_tracking.utilities.visualization as viz
import gol.subspace_tracking.pROST as st

datapath = sys.argv[1]
outpath = sys.argv[2]

#vizualizer = viz.Matplotlib_pROST_Visualizer()
vizualizer = viz.OpenCV_pROST_Visualizer()

im_shape = (160,120)
preprocessor = pp.Preprocessor(im_shape,filter_func = lambda x: cv2.GaussianBlur(x,(5,5),0.5),convert_bgr2rgb=False)

startStepU=5E-3
endStepU=1E-4
stepSizeParam= -log(endStepU/startStepU)/(cdn.get_ROI_start(datapath));

pROST = st.pROST(np.prod(im_shape)*3,15,p=0.25,mu=1E-2,t=0.15,phi=5*10E-5,
                 step_size_func=lambda iter: max(exp(-stepSizeParam*iter)*startStepU,endStepU),
                 y_iters=5,
                 median_filter_radius=3)

visualize = True
for name,image,save_func in cdn.stream_sequence(datapath,outpath):
    image_pp = preprocessor.preprocess(image)

    pROST.process_image(image_pp)

    if visualize:
        vizualizer.show_original(preprocessor.original)
        vizualizer.show_background(pROST.get_reconstruction()+0.5)
        vizualizer.show_foreground(pROST.get_error_image())
        vizualizer.show_segmentation(pROST.get_weights())
        vizualizer.update()

    resized_segmentation = cv2.resize(pROST.get_segmentation(),(preprocessor.original_size()[1],preprocessor.original_size()[0]))

    save_func(resized_segmentation)
