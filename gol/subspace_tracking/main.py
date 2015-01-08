__author__ = 'florian'

import numpy as np
from math import log, exp

import cv2

import gol.subspace_tracking.utilities.changedetection_net as cdn
import gol.subspace_tracking.utilities.preprocessing as pp
import gol.subspace_tracking.utilities.visualization as viz
import gol.subspace_tracking.pROST as st


sequence = '/media/florian/885AE8AA5AE895EA/changedetection.net/dataset/cameraJitter/traffic'


#vizualizer = viz.Matplotlib_pROST_Visualizer()
vizualizer = viz.OpenCV_pROST_Visualizer()

im_shape = (160,120)
preprocessor = pp.Preprocessor(im_shape,filter_func = lambda x: cv2.GaussianBlur(x,(5,5),0.5),convert_bgr2rgb=False)

startStepU=5E-3
endStepU=1E-4
stepSizeParam= -log(endStepU/startStepU)/(cdn.get_ROI_start(sequence));

pROST = st.pROST(np.prod(im_shape)*3,15,p=0.25,mu=1E-2,t=0.25,phi=5*10E-5,
                 step_size_func=lambda iter: max(exp(-stepSizeParam*iter)*startStepU,endStepU),
                 y_iters=5,
                 median_filter_radius=5)

visualize = True
for name,image,save_func in cdn.stream_sequence(sequence):
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
