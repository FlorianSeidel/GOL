__author__ = 'florian'

import cv2
import numpy as np

class Preprocessor:
    def __init__(self, size=(-1,-1), filter_func=lambda x: x,convert_bgr2rgb=False):
        self.average = None
        self.filter_func = filter_func
        self.size=size
        self.idx=0
        self.original=None
        self.convert=convert_bgr2rgb

    def preprocess(self, image):

        b, g, r = cv2.split(image)

        if self.convert:
            img = cv2.merge([r, g, b])
        else:
            img = cv2.merge([b,g,r])

        self.original=img

        if self.size!=(-1,-1):
            img = cv2.resize(img,self.size)

        if self.idx==0:
            self.average = img
        else:
            self.average*=self.idx/(self.idx+1.0)
            self.average+=img*(1.0/(self.idx+1.0))

        img -= self.average


        self.idx+=1
        return self.filter_func(img)

    def original_size(self):
        return self.average.shape

