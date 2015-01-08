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

