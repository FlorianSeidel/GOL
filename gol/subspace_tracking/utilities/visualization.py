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
import matplotlib.pyplot as plt
import cv2

class Matplotlib_pROST_Visualizer:

    def __init__(self):
        fig = plt.figure()
        self.original = fig.add_subplot(221)
        self.background = fig.add_subplot(222)
        self.foreground = fig.add_subplot(223)
        self.segmentation = fig.add_subplot(224)
        plt.ion()
        plt.show()


    def show_original(self,image):
        self.original.imshow(image)

    def show_background(self,image):
        self.background.imshow(image)

    def show_foreground(self,image):
        self.foreground.imshow(image)

    def show_segmentation(self,image):
        self.segmentation.imshow(image)

    def update(self):
        plt.pause(0.001)

class OpenCV_pROST_Visualizer:

    def __init__(self):
        pass

    def show_original(self,image):
        cv2.imshow("Original",image)

    def show_background(self,image):
        cv2.imshow("Background",image)

    def show_foreground(self,image):
        cv2.imshow("Foreground",image)

    def show_segmentation(self,image):
        cv2.imshow("Segmentation",image)

    def update(self):
        cv2.waitKey(1)