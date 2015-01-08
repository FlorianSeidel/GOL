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
__author__ = 'florian'


import cv2
import os
import os.path
import random
import numpy as np


def make_write_image_to_output_folder(folder, name):
    def write_image_to_output_folder(image):
        print "writing image to ", folder, " with name ", name
        cv2.imwrite(os.path.join(folder, name),image)

    return write_image_to_output_folder


def path_exists_or_die(sequence_path):
    if not os.path.exists(sequence_path):
        print "path does not exist ", sequence_path
        exit()


def read_roi_or_die(sequence_path):
    temporal_roi_file = os.path.join(sequence_path, 'temporalROI.txt')
    if not os.path.isfile(temporal_roi_file):
        print temporal_roi_file, " does not exist"
        exit()
    else:
        temporal_roi_file_handle = open(temporal_roi_file, 'r')
        roi_strings=temporal_roi_file_handle.readline().split()
        start_idx = int(roi_strings[0])
        end_idx = int(roi_strings[1])
    return start_idx,end_idx


def get_output_folder(sequence_path):
    output_folder_path = os.path.join(sequence_path, 'output')
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    return output_folder_path

def stream_images(sequence_path,output_folder_path,start_idx,end_idx,idx_func):
    for i in xrange(start_idx,end_idx):
        idx = idx_func(i,start_idx,end_idx)
        file_name = os.path.join(sequence_path, 'input', 'in'+('%6.6u' % idx) + ".jpg")
        print "Reading file ",file_name
        yield file_name,cv2.imread(file_name).astype(np.float32)/255.0, make_write_image_to_output_folder(output_folder_path, 'bin'+ ('%6.6u' % idx) + ".png")


def sample_from_beginning_of_sequence(sequence_path, samples):
    print "Sampling from sequence: ", sequence_path
    path_exists_or_die(sequence_path)
    #read temporalROI
    start_idx, _ = read_roi_or_die(sequence_path)

    print "Sampling from the first ", start_idx, " images of the sequence."
    output_folder_path = get_output_folder(sequence_path)

    return stream_images(sequence_path,output_folder_path,0,samples,lambda i,nope,end_idx:random.randint(0, start_idx - 1))

def get_ROI_start(sequence_path):
    return read_roi_or_die(sequence_path)[0]

def make_outpath(outpath):
        os.makedirs(outpath)

def stream_sequence(sequence_path,outpath,from_ROI_start=False):
    print "Streaming from sequence: ", sequence_path
    path_exists_or_die(sequence_path)
    #create output path
    if not os.path.exists(outpath):
        make_outpath(outpath)
    #read temporalROI
    start_idx,end_idx = read_roi_or_die(sequence_path)
    if not from_ROI_start:
        start_idx=1
    end_idx=end_idx+1
    print "Streaming from index ",start_idx," to index ", end_idx
    output_folder_path = outpath

    return stream_images(sequence_path,output_folder_path,start_idx,end_idx,lambda i,start_idx,end_idx:i)



