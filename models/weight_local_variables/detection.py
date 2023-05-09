from platform import release
import cv2
import os
from cv2 import imshow
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt

def m_kernel(target_size = 3):
    # Intensity Detection
    targetSize = target_size
    guardSize = targetSize
    filterSize = 3*targetSize
    filterParam = (1.0 / ((filterSize * filterSize) - (guardSize * guardSize)))
    padding = int((filterSize + 1)/2 - (guardSize-1)/2)
    # Make background filter
    # mFilter = np.zeros((filterSize, filterSize),np.float32)
    # filterParam = (1.0 / ((filterSize * filterSize) - (guardSize * guardSize)))
    # padding = int((filterSize + 1)/2 - (guardSize-1)/2)
    # for i in range(padding -1):
    #     mFilter[i,:] = filterParam
    #     mFilter[int(filterSize) - i - 1,:] = filterParam
    #     mFilter[:, i] = filterParam
    #     mFilter[:, int(filterSize) - i - 1] = filterParam

    mFilter2 = np.full((filterSize,filterSize),filterParam)
    centersize = filterSize - 2*padding + 2
    mFilter2[padding-1:padding+centersize-1,padding-1:padding+centersize-1] = np.zeros((centersize,centersize))
    return mFilter2

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def detect (img):
    # Apply Wiener Filter
    kernel = gaussian_kernel(5)
    filtered_img = wiener_filter(img, kernel, K = 10)
    filtered_img *= 255.0/filtered_img.max()

    # Highpass filter
    kernel = np.ones((3,3),np.float32)
    kernel = (-1)* kernel
    kernel[1,1] = 8
    dummy = convolve2d(filtered_img, kernel, mode = 'valid')
    # dummy *= 127.0/dummy.max()
    # dummy = dummy + 127.0
    
    targetSize = 3
    filterSize = 3*targetSize
    mFilter = m_kernel(3)
    SNR = 10.0

    # x = np.pad(img, pad_width=int(filterSize/2), mode='constant', constant_values=0)
    x = np.pad(dummy, pad_width=int(filterSize/2), mode='edge')
    x_b = np.pad(img, pad_width=int(filterSize/2), mode='edge')

    # background = convolve2d(x, mFilter, mode = 'valid')
    # foreground = dummy - background
    # binary_where = np.where(foreground >= SNR, 255, 0)
    # binary_where = binary_where.astype(np.uint8)

    background_b = convolve2d(x_b, mFilter, mode = 'valid')
    foreground_b = img - background_b
    binary_where_b = np.where(foreground_b > SNR, 255, 0)
    binary_where_b = binary_where_b.astype(np.uint8)

    #print(foreground[1])
    # display = [img, filtered_img, binary_where]
    # fig = plt.figure(figsize=(12, 10))
    # label = ['Original Image', 'Wiener Filter applied', 'HighPass Filter']
    # for i in range(len(display)):
    #     fig.add_subplot(2, 2, i+1)
    #     plt.imshow(display[i], cmap = 'gray', vmin=0, vmax=255)
    #     plt.title(label[i])
    #print(dummy[1:])
    #plt.imshow(binary_where, cmap = 'gray')
    #xprint(mFilter)



    contour, hierachy = cv2.findContours(binary_where_b,mode = cv2.RETR_TREE, method= cv2.CHAIN_APPROX_NONE)
    #binary_where_b = cv2.cvtColor(binary_where_b,cv2.COLOR_GRAY2BGR)
    for ctr in contour:
        x,y,w,h = cv2.boundingRect(ctr)
        if 3< max(w,h) < 11:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('detect', binary_where_b)

if __name__ == "__main__":
    # Read image from video 
    # /home/vee/Desktopssss/Sample/22-08-18-15-39-20.mp4
    global frame_no
    frame_no = 0
    expect_fps = 15
    vidcap = cv2.VideoCapture('/home/manhtb2/Desktop/manhtb2/1x2_Sim/Sample_2.avi')
    while (vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break

        # img = rgb2gray(frame)
        # cv2.imshow('Display', frame)
        # cv2.waitKey(10)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # Detection 
        if frame_no%expect_fps == 0:
        #cv2,imshow('Display',frame)
            detect(frame)
            cv2.imshow('original',frame)
        frame_no += 1
    
        if cv2.waitKey(1) == ord('x'):
            vidcap.release()

        # plt.show()
    

