from scipy.signal import gaussian, wiener, convolve2d
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

def get_wiener(img, k = 5):
    return np.uint8(cv.GaussianBlur(img,(k,k),1))

def old_wiener(img, k = 3):
    gaussian_kernel = gaussian(k,k/3).reshape(k,1)
    g = np.dot(gaussian_kernel, gaussian_kernel.transpose())
    g /= np.sum(g)
    dummy = np.fft.fft2(img)
    kernel = np.fft.fft2(g,img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + 10)
    dummy = dummy*kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    # print(np.max(dummy), np.sum(np.where(dummy>0.99*np.max(dummy))))
    dummy =  ((dummy - dummy.min()) * (1/(dummy.max() - dummy.min()) * 255)).astype('uint8')

    return dummy
    return np.uint8(wiener(img,k))

def get_highpass(img, mode = 1):
    kernel = -np.ones((3,3))
    kernel[1,1] = 8
    r =  convolve2d(img,kernel,mode = 'same',boundary = 'symm')
    #r = cv.filter2D(img,-1,kernel,borderType=cv.BORDER_REFLECT)
    r = np.where(r>0,r,0)
    if mode == 1:
        return np.uint8(np.where(r>10,255,0))
    if mode == 2:
        return ((r - r.min()) * (1/(r.max() - r.min()) * 255)).astype('uint8')
        return np.uint8(r)
    else:
        raise Exception('Wrong Argument')

def show_vid(PATH):
    vid = cv.VideoCapture(PATH)
    frame_no = 0
    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            break
        frame_no+=1
        if 1200>= frame_no >= 800 and frame_no%6 == 0:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)[20:1044,300:1580]
            fram = frame.copy()
            fram.astype('float64')
            cv.imshow('original',frame)

            #cv.imshow('filtered',np.uint8(old_wiener(frame)))

            cv.imshow('ghighpass',get_highpass(get_wiener(fram),2))
            hw = get_highpass(old_wiener(fram),2)
            #cv.imshow('whighpass',hw)
        # print(frame.shape, hw.shape)
        
        # cv.imshow('ghighpass',get_wiener(fram))
        # cv.imshow('whighpass',old_wiener(fram))

        # cv.imshow('highpass2',get_highpass((frame)))
        
        # tophat = cv.morphologyEx(frame,cv.MORPH_TOPHAT,np.ones((3,3)))
        # tophat = np.uint8(np.where(tophat>10,255,0))

        # cv.imshow('tophat', tophat)

        if cv.waitKey(1) == ord('x'):
            break


def show_img(PATH):
    img = cv.imread(PATH)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('original',img)

    cv.imshow('filtered',old_wiener(img))

    cv.imshow('highpass',get_highpass(old_wiener(img),2))
    print(get_highpass(old_wiener(img),2).shape, img.shape)
    
    if cv.waitKey(0000) == ord('x'):
        exit()

if __name__ == "__main__":
    PATH = '/home/irst/Manhtb2/IRST/IRST_Data/22-09-22/22-09-22-19-36-53.mp4'
    #PATH = '/home/irst/Manhtb2/IRST/IRST_Data/22-09-22/22-09-22-20-02-16.mp4'
    show_vid(PATH)

    # PATH = '/home/irst/Manhtb2/K/1.bmp'
    # show_img(PATH)
    # PATH = '/home/manhtb2/Desktop/opencv_python/paper imple/images/images/Misc_5.png'
    # show_img(PATH)