from fsm_thresh import FSM
from preprocessing import get_highpass, get_wiener
import numpy as np
import cv2 as cv
import pandas as pd
import sys

to_watch = [66,84,12,11]
beginf, endf = int(sys.argv[1]),int(sys.argv[2])
csv_result = {'Sample':[],'Adaptive param':[], 'Trusted detection rate %':[], 'Detection rate %':[], 'False alarm per frame': []}
for file_i in range(beginf, endf):
    data = dict()
    with open(f'/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/gt/Sample_{file_i}.txt','r') as f:
        for fline in f.readlines():
            flinez = fline.splitlines()[0]
            check = [int(i) for i in flinez.split(',')]
            data[check[0]] = check[1:]
    f.close()

    PATH = f'/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/Sample_{file_i}.avi'
    vid = cv.VideoCapture(PATH)

    frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    # fps = vid.get(cv.CAP_PROP_FPS)
    # vid_time = frames/fps

    frame_no = 2
    detected = 0
    detected_small = 0
    detected_10 = 0
    detected_small_10 = 0
    err = 0
    err_10 = 0

    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            break
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # cv.imshow('frame',np.uint8(frame))

        alarm = 0
        alarm_10 = 0

        #framez = get_highpass(get_wiener(frame),mode = 2)
        framez = frame.astype('float64')

        res,thresh = FSM(framez,0,50,0.04)
        res2, thresh2 = FSM(framez,10,50,0.04)

        sur = np.uint8(np.where(res>thresh,255,0))
        sur2 = np.uint8(np.where(res2>thresh2,255,0))

        contour, hiearachy = cv.findContours (sur,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour2, hiearachy2 = cv.findContours (sur2,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contour:
            if len(c)>=3:
                alarm += 1

        for c2 in contour2:
            if len(c2)>=3:
                alarm_10 += 1

        coor_x, coor_y = data[frame_no][:2]
        detect_val = np.sum(sur[max(0,coor_y-6):coor_y+6,coor_x-6:coor_x+6])
        detect_val2 = np.sum(sur2[max(0,coor_y-6):coor_y+6,coor_x-6:coor_x+6])

        if detect_val>= 255*3:
            err += alarm - 1
            detected += 1
            detected_small += 1
        else:
            if detect_val > 10:
                detected_small += 1
                err += len(contour) - 1
            else:
                err += len(contour)

        if detect_val2>= 255*3:
            err_10 += alarm_10 - 1
            detected_10 += 1
            detected_small_10 += 1
        else:
            if detect_val2 > 10:
                detected_small_10 += 1
                err_10 += len(contour2) - 1
            else:
                err_10 += len(contour2)
            #exit()

        if cv.waitKey(1) == ord('x'):
            vid.release()
        if frame_no%20 == 0:
            print(f'Sample {file_i} {frame_no} frames processed')
        frame_no += 1


    print (f'Ratio: {detected/(frames)*100}%')
    print (f'False alarms: {err/frames*100}')

    print (f'Ratio10: {detected_10/(frames)*100}%')
    print (f'False alarms10: {err_10/frames*100}')

    csv_result['Sample'].append(file_i)
    csv_result['Adaptive param'].append(0)
    csv_result['Trusted detection rate %'].append(f'{detected/(frames)*100}')
    csv_result['Detection rate %'].append(f'{detected_small/(frames)*100}')
    csv_result['False alarm per frame'].append(f'{err/frames}')

    csv_result['Sample'].append(file_i)
    csv_result['Adaptive param'].append(10)
    csv_result['Trusted detection rate %'].append(f'{detected_10/(frames)*100}')
    csv_result['Detection rate %'].append(f'{detected_small_10/(frames)*100}')
    csv_result['False alarm per frame'].append(f'{err_10/frames}')
print('done!')

df = pd.DataFrame(csv_result)
df.to_csv(f'220802sim_wlcv_result_{beginf}_{endf-1}.csv')
