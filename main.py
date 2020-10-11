# -*- coding: utf-8 -*-
import cv2
import serial
from threading import Thread
import numpy as np

# 스레드 쉬는 시간
threading_Time = 5/1000

# 38번 공용 변수
val_38 = False

#----- 읽고 값 전달 하는 스레드 -----
def RX_Receiving(ser):
    global receiving_exit, threading_Time
    global val_38

    receiving_exit = 1
    while True:
        if receiving_exit == 0:
            break
        time.sleep(threading_Time)

        while ser.inWaiting() > 0:
            result = ser.read(1)
            RX = ord(result)
            print("RX=" + str(RX))
#----------------------------------------

#----- 메인 -----
if __name__ == '__main__':

    #----- 카메라 세팅 -----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    cap.set(cv2.CAP_PROP_FPS, 30)
    #----------------------

    #----- 시리얼 포트 연결 및 스레드 생성 -----
    BPS = 4800
    serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
    serial_port.flush()  # serial cls
    time.sleep(0.5)

    serial_t = Thread(target=RX_Receiving, args=(serial_port,))
    serial_t.daemon = True
    serial_t.start()
    #--------------------------------------

    #----- 동작 메인 루프 -----
    while True:
        (grabbed, frame) = cap.read()
        cv2.imshow('FRAME', frame)

        key = 0xFF & cv2.waitKey(1)

        if key == 27:  # ESC  Key
            break
    camera.release()
    cv2.destroyAllWindows()
