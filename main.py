import cv2
import numpy as np
import serial.tools.list_ports

#serialInit = serial.Serial('COM3', 9600)
#listPort = serial.tools.list_ports.comports()

flag = 1

# 'http://192.168.1.4:8080/video'
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _,frame = cap.read()

    frame = cv2.resize(frame, (640, 480))
    # cv2.imshow('video', frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([30, 50, 50])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    opening_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((20, 20), np.uint8)
    opening_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    opening_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    erosion_green = cv2.erode(opening_green, kernel, iterations=1)
    erosion_yellow = cv2.erode(opening_yellow, kernel, iterations=1)

    kernel = np.ones((20, 20), np.uint8)
    erosion_red = cv2.erode(opening_red, kernel, iterations=1)

    def thresh_callback(val):
        threshold = val
        global flag
        print(flag)
        canny_output_green = cv2.Canny(erosion_green, threshold, threshold * 2)
        canny_output_red = cv2.Canny(erosion_red, threshold, threshold * 2)
        canny_output_yellow = cv2.Canny(erosion_yellow, threshold, threshold * 2)

        contours_green, _ = cv2.findContours(canny_output_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(canny_output_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(canny_output_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours_red:
            contours_poly_red = cv2.approxPolyDP(contours_red[0], 3, True)
            centers_red, radius_red = cv2.minEnclosingCircle(contours_poly_red)
            color = (0, 255, 0)
            if radius_red > 20 and flag == 1:
                cv2.rectangle(frame,
                              (int(centers_red[0]) - int(radius_red) - 8, int(centers_red[1]) - int(radius_red) - 8),
                              (int(centers_red[0]) + int(radius_red) + 8, int(centers_red[1]) + int(radius_red) + 8),
                              color,
                              2)
                #serialInit.write('r'.encode())
                #flag = 0

        elif contours_yellow:
            contours_poly_yellow = cv2.approxPolyDP(contours_yellow[0], 3, True)
            centers_yellow, radius_yellow = cv2.minEnclosingCircle(contours_poly_yellow)
            color = (0, 255, 0)
            if radius_yellow > 20 and flag == 1:
                cv2.rectangle(frame,
                              (int(centers_yellow[0]) - int(radius_yellow) - 8, int(centers_yellow[1]) - int(radius_yellow) - 8),
                              (int(centers_yellow[0]) + int(radius_yellow) + 8, int(centers_yellow[1]) + int(radius_yellow) + 8),
                              color,
                              2)
                #serialInit.write('y'.encode())
                #flag = 0

        elif contours_green:
            contours_poly_green = cv2.approxPolyDP(contours_green[0], 3, True)
            centers_green, radius_green = cv2.minEnclosingCircle(contours_poly_green)
            if radius_green > 20 and flag == 1:
                color = (0, 255, 0)
                cv2.rectangle(frame,
                              (int(centers_green[0]) - int(radius_green) - 8, int(centers_green[1]) - int(radius_green) - 8),
                              (int(centers_green[0]) + int(radius_green) + 8, int(centers_green[1]) + int(radius_green) + 8),
                              color,
                              2)
                #serialInit.write('g'.encode())
                #flag = 0

        else:
            flag = 1

        cv2.imshow('Contours', frame)

    thresh = 100  # initial threshold
    thresh_callback(thresh)

    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

