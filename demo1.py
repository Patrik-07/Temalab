import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# Get realtime webcam feed

# 0 -> webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Raw Webcam Feed', frame)

    # waitKey(0) will display the window infinitely until any keypress
    #
    # waitKey(10) will display a frame for 10 ms, after which display will
    # be automatically closed. Since the OS has a minimum time between
    # switching threads, the function will not wait exactly 10 ms, it will
    # wait at least 10 ms, depending on what else is running on your computer
    # at that time.
    # 
    # waitKey returns the code of the key pressed. If you have numlock activated,
    # the returned code might be different, but the last byte is the same whether
    # or not it is activated.
    # 
    # e.g:
    # key = cv2.waitKey(10)
    # print(key)
    # 
    # pressing c, this will output 1048675 when NumLock is activated
    # 99 otherwise. (1048675 = 100000000000001100011, 99 = 1100011: the last
    # byte is identical.
    #
    # when we write
    # 
    # key = cv2.waitKey(33) & 0b11111111
    # 
    # key will be the last byte of the key returned by cv2.waitKey
    # we can then compare this to ord('q'), and if they are equal, we can close
    # the video capture
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# releases the webcam resource
cap.release()

# closes the window
cv2.destroyAllWindows()
