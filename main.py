import cv2 as cv
print("hello world")

# dummy function , does nothing
def nothing(x):
    pass

media_location = "media\\vtest.avi"
# create a foreground mask to be used for subtraction
backSub = cv.createBackgroundSubtractorMOG2()

#video caputre object
vid = cv.VideoCapture(media_location)
print(vid.get(cv.CAP_PROP_FRAME_WIDTH))
print(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

#window for trackbar for threshold
cv.namedWindow("slide")

#creating tackbar
cv.createTrackbar("thres","slide",0,255,nothing)
cv.createTrackbar("mblur","slide",3,30,nothing)
cv.createTrackbar("gblur","slide",1,30,nothing)


while (True):

    ret, frame = vid.read()
    fgMask = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    fgMask = backSub.apply(frame)
    fgMask = cv.medianBlur(fgMask,9)
    #fgMask = cv.Canny(fgMask, 100, 1000)
    # cv.imshow('FG Mask', fgMask)
    threshold_value = cv.getTrackbarPos("thres","slide")
    ret, thr = cv.threshold(fgMask,threshold_value,255,cv.THRESH_BINARY)

    #applying contours
    conto, ret = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, conto, -1, (255, 0, 255), 2)
    # print(type(conto))
    for temp_cont in conto:
        if cv.contourArea(temp_cont) < 50 :

            continue
        (x,y,w,h) = cv.boundingRect(temp_cont)
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        cv.drawContours(frame,[temp_cont],-1,(255,0,255),-1)

    #original frame capture
    cv.imshow('frame',frame)

    #cv.imshow('mask',fgMask)

    #final thresholded image
    cv.imshow('threshwindow', thr)


    if cv.waitKey(100) & 0xFF == ord('q'):
        break


vid.release()

cv.destroyAllWindows()




