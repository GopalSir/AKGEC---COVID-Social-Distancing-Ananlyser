import cv2 as cv
print("hello world")

backSub = cv.createBackgroundSubtractorKNN(50)

vid = cv.VideoCapture(0)

while (True):

    ret, frame = vid.read()

    fgMask = backSub.apply(frame)
    cv.medianBlur(fgMask, 5, fgMask)
    cv.imshow('frame', frame)
    cv.imshow('FG Mask', fgMask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()

cv.destroyAllWindows()



