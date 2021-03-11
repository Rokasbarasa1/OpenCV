import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


#SHOW IMAGE-------------------------------------------------------------------------------------------------------------------------------------------------
# print("Package imported")
#
# img = cv2.imread("resources/lena.png")
#
# cv2.imshow("Output", img)
# cv2.waitKey(0)

#SHOW VIDEO-------------------------------------------------------------------------------------------------------------------------------------------------
# cap = cv2.VideoCapture("Resources/test_video.mp4")
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break;
#

#SHOW WEbCAM-------------------------------------------------------------------------------------------------------------------------------------------------
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break;

#SHOW GREY, blured AND CANNY IMAGE(borders)-------------------------------------------------------------------------------------------------------------------------------------------------
# kernel = np.ones((5,5), np.uint8)
# img = cv2.imread("resources/lena.png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgblur = cv2.GaussianBlur(imgGray, (17,17),0)
# imgCanny = cv2.Canny(img, 150,200)
# imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
#
# cv2.imshow("Gray image", imgGray)
# cv2.imshow("lur image", imgblur)
# cv2.imshow("Canny image", imgCanny)
# cv2.imshow("Dilation image", imgDialation)
# cv2.imshow("Eroded image", imgEroded)
# cv2.waitKey(0)

#SOME THEORY: IN OPEN CV Y AXIS IS FLIPED. SO WHERE IT WAS POSITIVE IT IS NEGATIVE-------------------------------------------------------------------------------------------------------------------------------------------------

#
# img = cv2.imread("resources/lambo.png")
# print(img.shape)
#
# imgResize = cv2.resize(img, (300,200))
# print(imgResize.shape)

#DEFINE START POINT AND END POINT OF HEIGHT AND LENGHT
# FIRST Y AXIS SPECIFIED THEN X AXIS
# imgCropped = img[0:200, 200:500]
#
# cv2.imshow("Image", img)
# cv2.imshow("Image resize", imgResize)
# cv2.imshow("Cropped image", imgCropped)
#
# cv2.waitKey(0)

#VARIOUS lines, circles text and rectangles.-------------------------------------------------------------------------------------------------------------------------------------------------
# img = np.zeros((512,512, 3), np.uint8)
# print(img.shape)
# [:] everything to everything
#Colors part of the thing blue
# img[200:300, 100:300] = 255,0,0
# img[:] = 255,0,0
#GREEN LINE FROM CENTER HOLY SHIT
# cv2.line(img,(0, 0), (300,300), (0,255,0),3)
#replace thickenss with cv2.FILLED for a filed rectangle

# cv2.rectangle(img, (0,0), (250,350), (0,0,255), 2)
#
# cv2.line(img,(0, 0), (img.shape[1],img.shape[0]), (0,255,0),3)
#
# cv2.circle(img, (400,50), 30, (255,255,255), 5)
#
# cv2.putText(img, " OPENCV ", (300,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,0), 3)
#
#
# cv2.imshow("LAck images", img)
#
# cv2.waitKey(0)

#GETTING THINGS OF THE TABLE AND WARPING THEM TO LOOK LIKE A PAPER INF FRONT OF VIEW-------------------------------------------------------------------------------------------------------------------------------------------------
#sTRETCHISNG

# img = cv2.imread("resources/cards.png")
# width, height = 250,350
# #this is a wierd way of doing it. He marks the points of the four edges of the card first
# #THen he makes an array of corners of where the thing is supposed to be
# #What this basicaly does it streches the image to fit the box thing of the second point
# #So its not magic but still very impresive
# pts1 = np.float32([[157,306],[397,263],[214,668],[483,608]])
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
#
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
#
# imgOutput = cv2.warpPerspective(img,matrix,(width,height))
#
# cv2.imshow("Image", img)
#
# cv2.imshow("Output warped", imgOutput)
#
# cv2.waitKey(0)

#STACKING IMAGES VERY SIMPLE THE FUNCTION IS A HELPER-------------------------------------------------------------------------------------------------------------------------------------------------

# #fixes some adding images problems
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# img = cv2.imread("resources/lena.png")
# #BOTH HAVE TO HAVE SAME COLOR QUALITY IF RGB BOTH RGB
#
# #imgHor = np.hstack((img,img))
# #imgVer = np.vstack((img,img))
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# imgStack = stackImages(0.2,([img,imgGray,img],[img,imgGray,img],[img,imgGray,img]))
#
# # cv2.imshow("Horizontal", imgHor)
# # cv2.imshow("Vertical", imgVer)
# cv2.imshow("Vertical", imgStack)
#
# cv2.waitKey(0)


## COLOR DETECTION-------------------------------------------------------------------------------------------------------------------------------------------------
# def empty(a):
#     pass
#
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# cv2.namedWindow("Trackbars")
# cv2.resizeWindow("Trackbars",640,240)
# cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty)
# cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty)
# cv2.createTrackbar("Saturation min", "Trackbars", 122, 255, empty)
# cv2.createTrackbar("Saturation max", "Trackbars", 255, 255, empty)
# cv2.createTrackbar("Val min", "Trackbars", 72, 255, empty)
# cv2.createTrackbar("Val max", "Trackbars", 255, 255, empty)
#
# while True:
#     img = cv2.imread("resources/lamborgini2.png")
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue min","Trackbars")
#     h_max = cv2.getTrackbarPos("Hue max", "Trackbars")
#     s_min = cv2.getTrackbarPos("Saturation min", "Trackbars")
#     s_max = cv2.getTrackbarPos("Saturation max", "Trackbars")
#     v_min = cv2.getTrackbarPos("Val min", "Trackbars")
#     v_max = cv2.getTrackbarPos("Val max", "Trackbars")
#
#     print(h_min, h_max, s_min, s_max, v_min, v_max)
#
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#
#     mask = cv2.inRange(imgHSV, lower, upper)
#
#     imgResult = cv2.bitwise_and(img, img, mask = mask)
#
#     imgStacked = stackImages(0.5, ([img,imgHSV],[mask,imgResult]))
#     cv2.imshow("Stacked", imgStacked)
#     cv2.waitKey(5)
#


#COUNTURING AND SHAPE DETECTION-------------------------------------------------------------------------------------------------------------
#FINDS PERIMETER AREA,APPLIES CONTOURS TO IMAGE and FINDS POINTS OF SHAPE
# def getContours(img):
#     contours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         print(area)
#         if area>200:
#             cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#             peri = cv2.arcLength(cnt, True)
#             # print(peri)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#             print(len(approx))
#             objCor = len(approx)
#             x, y, w, h = cv2.boundingRect(approx)
#
#             if objCor == 3 : objectType = "Tri"
#             elif objCor == 4 :
#                 aspratio = w/float(h)
#                 if aspratio>0.95 and aspratio < 1.05: objectType = "Sqr"
#                 else: objectType = "Rec"
#             elif objCor > 7 : objectType = "Cir"
#             else: objectType = "NULL"
#
#             cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(imgContour, objectType,
#                         (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
#                         (0, 0, 0), 2)
#
#
# img = cv2.imread("resources/shapes.png")
# imgContour = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
# imgCanny = cv2.Canny(imgBlur,50,50)
#
# imgBLank = np.zeros_like(img)
# getContours(imgCanny)
#
# imgStacked = stackImages(0.6, ([img,imgGray,imgBlur],
#                                [imgCanny,imgContour,imgBLank]))
# cv2.imshow("Output", imgStacked)
#
# cv2.waitKey(0)


#FACE DETECTION-------------------------------------------------------------------------------------------------------------

# faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
# img = cv2.imread("resources/crowd.png")
#
# imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(imgGrey, 1.1,4)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w, y+h), (0,0,255),2)
#
# cv2.imshow("Output", img)
# cv2.waitKey(0)

#FACE DETECTION WITH WEBCAM
# faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
# cap.set(10,150)
#
# while True:
#     success, img = cap.read()
#     imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(imgGrey, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     cv2.imshow("Video", img)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break;

#CONTOURING THE SELECTED SHAPE---------------------------------------------------------------
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
# cap.set(10,150)
#
# #ADD MORE COLORS POSSIBly. RIGHT TO ARRAY, but for that need for loop in find colors
# myColors = [[16,110,101,22,255,178], [16,110,101,22,255,178]]
# myColorValues = [[66,101,132], [66,101,132]] #bGR
# myPoints = []#[x,y,colorId]
#
#
# def findColor(img, myColors,myColorValues):
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     count = 0
#     newPoints = []
#     for color in myColors:
#         lower = np.array(color[0:3])
#         upper = np.array(color[3:6])
#         mask = cv2.inRange(imgHSV, lower, upper)
#         x,y = getContours(mask)
#         if x != 0 and y!=0:
#             newPoints.append([x,y,count])
#         count += 1
#     return newPoints
#
# def getContours(img):
#     contours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     x, y, w, h = 0,0,0,0
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area>200:
#             cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#             x, y, w, h = cv2.boundingRect(approx)
#     return x+w//2,y
#
# def drawOnCanvas(myPoints,myColorValues):
#
#     for point in myPoints:
#         cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)
#
# while True:
#     success, img = cap.read()
#     imgResult = img.copy()
#     newPoints = findColor(img, myColors,myColorValues)
#
#     if len(newPoints)!=0:
#         for newP in newPoints:
#             myPoints.append(newP)
#         drawOnCanvas(myPoints, myColorValues)
#
#     cv2.imshow("Video", imgResult)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break;

##DOCUMENT READER
# widthImg = 640
# heightImg = 905
# cap = cv2.VideoCapture(0)
# cap.set(3,widthImg)
# cap.set(4,heightImg)
# cap.set(10,150)
#
# def preProcessing(img):
#     imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgBlur = cv2.GaussianBlur(imgGrey, (5,5),1)
#     imgCanny = cv2.Canny(imgBlur, 200,200)
#     kernel = np.ones((5,5))
#     imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
#     imgThres = cv2.erode(imgDial, kernel, iterations=1)
#     return imgThres
#
# def getContours(img):
#     biggest = np.array([])
#     maxArea = 0
#     contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area>5000:
#             #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#             peri = cv2.arcLength(cnt,True)
#             approx = cv2.approxPolyDP(cnt,0.02*peri,True)
#             if area >maxArea and len(approx) == 4:
#                 biggest = approx
#                 maxArea = area
#     cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
#     return biggest
#
#
# def reorder (myPoints):
#     myPointsNew = np.zeros((4, 1, 2), np.int32)
#     if len(myPoints) != 0:
#         myPoints = myPoints.reshape((4,2))
#         myPointsNew = np.zeros((4,1,2),np.int32)
#         add = myPoints.sum(1)
#         #print("add", add)
#         myPointsNew[0] = myPoints[np.argmin(add)]
#         myPointsNew[3] = myPoints[np.argmax(add)]
#         diff = np.diff(myPoints,axis=1)
#         myPointsNew[1]= myPoints[np.argmin(diff)]
#         myPointsNew[2] = myPoints[np.argmax(diff)]
#         #print("NewPoints",myPointsNew)
#     return myPointsNew
#
# def getWarp(img,biggest):
#     biggest = reorder(biggest)
#     print(biggest.shape)
#     pts1 = np.float32(biggest)
#     pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
#
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     imgOutput = cv2.warpPerspective(img, matrix,(widthImg,heightImg))
#     return imgOutput
#
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (widthImg, heightImg))
#     imgContour = img.copy()
#     imgThres = preProcessing(img)
#     biggest = getContours(imgThres)
#     print(biggest)
#     imgWarped = getWarp(img,biggest)
#     if biggest.size !=0:
#         imgCropped = imgWarped[20:imgWarped.shape[0]-20, 20:imgWarped.shape[1]-20]
#         imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
#         imgArray = ([img, imgContour],
#                     [imgThres, imgWarped])
#     else:
#         imgArray = ([img,imgThres],
#                     [img, img])
#
#     stackedImages = stackImages(0.6, imgArray)
#     cv2.imshow("Video", stackedImages)
#     cv2.imshow("asda", imgWarped)
#
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break;



numberPlateCascade = cv2.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")
frameWidth = 640
frameHeight = 480
minArea = 500
color = (255,0,255)
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)
count = 0

while True:
    success, img = cap.read()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = numberPlateCascade.detectMultiScale(imgGrey, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if minArea > 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Number plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color,2)
            imgRoi =  img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Video", img)
    if cv2.waitKey(40) & 0xFF == ord('s'):
        cv2.imwrite("resources/Scanned/NoPlate_" + str(count) + ".jpg", imgRoi)
        cv2.rectangle(img, (0,200), (640, 300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan saved", (150, 265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
