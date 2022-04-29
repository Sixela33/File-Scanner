import cv2
import numpy as np
import utils

WEBCAM = True #Elijo si utilizo la webcam o un archivo
imagen = "Imagenes/Test.jpg" #Archivo a mirar si no e usa webcam
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg  = 480


utils.initializeTrackbars()
count=0

while True:

    if WEBCAM: _ , img = cap.read()
    else:img = cv2.imread(imagen)
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale

    # Modifico la imagen para facilitar encontrar contornos
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = utils.valTrackbars() # Tomo los valores de las barras
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) 
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
 
    # Copio las imagenes  
    imgContours = img.copy()
    imgBigContour = img.copy()

    # Encuentro los contornos y los dibujo
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 

    biggest, maxArea = utils.biggestContour(contours) # Elijo el mas grande
    if biggest.size != 0:
        biggest=utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # Dibujo el contorno mas grande
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)

        # Preparo los puntos para warpear la imagen
        pts1 = np.float32(biggest) # preparo 
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 

        # Warpeo la imagen
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
 
        # Limpio un poco (elimino 20 pixeles de cada lado)
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
 
 
        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold],
                      [imgContours, imgBigContour,imgWarpColored])
 
    else:
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
        imageArray = ([img,imgGray,imgThreshold],
                      [imgContours, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Threshold"],
              ["Contorno", "Mayor Contorno","Final"]]

    stackedImage = utils.stackImages(imageArray,0.5,lables)
    cv2.imshow("Result",stackedImage)

    # Salir cuando se presiona la 'q'
    if cv2.waitKey(1) == ord('q'):
        break

    # Tomar Screenshot cuando se presiona la 's' 
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1