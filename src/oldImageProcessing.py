import os
import cv2
import numpy as np

class imageProcessing:
    def __init__(self):
        
        self.updateData()
        
        #Statrt Block to normalize and transfrom image 
        #--------
        pointMassive, points, im = self.normalizationParametersFromPicture('Callibration_1.jpg')
        self.transposeImage(im, points, pointMassive)
        im = self.readImage(3)
        # self.showImage(im)
        # dst = cv2.warpPerspective(im,self.transformMatrix,(512*2,320*2))
        # print(self.transformMatrix)
        # self.showImage(dst)
        #--------
        #End Block to normalize and transfrom image 
        
        
         
        #Statrt Block to filter an image for next ellipse detection
        #------
        im = self.readImage(11)
        cv2.circle(im, (225,195 ),2,250,-1)  
        self.showImage(im)
        img = self.sharpFilter(im, paramGauss = False, paramMedian = 21, adaptThr = True, blockThresh =87, cThr =3)
        self.showImage(img)
        
        cv2.imwrite(self.adressImagesOutProcessed + 'test.jpg', img)
        
        #------------------ 
        #End Block to filter an image for next ellipse detection
         
         #Statrt Block to filter an meltp ellipse detection
        #-------       
        ellipse, ellipsePoints = self.ellipseApproxFilterdImage(img)
        cv2.ellipse(im, ellipse, 150, 1, lineType = cv2.LINE_8)
        
        #-------
        # End Block to filter an meltp ellipse detection
        # self.showImage(img1)
        self.showImage(im)

        cv2.destroyAllWindows()
        
    def ellipseApproxFilterdImage(self, im, nRays = 20):
        
        xSz = im.shape[1]
        ySz = im.shape[0]
        
        #Temp maunal center add more center from click
        #--
        # manualCenterMeltPool = (340,250)
        manualCenterMeltPool = (225,195)
        
        #--
        
        minR = min(xSz - manualCenterMeltPool[0],
                   ySz - manualCenterMeltPool[1],
                   manualCenterMeltPool[0],
                   manualCenterMeltPool[1]) - 1
        
        
        thetaRayArr = np.linspace(0, 2*np.pi, nRays)
        coordX = manualCenterMeltPool[0] + minR * np.cos(thetaRayArr)
        coordY = manualCenterMeltPool[1] + minR * np.sin(thetaRayArr)
        
        ellipsePoints = []
        for X, Y in zip(coordX, coordY):
            line = np.transpose(np.array(draw.line(manualCenterMeltPool[0], manualCenterMeltPool[1],
                                                   int(X), int(Y))))
            data = im[line[:,1], line[:,0]]

            test = np.asarray(data[:-1], dtype=np.float64) - np.asarray(data[1:], dtype=np.float64)
            param = np.argwhere(test>100)
            
            if param.size > 0:
                point = (int(param[0][0]))
                ellipsePoints.append((line[point,0], line[point,1]))
        
            
        ellipse = cv2.fitEllipse(np.asarray(ellipsePoints))
        # cv2.ellipse(im, ellipse, (0, 255, 0), 2)
        
        return ellipse, ellipsePoints
        
        
    def cornersAfterTransformation(self, im):
        xSz = im.shape[1]
        ySz = im.shape[0]
        
        M = self.transformMatrix
        
        P00 = [0,0]
        P01 = [xSz,0]
        P10 = [0, ySz]
        P11 = [xSz, ySz]
        
        xTransform = lambda x,y: (M[0,0]*x + M[0,1]*y + M[0,2]) // (M[2,0]*y + M[2,1]*x + M[2,2])
        yTransform = lambda x,y: (M[1,0]*x + M[1,1]*y + M[1,2]) // (M[2,0]*y + M[2,1]*x + M[2,2])
        
        xPoints = []
        yPoints = []
        
        xPoints.append(xTransform(P00[0],P00[1]))
        xPoints.append(xTransform(P01[0],P01[1]))
        xPoints.append(xTransform(P10[0],P10[1]))
        xPoints.append(xTransform(P11[0],P11[1]))

        yPoints.append(yTransform(P00[0],P00[1]))
        yPoints.append(yTransform(P01[0],P01[1]))
        yPoints.append(yTransform(P10[0],P10[1]))
        yPoints.append(yTransform(P11[0],P11[1]))
            
        print(min(xPoints), max(xPoints), min(yPoints), max(yPoints))
        
        return min(xPoints), max(xPoints), min(yPoints), max(yPoints)
    
    def transposeImage(self, im, points, pointMassive, pixelsShiftSize = 80):
        
        newPointMassive = []
        lengthOfSubPoint = len(points)
        
        #Find center of Mass of points and move it to the center
        for PointSet in points:
            cMX = 0
            cMY = 0
            
            for eachpoint in PointSet:
                cMX += pointMassive[eachpoint][0]/lengthOfSubPoint
                cMY += pointMassive[eachpoint][1]/lengthOfSubPoint
                
            xSz = im.shape[1]
            ySz = im.shape[0]
            
            difX = xSz/2 - cMX
            difY = ySz/2 - cMY
            
            newPointSet = []
            
            for eachpoint in PointSet:
                pX = pointMassive[eachpoint][0] + difX 
                pY = pointMassive[eachpoint][1] + difY
                newPointSet.append([pX, pY])
                
            newPointMassive.append(newPointSet)

        outputPointSet = []
        for i in range(lengthOfSubPoint):
            outputPointSet.append([0,0])
        
        for eachpoint in newPointMassive:
            for count, value in enumerate(eachpoint):
                outputPointSet[count][0] += value[0]/lengthOfSubPoint
                outputPointSet[count][1] += value[1]/lengthOfSubPoint
        
        for eachpoint in outputPointSet:
            im = cv2.circle(im,  (int(eachpoint[0]), int(eachpoint[1])), 3, (0, 255, 0), -1)
        
        
        #Start prospective transformation
        pts1 = np.float32(outputPointSet)
        pts2 = np.float32([
                           [xSz//2 + pixelsShiftSize, ySz//2 + pixelsShiftSize],[xSz//2 - pixelsShiftSize, ySz//2 + pixelsShiftSize],
                           [xSz//2 + pixelsShiftSize, ySz//2 - pixelsShiftSize],[xSz//2 - pixelsShiftSize, ySz//2 - pixelsShiftSize]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        self.transformMatrix = M
        
        print(im.shape[0])
        #show prospective picture on callibration image
        # dst = cv2.warpPerspective(im,M,(512,320))
        # self.showImage(dst)
        
        
    
    
    def normalizationParametersFromPicture(self, name):
        # Get coordinate of points from callibartion image
        adressCollibartionImage = '../callibration/'
        im = cv2.imread(adressCollibartionImage + name, cv2.IMREAD_COLOR)
        
         # set blue and green channels to 0 and treshold
        r = im.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0

        ret, thresh = cv2.threshold(r, 150, 255, cv2.THRESH_BINARY)
        img1 = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # create a masiive of points
        pointMassive = []
        for c in contours:
            cX = []
            cY = []
            for c1 in c:
                cX.append(c1[0][0])
                cY.append(c1[0][1])
            pointMassive.append([int(np.mean(cX)), int(np.mean(cY))])
            
        # Check if known image and grab points
        if name ==   'Callibration_1.jpg':
            points = [[0,1,2,4], [1,3,4,7], [4,7,8,12], [2,4,6,8]]
            points = [[1,5,8,15], [0,3,6,12], [1,5,8,15], [0,3,6,12]]

        elif name == 'Callibration_2.jpg':
            points = [[0,1,2,4], [1,3,4,7], [4,7,8,10], [3,6,7,9]]
        
        elif name == 'Callibration_3.jpg':
            points = [[0,2,3,5], [1,3,4,6], [3,5,6,8], [5,7,8,10]]
            #! 22.10.17 Bad picture points
        else:
            cv2.drawContours(im, contours, -1, (0,255,0), 1)
    
            i = 0
            for c in contours:
                cv2.putText(im, str(i), (pointMassive[i][0],pointMassive[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                i += 1
                
            self.showImage(im)
            print('Whict points to input? Choose 4 sets like [p1,p2,p3,p4]')
            s1 = input('Set 1')
            s2 = input('Set 2')
            s3 = input('Set 3')
            s4 = input('Set 4')
            
            points = [s1,s2,s3,s4]
            
        return pointMassive, points, im
        
    
    
    def sharpFilter(self, im, paramGauss = 0, paramMedian = 0, adaptThr = True, blockThresh = 0, cThr = 1):
        if not im.any():
            print("No image to filter")
            
        if paramGauss:
            gausImg = cv2.GaussianBlur(im, (paramGauss, paramGauss), 0)
            difImage =  im - gausImg 
            
            if paramMedian:
                difImage = cv2.medianBlur(difImage, paramMedian)
                

            return difImage
        
        if (blockThresh) is not False:
            
            img = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    cv2.THRESH_BINARY_INV,blockThresh,cThr)
            img = cv2.medianBlur(img, paramMedian)
            return img
        
        else:
            def nothing(x):
                pass
            
            cv2.imshow('sharpFilter',im)
            
            cv2.createTrackbar('Gaus\n filter\n parameter','sharpFilter',0,50,nothing)
            paramArr = np.arange(1,1001,2)
            
            cv2.createTrackbar('Meidan\n filter\n parameter','sharpFilter',0,100,nothing)
            
            cv2.createTrackbar('Block size thr','sharpFilter',1,100,nothing)
            cv2.createTrackbar('C thr','sharpFilter',0,20,nothing)
            
            switch = '0 : OFF \n1 : ON'
            cv2.createTrackbar(switch, 'sharpFilter',0,1,nothing)
            
            while(True):
            
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                
                i = cv2.getTrackbarPos('Gaus\n filter\n parameter','sharpFilter')
                j = cv2.getTrackbarPos('Meidan\n filter\n parameter','sharpFilter')
                s = cv2.getTrackbarPos(switch,'sharpFilter')
                
                paramGauss = paramArr[i]
                paramMedian = paramArr[j]
                
                if s == 0:
                    img = im
                    
                else:
                    gausImg = cv2.GaussianBlur(im, (paramGauss, paramGauss), 0)
                    img =  im - gausImg 
                img =  cv2.medianBlur(img, paramMedian)
                    
                    
                if adaptThr:
                    blockThreshindex = cv2.getTrackbarPos('Block size thr','sharpFilter')
                    cThr = cv2.getTrackbarPos('C thr','sharpFilter')
                    
                    if blockThreshindex == 0:
                        blockThreshindex = 1
                    blockThresh = paramArr[blockThreshindex]
                    # print(blockThresh)
                    
                    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    cv2.THRESH_BINARY_INV,blockThresh,cThr)

                    
                cv2.imshow('sharpFilter',img)
                
            print('Gauss filter parameter = ', paramGauss)
            print('Median filter parameter = ', paramMedian)
            if adaptThr:
                print('Block Thresh = ', blockThresh)
                print('cThr =', cThr)
            
            cv2.destroyAllWindows()
    
            return img
        
    def updateData(self):
        self.adressVideoIn = '../video/'
        self.arrVideoIn = os.listdir(self.adressVideoIn)
        self.adressImagesOut = '../images/'
        self.adressImagesOutProcessed = '../images/Processed/'
        self.adressImagesIn = '../images/'
        self.arrImagesIn = os.listdir(self.adressImagesIn) 
        
    
    def readImage(self, frameNumber):
        if not self.arrImagesIn :
            print("No image")
            return 1
        im = cv2.imread(self.adressImagesIn + self.arrImagesIn[int(frameNumber)], cv2.IMREAD_GRAYSCALE)
        print("Image: ", self.adressImagesIn + self.arrImagesIn[int(frameNumber)])
        return im
    
    def getImageProperties(self, im):
        imShape = im.shape
        return imShape

    def showImage(self, im, imName = 'image'):
        cv2.imshow(imName,im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def normalizeImageInt(self, im):
        c = (255*(im - np.min(im))/np.ptp(im)).astype('int')
        return np.uint8(c)
