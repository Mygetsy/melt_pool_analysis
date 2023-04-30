import cv2
import numpy as np
import sys
import os.path
import json
from datetime import datetime
from dataImageAndVideo import dataImageAndVideo
from elipseDetection import elipseDetection
from prosprectiveCorrection import prosprectiveCorrection



class videoProcessing(elipseDetection, prosprectiveCorrection):

    def __init__(self, adressVideo):

        dataImageAndVideo.__init__(self, adressVideo=adressVideo)
        sys.stdout.write('Process Start\n')


# Different public processing cases block START
    def extractInitialPictures(self, numberOfFrames, writeOut=True):
        self.imStack = self.extractImagesFromVideo(numberOfFrames=numberOfFrames)
        if writeOut:
            self.writeOutProcessedImages(self.imStack)
        sys.stdout.write('Process End')

    def correctProspectiveForImage(self, numberOfFrames, writeOut=False):
        self.imStack = self.extractImagesFromVideo(numberOfFrames)
        self.correctProspectiveInStack()
        if writeOut:
            self.writeOutProcessedImages(self.correctedProspectImStack)
        sys.stdout.write('Process End')

    def detectEllipseWithPrevFilterParam(self, numberOfFrames = 10,
                      nRays=100,
                      putEllipseOnImage=False,
                      putPointsOnImage=False,
                      putBoxOnImage=False, writeOutImages=False,
                      writeOutFilteredImages=False,
                      writeOutPropertiesUm=True,
                      writeOutParameters=True,
                      writeOutAvgIntensity=True):

        self.imStack = self.extractImagesFromVideo(numberOfFrames=numberOfFrames)
        self.firstImageFlag = True
        self.correctProspectiveInStack()
        self.imStack = self.correctedProspectImStack

        self.takeParametersFromPrevStep()

        self.ellipseDetectionInStack(safeFilteredImage=writeOutFilteredImages,
                                     nRays=nRays)
        self.avgIntensityInStack()

        if writeOutFilteredImages:
            self.imStack = self.imFilteredStack
        if putEllipseOnImage:
            self.putEllipseOnImage()
        if putPointsOnImage:
            self.putEllipsePointsOnImage()
        if putBoxOnImage:
            self.putEllipseBoxOnImage()
        if writeOutImages:
            self.writeOutProcessedImages(self.imStack)
        if writeOutPropertiesUm:
            self.writeOutProcessedVideoParameters()
        if writeOutPropertiesUm:
            self.writeOutVideoPropertiesUm()
        if  writeOutAvgIntensity:
            self.writeOutAvgIntensity()

        sys.stdout.write('Process End')
        
    def detectEllipse(self, numberOfFrames = 10,
                      nRays=100,
                      putEllipseOnImage=False,
                      putPointsOnImage=False,
                      putBoxOnImage=False, writeOutImages=False,
                      writeOutFilteredImages=False,
                      writeOutPropertiesUm=True,
                      writeOutParameters=True,
                      writeOutAvgIntensity=True):

        self.imStack = self.extractImagesFromVideo(numberOfFrames=numberOfFrames)
        self.firstImageFlag = True
        self.correctProspectiveInStack()
        self.imStack = self.correctedProspectImStack
        self.chooseImageFromStack()
        self.centerMeltPool = self.showImageAndGetPointData(self.firstImage)
        self.firstImageFlag = True
        self.getFilterParamForEllipseDetection()
        self.ellipseDetectionInStack(safeFilteredImage=writeOutFilteredImages,
                                     nRays=nRays)
        self.avgIntensityInStack()

        if writeOutFilteredImages:
            self.imStack = self.imFilteredStack
        if putEllipseOnImage:
            self.putEllipseOnImage()
        if putPointsOnImage:
            self.putEllipsePointsOnImage()
        if putBoxOnImage:
            self.putEllipseBoxOnImage()
        if writeOutImages:
            self.writeOutProcessedImages(self.imStack)
        if writeOutPropertiesUm:
            self.writeOutProcessedVideoParameters()
        if writeOutPropertiesUm:
            self.writeOutVideoPropertiesUm()
        if  writeOutAvgIntensity:
            self.writeOutAvgIntensity()

        sys.stdout.write('Process End')
# Different public processing cases block END

# Write Out block START
    def ellipsePxToUm(self):
        self.ellipseStackUm = []
        for count, ellipse in enumerate(self.ellipseStack):
            self.ellipseStackUm.append([])

            self.ellipseStackUm[count].append((
                self.umPixelDistanceCorect * ellipse[0][0],
                self.umPixelDistanceCorect * ellipse[0][1]))

            self.ellipseStackUm[count].append((
                self.umPixelDistanceCorect * ellipse[1][0],
                self.umPixelDistanceCorect * ellipse[1][1]))

            self.ellipseStackUm[count].append(ellipse[2])
            
    def writeOutVideoPropertiesPx(self):
        dt = datetime.now()
        name = self.adressDataOut + self.fileName + '/ellipsePx'
        frameTime = 0
        with open(name, "w") as myfile:
            myfile.write('\n'*2 + str(dt) + '\n')
            myfile.write('Time, ms ; ellipse properties [px]'
                         + ' ((center X, center Y),'
                         + ' (width, height),'
                         + ' angle)'
                         + '\n')
            for i, ellipse in enumerate(self.ellipseStack):
                frameTime += self.timeStep
                myfile.write(str(frameTime) + ' ; ')
                myfile.write(str(ellipse) + '\n')

    def writeOutAvgIntensity(self):
        dt = datetime.now()
        dOut = {'Time stamp' : str(dt),
                'Intensity list':  self.avgInensityStack}
        name = self.adressDataOut + self.fileName + '/avgInensity'
        
        with open(name,'w') as myfile:
            json.dump(dOut, myfile)
        
    def writeOutVideoPropertiesUm(self):
        
        self.ellipsePxToUm()
        dt = datetime.now()
        name = self.adressDataOut + self.fileName + '/ellipseUm'
        frameTime = 0
        with open(name, "w") as myfile:
            myfile.write('\n'*2 + str(dt) + '\n')
            myfile.write('Time, ms ; ellipse properties [µm]'
                         + ' ((center X, center Y),'
                         + ' (width, height),'
                         + ' angle)'
                         + '\n')
            for i, ellipse in enumerate(self.ellipseStackUm):
                frameTime += self.timeStep
                myfile.write(str(frameTime) + ' ; ')
                myfile.write(str(ellipse) + '\n')

    def writeOutProcessedVideoParameters(self):
        parametersDictionary = {}
        parametersDictionary['Tepr'] = 'Video'
        parametersDictionary['File name'] = self.fileName
        parametersDictionary['Time step'] = self.timeStep
        parametersDictionary['Melt pool center'] = self.centerMeltPool
        parametersDictionary['Filter type'] = self.fType
        parametersDictionary['F: Median filter'] = int(self.paramMedian)
        parametersDictionary['F: Block Thr'] = int(self.blockThresh)
        parametersDictionary['F: C Thr'] = int(self.cThr)
        parametersDictionary['F: kernelErDil'] = int(self.kernelErode)
        parametersDictionary['F: it Erode'] = int(self.itEr)
        parametersDictionary['F: it Dil'] = int(self.itDil)
        parametersDictionary['F: Laplacian'] = int(self.kLaplacian)

        try:
            parametersDictionary['px in um '] = float(
                                                self.umPixelDistanceCorect)
        except NameError:
            parametersDictionary['px in um '] = None

        parametersDictionary['Transformation matrix'] = self.transformMatrix.tolist()

        parametersDictionary['timeStamp'] = str(datetime.now())
        # jsonString = json.dumps(parametersDictionary)
        name = self.adressDataOut + self.fileName + '/ProcessedVideoParameters'
        with open(name, "a") as myfile:
            myfile.write('\n')
            json.dump(parametersDictionary, myfile)
            

    def takeParametersFromPrevStep(self):
        name = self.adressDataOut + self.fileName + '/ProcessedVideoParameters'
        if os.path.isfile(name):
            sys.stdout.write('Previous parameters in use\n')
            with open(name, "r") as myfile:
                pDict = json.loads(myfile.readlines()[-1])
            
        self.centerMeltPool = pDict['Melt pool center']
        self.initUpdateFilters(fType=pDict['Filter type'], 
                               paramMedian=pDict['F: Median filter'],
                              blockThresh=pDict['F: Block Thr'],
                              cThr=pDict['F: C Thr'],
                              kernelErode=pDict['F: kernelErDil'],
                              itEr=pDict['F: it Erode'],
                              itDil=pDict['F: it Dil'],
                              kLaplacian=pDict['F: Laplacian'])
            
# Write Out block END

# Prospective correction block START
    def correctProspectiveInStack(self):
        pointMassive, points, im = self.normalizationParametersFromPicture(
            'Callibration_1.jpg')
        self.transposeImage(self.imStack[0], points, pointMassive)

        self.correctedProspectImStack = []
        for img in self.imStack:
            dst = cv2.warpPerspective(img, self.transformMatrix, self.newSize)
            self.correctedProspectImStack.append(dst)

    def correctEllipsePointsInStack(self):
        self.ellipsePointsStackCorrected = []
        for count, ellipse in enumerate(self.ellipsePointsStack):
            self.ellipsePointsStackCorrected.append([])
            for point in ellipse:
                correctedVec = self.transformMatrix @ np.array([point[0],
                                                                point[1], 1, ])
                newPoint = (int(correctedVec[0]), int(correctedVec[1]))
                self.ellipsePointsStackCorrected[count].append(newPoint)

    def showCorrectedEllipseInStack(self):
        for count, img in enumerate(self.correctedProspectImStack):
            ellipse = cv2.fitEllipse(np.asarray(
                self.ellipsePointsStackCorrected[count]))
            cv2.ellipse(img, ellipse, 150, 2)
            self.showImage(img)
# Prospective correction block END

# Ellipse block START
    def ellipseDetectionInStack(self, safeFilteredImage=False, nRays=200):
        self.ellipseStack = []
        self.ellipsePointsStack = []
        if safeFilteredImage:
            self.imFilteredStack = []

        for img in self.imStack:
            fIm = self.applyFilter(img)

            ellipse, ellipsePoints = self.ellipseApproxFilterdImage(
                                    fIm,
                                    manualCenterMeltPool=self.centerMeltPool,
                                    nRays=nRays)
            if safeFilteredImage:
                self.imFilteredStack.append(fIm)
            self.ellipseStack.append(ellipse)
            self.ellipsePointsStack.append(ellipsePoints)

    def getFilterParamForEllipseDetection(self):
        try:
            self.firstImageIndex
        except NameError:
            self.chooseImageFromStack()

        if self.firstImageFlag:
            # self.sharpFilter(im=self.firstImage)
            self.filterChoice(im=self.firstImage)
            self.firstImageFlag = 0

    def putEllipseOnImage(self):
        for count, image in enumerate(self.imStack):
            if len(self.imStack[1].shape):
                self.imStack[count] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.ellipse(self.imStack[count], self.ellipseStack[count],
                        (0, 255, 0), 2)

    def putEllipsePointsOnImage(self, manualCenter=True):
        for count, image in enumerate(self.imStack):
            if len(self.imStack[1].shape):
                self.imStack[count] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.imStack[count] = cv2.circle(self.imStack[count],
                                             self.centerMeltPool,
                                             4, (250, 0, 0), -1)
            for i, point in enumerate(self.ellipsePointsStack[count]):
                self.imStack[count] = cv2.circle(self.imStack[count],
                                                 point,
                                                 8, (0, 255, 0), -1)

    def putEllipseBoxOnImage(self):
        for count, image in enumerate(self.imStack):
            if len(self.imStack[1].shape):
                self.imStack[count] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            x1, y1 = (self.ellipseStack[count][0][0]
                      + self.ellipseStack[count][1][0]//2,
                      self.ellipseStack[count][0][1]
                      + self.ellipseStack[count][1][1]//2)

            x2, y2 = (self.ellipseStack[count][0][0]
                      - self.ellipseStack[count][1][0]//2,
                      self.ellipseStack[count][0][1]
                      - self.ellipseStack[count][1][1]//2)

            cv2.rectangle(self.imStack[count],
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 3)

        
# Ellipse block END

# Intesity in the area detection START


    def avgIntensityInStack(self):      
        try:
           if not len(self.ellipseStack):
               sys.stderr('Ellipse points length zero!')
               sys.exit()
        except NameError:
            sys.stderr('No ellipse points defined!')
        
        self.avgInensityStack = []
        for im, ellipse in zip(self.imStack, self.ellipseStack):
            try:
                value = self.avgIntensity(im, ellipse)
            except TypeError:
                value = -1
            self.avgInensityStack.append(value)
        
        
    def avgIntensity(self, im, ellipse):

        numOfSegments = 20
        intEllipse = ((int(ellipse[0][0]),int(ellipse[0][1])),
                     (int(ellipse[1][0]/2),int(ellipse[1][1]/2)),
                     int(ellipse[2]))

        pts = cv2.ellipse2Poly(*intEllipse, arcStart=0, arcEnd=359, 
                                delta=360//numOfSegments)

        mask = cv2.fillPoly(np.zeros(im.shape, dtype="uint8"),
                            np.int32([pts]), 1)
        imForAnalysis = cv2.bitwise_and(im, im, mask=mask)

        nonZero = float(np.count_nonzero(imForAnalysis))
        
        avgIntensityInArea = np.sum(imForAnalysis) / nonZero
        
        return  avgIntensityInArea
           
# Intesity in the area detection END

    def chooseImageFromStack(self, imageNumber=False):
        if imageNumber:
            self.firstImage = self.imStack[int(imageNumber)]
            self.firstImageIndex = imageNumber
        else:
            def nothing(x):
                pass
            cv2.imshow('firstImage', self.imStack[0])
            cv2.createTrackbar('ChooseImage', 'firstImage',
                               0, len(self.imStack), nothing)
            while(True):
                k = cv2.waitKey(1)
                if k == 27:
                    break
                i = cv2.getTrackbarPos('ChooseImage', 'firstImage')
                cv2.imshow('firstImage', self.imStack[i])

            self.firstImage = self.imStack[i]
            self.firstImageIndex = i
