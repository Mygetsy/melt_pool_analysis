import cv2
import numpy as np
from dataImageAndVideo import dataImageAndVideo


class prosprectiveCorrection(dataImageAndVideo):
    ''' Class for correcting prospective of input image or Video according
    some callibration image. Callibration image should have red dots for collibrating.'''

    def __init__(self, adressVideo=False, adressImage=False,
                 initialIm=False, callibartionImageName=False):
        '''Initalize class with adress of Video or Image,
        some initial image on which correction will be done '''

        if initialIm:
            self.initialIm = initialIm
        else:
            super().__init__(adressVideo, adressImage)
        
        
        if callibartionImageName is False:
            callibartionImageName = 'Callibration_2.jpg'
            
        pointMassive, points, im =\
            self.normalizationParametersFromPicture(callibartionImageName)
        self.transposeImage(self.initialIm, points, pointMassive)

        # TODO: put full image in square

        dst = cv2.warpPerspective(self.initialIm, self.transformMatrix,
                                  self.newSize)
        self.showImage(dst)
        # self.transformMatrix
        # xMin, xMax, yMin, yMax =\
        #   self.cornersAfterTransformation(self.initialIm)

        # self.corners(self.initialIm, self.transformMatrix)
        # inputCor

    def correctTransformMatrix(self, im, M, pts1, pts2):
        ''' Correct image to put in proper rectangular'''
        xSz = im.shape[1]
        ySz = im.shape[0]

        src = np.array([[0, 0], [xSz, 0], [0, ySz], [xSz, ySz]],
                       dtype=np.float32)

        ocroners = cv2.perspectiveTransform(src[None, :, :], M)
        br = cv2.boundingRect(ocroners)
        
        for point in pts2:
            point[0] -= br[0]
            point[1] -= br[1]

        M = cv2.getPerspectiveTransform(pts1, pts2)
        self.newSize = (br[2], br[3])
        self.transformMatrix = M

    def transposeImage(self, im, points, pointMassive, pixelsShiftSize=250):
        '''
        Transpose the input image in proper way based on input image and callibration points from the callibratuion image.

        Parameters
        ----------
        im : np array
            input image
        points : list of tuples
            All point coordinate from normalizationParametersFromPicture
        pointMassive : list of int in list
            Indexes of selected points for transpose
        pixelsShiftSize : int
            How much pixels between clibration points on transposed image.
            The default is 250.

        Returns self.transformMatrix
        -------
        None.

        '''
        newPointMassive = []
        lengthOfSubPoint = len(points)

        # Find center of Mass of points and move it to the center
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
            outputPointSet.append([0, 0])

        for eachpoint in newPointMassive:
            for count, value in enumerate(eachpoint):
                outputPointSet[count][0] += value[0]/lengthOfSubPoint
                outputPointSet[count][1] += value[1]/lengthOfSubPoint

        for eachpoint in outputPointSet:
            im = cv2.circle(im, (int(eachpoint[0]), int(eachpoint[1])),
                            3, (0, 255, 0), -1)

        # Start prospective transformation
        pts1 = np.float32(outputPointSet)
        pts2 = np.float32([
                           [pixelsShiftSize, pixelsShiftSize],
                           [0, pixelsShiftSize],
                           [pixelsShiftSize, 0],
                           [0, 0]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        self.transformMatrix = M
        # Change transform matrix to put it in sepcial defined square
        self.correctTransformMatrix(im, self.transformMatrix, pts1, pts2)
        # Correction pxtoUm
        self.umPixelDistanceCorect = self.umPixelDistance / pixelsShiftSize

    def normalizationParametersFromPicture(self, name, umPixelDistance=250):
        '''Get coordinate of points from callibartion image with known distance
         and data how µm in px point'''
        adressCollibartionImage = '../callibration/'
        im = cv2.imread(adressCollibartionImage + name, cv2.IMREAD_COLOR)

        # set blue and green channels to 0 and treshold
        r = im.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0

        ret, thresh = cv2.threshold(r, 150, 255, cv2.THRESH_BINARY)
        img1 = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

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
        if name == 'Callibration_1.jpg':
            points = [[0, 1, 2, 4], [1, 3, 4, 7], [4, 7, 8, 12],
                      [2, 4, 6, 8]]
            points = [[1, 5, 8, 15], [0, 3, 6, 12], [1, 5, 8, 15],
                      [0, 3, 6, 12]]

        elif name == 'Callibration_2.jpg':
            points = [[0, 1, 2, 4], [1, 3, 4, 7], [4, 7, 8, 10],
                      [3, 6, 7, 9]]

        elif name == 'Callibration_3.jpg':
            points = [[0, 2, 3, 5], [1, 3, 4, 6], [3, 5, 6, 8], [5, 7, 8, 10]]
        # ! 22.10.17 Bad picture points
        else:
            cv2.drawContours(im, contours, -1, (0, 255, 0), 1)

            i = 0
            for c in contours:
                cv2.putText(im, str(i), (pointMassive[i][0],
                            pointMassive[i][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                i += 1
                
            self.showImage(im)
            print('Whict points to input? Choose 4 sets like [p1,p2,p3,p4]')
            s1 = input('Set 1')
            s2 = input('Set 2')
            s3 = input('Set 3')
            s4 = input('Set 4')

            points = [s1, s2, s3, s4]
            
        self.umPixelDistance = umPixelDistance
        return pointMassive, points, im
    
