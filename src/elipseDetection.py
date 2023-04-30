import cv2
import random
import sys
import numpy as np
from skimage import draw
from dataImageAndVideo import dataImageAndVideo


class elipseDetection(dataImageAndVideo):
    """Melt pool boundary detection class with approximation of it by
    ellipse."""

    def __init__(self, adressVideo=False, adressImage=False,
                 initialIm=False):
        if initialIm:
            self.initialIm = initialIm
        else:
            super().__init__(adressVideo, adressImage)

        self.sharpFilter(self.initialIm, paramGauss=False, paramMedian=21,
                         adaptThr=True, blockThresh=87, cThr=3)
        self.showFilterImage()
        ellipse, ellipsePoints = self.ellipseApproxFilterdImage(
                                                self.filteredImg)

    def filterChoice(self, im):
        '''
        Should help to choose a proper filter for the image.

        Parameters
        ----------
        im : np array
            Imput image

        Modify
        -------
        returns Image as self.filteredImg 
        and also modify parameter members through 
        self.initUpdateFilters(..)

        Returns
        -------
        None.
        '''

        if not im.any():
            print("No image to filter")

        def nothing(x):
            pass

        def createNewWindow(value):
            # Mb will not destroy all windows
            cv2.destroyWindow('filterChoice')
            cv2.namedWindow('filterChoice')
            cv2.createTrackbar('Filter type', 'filterChoice',
                               value, 2, nothing)
        img = np.copy(im)
        createNewWindow(0)
        cv2.imshow('filterChoice', img)
        i = 0
        
        while(True):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            i = cv2.getTrackbarPos('Filter type', 'filterChoice')
            if i == 0:
                img = im
                createNewWindow(i)

                while(i == 0):
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        cv2.destroyAllWindows()
                        return self.initUpdateFilters(fType=i)

                    i = cv2.getTrackbarPos('Filter type', 'filterChoice')
                    cv2.imshow('filterChoice', img)
            if i == 1:
                createNewWindow(i)
                cv2.createTrackbar('Meidan\n filter\n parameter',
                                   'filterChoice',
                                   0, 100, nothing)
                cv2.createTrackbar('Block size thr',
                                   'filterChoice',
                                   1, 100, nothing)
                cv2.createTrackbar('C thr',
                                   'filterChoice',
                                   0, 20, nothing)
                cv2.createTrackbar('Erode',
                                   'filterChoice',
                                   0, 7, nothing)
                cv2.createTrackbar('Dilate',
                                   'filterChoice',
                                   0, 7, nothing)
                paramArr = np.arange(1, 1001, 2)

                while(i == 1):
                    img = im
                    k = cv2.waitKey(1) & 0xFF
                    i = cv2.getTrackbarPos('Filter type', 'filterChoice')
                    j = cv2.getTrackbarPos('Meidan\n filter\n parameter',
                                           'filterChoice')
                    paramMedian = paramArr[j]
                    blockThreshindex = cv2.getTrackbarPos('Block size thr',
                                                          'filterChoice')
                    cThr = cv2.getTrackbarPos('C thr',
                                              'filterChoice')

                    blockThresh = paramArr[blockThreshindex]
                    kernelErode = 5
                    itEr = cv2.getTrackbarPos('Erode',
                                              'filterChoice')
                    itDil = cv2.getTrackbarPos('Dilate',
                                               'filterChoice')

                    img = self.adaptiveThrFilter(img,
                                                 paramMedian=paramMedian,
                                                 blockThresh=blockThresh,
                                                 cThr=cThr,
                                                 kernelErode=kernelErode,
                                                 itEr=itEr, itDil=itDil)

                    cv2.imshow('filterChoice', img)
                    if k == 27:
                        cv2.destroyWindow('filterChoice')
                        return self.initUpdateFilters(fType=i,
                                                      paramMedian=paramMedian,
                                                      blockThresh=blockThresh,
                                                      cThr=cThr,
                                                      kernelErode=kernelErode,
                                                      itEr=itEr,
                                                      itDil=itDil)
            if i == 2:
                createNewWindow(i)
                cv2.createTrackbar('Laplacian filter',
                                   'filterChoice',
                                   0, 20, nothing)
                cv2.createTrackbar('Erode',
                                   'filterChoice',
                                   0, 7, nothing)
                cv2.createTrackbar('Dilate',
                                   'filterChoice',
                                   0, 7, nothing)
                paramArr = np.arange(1, 1001, 2)

                while(i == 2):
                    img = im
                    k = cv2.waitKey(1) & 0xFF
                    i = cv2.getTrackbarPos('Filter type', 'filterChoice')
                    k = cv2.getTrackbarPos('Laplacian filter',
                                           'filterChoice')
                    kLaplacian = paramArr[k]
                    kernelErode = 3
                    itEr = cv2.getTrackbarPos('Erode',
                                              'filterChoice')
                    itDil = cv2.getTrackbarPos('Dilate',
                                               'filterChoice')
                    img = self.laplacianThrFilter(img,
                                                  kLaplacian=kLaplacian,
                                                  kernelErode=kernelErode,
                                                  itEr=itEr,
                                                  itDil=itDil)

                    cv2.imshow('filterChoice', img)
                    if k == 27:
                        cv2.destroyWindow('filterChoice')
                        return self.initUpdateFilters(fType=i,
                                                      kLaplacian=kLaplacian,
                                                      kernelErode=kernelErode,
                                                      itEr=itEr,
                                                      itDil=itDil)

    def adaptiveThrFilter(self, im,
                          paramMedian=0,
                          blockThresh=False,
                          cThr=1, kernelErode=5,
                          itEr=5, itDil=2):

        if blockThresh == 1:
            blockThresh = 3
        img = cv2.adaptiveThreshold(im, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    blockThresh, cThr)
        img = cv2.medianBlur(img, paramMedian)
        kernel = np.ones((int(kernelErode), int(kernelErode)), np.uint8)
        img = cv2.erode(img, kernel, iterations=itEr)
        img = cv2.dilate(img, kernel, iterations=itDil)
        return img

    def laplacianThrFilter(self, im, kLaplacian=3, kernelErode=5,
                           itEr=5, itDil=2, paramGauss=5):

        img = cv2.GaussianBlur(im, (paramGauss, paramGauss), 0)
        img = cv2.Laplacian(img, cv2.CV_64F, ksize=kLaplacian)
        kernel = np.ones((int(kernelErode), int(kernelErode)), np.uint8)
        img = cv2.erode(img, kernel, iterations=itEr)
        img = cv2.dilate(img, kernel, iterations=itDil)
        return img

    def applyFilter(self, im):
        if self.fType == 0:
            sys.exit('No filter has been choosen!')

            return im
        elif self.fType == 1:
            return self.adaptiveThrFilter(im,
                                          paramMedian=self.paramMedian,
                                          blockThresh=self.blockThresh,
                                          cThr=self.cThr,
                                          kernelErode=self.kernelErode,
                                          itEr=self.itEr,
                                          itDil=self.itDil)
        elif self.fType == 2:
            return self.laplacianThrFilter(im,
                                           kLaplacian=self.kLaplacian,
                                           kernelErode=self.kernelErode,
                                           itEr=self.itEr,
                                           itDil=self.itDil)

    def initUpdateFilters(self, fType=0, paramMedian=False,
                          blockThresh=False, cThr=False,
                          kernelErode=False, itEr=False, itDil=False,
                          kLaplacian=False):
        self.fType = fType
        self.paramMedian = paramMedian
        self.blockThresh = blockThresh
        self.cThr = cThr
        self.kernelErode = kernelErode
        self.itEr = itEr
        self.itDil = itDil
        self.kLaplacian = kLaplacian

    def sharpFilter(self, im, paramGauss=False, paramMedian=0,
                    adaptThr=True, blockThresh=False, cThr=1):
        """
        Filter image to get counter
        if input parameters are given -> filter according the
        parameters
        if input parameters are NOT given -> provide a window with track bar
        to adjust the parameters

        Parameters
        ----------
        im : np array
            Imput image
        paramGauss : int or bool
            Kernel for Gauss filter. The default is False.
        paramMedian : int
            Kernele for median filter. The default is 0.
        adaptThr : bool
            Bool for turn on/off cv2.adaptiveThreshold. The default is True.
        blockThresh : int
            block for cv2.adaptiveThreshold. The default is False.
        cThr : int > 1
            C for cv2.adaptiveThreshold. The default is 1.

        Returns
        -------
            returns Image as  and also modify paramete members
            self.filteredImg
        """

        if not im.any():
            print("No image to filter")

        if int(paramGauss):
            gausImg = cv2.GaussianBlur(im, (paramGauss, paramGauss), 0)
            difImage = im - gausImg

            if paramMedian:
                difImage = cv2.medianBlur(difImage, paramMedian)

            self.filteredImg = difImage

        if (blockThresh) is not False:
            img = cv2.adaptiveThreshold(im, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        blockThresh, cThr)
            img = cv2.medianBlur(img, paramMedian)
            kernel = np.ones((5, 5), np.uint8)
            img = cv2.erode(img, kernel, iterations=5)
            img = cv2.dilate(img, kernel, iterations=2)
            self.filteredImg = img

        else:
            def nothing(x):
                pass

            # cv2.imshow('sharpFilter', im)
            cv2.namedWindow('sharpFilter')

            cv2.createTrackbar('Gaus\n filter\n parameter', 'sharpFilter',
                               0, 50, nothing)

            paramArr = np.arange(1, 1001, 2)

            cv2.createTrackbar('Meidan\n filter\n parameter', 'sharpFilter',
                               0, 100, nothing)

            cv2.createTrackbar('Block size thr', 'sharpFilter',
                               1, 100, nothing)

            cv2.createTrackbar('C thr', 'sharpFilter',
                               0, 20, nothing)

            switch = '0 : OFF \n1 : ON'
            cv2.createTrackbar(switch, 'sharpFilter',
                               0, 1, nothing)

            while(True):

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

                i = cv2.getTrackbarPos('Gaus\n filter\n parameter',
                                       'sharpFilter')
                j = cv2.getTrackbarPos('Meidan\n filter\n parameter',
                                       'sharpFilter')
                s = cv2.getTrackbarPos(switch,
                                       'sharpFilter')

                paramGauss = paramArr[i]
                paramMedian = paramArr[j]

                if s == 0:
                    img = im

                else:
                    gausImg = cv2.GaussianBlur(im, (paramGauss, paramGauss), 0)
                    img = im - gausImg
                img = cv2.medianBlur(img, paramMedian)

                if adaptThr:
                    blockThreshindex = cv2.getTrackbarPos('Block size thr',
                                                          'sharpFilter')
                    cThr = cv2.getTrackbarPos('C thr',
                                              'sharpFilter')

                    if blockThreshindex == 0:
                        blockThreshindex = 1
                    blockThresh = paramArr[blockThreshindex]

                    img = cv2.adaptiveThreshold(img, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV,
                                                blockThresh, cThr)

                kernel = np.ones((5, 5), np.uint8)
                img = cv2.erode(img, kernel, iterations=5)
                img = cv2.dilate(img, kernel, iterations=2)
                cv2.imshow('sharpFilter', img)

            print('Gauss filter parameter = ', paramGauss)
            print('Median filter parameter = ', paramMedian)

            if adaptThr:
                print('Block Thresh = ', blockThresh)
                print('cThr =', cThr)

            cv2.destroyAllWindows()

            self.filteredImg = img

            if paramGauss == 1:
                self.paramGauss = False
            else:
                self.paramGauss = 1

            self.paramMedian = paramMedian
            self.adaptThr = adaptThr
            self.blockThresh = blockThresh
            self.cThr = cThr

    def ellipseApproxFilterdImage(self, im, nRays=50,
                                  manualCenterMeltPool=(225, 195)):
        """
        Ellipse approximation of melt pool shape on the filered binary picture.
        Steps:
            1. Manual approximation of centers of melt pool.
            2. Rays starts from the defined centers.
            3. The first point on outer surface defines by the first
                positive gradient.
            4. Points further than median distance filtered.

        Parameters
        ----------
        im : np array
            Imput image
        nRays : int
            Number of output Rays. The default is 50.
        manualCenterMeltPool : tuple, (int, int)
            Center of melt pool. The default is (225, 195).

        Returns
        -------
        ellipse : (x,y),(MA/2, ma/2),angle
            from fitellipse()
        ellipsePoints : list of tuples, [(),()]
            points defined
        """
        xSz = im.shape[1]
        ySz = im.shape[0]

        minR = min(xSz - manualCenterMeltPool[0],
                   ySz - manualCenterMeltPool[1],
                   manualCenterMeltPool[0],
                   manualCenterMeltPool[1]) - 1

        thetaRayArr = np.linspace(0, 2*np.pi, nRays)
        coordX = manualCenterMeltPool[0] + minR * np.cos(thetaRayArr)
        coordY = manualCenterMeltPool[1] + minR * np.sin(thetaRayArr)

        ellipsePoints = []
        # Line gradient alg part ---
        for X, Y in zip(coordX, coordY):
            line = np.transpose(np.array(draw.line(
                                         manualCenterMeltPool[0],
                                         manualCenterMeltPool[1],
                                         int(X), int(Y))))

            data = im[line[:, 1], line[:, 0]]
            test = (np.asarray(data[:-1], dtype=np.float64)
                    - np.asarray(data[1:], dtype=np.float64))

            param = np.argwhere(test > 100)

            if param.size > 0:
                point = (int(param[0][0]))
                ellipsePoints.append((line[point, 0], line[point, 1]))
            if param.size > 1:
                ellipsePoints.append((line[point, 0], line[point, 1]))

        # Line gradient alg part ---
        if len(ellipsePoints) < 5:
            ellipse = [(1, 1), (1, 1), 0]
            return ellipse, ellipsePoints

        # Filter points START
        filterPoints = np.asarray(ellipsePoints)
        filterCenter = np.asarray(manualCenterMeltPool)
        residualFromCenter = filterPoints - filterCenter
        residualDistFromCenter = np.sqrt(np.sum(
                                    np.power(residualFromCenter, 2), axis=1))

        resMed = residualDistFromCenter / np.median(residualDistFromCenter)
        indexesToRemove = []
        for i, residual in enumerate(resMed):
            if residual > 1.5 or residual < 0.7:
                indexesToRemove.append(i)

        for i in reversed(indexesToRemove):
            ellipsePoints.remove(ellipsePoints[i])
        # Filter points END
        if len(ellipsePoints) < 5:
            ellipse = [(1, 1), (1, 1), 0]
        else:
            ellipse = cv2.fitEllipse(np.asarray(ellipsePoints))
        return ellipse, ellipsePoints

    def randomPointsEllipseDetection(self, points):
        '''Test bad function'''
        centerX = []
        centerY = []
        height = []
        width = []
        angle = []

        for point in points:
            try:
                randomPointsList = random.choices(points, k=5)
                ellipse = cv2.fitEllipse(np.asarray(randomPointsList))
                centerX.append(ellipse[0][0])
                centerY.append(ellipse[0][1])
                height.append(ellipse[1][0])
                width.append(ellipse[1][1])
                angle.append(ellipse[2])
            finally:
                continue

        centerX = np.median(np.asarray(centerX))
        centerY = np.median(np.asarray(centerY))
        height = np.median(np.asarray(height))
        width = np.median(np.asarray(width))
        angle = np.median(np.asarray(angle))
        return ((centerX, centerY), (height, width), angle)

    def showFilterImage(self):
        ''' Shows self.filterImg'''
        cv2.imshow('Filtered image', self.filteredImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fit_ellipse(self, points):
        """
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0
        to the provided arrays of data
        points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
        """
        x = np.zeros((len(points)))
        y = np.zeros((len(points)))
        for i, point in enumerate(points):
            x[i], y[i] = point[0], point[1]

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel()

