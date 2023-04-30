import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


class dataImageAndVideo:

    """ Base class for importing of video and image via link
        and updating all data about the input and output file.
    """

    def __init__(self, adressVideo=False, adressImage=False):
        self.updateData()
        if adressImage:
            self.adressImage = adressImage
            self.initialIm = self.readImage(self.adressImage)
            self.getImageProperties(self.initialIm)
        elif adressVideo:
            self.vidcap = cv2.VideoCapture(adressVideo)
            self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fpsTrue = 42000.
            self.lengthTrue = self.frame_count / 42000.
            self.fileName = adressVideo.split('/')[-1]

    def extractImagesFromVideo(self,
                               numberOfFrames=10,
                               eachFrames=False,
                               eachMSeconds=False):
        """
        Function for exctaracting pictures from input video
        Parameters
        ----------
        numberOfFrames : int
             Number of extracted images
        eachFrames : int
            Each Nth frame. The default is False.
        eachMSeconds : float
            each real mSecond of real video. The default is False.

        Returns
        -------
        imageStack : list of images
            list of extracted GRAY images
        """
        numberOfFrames += 1
              
        if numberOfFrames:
            stepFrames = self.frame_count // (numberOfFrames - 1)
            if stepFrames == 0:
                stepFrames = 1
        elif eachFrames:
            stepFrames = eachFrames
        elif eachMSeconds:
            numberOfFrames = 1000 * self.lengthTrue / eachMSeconds
            stepFrames = self.frame_count // (numberOfFrames - 1)

        self.timeStep = self.lengthTrue * stepFrames / self.frame_count

        count = 0
        i = 0
        success, image = self.vidcap.read()
        success = True
        imageStack = []

        tqdm_i = tqdm(total = numberOfFrames-1)

        while success:
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, (count))
            success, image = self.vidcap.read()
            
            if success is False:
                break
            imageStack.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            count = count + stepFrames
            i += 1
            tqdm_i.update()
        tqdm_i.close()
            
        sys.stdout.write('\nOpened ' + str(numberOfFrames) + ' frames\n')   
        return imageStack

    def updateData(self):
        """ Udate links to derictories"""
        self.adressVideoIn = '../video/'
        self.arrVideoIn = os.listdir(self.adressVideoIn)
        self.adressImagesOut = '../images/'
        self.adressImagesOutProcessed = '../images/Processed/'
        self.adressImagesIn = '../images/'
        self.arrImagesIn = os.listdir(self.adressImagesIn)
        self.loging = '../logfiles/'
        self.logingFile = os.listdir(self.loging)
        self.adressDataOut = '../data/'

    def readImage(self, adress):
        """ Read GRAY image"""
        im = cv2.imread(adress, cv2.IMREAD_GRAYSCALE)
        if im.any():
            return im
        else:
            print("NO IMAGE")
            return None

    def getImageProperties(self, im):
        """Read image propeties and make them as public"""
        self.imShape = im.shape

    def showInitialImage(self):
        """Shows object at initial image and wait key be pressed"""
        cv2.imshow('Initial image', self.initialIm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showImage(self, im):
        """Shows any input image and wait key be pressed"""
        cv2.imshow('Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showImageAndGetPointData(self, im, numberOfPoints=1):
        """
        Manually get points location by clicking from a new copy of input image
        and draw them on the copy.

        Parameters
        ----------
        im : np array
            Image
        numberOfPoints : int
            Desired number of points

        Returns
        -------
        points : list(tuple), [(int,int),...]
            List of points

        """
        im = np.copy(im)
        cv2.imshow('Image for click', im)
        points = []

        def draw_circle(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(im, (x, y), 5, 150, -1)
                points.append((x, y))

        while(1):
            cv2.setMouseCallback('Image for click', draw_circle)
            cv2.imshow('Image for click', im)
            k = cv2.waitKey(20) & 0xFF

            if k == 27:
                break
            if len(points) == int(numberOfPoints):
                cv2.imshow('Image for click', im)
                k = cv2.waitKey(20) & 0xFF
                if int(numberOfPoints) == 1:
                    points = points[0]
                break

        cv2.destroyAllWindows()
        return points

    def writeOutProcessedImages(self, im, filetype='.png',
                                frameName='frame'):
        try:
            os.mkdir(self.adressDataOut)
            print('data folder created')
        except FileExistsError:
            pass

        try:
            os.mkdir(self.adressDataOut + self.fileName)
            print('File folder created')
        except FileExistsError:
            pass
        imageAdress = self.adressDataOut + self.fileName + '/'
        for count, eachImage in enumerate(im):
            imageName = frameName + str(count) + filetype
            if eachImage.any() is not None:
                cv2.imwrite(imageAdress + imageName, eachImage)


