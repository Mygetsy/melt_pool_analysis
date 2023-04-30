#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:18:38 2022

@author: macbook
"""
import cv2
import os
import numpy as np
from dataImageAndVideo import dataImageAndVideo

class substractingImages(dataImageAndVideo): 
    def __init__(self, adressVideo=False, adressImage=False):
        pass
        self.imAdress = '/Users/macbook/meltPoolVideoProcess/meltPoolProcessingProject/images/imSubst/'
        images = os.listdir(self.imAdress)
        im1 = self.readImage(self.imAdress + 'frame29.png')
        im2 = self.readImage(self.imAdress + 'frame30.png')
        im3 = self.readImage(self.imAdress + 'frame31.png')
        
        self.__initAll()
        
        shiftPx = self.calculateShiftPxConstSpeed()
        self.subStractTwoImage(im1, im2, im3, shiftPx)
        
        
    def __initAll(self):
        self.__VscanX = 25e3 #um/s
        self.__VscanY = 0 #um/s
        self.__pxToum = 1 
        
      
    def subStractTwoImage(self, im1, im2, im3, shiftPx):
         
        shiftedIm2 = cv2.warpAffine(im2, shiftPx, (im2.shape[1], im2.shape[0]))
        shiftedIm3 = cv2.warpAffine(im3, shiftPx, (im2.shape[1], im2.shape[0]))
        shiftedIm3 = cv2.warpAffine(shiftedIm3, shiftPx, (im2.shape[1], im2.shape[0]))
        subtracted = cv2.subtract(shiftedIm2, im1)
        subtracted2 = cv2.subtract(shiftedIm3, im1)
        
        dst = cv2.addWeighted(subtracted,0.5, subtracted2,-0.5,0)
        _,thresh = cv2.threshold(dst, 1,255,cv2.THRESH_BINARY)
        # self.showImage(shiftedIm2)
        # self.showImage(shiftedIm2)
        cv2.imwrite(self.imAdress+'im2Sh.png', shiftedIm2)
        cv2.imwrite(self.imAdress+'imSubst.png', thresh)
        
    def calculateShiftPxConstSpeed(self, dt = 0):
        if dt == 0:
            dt = 0.007166
            # dt = 0.00364
        M = np.float32([
            [1, 0, 43],
            [0, 1, 1*self.__pxToum * self.__VscanY * dt]])
        return M
    
    def testPropertiesOnTestImages(self):
        self.checkAll

if __name__ == "__main__":
    testSubst = substractingImages()