#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 09:51:29 2022

@author: macbook
"""
import os
import sys
import re
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime



class dataProcessing:
    '''
    height, width according to picture -> height in y derection equal to 
    melt pool width at crossection
    '''

    def __init__(self, adressVideo='.', processWholeDataFolder=False):
        
        if processWholeDataFolder:
            pass
            self.processWholeDataFolderFunc(plotFigs=False)
        else:
            self.checkDataExistanse(adressVideo)
            time, otherData, timeStamp, header = self.grabDataFromEllipseFile()
  
            # avIntensity = self.grabDataFromAvIntensityFile()
            # self.plotWidthHeightOverTime(time, otherData)
            # self.plotMeltPoolCenterOverTime(time, otherData)
            # self.calcMeltPoolWidthHeight_mean(time, otherData,0.2, 0.6)
            # self.calcAvgInetensity(time, avIntensity)

        
    def processWholeDataFolderFunc(self, plotFigs=True):
        '''
        create full_process folder
        create if not created whole process 
        putTimeStamp
        loop for every folder in Data excep the full process
        plot and save there graphs with title of folder
        append to the file the information:
        name; meanH, meanW; ... ;StdW
        '''
        self._updateData()
        valueDict = {}
        
        pas =  os.path.abspath(self.adressDataOut)
        folderPas = os.path.join(pas, 'full_process')
        
        try:
            os.makedirs(folderPas)
        except OSError:
                pass
        
        # directories = [x[1] for x in os.walk(pas)]
        directories = os.listdir(pas)
        print(directories)
        directories.remove('full_process')
        directories.remove('.DS_Store')
        
        for each in directories:
            
            file = os.path.join(pas, each,'ellipseUm')
            avIntensityFile = os.path.join(pas, each, 'avgInensity')
            self.filePass = file
            self.folderPass = pas
            self.avIntensityFilePass = avIntensityFile
            
            time, otherData, _, _ = self.grabDataFromEllipseFile()
            avIntensity = self.grabDataFromAvIntensityFile()
            valueDict[each] = {}
            valueDict[each]['MeltPoolSize'] = self.calcMeltPoolWidthHeight_mean(time,
                                                                otherData,
                                                                t1=0.2,t2=0.6)
            valueDict[each]['Av_intensity'] = self.calcAvgInetensity(time,
                                                         avIntensity,
                                                         t1=0.2,t2=0.6)
            if plotFigs:
                self.plotWidthHeightOverTime(time, otherData, True).savefig(
                                    folderPas + '/' + str(each).replace('.','_') +
                                    '_wh_plot', format='png')
            
        with open(folderPas+'/values', 'w') as f:
            json.dump(valueDict, f)
      
        
    def plottingAvIntesityForSteel(self):
        pas =  os.path.abspath(self.adressDataOut)
        filePas = os.path.join(pas, 'full_process', 'values')
        
        with open(filePas, 'r') as f:
            dictAll = json.load(f)
        
        '''
        В каждом имени файла найти DEDS* -> 1,2,3,4
        В каждом имени файла найти мощность _P**_ -> 110,200,250
        СобратьDict {1{power:[parameres]},2,3,4}
        
        Построить для одного материал тайпа график
        '''
        dictOut = {1:{}, 2:{}, 3:{}, 4:{}}
        for name in dictAll:
            materialType = int(re.findall('DEDS([0-4])', name)[0])
            power = int(re.findall('_P([0-9]+)_', name)[0])
            if power == 1110:
                power = 110
            
            dictOut[materialType][power] = dictAll[name]
            
        plt.figure()
        
        for matType in dictOut:
            powerList = []
            avIntMean = []
            avIntStd = []
            heightMed = []
            heightStd = []
            
            for power in dictOut[matType]:
                mean = dictOut[matType][power]['Av_intensity'][0]
                std =  dictOut[matType][power]['Av_intensity'][1]
                
                powerList.append(power)
                avIntMean.append(mean)
                avIntStd.append(std)
                
                hMed = dictOut[matType][power]['MeltPoolSize'][1][0]
                hStd =  dictOut[matType][power]['MeltPoolSize'][2][0]
                
                heightMed.append(hMed)
                heightStd.append(hStd)
                
            # plt.errorbar(powerList, avIntMean, yerr=avIntStd,
            #              label=str(matType), fmt="o")
            plt.errorbar(powerList, heightMed, yerr=heightStd,
                         label=str(matType), fmt="o")
        
        plt.legend()
        # plt.xlim(right=430)
        plt.ylim(0,500)
        plt.show()
                
    def calcAvgInetensity(self, time=False, avIntensity=False, 
                         t1=False, t2=False):
        if not (time and avIntensity):
            time, otherData, timeStamp, header = self.grabDataFromEllipseFile()
            avIntensity = self.grabDataFromAvIntensityFile()
        
        def find_nearest_idx(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
            
        if t1 is False: t1=time[0]
        if t2 is False: t2=time[-1]
        
        avIntensity = np.array(avIntensity)
        
        t1Idx = find_nearest_idx(time, t1)
        t2Idx = find_nearest_idx(time, t2)
        forCals = avIntensity[t1Idx:t2Idx]
        forCals = forCals[(forCals > 0)]
        meanAvgInt = np.mean(forCals)
        stdAvgInt = np.std(forCals)

        
        return [meanAvgInt, stdAvgInt]
        
    def calcMeltPoolWidthHeight_mean(self, time=False, otherData=False,
                                     t1=False, t2=False):
        if not (time and otherData):
            sys.exit('No time or other data')
        
        def find_nearest_idx(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
            
        # width = np.array([item[2] for item in otherData])
        # height = np.array([item[3] for item in otherData])
        width, height = self.widthHeightEllipseToBox(otherData)
    
        if t1==False: t1=time[0]
        if t2==False: t2=time[-1]
        
        t1Idx = find_nearest_idx(time, t1)
        t2Idx = find_nearest_idx(time, t2)
        
        params = []
        
        meanW = width[t1Idx:t2Idx].mean()
        meanH = height[t1Idx:t2Idx].mean()
        params.append((meanW, meanH))
        
        medianW = np.median(width[t1Idx:t2Idx])
        medianH = np.median(height[t1Idx:t2Idx])
        params.append((medianW, medianH))
        
        stdW = np.std(width[t1Idx:t2Idx])
        stdH = np.std(height[t1Idx:t2Idx])
        params.append((stdW, stdH))
        
        return params
        
    def plotWidthHeightOverTime(self, time=False, otherData=False,
                                turnOffIOplot=False):
        if not (time and otherData):
            time, otherData, _, _ = self.grabDataFromEllipseFile()
            # sys.exit('No time or other data')

        self.plotProperties()
        # width = [item[2] for item in otherData]
        # height = [item[3] for item in otherData]
        width, height = self.widthHeightEllipseToBox(otherData)
        
        
        if turnOffIOplot:
            plt.ioff()
        
        fig = plt.figure(figsize=(self.xsc, self.ysc))
        plt.plot(time, width, label='Width')
        plt.plot(time, height, label='Height')

        plt.xlabel('Time, µm', fontsize=self.fontSize)
        plt.ylabel('Width, µm', fontsize=self.fontSize)
        plt.ylim(0, 700)

        plt.xticks(fontsize=self.fontSize2)
        plt.yticks(fontsize=self.fontSize2)
        plt.legend(fontsize=self.fontSize2)
        return fig

    def plotMeltPoolCenterOverTime(self, time=False, otherData=False,
                                   turnOffIOplot=False):
        if not (time and otherData):
            time, otherData, _, _ = self.grabDataFromEllipseFile()
            # sys.exit('No time or other data')

        self.plotProperties()
        x = [item[0] for item in otherData]
        y = [item[1] for item in otherData]
        
        if turnOffIOplot:
            plt.ioff()

        plt.figure(figsize=(self.xsc, self.ysc))
        plt.plot(time, x, label='X coord')
        plt.plot(time, y, label='Y coord')

        plt.xlabel('Time, µm', fontsize=self.fontSize)
        plt.ylabel('Coordinate, µm', fontsize=self.fontSize)
        plt.ylim(0, max(np.median((x)), np.median((y)))+100)

        plt.xticks(fontsize=self.fontSize2)
        plt.yticks(fontsize=self.fontSize2)
        plt.legend(fontsize=self.fontSize2)

    def plotProperties(self):
        self.xsc = 8
        self.ysc = 8
        self.fontSize = 20
        self.fontSize2 = 18
        self.dpi = 150
        self.format = '.png'

    def checkDataExistanse(self, adressVideo):
        self._updateData()
        videoName = adressVideo
        pas = os.path.join(self.adressDataOut, videoName)
        file = os.path.join(pas, 'ellipseUm')
        avIntensityFile = os.path.join(pas, 'avgInensity')
        if not os.path.isdir(pas):
            sys.exit('No folder found: ' + pas)
        if not os.path.isfile(file):
            sys.exit('No file found')
        self.filePass = file
        self.avIntensityFilePass = avIntensityFile
        self.folderPass = pas
        
    
    def grabDataFromAvIntensityFile(self):
        if not self.avIntensityFilePass:
            sys.exit('Err: No intensity file or file pass found')
        
        with open(self.avIntensityFilePass, "r") as myfile:
            valDcit = json.loads(myfile.readline())
            
        return valDcit['Intensity list']
    
    def grabDataFromEllipseFile(self):
        if not self.filePass:
            sys.exit('Err: No file or file pass found')

        with open(self.filePass, "r") as f:
            lines = f.readlines()

        for line in lines:
            try:
                lines.remove('\n')
            except ValueError:
                pass
            finally:
                pass

        timeStamp = lines[0]
        header = lines[1]
        lines = lines[2:]

        time = []
        otherData = []
        for line in lines:
            timeTmp = float(line.split(';')[0])
            dataTmp = re.sub(r"[\([{})\]\n ]",
                             "", line.split(';')[1]).split(',')
            dataTmp = [float(i) for i in dataTmp]

            time.append(timeTmp)
            otherData.append(dataTmp)
        return time, otherData, timeStamp, header

    def widthHeightEllipseToBox(self, otherData):
        width = np.array([item[2] for item in otherData])
        height = np.array([item[3] for item in otherData])
        angle = np.array([item[4] for item in otherData])* np.pi/180
        angleCos = np.cos(angle)
        angleSin = np.sin(angle)
        
        widthBox = np.sqrt((width*angleCos)**2 + (height*angleSin)**2)
        heightBox = np.sqrt((width*angleSin)**2 + (height*angleCos)**2)
        
        return widthBox, heightBox
        
    def _updateData(self):
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

if __name__ == "__main__":
    # adressVideo='SS316L_DEDS4_91_d500_P110_V25_42000fp_C001H001S0001.mp4'
    # data =  dataProcessing(adressVideo=adressVideo)
    # data = dataProcessing(processWholeDataFolder = True)
    
    
    defAdress = '../video/'
    folder_name = 'SS316L_ALL/S2/'
    video_name = 'SS316L_DEDS2_104_d500_P1110_V25_42000fp_C001H001S0001_C001H001S0001.mp4'
    adressVideo = defAdress + folder_name + video_name +'/'+video_name
    
    testDataVideoProcessing = dataProcessing(video_name)
    testDataVideoProcessing.plotWidthHeightOverTime()
    # print(testDataVideoProcessing.plotMeltPoolCenterOverTime())
    print(testDataVideoProcessing.calcAvgInetensity(t1=0.2,t2=0.6))
    
    # data.plottingAvIntesityForSteel()
    
    