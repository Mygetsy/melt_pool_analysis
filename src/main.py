from videoProcessing import videoProcessing

# dataImageAndVideo.DataImageAndVideo
if __name__ == "__main__":

    defAdress = '/Users/macbook/meltPoolVideoProcess/meltPoolProcessingProject/video/'
    folder_name = 'SS316L_ALL/S2/'
    video_name = 'SS316L_DEDS2_104_d500_P1110_V25_42000fp_C001H001S0001_C001H001S0001'
    adressVideo = defAdress + folder_name + video_name +'/'+video_name+'.mp4'
    print('-'*6)
    # print(adressVideo)
    testViseo_peocessing = videoProcessing(adressVideo) 
    # testViseo_peocessing.extractInitialPictures(150, writeOut=True)
    # testViseo_peocessing.takeParametersFromPrevStep()
    # testViseo_peocessing.detectEllipse(numberOfFrames = 150,
    #                       nRays=200,
    #                       putEllipseOnImage=True,
    #                       putPointsOnImage=False,
    #                       putBoxOnImage=False, writeOutImages=True,
    #                       writeOutFilteredImages=False,
    #                       writeOutPropertiesUm=True,
    #                       writeOutParameters=True)              
    
    testViseo_peocessing.detectEllipse(numberOfFrames = 10,
                          nRays=200,
                          putEllipseOnImage=True,
                          putPointsOnImage=False,
                          putBoxOnImage=False, writeOutImages=False,
                          writeOutFilteredImages=False,
                          writeOutPropertiesUm=True,
                          writeOutParameters=False)     