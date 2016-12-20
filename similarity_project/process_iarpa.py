'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
from natsort import natsorted
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
import matplotlib
from readline import get_endidx
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iarpa_settings import *

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
desired_im_sz = (72, 128) #half of original image

# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    nt=20 #number of transformations per image
    numTransf = 100
    step = 5 # choose obj every 10 degrees of movement
    
    numTrainObjs =4000 #due to memory constraints(2k objs, 10 nt)
    numValObjs = 200
    numSVMObjs =700

    masterSet = '/n/coxfs01/wscheirer/stimuli/Master_Set/' #4129
    set3 = '/n/coxfs01/wscheirer/stimuli/Set3/' #887

    print 'start of process...'
    #for training 4k objs
    startID=0
    endID=numTrainObjs #4000
    dataDir,objDirs,_ = os.walk(masterSet).next()
    objDirs=natsorted(objDirs) #natural sort of objList
    
    
    X_data = np.zeros((numTrainObjs,) + (nt,) + desired_im_sz + (3,), np.uint8)
    for objID in range(startID,endID): #0-4000
        _,_,transFiles = os.walk(os.path.join(dataDir,objDirs[objID],'0')).next()
        transFiles = natsorted(transFiles) #natural sort, to keep the transformation order for training
        for transID in range(0, numTransf, step): #starts at 0, up to not including nt
            image=imread(os.path.join(dataDir,objDirs[objID],'0',transFiles[transID]))
            X_data[objID-startID, (transID/step)] = process_im(image, desired_im_sz)
        print objDirs[objID]
    X_data = np.transpose(X_data,(0,1,4,2,3)) #changing the position of numChannels
    X_data = (X_data.astype(np.float32))/255 #normalize the image
    hkl.dump(X_data, os.path.join(DATA_DIR,'X_train_4k.hkl'))  
    print 'X_train.hkl created...'

    
    #Generate Val Data (129 remaining)
    startID=numTrainObjs #4k
    endID=len(objDirs) #4129
    
    #X_data has 129  objects
    X_data = np.zeros((numValObjs,) + (nt,) + desired_im_sz + (3,), np.uint8)
    for objID in range(startID,endID): 
        _,_,transFiles = os.walk(os.path.join(dataDir,objDirs[objID],'0')).next()
        transFiles = natsorted(transFiles) #natural sort, to keep the transformation order for training
        for transID in range(0, numTransf,step): #starts at 0, up to not including 100 <= len(transFiles)
            image=imread(os.path.join(dataDir,objDirs[objID],'0',transFiles[transID]))
            X_data[objID-startID, (transID/step)] = process_im(image, desired_im_sz)
        print objDirs[objID]
    
    #predNet evaluationDataSet for training
    startID = 0
    endID = numValObjs-129 #71
    dataDir,objDirs,_ = os.walk(set3).next() 
    objDirs=natsorted(objDirs) #natural sort of objList
    i=objDirs.index('testObj52')
    del objDirs[i]
    i=objDirs.index('testObj89')
    del objDirs[i]
    i=objDirs.index('testObj97')
    del objDirs[i]
    i=objDirs.index('testObj117')
    del objDirs[i]
    i=objDirs.index('testObj138')
    del objDirs[i]
    i=objDirs.index('testObj137')
    del objDirs[i]
    i=objDirs.index('testObj180')
    del objDirs[i]
    i=objDirs.index('testObj183')
    del objDirs[i]  
    i=objDirs.index('testObj279')
    del objDirs[i]
    i=objDirs.index('testObj307')
    del objDirs[i]
    i=objDirs.index('testObj344')
    del objDirs[i]         
    i=objDirs.index('testObj361')
    del objDirs[i]
    i=objDirs.index('testObj371')
    del objDirs[i]    
    i=objDirs.index('testObj397')
    del objDirs[i]
    i=objDirs.index('testObj412')
    del objDirs[i]
    i=objDirs.index('testObj424')
    del objDirs[i]
    i=objDirs.index('testObj429')
    del objDirs[i]
    i=objDirs.index('testObj435')
    del objDirs[i]
    i=objDirs.index('testObj443')
    del objDirs[i]
    i=objDirs.index('testObj444')
    del objDirs[i]
    i=objDirs.index('testObj445')
    del objDirs[i]
    i=objDirs.index('testObj491')
    del objDirs[i]
    i=objDirs.index('testObj512')
    del objDirs[i]
    i=objDirs.index('testObj516')
    del objDirs[i] 
    i=objDirs.index('testObj520')
    del objDirs[i] 
    i=objDirs.index('testObj521')
    del objDirs[i] 
    i=objDirs.index('testObj527')
    del objDirs[i] 
    i=objDirs.index('testObj553')
    del objDirs[i]   
    i=objDirs.index('testObj570')
    del objDirs[i] 
    i=objDirs.index('testObj575')
    del objDirs[i] 
    i=objDirs.index('testObj600')
    del objDirs[i] 
    i=objDirs.index('testObj603')
    del objDirs[i] 
    i=objDirs.index('testObj607')
    del objDirs[i] 
    i=objDirs.index('testObj612')
    del objDirs[i] 
    i=objDirs.index('testObj619')
    del objDirs[i] 
    i=objDirs.index('testObj672')
    del objDirs[i]
    i=objDirs.index('testObj685')
    del objDirs[i] 
    i=objDirs.index('testObj695')
    del objDirs[i]    
    i=objDirs.index('testObj714')
    del objDirs[i] 
    i=objDirs.index('testObj715')
    del objDirs[i] 
    i=objDirs.index('testObj744')
    del objDirs[i] 
    i=objDirs.index('testObj752')
    del objDirs[i] 
    i=objDirs.index('testObj763')
    del objDirs[i] 
    i=objDirs.index('testObj773')
    del objDirs[i] 
    i=objDirs.index('testObj774')
    del objDirs[i] 
    i=objDirs.index('testObj788')
    del objDirs[i] 
    i=objDirs.index('testObj795')
    del objDirs[i] 
    i=objDirs.index('testObj801')
    del objDirs[i]       
    i=objDirs.index('testObj813')
    del objDirs[i] 
    i=objDirs.index('testObj821')
    del objDirs[i] 
    i=objDirs.index('testObj823')
    del objDirs[i] 
    i=objDirs.index('testObj853')
    del objDirs[i] 
    i=objDirs.index('testObj854')
    del objDirs[i] 
    i=objDirs.index('testObj892')
    del objDirs[i] 
    i=objDirs.index('testObj903')
    del objDirs[i] 
    i=objDirs.index('testObj904')
    del objDirs[i] 
    
    
    for objID in range(startID,endID): #71-129
        _,_,transFiles = os.walk(os.path.join(dataDir,objDirs[objID],'0')).next()
        transFiles = natsorted(transFiles) #natural sort, to keep the transformation order for training
        for transID in range(0, numTransf, step): #starts at 0, up to not including nt
            image=imread(os.path.join(dataDir,objDirs[objID],'0',transFiles[transID]))
            X_data[objID+129, (transID/step)] = process_im(image, desired_im_sz)
        print objDirs[objID]
    X_data = np.transpose(X_data,(0,1,4,2,3)) #changing the position of numChannels
    X_data = (X_data.astype(np.float32))/255 #normalize the image
    hkl.dump(X_data, os.path.join(DATA_DIR,'X_val_200.hkl'))  
    print 'X_val.hkl created...'
    
    #SVM dat
    startID = endID #71
    endID = startID + numSVMObjs #871
    X_data = np.zeros((numSVMObjs,) + (nt,) + desired_im_sz + (3,), np.uint8)
    for objID in range(startID, endID): 
        _,_,transFiles = os.walk(os.path.join(dataDir,objDirs[objID],'0')).next()
        transFiles = natsorted(transFiles) #natural sort, to keep the transformation order for training
        for transID in range(0, numTransf, step): #starts at 0, up to not including 100 <= len(transFiles)
            image=imread(os.path.join(dataDir,objDirs[objID],'0',transFiles[transID]))
            X_data[objID-startID, (transID/step)] = process_im(image, desired_im_sz)
        print objDirs[objID]
    X_data = np.transpose(X_data,(0,1,4,2,3)) #changing the position of numChannels
    X_data = (X_data.astype(np.float32))/255 #normalize the image
    hkl.dump(X_data, os.path.join(DATA_DIR,'X_svm_700.hkl'))  
    print 'X_test.hkl created...'
    

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = (im.shape[1] - desired_sz[1]) / 2
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    process_data()
