'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
import random
from operator import itemgetter
from scipy.spatial import distance as dist #for computing distance of vectors
import scipy.io as sio
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.text
#from sklearn.metrics.pairwise import cosine_similarity

import hickle as hkl


from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from faces_settings import *

batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_faces_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_faces_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output representation layers)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'all_R'
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
repLayers = test_prednet(inputs)
test_model = Model(input=inputs, output=repLayers)

#Get Representation layers vector for the given testing set
X_test = hkl.load(test_file)
X_hat = test_model.predict(X_test, batch_size)


#Create distribution Within Object Transformations
        #to ensure the same random numbers are computed on the next run
numObjs = X_hat.shape[0]    #number of different objects
numTransf = X_hat.shape[1]  #number of transformations/time-steps per object
numSamplesToDraw=500

#testing cosine similarity , use a random normal distribution
cosSimTest = np.zeros((numSamplesToDraw,1))
for i in range(numSamplesToDraw):
    cosSimTest[i] = 1-dist.cosine(np.random.normal(size=X_hat.shape[2]), np.random.normal(size=X_hat.shape[2]))
    
#Cos Sim Within Same Objs
distWithinObjs = np.zeros((numSamplesToDraw,1))
sampleObjIDs_1 = np.random.choice(200,size=(numSamplesToDraw,1),replace=True)
#ignore the rep layer for nt=0 as this will all be the same for all objects as the network
#lags behind one step since it has not seen anything yet
sampleTransIDs_1 = np.random.choice(9,size=(numSamplesToDraw,1),replace=True)
sampleTransIDs_2 = np.random.choice(9,size=(numSamplesToDraw,1),replace=True)
sampleTransIDs_1 = sampleTransIDs_1+1
sampleTransIDs_2 = sampleTransIDs_2+1
for i in range(numSamplesToDraw):
    distWithinObjs[i] = 1-dist.cosine(X_hat[sampleObjIDs_1[i],sampleTransIDs_1[i],:],X_hat[sampleObjIDs_1[i],sampleTransIDs_2[i],:])

'''  
for i in range((numSamplesToDraw,1)):
    objIdx=random.randint(0,numObjs-1)
    transIdx_1= random.randint(1,numTransf-1)
    transIdx_2 = random.randint(1,numTransf-1)
    distWithinObjs[i] = 1-dist.cosine(X_hat[objIdx, transIdx_1,:],X_hat[objIdx, transIdx_2, :] )
'''
    
#Cosine Similarity Across Objects
distBetweenObjs = np.zeros((numSamplesToDraw,1))
sampleObjIDs_1 = np.random.choice(200,size=(numSamplesToDraw,1),replace=True)
sampleObjIDs_2 = np.random.choice(200,size=(numSamplesToDraw,1),replace=True)
#ignore the rep layer for nt=0 as this will all be the same for all objects as the network
#lags behind one step since it has not seen anything yet
sampleTransIDs_1 = np.random.choice(9,size=(numSamplesToDraw,1),replace=True)
sampleTransIDs_2 = np.random.choice(9,size=(numSamplesToDraw,1),replace=True)
sampleTransIDs_1 = sampleTransIDs_1+1
sampleTransIDs_2 = sampleTransIDs_2+1
for i in range(numSamplesToDraw):
    distBetweenObjs[i] = 1-(dist.cosine(X_hat[sampleObjIDs_1[i],sampleTransIDs_1[i],:],X_hat[sampleObjIDs_2[i],sampleTransIDs_2[i],:]))



#Create histogram
'''
numSamplesHist = 1000
rankHistogramArray = np.zeros(numObjs)
#tmpRandSamples = np.zeros((numObjs, X_hat.shape[2]))
samplesObjDistArray = np.zeros((numObjs,2))
for k in range (numSamplesHist):
    #object to compare to
    trueObjIdx = random.randint(0,numObjs-1)
    trueTransIdx = random.randint(0,numTransf-1)
    
    #select a random transformation/sample for each object and compute cosine distance
    samplesObjDistArray[:,0]=range(numObjs)#initialize first column to have the objID in order
    samplesObjDistArray[:,1]= 0.0 #reset the distance value
    for m in range(numObjs):
        samplesObjDistArray[m,1] = dist.cosine(X_hat[trueObjIdx, trueTransIdx,:],X_hat[m,random.randint(0,numTransf-1),:])
        
    #sort array by distance
    samplesObjDistArray=samplesObjDistArray.tolist()#make a list 
    samplesObjDistArray.sort(key=itemgetter(1))#sort by the first column =distance
    
    #find trueObject position within the array
    samplesObjDistArray = np.array(samplesObjDistArray) #change back to np array
    trueRankIdx = np.where(samplesObjDistArray[:,0]==trueObjIdx)[0][0] #find distance position of true object
    
    #increment proper rankIndx for histogram
    rankHistogramArray[trueRankIdx] = rankHistogramArray[trueRankIdx]+1  #increment true rank distance position by 1
'''    

#ptest permute the labels and compute mean differences
numPermTests=200
shuffledDistancesArray = np.concatenate((distWithinObjs,distBetweenObjs),axis=0)
middlePosition = shuffledDistancesArray.shape[0]/2
permMeanDiffArray = np.zeros((numPermTests,1))
for i in range(numPermTests):
    np.random.shuffle(shuffledDistancesArray)
    rndMeanWithin=np.mean(shuffledDistancesArray[0:middlePosition])
    rndMeanBetween = np.mean(shuffledDistancesArray[middlePosition:])
    permMeanDiffArray[i] = rndMeanBetween-rndMeanWithin
        
# Plot distributions and histogram
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'distribution_plots/')
fig = plt.figure() #3 plots

#plot cos sim test 
ax=fig.add_subplot(5,1,1)
ax.set_title("Cos Sim for Norm Distribution")
ax.hist(cosSimTest,facecolor='purple')

#plot cos sim within same objs
ax = fig.add_subplot(5,1,2)
sampledMean_within = np.mean(distWithinObjs)
ax.set_title('Sampled Cosine Similarity Within Objects, (mean = %s)'%(sampledMean_within))
ax.hist(distWithinObjs,facecolor='green')
#ax.plot(distWithinObjs)
#plt.show()

ax=fig.add_subplot(5,1,3)
sampledMean_between=np.mean(distBetweenObjs)
ax.set_title('Sampled Cosine Similarity Between Objects, (mean = %s)'%(sampledMean_between))
ax.hist(distBetweenObjs,facecolor ='blue')
#ax.plot(distBetweenObjs)
#plt.show()

#plots interleave
ax=fig.add_subplot(5,1,4)
sampledMean_difference=sampledMean_between-sampledMean_within
ax.set_title('Sampled Cosine Similarity Within and Between Objects, (mean = %s)'%(sampledMean_difference))
ax.hist(distWithinObjs,facecolor ='green')
ax.hist(distBetweenObjs,facecolor ='blue', alpha=0.5)
#ax.legend(['y=distWithin', 'y=distBetween'], loc='upper left')
plt.tight_layout()

#plot permuted mean distribution
ax=fig.add_subplot(5,1,5)
permMeanDiff = np.mean(permMeanDiffArray)
ax.set_title('Permuted Mean Difference Distribution, (permMeanDiff = %s)' %(permMeanDiff))
ax.set_ylabel("Mean Frequency")
ax.hist(permMeanDiffArray,facecolor ='purple')
#sampledMeanPosition = np.where(permMeanDiffArray[:,0]==sampledMeanDiff)[0][0]
#ax.axvline(x=sampledMeanPosition, color='red')
#ax.legend(['y=perm mean diff'], loc='upper left')


#ax=fig.add_subplot(3,1,3)
#ax.set_title("Rank Histogram (1k samples)")
#ax.set_xlabel("Bins")
#ax.set_ylabel("Rank Frequency")
#ax.hist(rankHistogramArray, bins=200)


fig.subplots_adjust(hspace = 1)
plt.tight_layout()
plt.savefig(plot_save_dir +  'plot_distributions_3layers.png')
#sio.savemat(plot_save_dir+'permMeanDiffArray_4layers.mat',{'permMeanDiffArray':permMeanDiffArray})
#sio.savemat(plot_save_dir+'cosDist_withinObjs_4layers.mat', {'distWithinObjs':distWithinObjs})
#sio.savemat(plot_save_dir+'cosDist_acrossObjs_4layers.mat',{'distBetweenObjs':distBetweenObjs})
#sio.savemat(plot_save_dir+'rankHistogram.mat_4layers',{'rankHistogramArray2':rankHistogramArray})

#gs = gridspec.GridSpec(2)
#gs.update(wspace=0., hspace=0.)
#plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'distribution_plots/')
#if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
#plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
#for i in plot_idx:
#   for t in range(nt):
#        plt.subplot(gs[t])
#        plt.imshow(X_test[i,t,:,:,0], interpolation='none') #imshow input(2D) or (3D with the last dimension being 3 or greater) if 1 channel od imshow(2D)
#        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#        if t==0: plt.ylabel('Actual', fontsize=10)

#        plt.subplot(gs[t + nt])
#        plt.imshow(X_hat[i,t,:,:,0], interpolation='none')
#        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#        if t==0: plt.ylabel('Predicted', fontsize=10)

#    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#    plt.clf()



