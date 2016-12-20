'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os, pdb
import numpy as np
import random
from operator import itemgetter
from scipy.spatial import distance as dist #for computing distance of vectors
import scipy.io as sio
from six.moves import cPickle
import matplotlib
import time
#from matplotlib.backends.wx_compat import fontweights
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
from iarpa_settings import *

batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_iarpa_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_iarpa_model.json')
test_file = os.path.join(DATA_DIR, 'X_svm_700.hkl')
svm_model_file = os.path.join(WEIGHTS_DIR,'svm_model.pkl')


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
numObjs=X_test.shape[0]
numTrans=X_test.shape[1]


'''
#Get Pixel Data
X_test = hkl.load(test_file)
X_hat=np.zeros((X_test.shape[0],X_test.shape[1], (X_test.shape[2]*X_test.shape[3]*X_test.shape[4])))
for obj in range(numObjs):
    for trans in range(numTrans):
        X_hat[obj,trans] = X_test[obj,trans].flatten()
'''



###svm stuff
feature_size = X_hat.shape[2] #size of the representation layer vector
n_train_pairs = 500 #500 pairs same, 500 pairs diff
n_test_pairs = 100 #100 pairs same, 500 pairs diff
objs_for_train=600 #objects used for training
objs_for_test=100 #objects used for testing

#Generate training samples for same objects
X_same = np.zeros((n_train_pairs, feature_size))
for i in range(n_train_pairs):
    idx = np.random.randint(objs_for_train) #0-599
    t1 = np.random.randint(5, numTrans) #start with 5
    t2 = None
    while t2 is None:
        prop_t2 = np.random.randint(5, numTrans) #start with 5
        if prop_t2 != t1:
            t2 = prop_t2
    d = (X_hat[idx, t1] - X_hat[idx, t2])**2
    X_same[i] = d

#Generate training samples for different objects
X_diff = np.zeros((n_train_pairs, feature_size))
for i in range(n_train_pairs):
    idx1 = np.random.randint(objs_for_train) #0-599
    idx2 = None
    while idx2 is None:
        prop_idx2 = np.random.randint(objs_for_train)#0-599
        if prop_idx2 != idx1:
            idx2 = prop_idx2
    t1 = np.random.randint(5, numTrans)#start with 5
    t2 = np.random.randint(5, numTrans) #start with 5
    d = (X_hat[idx1, t1] - X_hat[idx2, t2])**2
    X_diff[i] = d

X_train = np.concatenate((X_same, X_diff), axis=0)
y_train = np.zeros(2*n_train_pairs)
y_train[:n_train_pairs] = 1 #label 1 = same object, 0 for different


#Generate Testing Samples
#Generate training samples for same objects
X_same = np.zeros((n_test_pairs, feature_size))
for i in range(n_test_pairs):
    idx = np.random.randint(objs_for_train, numObjs)#600-699
    t1 = np.random.randint(5, numTrans) #start with 5
    t2 = None
    while t2 is None:
        prop_t2 = np.random.randint(5, numTrans) #start with 5
        if prop_t2 != t1:
            t2 = prop_t2
    d = (X_hat[idx, t1] - X_hat[idx, t2])**2
    X_same[i] = d

#Generate training samples for different objects
X_diff = np.zeros((n_test_pairs, feature_size))
for i in range(n_test_pairs):
    idx1 = np.random.randint(objs_for_train,numObjs)
    idx2 = None
    while idx2 is None:
        prop_idx2 = np.random.randint(objs_for_train,numObjs)
        if prop_idx2 != idx1:
            idx2 = prop_idx2
    t1 = np.random.randint(5, numTrans)#start with 5
    t2 = np.random.randint(5, numTrans) #start with 5
    d = (X_hat[idx1, t1] - X_hat[idx2, t2])**2
    X_diff[i] = d
    
X_test = np.concatenate((X_same, X_diff), axis=0)
y_test = np.zeros(2*n_test_pairs)
y_test[:n_test_pairs] = 1


#Training
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
#clf = LinearSVC()
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print 'mean confidence score %s',score
joblib.dump(clf,svm_model_file)
print 'model saved'

#from sklearn.svm import LinearSVC, SVC
#from sklearn.externals import joblib

#clf = joblib.load(svm_model_file)
#print 'model loaded'
#startTime = time.time()
decision_scores = clf.decision_function(X_test) #get socres
probs = clf.predict_proba(X_test) #get probabilities
#endTime = time.time()
#elapsedTime = endTime-startTime
#print 'elapsed time: %s',elapsedTime
#print 'numPairs: %s',X_test.shape[0]


#Save Scores
probs_file = os.path.join(RESULTS_SAVE_DIR, 'X_PROB_2.hkl')  
scores_file = os.path.join(RESULTS_SAVE_DIR,'X_SCORES_2.hkl')
hkl.dump(probs, probs_file)
hkl.dump(decision_scores,scores_file)


'''
#Read scores and prob from file
probs_file = os.path.join(RESULTS_SAVE_DIR, 'X_PROB.hkl')  
scores_file = os.path.join(RESULTS_SAVE_DIR,'X_SCORES.hkl')
probs = hkl.load(probs_file)
decision_scores = hkl.load(scores_file)
'''

'''
numPermTests= 10000
permMeanDiffArray= np.zeros((numPermTests,1))
shuffledDistancesArray=np.zeros((200,1))
shuffledDistancesArray = hkl.load(scores_file)
for i in range(numPermTests):
    np.random.shuffle(shuffledDistancesArray)
    rndMeanWithin=np.mean(shuffledDistancesArray[0:100])
    rndMeanBetween = np.mean(shuffledDistancesArray[100:])
    permMeanDiffArray[i] = rndMeanBetween-rndMeanWithin
    
'''

#computing the means
sampledMean_within = np.mean(decision_scores[:100])
sampledMean_between =np.mean(decision_scores[100:])
sampledMean_difference=sampledMean_between-sampledMean_within
    
# Plot distributions and histogram
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'distribution_plots/')
fig = plt.figure() #3 plots
#ax=fig.add_subplot(2,1,1)
plt.title('SVM Sampled Distance to the Separating Hyperplane \n meanWithinObjs = %s, meanAcrossObjs = %s \n meanDiff = %s'%(sampledMean_within,sampledMean_between,sampledMean_difference))
plt.hist(decision_scores[:100],facecolor ='green',label='within', bins=30, range=[-5,5])
plt.hist(decision_scores[100:],facecolor ='blue', alpha=0.5,label='between',bins=30, range=[-5,5])
plt.legend(['DistanceWithin', 'DistanceBetween'], loc='upper left')
plt.tight_layout()

#plot permuted mean distribution
#ax=fig.add_subplot(2,1,2)
#permMeanDiff = np.mean(permMeanDiffArray)
#permSTD = np.std(perMeanDiffArray)
#ax.set_title('Permuted Mean Difference Distribution \n mean = %s \n std =%s' %(permMeanDiff,permSTD))
#ax.hist(permMeanDiffArray,facecolor ='purple',bins=30)
#ax.axvline(x=abs(sampledMean_difference), color = 'r', linestyle='dashed',linewidth=2)
#fig.subplots_adjust(hspace = 1)
#plt.tight_layout()

#Save Images
plt.savefig(plot_save_dir +  'plot_distribution_svmScores.png')
plt.savefig(plot_save_dir +  'plot_distribution_svmScores.pdf')

#pdb.set_trace()

