# prednet

Code and models accompanying [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning] (https://arxiv.org/abs/1605.08104) by Bill Lotter, Gabriel Kreiman, and David Cox.

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005).
**Check out example prediction videos [here] (https://coxlab.github.io/prednet/).**

The architecture is implemented as a custom layer<sup>1</sup> in [Keras] (http://keras.io/). Tested on Keras 1.0.7 with [theano] (http://deeplearning.net/software/theano/) backend.
See http://keras.io/ for instructions on installing Keras and its list of dependencies.
For Torch implementation, see [torch-prednet] (https://github.com/e-lab/torch-prednet).
<br>

## changes to prednet

This version of prednet has an additional output_mode ="allR" which can output all layers of the network. 
This code will output the vector representation of the last 2 layers of the network 
These two layers were chosen based on the best resutls of the similarity analysis
<br>

### IARPA Demo

We include code for reading stimuli object files to numpy arrays as hkl files. We also include code for training and evaluating the model to support a similarity feature.

#### Steps
1. **Read stimuli object files to numpy arrays as hkl files**
	```bash
	python process_iarpa.py
	```
	This will create 3 types of files on the data directory:
	./iarpa_data/
	X_train_4k.hkl : HKL file holding 4k stimuli objects with 10 timesteps/transformations for training the predNet
	X_val_200.hkl  : HKL file holding 200 stimuli objects with 10 timesteps/transformation for evaluation of the predNet training
	X_svm_700.hkl  : HKL file holding 700 stimuli objects with 10 timesteps/transformations, 600 used for training and 100 used for testing
	
	Object stimuli are being read from:  
 	'/n/coxfs01/wscheirer/stimuli/Master_Set/'  
	'/n/coxfs01/wscheirer/stimuli/Set3/'
    
    If hkl files already exists, no need to re-run it.  
    HKL files generated from the Notre Dame Stimuli dataset can be found at:  
    /n/coxfs01/ygonzalez/prednet_similarity_hkl_data/similarity_project/iarpa_data/
    
	<br>
	<br>
	
2. **Train predNet model**
	```bash
	python iarpa_train.py
	```
	This will train a PredNet model for t+1 prediction. (10)
	See [Keras FAQ] (http://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu) on how to run using a GPU.
	
	Estimated training time for 4k stimuli and 10 transformations is about 18 hours. 
	
	<br>
	<br>
	
3. **Evaluate model**
	```bash
	python iarpa_eval_svm_similarity.py
	```
	
	This will train an svm to compute the similarity feature between objects using the representation layers created by the predNet. 
	This code takes about 15 minutes to run. Plots with the SVM Sampled Distance to the Separating Hyperplane are generated and saved under
	iarpa_results/distribution plots. 
	
	```bash
	python iarpa_eval_cos_similarity.py
	```
	Similarity evaluation variation using cosine distances
	
	```bash
	python iarpa_eval_cos_similarity.py
	```
	
	This will output the mean-squared error for predictions as well as make plots comparing predictions to ground-truth.
	Results are saved under ./iarpa_results/prediction_plots

<br>		

<sup>1</sup> Note on implementation:  PredNet inherits from the Recurrent layer class, i.e. it has an internal state and a step function. Given the top-down then bottom-up update sequence, it must currently be implemented in Keras as essentially a 'super' layer where all layers in the PredNet are in one PredNet 'layer'. This is less than ideal, but it seems like the most efficient way as of now. We welcome suggestions if anyone thinks of a better implementation.  
