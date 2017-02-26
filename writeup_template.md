#**Traffic Sign Recognition** 
-

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./1.jpg "Traffic Sign 1"
[image5]: ./2.jpg "Traffic Sign 2"
[image6]: ./3.jpg "Traffic Sign 3"
[image7]: ./4.jpg "Traffic Sign 4"
[image8]: ./5.jpg "Traffic Sign 5"

---

####You're reading it! and here is a link to my [project code](https://github.com/yifenghuang/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (after split validation data)
* The size of test set is 12630
* The shape of a traffic sign image is 32*32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

I ploted the traffic sign images randomly using matplotlib

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I convert the images to grayscale but compare to RGB image data this convertion reduce the amount of informations of each image. So I still using RGB images to train. After that I normalized and regularsed the data because I want the data in same scale.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I' m living in China and for some reason I can not download original data file even using my VPN. So the data I download is from Baiduyun and the validation data is already splited and saved. 

but I still write a function to split the data and it works.

    def split(X_train_o, y_train_o, valid=6000): 
     
    	num   = len(y_train_o)
    	index = list(range(num))
    	random.shuffle(index)
    
    	train_index=index[valid:]
    	valid_index=index[:valid]

   		X_train = X_train_o[train_index] 
    	y_train = y_train_o[train_index]
    	X_valid = X_train_o[valid_index] 
   		y_valid = y_train_o[valid_index] 
   
    	return  X_train, y_train,  X_valid, y_valid



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model(lenet5 add dropout) is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x6x16	    | etc.      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten 5x5x16	      	| output 400 				|
| Fully connected 400		|output 120       									|
| dropout	      	| keep 0.9 				|
| Fully connected 120		|output 84       									|
| dropout	      	| keep 0.8 				|
| Fully connected 84		|output 43       									|
| Softmax				| etc.        									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth and seventh cell of the ipython notebook. 

I used adam optimizer to train the moudle. the loss function is cross entropy loss. learning rate is setted as 0.001.

the batch size is 2048 and the number of epochs is 2000. The training time is about 30min in nvidia GTX1070. I checked the validation accuracy while training and after training the accuracy is about 95%.

I remove the drop layer and find out the lenet5 architecture without dropout only can reach 90% of validation accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 95.5%
* test set accuracy of 93.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
the first architecture is classic Lenet5 with gray scale images data. I chosed it in the beginning because it is shown in the CNN lesson and easy to calibrate.

* What were some problems with the initial architecture?
I only get 85% validate accuracy after training. nomatter how I change the hyper parameters the accuracy is still very low.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
first I used RGB files instead of gray scale images. this can provide 2% improvement of validation accuracy. then I added two drop out layer and make the valid accuracy up to 95%. in this moudle the test accuracy is 93.7% which is ok.

* Which parameters were tuned? How were they adjusted and why?
the training rate and epochs were tuned. and the batch_size is depend on your ram size. I first try the learning rate as 0.0001 but the training speed is too slow and need too much epochs to stable. so I rise the learning rate to 0.001 and after 40 epochs the accuracy is almost stable. and the more epochs you run, the more accuracy you will get.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
the dropout layer can avoid overfitting so we can get higher validation accuracy and test accuracy. to make the data in same scale the normalization and regularition is chosen to work.

If a well known architecture was chosen:
* What architecture was chosen?
lenet5

* Why did you believe it would be relevant to the traffic sign application?
because it is shown in the CNN lesson and it can classify hand write digital in very high accuracy. and I belived it will relevant to the fraffic sign application.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 the final test accuracy is 93.7% and the validation accuracy is 95.5%. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
the fourth image is slightly dificult to classify because when scale it to 32x32, it looks just like a Roundabout mandatory, turn right ahead , turn left ahead, ahead only, go straight or left and other blue background round shape sign.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry     			| No entry 										|
| speed limit 60km/h		| speed limit 60km/h							|
| keep right			| roundabout mandatory     							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.7% because of the amount is tiny.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability almost 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	     				| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         	| Stop sign   					| 
| 2.49e-14    		| speed limit 80km/h	|
| 3.76e-15| speed limit 60kmh				|
| 2.32e-19	| general caution 				|
| 1.36e-22	| bicycles crossing			|


For the second image:

| Probability         	|     Prediction	     				| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		| no entry   					| 
| 0    		| speed limit 20km/h	|
| 0	| speed limit 30kmh				|
| 0	| speed limit 50kmh 				|
| 0	| speed limit 60kmh			|

For the 3th image:

| Probability         	|     Prediction	     				| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         	| speed limit 60kmh   					| 
| 4.62e-24    		| speed limit 50km/h	|
| 2.02e-29| keep right				|
| 5.19e-32	| yield				|
| 3.53e-32	| slippery road			|

For the 4th image:

| Probability         	|     Prediction	     				| 
|:---------------------:|:---------------------------------------------:| 
| 0.98        	| roundabout mandatory   					| 
| 0.02    		| double curve	|
| 2.68e-9| speed limit 30kmh				|
| 1.45e-10	| go straight or right 				|
| 4.73e-13	| keep left			|

For the 5th image:

| Probability         	|     Prediction	     				| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         	| right-of-way at the next intersection		| 
| 2.23e-29    		| beware of ice	|
| 9.21e-36| end of no passing by vehicles over 3.5 tons|
| 0	| speed limit 20km/h			|
| 0	| speed limit 30km/h		|