import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
import json
from numpy.random import seed
from tensorflow import set_random_seed
from FullNetwork import FNet


#Model of Convolutional NN to elaborate Images
class ModelCNN():

    #FIELDS
    # self.session                  -> model session
    # self.accuracy                 -> accuracy model
    # self.x                        -> input of the model (placeholder)
    # self.y_true                   -> output (true values) of the model to compare with the predicted (placeholder)
    # self.y_true_class             -> class of true value of the output
    # self.layer_flat               -> flat layer of the model
    # self.fullConnectedLayers      -> layers of full connected net
    # self.y_pred                   -> predicted output
    # self.y_pred_class             -> predicted class
    # self.cross_entropy            -> entropy of the model
    # self.cost                     -> final cost of the model
    # self.optimizer                -> optimizer for back propagation
    # self.correct_prediction       -> prediction 
    # self.modelPath                -> path of the model to load-save
    # self.modelName                -> name of the model
    # self.nClasses                 -> number of classes
    # self.nInput                   -> numer of inputs

    def __init__(self):
        self.__getModelPath__()
        self.nClasses = 5
        self.nInput =3
        self.batchSize = 100

        data = json.load(open(".\\config.json"))
        #MODEL CREATION
        #creation of the session
        self.session = tf.Session()
        #input of the network
        self.x = tf.placeholder(tf.float32, shape=[None, self.nInput], name='x')
        #labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.nClasses], name='y_true')
        self.y_true_class = tf.argmax(self.y_true, dimension=1)

        #creation of full connected layers
        self.fullConnectedLayers = []

        if(data["full_connected"][len(data["full_connected"])-1]["output_size"] != self.nClasses):
            raise ValueError('In config.json, number of classes must be equal of the last full connected output layer variables!')

        for fc in data["full_connected"]:
            inputlayer = self.x
            num_input = self.nInput
            if(len(self.fullConnectedLayers) != 0):
                inputlayer = self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer
                num_input = self.fullConnectedLayers[len(self.fullConnectedLayers)-1].output_size
            num_output = fc["output_size"]
            use_re = fc["use_relu"]

            newFC = self.__create_fc_layer__(input = inputlayer, num_inputs = num_input, num_outputs = num_output, use_relu = use_re )
            
            self.fullConnectedLayers.append( FNet(input_size = num_input, output_size = num_output, layer = newFC))
        
        
        #predictions
        self.y_pred = tf.nn.softmax(self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer, name='y_pred') #standard use of softmax -> 0 - 1 probability spread to the output
        self.y_pred_class = tf.argmax(self.y_pred, dimension=1) #return the index of the max value output
        
        #initialize variables
        self.session.run(tf.global_variables_initializer())

        #set the minimization cost like cross-entropy (ok not the normal norm)
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= self.fullConnectedLayers[len(self.fullConnectedLayers)-1].layer, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = data["learning_rate"]).minimize(self.cost)
        self.correct_prediction = tf.equal(self.y_pred_class, self.y_true_class)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.session.run(tf.global_variables_initializer()) 


    ### reads the path to save-load the model from config.json
    def __getModelPath__(self):
        data = json.load(open(".\\config.json"))
        self.modelPath = data["model_path"]
        self.modelName = data["model_name"]


    ### creates an array of variables to weights 
    def __create_weights__(self, shape):

        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    ### creates an array of variables to bias
    def __create_biases__(self, size):

        return tf.Variable(tf.constant(0.05, shape=[size]))


    ### create the full connected layer
    def __create_fc_layer__(self, input, num_inputs, num_outputs, use_relu=True):

        #Let's define trainable weights and biases.
        weights = self.__create_weights__(shape=[num_inputs, num_outputs])
        biases = self.__create_biases__(num_outputs)
        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.add(tf.matmul(input, weights) , biases)
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    ### show the progress of training
    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        #Calculate the accuracy of training data
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        #Calculate the accuracy of validation data
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)

        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f},  Cost: {3:.3f}"

        print(msg.format(epoch + 1, acc, val_acc, val_loss, self.cost))

    ### trains the model and saves it into the pre defined location
    def train(self, iterations, data_train, label_train, data_eval, label_eval):
        saver = tf.train.Saver()
        for i in range(0, iterations):
            index = int(i % self.batchSize) * int(len(data_train)/self.batchSize)
            x_batch = data_train[index: index+self.batchSize]
            y_true_batch = label_train[index: index+self.batchSize]
            x_valid_batch = data_eval
            y_valid_batch = label_eval
            #feeding
            feed_dict_tr = {self.x: x_batch, self.y_true: y_true_batch}
            feed_dict_val = {self.x: x_valid_batch, self.y_true: y_valid_batch}
            #running
            self.session.run(self.optimizer, feed_dict=feed_dict_tr)
            
            if i % 10 == 0: 
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / 10)    
                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(self.session, self.modelPath+self.modelName) 

            if i == iterations-1:
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / 10)    
                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(self.session, self.modelPath+self.modelName)
    
    ### evaluates data from the model, and load it from disk if needed
    def evaluate(self, data_in, load_model = False):

        if ( load_model ):
            saver = tf.train.Saver()
            # Step-1 load graph
            saver = tf.train.import_meta_graph(self.modelPath+self.modelName+'.meta')
            # Step-2: Now let's load the weights saved using the restore method.
            saver.restore(self.session, tf.train.latest_checkpoint(self.modelPath))
            # Accessing the default graph which we have restored
            graph = tf.get_default_graph()
            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            self.y_pred = graph.get_tensor_by_name("y_pred:0")
            ## Let's feed the images to the input placeholders
            self.x= graph.get_tensor_by_name("x:0") 
            self.y_true = graph.get_tensor_by_name("y_true:0")
        
        feed_dict_testing = {self.x: data_in, self.y_true: np.zeros((1, self.nClasses))}
        result = self.session.run(self.y_pred, feed_dict=feed_dict_testing)
        return result

