import tensorflow as tf
import numpy as np
import pickle
import os
class CNNClassifier():
    def load(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        pickle_directory = directory + "/pickle_files/"
        pickle_directory = os.fsencode(pickle_directory)
        all_emps = []
        for file in os.listdir(pickle_directory):
            filename = os.fsdecode(file)
            if filename == "labels.pickle":
                labels_file = open(file, "rb")
                labels_data = pickle.load(labels_file)
            elif filename.endswith(".pickle"):
                loaded_pickle = open(file,"rb")
                emp = pickle.load(loaded_pickle)
        
            else:
                continue
            all_emps.append(emp)
            return all_emps, labels_file

    def __init__(self, batchsize = 32, learning_rate = 0.01, epochs = 1):
        #batchsize= the number of samples that will be propagated through the network  
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.epochs = epochs

    
    def build_model(self):
        
        #Output: probability distribution of size [10]
        
        #placeholder is a value that we'll when Tensorflow runs this program
        #second agrument in placeholder function corresponds to the dimensions of
        #the matrix
        images = tf.placeholder(tf.float32, [self.batchsize, 150, 4, 1])
        target = tf.placeholder(tf.int32, [self.batchsize,10])
        #get variable creates a new variable 'kernel'
        #A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        kernel = tf.get_variable("conv1weights", [5,4,1,16], initializer = tf.random_normal_initializer(stddev =0.02))
        #creates a 2-D convolution, the input is 'images',filter=kernel,[1,1,1,1]:pad dimension
        #padding:pad evenly
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = "SAME")
        bias = tf.get_variable("conv1bias", [16], initializer= tf.constant_initializer(0.0))
        #add bias to the weights allows you to shift the activation 
        #function to the left or right, which may be critical for successful learning.
        output = tf.nn.bias_add(conv,  bias)
        #Given a tensor,output, this operation returns a tensor that 
        #has the same values as tensor with shape shape.
        output = tf.reshape(output, [output.get_shape().as_list()[0], output.get_shape().as_list()[1]*output.get_shape().as_list()[2]*output.get_shape().as_list()[3]])
        #initial weights are chosen randomly
        weights_linear1 = tf.get_variable("weightslinear", [9600, 512], initializer = tf.random_normal_initializer(stddev = 0.02))
        #computes matrix multiplication between output and weights_linear1
        outputs_hidden = tf.matmul(output, weights_linear1)
        weights_linear2 = tf.get_variable("weightslinear2", [512, 10], initializer = tf.random_normal_initializer(stddev = 0.02))
        outputs = tf.matmul(outputs_hidden, weights_linear2)
        #a generalization of the logistic function that "squashes" a K-dimensional 
        #vector of arbitrary real values to a K-dimensional vector of real values in the range [0, 1] that add up to 1
        outputs = tf.nn.softmax(outputs)
        loss = self.loss_func(outputs, target)
        optimizer = self.optim(loss)
        print(outputs.get_shape().as_list())
        print("Built Model")
    #determines the difference between the predicted and target probabiblities/measures
    #badness or error in our neural network
    def loss_func(self, predict_proba, target_proba):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_proba, labels = target_proba))
    
    #Optimisation functions calculate the partial derivative of loss 
    #function with respect to weightsWeek 2
    def optim(self, loss):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
    
    def train(self):
        print("Training...........")
        self.sess = tf.InteractiveSession()
        #global_variables_initializer is iterating through the variables of the
        #GLOBAL_VARIABLES collection and calling their initializer.
        tf.global_variables_initializer().run()
       
        #Load data

        true_data, true_labels = self.load()   
        id_matrix = np.matlib.identity(len(true_labels))
        for row in range(len(id_matrix)):
            id_matrix = [row]
        #runs the TensorFlow operations
        for e in range(epochs):
            for it in range(int(len(true_labels)/self.batchsize)):
                x = true_data[self.batchsize*it : self.batchsize*(it+1)] 
                y = true_labels[self.batchsize*it : self.batchsize*(it+1)] 
                #get the values of many tensors
                _, l = self.sess.run([opt, loss], feed_dict = {images: x, target: y})
                print(l)
                
if __name__ == "__main__":
    cnnclassifier = CNNClassifier()
    cnnclassifier.build_model()
    cnnclassifier.train()
