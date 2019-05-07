import tensorflow as tf
import numpy as np
import pickle
import os
import pdb
class CNNClassifier():
    
    # Loads data from a single directory of pickle files
    def load(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        pickle_directory = directory + "/pickle_files/"
        dataset_file = open(os.path.join(pickle_directory, "dataset.pickle"), "rb")
        dataset_data = pickle.load(dataset_file)
        np.random.shuffle(dataset_data)
        true_labels = []
        true_data = []
        # Splits array of tuples into two arrays of data and labels
        # MAKE THIS MODULAR AKA 1000 per file
        for j in range(20000):
            true_data.append(dataset_data[j][0])
            true_labels.append(dataset_data[j][1])

        return true_data,true_labels
      

    # Sets up variables needed for model
    def __init__(self, batchsize = 32, learning_rate = 0.001, epochs = 1):
        #batchsize= the number of samples that will be propagated through the network  
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.epochs = epochs

    
    #Output: probability distribution of size [10]
    def build_model(self):
        
        # Placeholder is a value that we'll when Tensorflow runs this program
        # Second agrument in placeholder function corresponds to the dimensions of the matrix
        self.images = tf.placeholder(tf.float32, [self.batchsize, 150, 4, 1])
        self.target = tf.placeholder(tf.int32, [self.batchsize,2])
        
        # Get variable creates a new variable 'kernel'
        # A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        kernel = tf.get_variable("conv1weights", [3,3,1,32], initializer = tf.random_normal_initializer(stddev =0.02))
        # Creates a 2-D convolution, the input is 'images',filter=kernel,[1,1,1,1]:pad dimension
        # padding:pad evenly
        conv = tf.nn.conv2d(self.images, kernel, [1,1,1,1], padding = "SAME")
        bias = tf.get_variable("conv1bias", [32], initializer= tf.constant_initializer(0.0))
        output1 = tf.nn.bias_add(conv,  bias)
        print("Conv1", output1.get_shape().as_list())

        #150x4 becomes 150x2
        pool1 = tf.nn.max_pool(output1, [1,1,2,1], [1,1,2,1], padding="SAME", data_format="NHWC")
        #tf.layers.max_pool2D
        print("Pool1", pool1.get_shape().as_list())
        
        kernel2 = tf.get_variable("conv2weights", [2,2,32,32], initializer = tf.random_normal_initializer(stddev =0.02))
        conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding = "SAME")
        bias2 = tf.get_variable("conv2bias", [32], initializer= tf.constant_initializer(0.0))
        # Add bias to the weights allows you to shift the activation 
        # Function to the left or right, which may be critical for successful learning.
        output2 = tf.nn.bias_add(conv2,  bias2)
        print("Conv2", output2.get_shape().as_list())
        
        #150x2 becomes 150x2
        pool2 = tf.nn.max_pool(output2, [1,1,1,1], [1,1,1,1], padding="SAME", data_format="NHWC")
        #tf.layers.max_pool2D
        print("Pool2", pool2.get_shape().as_list())


        kernel3 = tf.get_variable("conv3weights", [2,1,32,32], initializer = tf.random_normal_initializer(stddev =0.02))
        conv3 = tf.nn.conv2d(pool2, kernel3, [1,1,1,1], padding = "SAME")
        bias3 = tf.get_variable("conv3bias", [32], initializer= tf.constant_initializer(0.0))
        # Add bias to the weights allows you to shift the activation 
        # Function to the left or right, which may be critical for successful learning.
        output3 = tf.nn.bias_add(conv3,  bias3)
        print("Conv3", output3.get_shape().as_list())
        # Given a tensor,output, this operation returns a tensor that 
        # has the same values as tensor with shape shape.
        

        #150x2 becomes 75x1
        pool3 = tf.nn.max_pool(output3, [1,2,2,1], [1,2,2,1], padding="SAME", data_format="NHWC")
        #tf.layers.max_pool2D
        print("Pool3", pool3.get_shape().as_list())


        kernel4 = tf.get_variable("conv4weights", [2,1,32,32], initializer = tf.random_normal_initializer(stddev =0.02))
        conv4 = tf.nn.conv2d(pool3, kernel4, [1,1,1,1], padding = "SAME")
        bias4 = tf.get_variable("conv4bias", [32], initializer= tf.constant_initializer(0.0))
        # Add bias to the weights allows you to shift the activation 
        # Function to the left or right, which may be critical for successful learning.
        output4 = tf.nn.bias_add(conv4,  bias4)
        print("Conv4", output4.get_shape().as_list())
        # Given a tensor,output, this operation returns a tensor that 
        # has the same values as tensor with shape shape.
        

        #75x1 becomes 25x1
        pool4 = tf.nn.max_pool(output4, [1,3,1,1], [1,3,1,1], padding="SAME", data_format="NHWC")
        #tf.layers.max_pool2D
        print("Pool4", pool4.get_shape().as_list())


        kernel5 = tf.get_variable("conv5weights", [2,1,32,16], initializer = tf.random_normal_initializer(stddev =0.02))
        conv5 = tf.nn.conv2d(pool4, kernel5, [1,1,1,1], padding = "SAME")
        bias5 = tf.get_variable("conv5bias", [16], initializer= tf.constant_initializer(0.0))
        # Add bias to the weights allows you to shift the activation 
        # Function to the left or right, which may be critical for successful learning.
        output = tf.nn.bias_add(conv5,  bias5)
        print("Conv5", output.get_shape().as_list())
        # Given a tensor,output, this operation returns a tensor that 
        # has the same values as tensor with shape shape.
        #output has size 32x400 because 32 batches
        output = tf.reshape(output, [output.get_shape().as_list()[0], 
            output.get_shape().as_list()[1]*output.get_shape().as_list()[2]*output.get_shape().as_list()[3]])


        # Initial weights are chosen randomly
        #second parameter [,output size for weight input size]
        weights_linear1 = tf.get_variable("weightslinear", [400, 32], initializer = tf.random_normal_initializer(stddev = 0.02))
        
        # Computes matrix multiplication between output and weights_linear1
        #outputs_hidden is 32x32
        outputs_hidden = tf.matmul(output, weights_linear1)
        outputs_hidden = tf.nn.relu(outputs_hidden)
        weights_linear2 = tf.get_variable("weightslinear2", [32, 2], initializer = tf.random_normal_initializer(stddev = 0.02))
        self.outputs = tf.matmul(outputs_hidden, weights_linear2)
        
        # A generalization of the logistic function that "squashes" a K-dimensional 
        # Vector of arbitrary real values to a K-dimensional vector of real values in the range [0, 1] that add up to 1
        self.outputs = tf.nn.softmax(self.outputs)
        self.loss = self.loss_func(self.outputs, self.target)
        self.optimizer = self.optim(self.loss)
        print(self.outputs.get_shape().as_list())
        print("Built Model")



    # Determines the difference between the predicted and target probabiblities/measures
    # badness or error in our neural network
    def loss_func(self, predict_proba, target_proba):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_proba, labels = target_proba))
    

    # Optimisation functions calculate the partial derivative of loss 
    # function with respect to weightsWeek 2
    def optim(self, loss):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
    
    # Trains the model given the built model above
    def train(self):
        print("Training...........")
        self.sess = tf.InteractiveSession()
        #global_variables_initializer is iterating through the variables of the
        #GLOBAL_VARIABLES collection and calling their initializer.
        tf.global_variables_initializer().run()
        #Load data
        true_data, true_labels = self.load()
        
        true_labels = np.asarray(true_labels) #10000*1
        true_labels_hot = np.zeros((true_labels.shape[0], 2))
        for i in range(true_labels.shape[0]):
            true_labels_hot[i, true_labels[i]] = 1 #check this
      
        #Lengths of different sets
        train_length = int(len(true_labels)*0.7)
        test_length = int(len(true_labels)*0.2)
        val_length = int(len(true_labels)*0.1)
        
        #Split data sets
        training_set = true_data[0:train_length]
        test_set = true_data[train_length:train_length+test_length]
        val_set = true_data[train_length+test_length:train_length+test_length+val_length]

        #Trainging and Validation labels
        training_labels_hot = true_labels_hot[0:train_length] 
        training_labels = true_labels[0:train_length]
        val_labels_hot = true_labels_hot[train_length+test_length:train_length+test_length+val_length]
        val_labels = true_labels[train_length+test_length:train_length+test_length+val_length]
        
        #runs the tensorflow operation
        epochs = 500
        print("Learning rate: " + str(self.learning_rate))
        print("Epochs: " + str(epochs))
        print("-----------------------------------------------")
        for e in range(epochs):
            count_val = 0
            count_train = 0
            for it in range(int(len(training_set)/self.batchsize)): #check this
                x = training_set[self.batchsize*it : self.batchsize*(it+1)] 
                x = np.reshape(x,(self.batchsize,150,4,1))
                y = training_labels_hot[self.batchsize*it : self.batchsize*(it+1)] 
                #get the values of many tensors
                _, l = self.sess.run([self.optimizer, self.loss], feed_dict
                    ={self.images:x, self.target:y})
                #print(l)
			#validation
           
            for it in range(int(len(val_set)/self.batchsize)):
                #validation accuracy
                x = val_set[self.batchsize*it : self.batchsize*(it+1)] 
                x = np.reshape(x,(self.batchsize,150,4,1))
                y = val_labels_hot[self.batchsize*it : self.batchsize*(it+1)] 
                #get the values of many tensors
                v = self.sess.run([self.outputs], feed_dict
                    ={self.images:x, self.target:y})
                val_max = np.argmax(v[0], 1)
                acc = val_labels[self.batchsize*it : self.batchsize*(it+1)]
                #compare
                val_max = np.array(val_max)
                acc = np.array(acc)
                equal = np.sum(val_max == acc)
                count_val += equal
            
            for it in range(int(len(training_set)/self.batchsize)): 
                #training accuracy
                x  = training_set[self.batchsize*it : self.batchsize*(it+1)]
                x = np.reshape(x,(self.batchsize,150,4,1))
                y = training_labels_hot[self.batchsize*it : self.batchsize*(it+1)]
                #get the values of many tensors
                v = self.sess.run([self.outputs], feed_dict
                    ={self.images:x, self.target:y})
                val_max = np.argmax(v[0], 1)
                acc = training_labels[self.batchsize*it : self.batchsize*(it+1)]
                #compare
                val_max = np.array(val_max)
                acc = np.array(acc)
                equal = np.sum(val_max == acc)
                count_train += equal
            print("Epoch " + str(e) + " Validation accuracy: " +
                str(count_val/len(val_set)))
           
            print("Epoch " + str(e) + " Training accuracy: " +
                str(count_train/len(training_set)))
            print("_________")

				

if __name__ == "__main__":
    cnnclassifier = CNNClassifier()
    cnnclassifier.build_model()
    cnnclassifier.train()
    
