import tensorflow as tf 


class CNNClassifier():
    def __init__(self, batchsize):
        self.batchsize = batchsize

    
    def model(self, images):
        """
        Input: images of size [batchsize, 150, 4]
        Output: probability distribution of size [10]
        """
        kernel = tf.get_variable("conv1weights", [5,4,1,16], initializer = tf.random_normal_initializer(stddev =0.02))
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = "SAME")
        bias = tf.get_variable("conv1bias", [16], initializer= tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv,  bias)
    def train():



