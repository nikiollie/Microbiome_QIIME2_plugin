import tensorflow as tf 


class CNNClassifier():
    def __init__(self, batchsize = 32, learning_rate = 0.01, epochs = 1):
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.epochs = epochs

    
    def build_model(self):
        """
        Output: probability distribution of size [10]
        """
        images = tf.placeholder(tf.float32, [self.batchsize, 150, 4])
        target = tf.placeholder(tf.int32, [self.batchsize])
        kernel = tf.get_variable("conv1weights", [5,4,1,16], initializer = tf.random_normal_initializer(stddev =0.02))
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = "SAME")
        bias = tf.get_variable("conv1bias", [16], initializer= tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv,  bias)
        output = tf.reshape(output, [output.get_shape().as_list()[0], output.get_shape().as_list()[1]*output.get_shape().as_list()[2]*output.get_shape().as_list()[3]])
    
    def loss_func(self, predict_proba, target_proba):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_proba, labels = target_proba))
    
    def optim(self, loss):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)

    def train(self):
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        """
        Load data
        """
        true_data = load()
        true_labels = load_labels() # size = 10000*1

        for e in range(epochs):
            for it in range(len(true_labels)/self.batchsize):
                x = self.batchsize * 150 * 4 
                y = self.batchsize * 1
                _, l = self.sess.run([opt, loss], feed_dict = {images: x, target: y})
                print(l)

