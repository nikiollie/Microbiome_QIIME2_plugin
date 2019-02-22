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
        images = tf.placeholder(tf.float32, [self.batchsize, 150, 4, 1])
        target = tf.placeholder(tf.int32, [self.batchsize,10])
        kernel = tf.get_variable("conv1weights", [5,4,1,16], initializer = tf.random_normal_initializer(stddev =0.02))
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = "SAME")
        bias = tf.get_variable("conv1bias", [16], initializer= tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv,  bias)
        output = tf.reshape(output, [output.get_shape().as_list()[0], output.get_shape().as_list()[1]*output.get_shape().as_list()[2]*output.get_shape().as_list()[3]])
        weights_linear1 = tf.get_variable("weightslinear", [9600, 512], initializer = tf.random_normal_initializer(stddev = 0.02))
        outputs_hidden = tf.matmul(output, weights_linear1)
        weights_linear2 = tf.get_variable("weightslinear2", [512, 10], initializer = tf.random_normal_initializer(stddev = 0.02))
        outputs = tf.matmul(outputs_hidden, weights_linear2)
        outputs = tf.nn.softmax(outputs)
        loss = self.loss_func(outputs, target)
        optimizer = self.optim(loss)
        print(outputs.get_shape().as_list())
        print("Built Model")

    def loss_func(self, predict_proba, target_proba):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_proba, labels = target_proba))
    
    def optim(self, loss):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)

    def train(self):
        print("Training...........")
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        """
        Load data
        """
        true_data = load()
        true_labels = load_labels() # size = 10000*1

        for e in range(epochs):
            for it in range(int(len(true_labels)/self.batchsize)):
                x = true_data[self.batchsize*it : self.batchsize*(it+1)] 
                y = true_labels[self.batchsize*it : self.batchsize*(it+1)] 
                _, l = self.sess.run([opt, loss], feed_dict = {images: x, target: y})
                print(l)

if __name__ == "__main__":
    cnnclassifier = CNNClassifier()
    cnnclassifier.build_model()
    cnnclassifier.train()
