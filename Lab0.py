
import os
import numpy as np
from scipy.linalg import expm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"

#########################################################################
# the comment of implementation is at the README file of my github page #
#########################################################################


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        #TODO: implement
        return 1.0 / (1.0 + np.exp(-1 * x))


    # Activation prime function.
    def __sigmoidDerivative(self, x):
        # pass   #TODO: implement
        return self.__sigmoid(x) * (1.0 - self.__sigmoid(x));

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 500, minibatches = True, mbs = 100): # epochs = 10000 mbs = 100
        xBatch_generator = self.__batchGenerator(xVals, mbs)
        yBatch_generator = self.__batchGenerator(yVals, mbs)
        # xVals is flattened already so it has 28 x 28 = 784 columns and 60000 rows
        for j in range (epochs):
            mini_xVals = next(xBatch_generator)
            mini_yVals = next(yBatch_generator)

            layer1 = self.__sigmoid(np.dot(mini_xVals, self.W1))
            layer2 = self.__sigmoid(np.dot(layer1, self.W2))

            loss = mini_yVals - layer2
            layer2_delta = loss * self.__sigmoidDerivative(layer2)
            layer1_error = np.dot (layer2_delta, self.W2.T)
            layer1_delta = layer1_error * self.__sigmoidDerivative(layer1)
            layer1_adjustment = np.dot(mini_xVals.T, layer1_delta) * self.lr  # this is the change in W1
            layer2_adjustment = np.dot(layer1.T, layer2_delta) * self.lr # this is the change in W2

            self.W1 = self.W1 + layer1_adjustment
            self.W2 = self.W2 + layer2_adjustment

        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


    def train_keras(self, x, y, eps = 5):
        model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(512, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        invert_categorical_y = np.argmax(y, axis=1)
        model.fit(x, invert_categorical_y, epochs=eps, verbose=0 )
        return model



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = (xTrain * 1.0) / 255.0
    xTest = (xTest * 1.0) / 255.0


    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    xTrain_flat = xTrain.reshape (60000, 784)
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        #TODO: Write code to build and train your custon neural net.
        model = NeuralNetwork_2Layer(784, 10, 1000)
        model.train(xTrain_flat, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        # print("Building and training TF_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        model = NeuralNetwork_2Layer(784, 10, 10000)
        keras_model = model.train_keras(xTrain, yTrain)
        return keras_model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    # print (data.shape)
    flattened_data = data.reshape(10000, 784)
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        #TODO: Write code to run your custon neural net.
        return to_categorical(np.argmax(model.predict(flattened_data), axis=1), NUM_CLASSES)
    elif ALGORITHM == "tf_net":
        #TODO: Write code to run your keras neural net.
        return to_categorical(np.argmax(model.predict(data) , axis=1), NUM_CLASSES)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    y_true = np.argmax(yTest, axis=1)
    y_pred = np.argmax(preds, axis=1)
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s \n" % ALGORITHM)
    print("Classifier confusion matrix: ")
    print(confusion_matrix(y_true, y_pred))
    print("Classifier F1 score (macro): " '{0:6f}' .format(f1_score(y_true, y_pred, average='macro')))
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
