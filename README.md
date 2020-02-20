# CS390NIP-lab0

This is the python file for CS39NIP lab0
# Custom neural net:
1. implemented _sigmoid function

2. implemented _sigmoidDerivative function (based on https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x)

3. implement train function
4. change epochs to 500 and mbs to 100
5. get small Batch for x and y by calling _batchGenerator()
6. create loop for how many epochs to run the forward and back propagation
7. get matrix of x and y by calling next() and save to mini_xVal and mini_yVal
8. run forward propagation
9. calculate back propagation based on the slide 2 page 52
10. update the W1 and W2 by new values.

11. implement preprocessData function
12. set the range of xTrain and xTest into the range between (0.0 to 1.0) by dividing original matrix by 255.0

13. implement trainModel function
14. use .reshape function to flatten the xTrain matrix
15. Write code for ALGORITHM == "custom_net" part
16. call NeuralNetwork_2Layer function and insert parameter 28 x 28 = 784 (flattened matrix), xVal = 10, yVal = 1000
17. run the training by calling model.train function
18. return model

19. implement runModel function
20. flatten data by calling reshape function
21. call model.predict to test on testing data
22. modify the result from model.predict to formate that can run evalResults ([0.00001] -> [1,0,0,0,0,0,0,0,0,0])


#  TF neural net:

1. create  train_keras function in NeuralNetwork_2Layer() class
2. implement the code from slide 3 page 45
3. create keras Sequential which order that keras should do, which flatten the data and create first layer and activation function of relu and then second layer activation function of softmax (same as the slide)
4. compile the model by calling model.compile function which the optimizer is adam and loss = sparse_categorical and matrics = ['accuracy']) same as the slide as well
5. adjust the result by calling np.argmax so that [0,1,0,0,0,0,0,0,0,0] -> [1]
6. train the model by calling model.fit function

7. implement trainModel function
8. work on ALGORITHM == 'tf_net' part
9. create model as previous section
10. call train_keras function we created before and get the trained model
11. return the model

12. implement runModel function
13. call ALGORITHM == 'tf_net' part
14 same as previous part get the predict result and modify result matrix to be able to run evalResult

# Pipeline & misc:

1. The image data values are between 0 and 255. Preprocess this such that values are between 0.0 and 1.0 is done in first section
2. implement evalResult function
3. use np.argmax to adjust the yTest and yPred to run the F1 score calculation function and generating confusion matrix function
4. import f1_score and confusion_matrix function from sklearn.matrix library
5. call those functions





