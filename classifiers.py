import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        # sum ranges or denominator isn't needed
        # initialize vocab size, likelihood, and prior belief values
        self.vocab_size = None
        self.likelihood = None
        self.prior = None
        

    def fit(self, X, Y):
        # Add your code here!
        # set X and Y to numpy arrays
        X = np.array(X)
        Y - np.array(Y)

        # get the vocab size
        self.vocab_size = X.shape[1]

        # get the prior probability
        self.prior = np.zeros(2)
        for i in range(2):
            # np.sum(Y == i) get the instances that match the class
            # len(Y) gets total length of values
            self.prior[i] = np.sum(Y == i)/len(Y)

        # now calculate the likelihood
        self.likelihood = np.zeros((2, self.vocab_size))
        for label in range(2):
            # Select only instances with the current label
            X_label = X[Y == label]
            # Calculate word counts for each feature
            word_counts = np.sum(X_label, axis=0)
            # Calculate likelihood with add-1 smoothing
            self.likelihood[label] = (word_counts + 1) / (np.sum(X_label) + self.vocab_size)
    
    def predict(self, X):
        # Add your code here!
        X = np.array(X)

        predictions = []

        for instance in X:
            log_p = np.zeros(2)

            for i in range(2):
                # calculate the log value
                log_p[i] = np.log(self.prior[i]) + np.sum(np.log(self.likelihood[i]) * instance)

            # pick the largest value out of the two classes and append to predictions
            label = np.argmax(log_p)
            predictions.append(label)


        return predictions



# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier"""
    
    def __init__(self):
        self.lr = 0.01
        self.epochs = 1000
        self.weights = None
        self.lambd = 10

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y)) + (2*self.lambd * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - Y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls



# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
