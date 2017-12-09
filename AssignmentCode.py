import pandas as pd
import numpy as np

# Data with features and target values
# Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
# Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")


# ========================================== Data Helper Functions ==========================================

# Normalize values between 0 and 1
# dataset: Pandas dataframe
# categories: list of columns to normalize, e.g. ["column A", "column C"]
# Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData


# Encode categorical values as mutliple columns (One Hot Encoding)
# dataset: Pandas dataframe
# categories: list of columns to encode, e.g. ["column A", "column C"]
# Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)


# Split data between training and testing data
# dataset: Pandas dataframe
# ratio: number [0, 1] that determines percentage of data used for training
# Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset) * ratio)
    return dataset[:tr], dataset[tr:]


# Convenience function to extract Numpy data from dataset
# dataset: Pandas dataframe
# Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam", "winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels


# Convenience function to extract data from dataset (if you prefer not to use Numpy)
# dataset: Pandas dataframe
# Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()


# Calculates accuracy of your models output.
# solutions: model predictions as a list or numpy array
# real: model labels as a list or numpy array
# Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)
    return (predictions == labels).sum() / float(labels.size)


# ===========================================================================================================
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def step_function(inX):
    try:
        return 1 if inX >= 0 else 0
    except:
        try:
            return [1 if inX_ele >= 0 else 0 for inX_ele in inX]
        except:
            raise Exception()

def step_binary_activation(inX):
    return step_function(inX)

def sigmoid_binary_activation(inX):
    sigmoid_inX = sigmoid(inX)
    sigmoid_inX[sigmoid_inX >= 0.5] = 1
    sigmoid_inX[sigmoid_inX < 0.5] = 0
    return sigmoid_inX

def euc_dis(vector1, vector2):
    if len(vector1) != len(vector2):
        raise AssertionError()
    diff_square_list = list(map(lambda x:(x[0] - x[1]) ** 2, zip(vector1, vector2)))
    dis = 0
    for diff_square in diff_square_list:
        dis += diff_square
    return dis

class KNN:
    def __init__(self, k=3):
        self.k = k

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        self.features = features
        self.labels = labels

    def predict(self, features, distance_function=euc_dis):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        result_array = []
        for row in features:
            distances = [distance_function(stored_row, row) for stored_row in self.features]
            label_dis_array = [(self.labels[index], dis) for index, dis in enumerate(distances)]

            label_dis_array.sort(key = lambda x:x[1])
            shortest_dis_index = [label for label,_ in label_dis_array[0:self.k]]
            sum_result = 1 if sum(shortest_dis_index) * 2 >= self.k else 0
            result_array.append(sum_result)
        return result_array

class Perceptron:
    def __init__(self, learning_rate=0.01, activation_fun=sigmoid_binary_activation, epoch=200):
        # Perceptron state here
        # Feel free to add methods
        self.W = None
        self.b = None
        self.learning_rate = learning_rate
        self.activation_fun = activation_fun
        self.epoch = epoch

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        print("Perceptron training")
        try:
            self.W = np.random.randn(features.shape[1])
            self.b = np.random.randn(1)
            sample_num = features.shape[0]
            for current_epoch in range(self.epoch):
                correct_classified = 0
                for feature, label in zip(features, labels):
                    label_pred = self.activation_fun(np.dot(self.W, feature) + self.b)
                    if (label == label_pred):
                        correct_classified += 1

                    error_value = label - label_pred
                    self.W += self.learning_rate * error_value * feature
                    self.b += self.learning_rate * error_value
                accuracy = correct_classified / sample_num
                print(f"epoch {current_epoch}: training accuracy -> {accuracy}")

        except Exception:
            raise AssertionError()

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        return self.activation_fun(np.dot(features, self.W) + self.b)

# class MLP:
#     def __init__(self):
#         # Multilayer perceptron state here
#         # Feel free to add methods
#         self.weight = None
#         self.learningrate = 0.01
#     pass
#
#     def train(self, features, labels):
#         # training logic here
#         # input is list/array of features and labels
#         self.weight = np.random.uniform(-0.2, 0.2, len(features[0]) + 1)
#
#         pass
#
#     def predict(self, features):
#         # Run model here
#         # Return list/array of predictions where there is one prediction for each set of features
#         pass
#
#     def sigmoid(self, inX):
#         return 1.0 / (1 + np.exp(-inX))

# class ID3:
# 	def __init__(self):
# 		#Decision tree state here
# 		#Feel free to add methods

# 	def train(self, features, labels):
# 		#training logic here
# 		#input is list/array of features and labels

# 	def predict(self, features):
# 		#Run model here
# 		#Return list/array of predictions where there is one prediction for each set of features

def prepare_data(dataset, training_split_rate = 0.8, one_hot_encoding=True, use_numpy=True):
    numeric_features = ['net_ope_exp', 'net_con', 'tot_loa']
    text_features = ['can_off', 'can_inc_cha_ope_sea']
    if one_hot_encoding:
        dataset = encodeData(dataset, text_features)
    dataset = normalizeData(dataset, numeric_features)
    train, test = trainingTestData(dataset, training_split_rate)
    X_train, y_train = getPythonList(train) if not use_numpy else getNumpy(train)
    X_test, y_test = getPythonList(test) if not use_numpy else getNumpy(train)
    return X_train, y_train, X_test, y_test

def test_knn_model(dataset, k_range=[1,3]):
    print("kNN model")
    # prepare data
    X_train, y_train, X_test, y_test = prepare_data(dataset, use_numpy=False)
    for k in k_range:
        knn_model = KNN(k = k)
        knn_model.train(X_train, y_train)
        score = evaluate(y_test, knn_model.predict(X_test))
        print("k =", k, "test accuracy =", score)

def test_perceptron_model(dataset):
    X_train, y_train, X_test, y_test = prepare_data(dataset)
    perceptron_model = Perceptron(learning_rate=0.01, activation_fun=sigmoid_binary_activation)
    perceptron_model.train(X_train, y_train)
    print("Perceptron model test accuracy:", evaluate(y_test, perceptron_model.predict(X_test)))

def test_models(dataset):
    # test kNN model
    # test_knn_model(dataset, k_range=[1,3,5])

    # test perceptron model
    test_perceptron_model(dataset)

if __name__ == '__main__':
    test_models(dataset)