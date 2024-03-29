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

def sigmoid_deriv(sigmoid_value):
    return sigmoid_value * (1 - sigmoid_value)

def sigmoid_binary_classifier(sigmoid_value):
    try:
        label = 1 if sigmoid_value >= 0.5 else 0
        return label
    except:
        try:
            labels = [1 if sigmoid_ele_value >= 0.5 else 0 for sigmoid_ele_value in sigmoid_value]
            return labels
        except:
            raise AssertionError()

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
        self.features = features.copy()
        self.labels = labels.copy()

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
    def __init__(self, learning_rate=0.01, activation_fun=sigmoid, classifier=sigmoid_binary_classifier, epoch=200):
        # Perceptron state here
        # Feel free to add methods
        self.W = None
        self.b = None
        self.learning_rate = learning_rate
        self.activation_fun = activation_fun
        self.classifier = classifier
        self.epoch = epoch
    
    def _init_parameters(self, input_shape):
        self.W = np.random.randn(input_shape)
        self.b = np.random.randn()
    
    def _forward_prop(self, current_epoch, features, labels, sample_num):
        correct_classified = 0
        for feature, label in zip(features, labels):
            y = self.activation_fun(np.dot(self.W, feature) + self.b)
            label_pred = self.classifier(y)
            if (label == label_pred):
                correct_classified += 1

            error_value = label - label_pred
            self.W += self.learning_rate * error_value * feature
            self.b += self.learning_rate * error_value
        accuracy = correct_classified / sample_num
        print(f"epoch {current_epoch}: training accuracy -> {accuracy}")

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        self._init_parameters(input_shape=features.shape[1])
        for current_epoch in range(self.epoch):
            self._forward_prop(current_epoch, features, labels, sample_num=features.shape[0])

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        
        return self.classifier(self.activation_fun(np.dot(features, self.W) + self.b))

# class MLP:
#     def __init__(self, learning_rate=0.01, activation_fun=sigmoid_binary_activation, hidden_layer_size=5, epoch=200):
#         # Multilayer perceptron state here
#         # Feel free to add methods
#         pass

#     def train(self, features, labels):
#         # training logic here
#         # input is list/array of features and labels
#         pass  

#     def predict(self, features):
#         # Run model here
#         # Return list/array of predictions where there is one prediction for each set of features
#         pass

class DecisionTreeNode():
    def __init__(self, left_child, right_child, is_leaf, criteria, label):
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.criteria = criteria
        self.label = label

    def decide(self, feature):
        if self.is_leaf:
            return self.label

        deciison_branch = self.criteria(feature)
        if decision_branch == 'left':
            return self.left_child
        elif deicision_branch == 'right':
            return self.right_child
        else:
            raise AssertionError()

class DecisionTree():
    def __init__(self, root):
        self.root = root

    def get_tree(self):
        return self.root

class ID3:
	def __init__(self, bucket_size=5):
		#Decision tree state here
		#Feel free to add methods
        self.bucket_size = bucket_size
        self.tree = None
    
    def _create_decision_tree(self, features, labels):
        root = None
        positive_sample = 0
        negative_sample = 0
        for label in labels:
            if label == 0:
                negative_sample += 1
            else:
                positive_sample += 1
        
        remained_columns = [i for i in range(len(features))】
        if positive_sample == 0 or negative_sample == 0 or len(remained_columns) == 0:
            root = DecisionTreeNode(None, None, True, None, label = 1 if positive_sample > negative_sample else 0)
            self.tree = DecisionTree(root)
        else:
            attr, threshold = self.decide_attr(features, labels, remained_columns)


	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
        _create_decision_tree(features, labels)

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

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
    perceptron_model = Perceptron(learning_rate=0.01)
    perceptron_model.train(X_train, y_train)
    print("Perceptron model test accuracy:", evaluate(y_test, perceptron_model.predict(X_test)))

def test_models(dataset):
    # test kNN model
    # test_knn_model(dataset, k_range=[1,3,5])

    # test perceptron model
    test_perceptron_model(dataset)

if __name__ == '__main__':
    test_models(dataset)