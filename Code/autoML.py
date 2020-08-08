import numpy as np
from tpot import TPOTClassifier

#MNIST
'''
from mlxtend.data import loadlocal_mnist

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25, random_state=34)

training_images, training_labels = loadlocal_mnist(
        images_path='F:\\Sujan\\PythonProjects\\Datasets\\train-images-idx3-ubyte', 
        labels_path='F:\\Sujan\\PythonProjects\\Datasets\\train-labels-idx1-ubyte')
testing_images, testing_labels = loadlocal_mnist(
        images_path='F:\\Sujan\\PythonProjects\\Datasets\\t10k-images-idx3-ubyte', 
        labels_path='F:\\Sujan\\PythonProjects\\Datasets\\t10k-labels-idx1-ubyte')
'''

#AlphanumKaggle
training_images = np.load('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\alphanumeric-handwritten-dataset\\alphanum-hasy-data-X.npy')
training_labels = np.load('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\alphanumeric-handwritten-dataset\\alphanum-hasy-data-Y.npy')

training_images=training_images.reshape(4658,1024)

tpot = TPOTClassifier(scoring='accuracy', generations=5, population_size=40, cv=5, verbosity=2, n_jobs=-1)

tpot.fit(training_images, training_labels)
print(tpot.score(training_images, training_labels))
