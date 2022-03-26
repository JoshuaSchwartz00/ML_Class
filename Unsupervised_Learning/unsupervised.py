import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_data(num=0):
    if num == 0:
        data = load_breast_cancer(return_X_y=True)
        name = "cancer"
    else: 
        raw_data = pd.read_csv("data_banknote_authentication.csv", header=None)
        raw_data = raw_data.values
        data = []
        data.append(raw_data[:, :-1])
        data.append(raw_data[:, -1])
        
        name = "bank"
    
    return data[0], data[1], name

def graph_dataset(title, x_data, y_data=None):
    if y_data is None:
        plt.scatter(x_data[:, 0], x_data[:, 1])
    else:
        if x_data.shape[1] == 1:
            zeros = np.zeros(x_data.shape[0])
            plt.scatter(x_data, zeros, c=y_data, cmap="RdYlBu")
        else:
            plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap="RdYlBu")
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()

def kmeans(X, y):
    model = KMeans(n_clusters=2)
    new_y = model.fit_predict(X)
    
    return new_y

def em(X, y):
    model = GaussianMixture(n_components=2)
    new_y = model.fit_predict(X)
    
    return new_y

def pca(X, y):
    model = PCA(n_components=2)
    new_X = model.fit_transform(X)
    
    print(model.components_)
    
    return new_X, y

def ica(X, y):
    model = FastICA(n_components=2, tol=3e-4)
    new_X = model.fit_transform(X)
    
    print(model.components_)
    
    return new_X, y

def randomized_projection(X, y): 
    model = GaussianRandomProjection(n_components=2)
    new_X = model.fit_transform(X)
    
    return new_X, y

def lda(X, y):
    model = LinearDiscriminantAnalysis(n_components=1)
    new_X = model.fit_transform(X, y)
    
    return new_X, y

def old_graph_results(name, train_accuracy, test_accuracy, x_label="Data Size", x_data=None):
    if x_data is None:
        plt.plot(train_accuracy, label="Train Accuracy")
        plt.plot(test_accuracy, label="Test Accuracy")
    else:
        plt.plot(x_data, train_accuracy, label="Train Accuracy")
        plt.plot(x_data, test_accuracy, label="Test Accuracy")
    plt.legend()
    plt.title(f"{name}")
    plt.xlabel(f"{x_label}")
    plt.ylabel("Accuracy")
    plt.savefig(f"{name}.png")
    plt.clf()
    
def old_graph_time(name, x_data, time):
    plt.plot(x_data, time)
    plt.title(f"{name}")
    plt.xlabel(f"Data Size")
    plt.ylabel("Time")
    plt.savefig(f"{name}.png")
    plt.clf()

def graph_loss(name, loss):
    plt.plot(loss)
    plt.title(f"{name}")
    plt.xlabel(f"Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{name}.png")
    plt.clf()

def neural_net(X, y, reduction, data_name):
    total_train_acc = []
    total_test_acc = []
    times = []
     
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
    x_data = [x_data[i*5] for i in range(len(x_data)//5)]
        
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        model = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu", max_iter=2000)
        model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        times.append(train_time)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    loss = model.loss_curve_
    
    old_graph_results(f"Neural Network data size vs accuracy on {reduction}-reduced {data_name} dataset", total_train_acc, total_test_acc, x_data=x_data)
    
    old_graph_time(f"Neural Network data size vs time on {reduction}-reduced {data_name} dataset", x_data, times)
    
    graph_loss(f"Neural Network loss curve on {reduction}-reduced {data_name} dataset", loss)

if __name__ == "__main__":
    #part 1
    #part 5
    X, y, name = get_data()
    neural_net(X, y, "non", name)
    
    title = f"KMeans on unreduced {name} dataset"
    new_y = kmeans(X, y)
    print(y, new_y)
    neural_net(X, new_y, "KMeans", name)
    
    print()
    
    title = f"Expectation Maximization on unreduced {name} dataset"
    new_y = em(X, y)
    print(y, new_y)
    neural_net(X, new_y, "Expectation Maximization", name)
    
    
    print()
    
    X, y, name = get_data(1)
    
    title = f"KMeans on unreduced {name} dataset"
    new_y = kmeans(X, y)
    print(y, new_y)
    
    print()
    
    title = f"Expectation Maximization on unreduced {name} dataset"
    new_y = em(X, y)
    print(y, new_y)
    
    dimensions = ["PCA", "ICA", "Randomized Projection", "LDA"]
    
    for a in range(2):
        for b in dimensions:
            print(a, b)
            X, y, name = get_data(a)
            
            #part 2  
            if b == "PCA":
                X, y = pca(X, y)
            elif b == "ICA":
                X, y = ica(X, y)
            elif b == "Randomized Projection":
                for i in range(5):
                    X, y = randomized_projection(X, y)
                    title = f"{b}-reduced {name} dataset {i}"
                    graph_dataset(title, X, y)
                    X, y, name = get_data(a)
                
                X, y = randomized_projection(X, y)
            elif b == "LDA":
                X, y = lda(X, y)
                
            title = f"{b}-reduced {name} dataset"
            graph_dataset(title, X, y)
            
            #part 4
            neural_net(X, y, b, name)
            
            #part 3
            c = "KMeans"
            title = f"{c} on {b}-reduced {name} dataset"
            
            new_y = kmeans(X, y)
            
            graph_dataset(title, X, new_y)
            
            
            c = "Expectation Maximization"
            title = f"{c} on {b}-reduced {name} dataset"
            
            new_y = em(X, y)
            
            graph_dataset(title, X, new_y)
                    