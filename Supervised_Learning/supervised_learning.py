import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_data(num = 1):
    if num:
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

def graph_results(data_name, name, train_accuracy, test_accuracy, x_label="Data Size", x_data=None):
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
    plt.savefig(f"{name}_{data_name}.png")
    plt.clf()
    
def graph_time(data_name, name, x_data, time):
    plt.plot(x_data, time)
    plt.title(f"{name}")
    plt.xlabel(f"Data Size")
    plt.ylabel("Time")
    plt.savefig(f"{name}_{data_name}.png")
    plt.clf()

def decision_tree(X, y, data_name):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.6)
    
    total_train_acc = []
    total_test_acc = []
    
    node_count = []
    depth = []
    
    model = DecisionTreeClassifier(criterion="entropy")
    path = model.cost_complexity_pruning_path(X_train, y_train)
    
    for ccp_alpha in path.ccp_alphas:
        model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
        node_count.append(model.tree_.node_count)
        depth.append(model.tree_.max_depth)
        
    ccp_alphas = path.ccp_alphas[:-1]
    total_train_acc = total_train_acc[:-1]
    total_test_acc = total_test_acc[:-1]
    node_count = node_count[:-1]
    depth = depth[:-1]
    
    plt.plot(ccp_alphas, node_count)
    plt.xlabel("alpha")
    plt.ylabel("Number of Nodes")
    plt.title("Alpha vs Number of Nodes")
    plt.savefig(f"nodes_alpha_{data_name}.png")
    
    plt.clf()
    
    plt.plot(ccp_alphas, depth)
    plt.xlabel("alpha")
    plt.ylabel("Depth")
    plt.title("Alpha vs Depth")
    plt.savefig(f"depth_alpha_{data_name}.png")
    
    plt.clf()
    
    graph_results(data_name, "Decision Tree alpha vs accuracy", total_train_acc, total_test_acc, x_label="Alpha", x_data=ccp_alphas)
    
    alpha = ccp_alphas[np.argsort(total_test_acc)[-1]]
    
    total_train_acc = []
    total_test_acc = []
    times = []
    
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
    
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha)
        model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        times.append(train_time)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, "Decision Tree data size vs accuracy", total_train_acc, total_test_acc, x_data=x_data)
    
    graph_time(data_name, "Decision tree data size vs time", x_data, times)
    
    return alpha
    

def neural_net(X, y, data_name):
    total_train_acc = []
    total_test_acc = []
    times = []
     
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
        
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        model = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu", max_iter=1000)
        model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        times.append(train_time)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, f"Neural Network data size vs accuracy", total_train_acc, total_test_acc, x_data=x_data)
    
    graph_time(data_name, "Neural Network data size vs time", x_data, times)

def boosting(X, y, data_name, alpha):
    total_train_acc = []
    total_test_acc = []
    times = []
     
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
        
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha, max_depth=5))
        adaboost_model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        times.append(train_time)
        
        train_accuracy = adaboost_model.score(X_train, y_train)
        test_accuracy = adaboost_model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
        
    graph_results(data_name, "Adaboost data size vs accuracy", total_train_acc, total_test_acc, x_data=x_data)
    
    graph_time(data_name, "Adaboost data size vs time", x_data, times)

def svm(X, y, data_name):
    poly_total_train_acc = []
    poly_total_test_acc = []
    linear_total_train_acc = []
    linear_total_test_acc = []
    poly_times = []
    linear_times = []
     
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
        
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        poly_model = SVC(kernel="poly")
        poly_model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        poly_times.append(train_time)
        
        curr_time = datetime.now()
        linear_model = SVC(kernel="linear")
        linear_model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        linear_times.append(train_time)
        
        poly_train_accuracy = poly_model.score(X_train, y_train)
        poly_test_accuracy = poly_model.score(X_test, y_test)
        linear_train_accuracy = linear_model.score(X_train, y_train)
        linear_test_accuracy = linear_model.score(X_test, y_test)
        
        poly_total_train_acc.append(poly_train_accuracy)
        poly_total_test_acc.append(poly_test_accuracy)
        linear_total_train_acc.append(linear_train_accuracy)
        linear_total_test_acc.append(linear_test_accuracy)
    
    plt.plot(x_data, poly_total_train_acc, label="Poly SVM Train")
    plt.plot(x_data, poly_total_test_acc, label="Poly SVM Test")
    plt.plot(x_data, linear_total_train_acc, label="Linear SVM Train")
    plt.plot(x_data, linear_total_test_acc, label="Linear SVM Test")
    plt.legend()
    plt.xlabel("Data Size")
    plt.ylabel("Accuracy")
    plt.title("Comparison of SVMs using Poly and Linear Kernels")
    plt.savefig(f"svm_poly_lin_{data_name}.png")
    
    plt.clf()
    
    title = "SVM Comparison data size vs time"
    
    plt.plot(x_data, poly_times, label="Poly kernel training times")
    plt.plot(x_data, linear_times, label="Linear kernel training times")
    plt.legend()
    plt.title(title)
    plt.xlabel("Data Size")
    plt.ylabel("Time")
    plt.savefig(f"{title}_{data_name}.png")
    
    plt.clf()

def knn(X, y, data_name, k=(1, 16)):
    total_train_acc = []
    total_test_acc = []
    times = []
     
    x_data = np.array([i for i in range(int(X.shape[0]*.2), int(X.shape[0]*.8))])
        
    for i in x_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)
        
        curr_time = datetime.now()
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        train_time = (datetime.now() - curr_time).total_seconds()
        times.append(train_time)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, "KNN data size vs accuracy", total_train_acc, total_test_acc, x_data=x_data)
    
    graph_time(data_name, "KNN data size vs time", x_data, times)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    k_arr = [i for i in range(k[0], k[1])]
    
    total_train_acc = []
    total_test_acc = []
    
    for i in range(k[0], k[1]):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, "KNN K-value vs accuracy", total_train_acc, total_test_acc, x_label="K", x_data=k_arr)

if __name__ == "__main__":
    X, y, data_name = get_data()
    
    alpha = decision_tree(X, y, data_name)
    neural_net(X, y, data_name)
    boosting(X, y, data_name, alpha)
    svm(X, y, data_name)
    knn(X, y, data_name)
    
    X, y, data_name = get_data(num=0)
    
    alpha = decision_tree(X, y, data_name)
    neural_net(X, y, data_name)
    boosting(X, y, data_name, alpha)
    svm(X, y, data_name)
    knn(X, y, data_name)