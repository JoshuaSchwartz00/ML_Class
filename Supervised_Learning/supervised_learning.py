import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
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
        data = load_iris(return_X_y=True)
        name = "iris"
    
    return data[0], data[1], name

def graph_results(data_name, name, train_accuracy, test_accuracy, x_label="Time", x_data=None):
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

def decision_tree(X, y, data_name, depth=(1, 15)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    
    total_train_acc = []
    total_test_acc = []
    
    for i in range(depth[0], depth[1]):
        model = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, "Decision Tree depth vs accuracy", total_train_acc, total_test_acc, x_label="Max Depth")
    
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
    

def neural_net(X, y, data_name, hidden_layers=(1, 4), hidden_nodes=(20, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    
    hidden_node_arr = [i for i in range(hidden_nodes[0], hidden_nodes[1])]
    
    for layer in range(hidden_layers[0], hidden_layers[1]):
        total_train_acc = []
        total_test_acc = []
        
        for node in range(hidden_nodes[0], hidden_nodes[1]):
            hls = (node,)*layer
            model = MLPClassifier(hidden_layer_sizes=hls ,activation="relu", max_iter=1000)
            model.fit(X_train, y_train)
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            total_train_acc.append(train_accuracy)
            total_test_acc.append(test_accuracy)
    
        graph_results(data_name, f"Neural Network with {layer} hidden layers and relu Accuracy", total_train_acc, total_test_acc, x_label="Hidden Nodes per Layer", x_data=hidden_node_arr)

def boosting(X, y, data_name, n_estimators=(1, 101)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    
    adaboost_total_train_acc = []
    adaboost_total_test_acc = []
    
    for i in range(n_estimators[0], n_estimators[1]):
        adaboost_model = AdaBoostClassifier(n_estimators=i)
        
        adaboost_model.fit(X_train, y_train)
        
        adaboost_train_accuracy = adaboost_model.score(X_train, y_train)
        adaboost_test_accuracy = adaboost_model.score(X_test, y_test)
        
        adaboost_total_train_acc.append(adaboost_train_accuracy)
        adaboost_total_test_acc.append(adaboost_test_accuracy)
        
    graph_results(data_name, "Adaboost estimators vs accuracy", adaboost_total_train_acc, adaboost_total_test_acc, x_label="Number of Estimators")

def svm(X, y, data_name, degree=(0, 20), coef0=(0, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    
    poly_total_train_acc = []
    poly_total_test_acc = []
    linear_total_train_acc = []
    linear_total_test_acc = []
    
    for i in range(coef0[0], coef0[1]):
        poly_model = SVC(kernel="poly")
        linear_model = SVC(kernel="linear")
        
        poly_model.fit(X_train, y_train)
        linear_model.fit(X_train, y_train)
        
        poly_train_accuracy = poly_model.score(X_train, y_train)
        poly_test_accuracy = poly_model.score(X_test, y_test)
        linear_train_accuracy = linear_model.score(X_train, y_train)
        linear_test_accuracy = linear_model.score(X_test, y_test)
        
        poly_total_train_acc.append(poly_train_accuracy)
        poly_total_test_acc.append(poly_test_accuracy)
        linear_total_train_acc.append(linear_train_accuracy)
        linear_total_test_acc.append(linear_test_accuracy)
    
    graph_results(data_name, f"SVM with poly kernel Accuracy", poly_total_train_acc, poly_total_test_acc, x_label="Coefficient 0")
    graph_results(data_name, f"SVM with linear kernel Accuracy", linear_total_train_acc, linear_total_test_acc, x_label="Coefficient 0")
    
    plt.plot(poly_total_train_acc, label="Poly SVM Train")
    plt.plot(poly_total_test_acc, label="Poly SVM Test")
    plt.plot(linear_total_train_acc, label="Linear SVM Train")
    plt.plot(linear_total_test_acc, label="Linear SVM Test")
    plt.legend()
    plt.xlabel("coef0")
    plt.ylabel("Accuracy")
    plt.title("Comparison of SVMs using Poly and Linear Kernels")
    plt.savefig(f"svm_poly_lin_{data_name}.png")
    
    plt.clf()
    
    total_train_acc = []
    total_test_acc = []
    
    for i in range(degree[0], degree[1]):
        model = SVC(kernel="poly", degree=i)
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        total_train_acc.append(train_accuracy)
        total_test_acc.append(test_accuracy)
    
    graph_results(data_name, "SVM with poly kernel and n degrees Accuracy", total_train_acc, total_test_acc, x_label="Degrees")

def knn(X, y, data_name, k=(1, 16)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    
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
    
    decision_tree(X, y, data_name)
    neural_net(X, y, data_name)
    boosting(X, y, data_name)
    svm(X, y, data_name)
    knn(X, y, data_name)
    
    X, y, data_name = get_data(num=0)
    
    decision_tree(X, y, data_name)
    neural_net(X, y, data_name)
    boosting(X, y, data_name)
    svm(X, y, data_name)
    knn(X, y, data_name)