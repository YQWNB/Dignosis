"""
This code does training, validation and testing of the model for automated cardiac disease diagnosis.

"""
from __future__ import print_function
from joblib.logger import PrintTime
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import itertools
import sys
from numpy.core.fromnumeric import shape
from numpy.lib.financial import npv
import pandas as pd
from scipy.sparse import data
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier)
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn
from datetime import datetime
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sknn.mlp import Classifier, Convolution, Layer


sys.path.append("../")

# Disease Mapping
NOR = 'NOR'
# 30 patients with previous myocardial infarction
# (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction) - MINF
MINF = 'MINF'
# 30 patients with dilated cardiomyopathy
# (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) - DCM
DCM = 'DCM'
# 30 patients with hypertrophic cardiomyopathy
# (left ventricular cardiac mass high than 110 g/m2,
# several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) - HCM
HCM = 'HCM'
# 30 patients with abnormal right ventricle (volume of the right ventricular
# cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) - RV
RV = 'RV'
heart_disease_label_map = {NOR:0, MINF:1,DCM:2,HCM:3, RV:4}

# Path to cardiac features generated from given manually annotated training data
# These features are used for training and validation of model
full_training = './training_data/Cardiac_parameters_training.csv'

train = './training_data/Cardiac_parameters_train.csv'
validation = './training_data/Cardiac_parameters_validation.csv'

# Path to cardiac features generated from segmentations predicted by the segmentation network
test_on_prediction = './prediction_data/Cardiac_parameters_prediction.csv'

# test_on_prediction = './prediction_data/Cardiac_parameters_minmax_k_16.csv'

# Features columns selection
START_COL = 1
END_COL = 21

class_names = [NOR, MINF, DCM, HCM, RV]
class_names_for_cm = [NOR, MINF, DCM, HCM, 'ARV']

def visualize_tree(tree, feature_names, save_dir='./'):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open(save_dir+'/'+"dt.dot", 'w') as f:
        # export_graphviz(tree, out_file=f,
        #                 feature_names=feature_names)
        pass

    command = "dot"+" -Tpng "+ save_dir+"/dt.dot "+"-o "+ save_dir+"/dt.png"
    # try:
    os.system(command)
    # except:
    #     exit("Could not run dot, ie graphviz, to "
    #          "produce visualization")

def encode_target(df, target_column, label_map):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    # map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(label_map)

    return (df_mod, targets)

def load_dataframe(csv_file, shuffle=False):
    """
    Load Patient information from the csv file as a dataframe
    """
    df = pd.read_csv(csv_file)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # patient_data = df.to_dict("records")
    # return patient_data
    return df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)
    # CardiacDiagnosisModelTester(EN_clf, test_on_prediction, name='EnsembleOnFinalTestSet', scaler=scaler, save_dir=save_dir, label_available=False,
    #                             prediction_csv=test_on_prediction)
def CardiacDiagnosisModelTester(clf, final_test_path, name, scaler, save_dir='./', label_available=False, prediction_csv=None):
    """
    This code does the cardiac disease classification (5-classes)
    """
    class_names = [NOR, MINF, DCM, HCM, RV]
    df = load_dataframe(final_test_path)
    features = list(df.columns[np.r_[START_COL:END_COL]])
    X_df = df[features]
    # print (features)
    X_scaled = scaler.transform(X_df)  
    y_pred = clf.predict(X_scaled)
    # y_pred = y_pred.reshape(-1)
    print ("Writing predictions to file", name)
    target = open(save_dir+'/'+ name+'predictions_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S")), 'w')
    classes = {NOR: 0,MINF:0,DCM:0,HCM:0,RV:0}
    for pid, pred in zip(df['Name'], y_pred):
        classes[class_names[pred]] +=1
        line = '{} {}'.format(pid, class_names[pred])
        target.write(line)
        target.write("\n")
    target.close()
    print (classes)
    if label_available:
        y_true,_ = encode_target(df, 'GROUP', heart_disease_label_map)
        accuracy = accuracy_score(y_true['GROUP'], y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    else:
        if prediction_csv:
            df = load_dataframe(test_on_prediction)
            df['GROUP'] = [class_names[pred] for pred in y_pred]
            df.to_csv(prediction_csv,  index=False)
def getevalution(cnf_matrix,name):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
# Sensitivity, hit rate, recall, or true positive rate
    TPR = sum(TP/(TP+FN))/5
# Specificity or true negative rate
    TNR = sum(TN/(TN+FP))/5
# Precision or positive predictive value
    PPV = sum(TP/(TP+FP))/5
# Overall accuracy
    ACC = sum((TP+TN)/(TP+FP+FN+TN))/5
    
    print("{}‘s TPR={},TNR={},PPV={},ACC={}".format(name,TPR,TNR,PPV,ACC))
    return TPR

def E(acc,num1,num2):  #根据cost计算MC（Loss）和E（weight）
    cost1=30
    cost2=1
    MC=(num1*cost1+num2*cost2)/100
    E = acc-MC/(cost1+cost2)*8
    return MC,E
def getelf(acc):   #根据给定的子分类器得到集成分类器的参数
    W = {}
    for label in acc.keys():
        accuracy = acc[label][0]
        num1 = acc[label][1]
        num2 = acc[label][2]
        MC,e = E(accuracy,num1,num2)
        W[label] = e
    d_order=sorted(W.items(),key=lambda x:x[1],reverse=True)
    tagindex = {}
    weight = []
    for i in range(5):
        tagindex[d_order[i][0]] = i
        weight.append(d_order[i][1])
    standard = d_order[6][1]
    total = 0
    for i in range(5):
        weight[i] = weight[i]-standard
        total+=weight[i]
    for i in range(5):
        weight[i] = weight[i]/total
    for i in tagindex.keys():
        tagindex[i] = weight[tagindex[i]]    
    return tagindex
def evaluate(acc):
    for label in acc.keys():
        accuracy = acc[label][0]
        num1 = acc[label][1]
        num2 = acc[label][2]
        MC,e = E(accuracy,num1,num2)
        acc[label]=e
        print("{}’s MC={} ,E = {}".format(label,MC,e))    
if __name__ == '__main__':

    save_dir ='./Predictions{}'.format(time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir)   
    # Prepare the feature vectors for training and validation
    train_df, train_targets = encode_target(load_dataframe(train), 'GROUP', heart_disease_label_map)
    valid_df, _ = encode_target(load_dataframe(validation), 'GROUP', heart_disease_label_map)
    # Select the features:
    features = list(train_df.columns[np.r_[START_COL:END_COL]])
    print("* features:", features, sep="\n")
    X_train = train_df[features]
    y_train = train_df['GROUP']
    X_valid = valid_df[features]
    y_valid = valid_df['GROUP']

    # ################ Model trained on simple decision train #############
    DT_clf = DecisionTreeClassifier(min_samples_split=10, random_state=99)
    DT_clf.fit(X_train, y_train)
    print ('Generating features learnt from decision trees')
    # Code to visualize learnt features
    visualize_tree(DT_clf, features, save_dir=save_dir)
    print ('Results of training on Decision trees:')
    print ('Score on Manually segmented Test Set')
    print (DT_clf.predict(X_valid))
    print (DT_clf.score(X_valid, y_valid))
    cnf_matrix = confusion_matrix(y_valid, DT_clf.predict(X_valid))
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                          title='Confusion matrix: DT classifier on Validation Set')
    # plt.show()
    plt.savefig(save_dir+'/confusion_matrix_Decision tree')    
    
    # # ################# Model trained on Random Forest ##################
    n_estimators = 1000
    RF_clf = RandomForestClassifier(n_estimators=n_estimators)
    RF_clf.fit(X_train, y_train)
    print ('Results of training on Random Forest:')
    print ('Score on Manually segmented Test Set')
    y_pred = RF_clf.predict(X_valid)
    print (y_pred)
    print (RF_clf.score(X_valid, y_valid))
    # Predict on Manually segmentation results
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_valid, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                          title='Confusion matrix: Random Forest classifier on Validation Set')
    # plt.show()
    plt.savefig(save_dir+'/confusion_matrix_Manual_RF')
    

    ################# Model trained on MLP ##################
    print ('Results of training MLP:')
    scaler = StandardScaler() 
    scaler.fit(X_train) 
    X_scaled = scaler.transform(X_train)
    MLP_clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000)
    MLP_clf.fit(X_scaled, y_train)

    print ('Score on manually segmented Validation Set')
    y_pred = MLP_clf.predict(scaler.transform(X_valid))
    print (y_pred)
    print (MLP_clf.score(scaler.transform(X_valid), y_valid))

    # Predict on Manually segmentation results
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_valid, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                          title='Confusion matrix: MLP classifier on Validation Set')
    # plt.show()
    plt.savefig(save_dir+'/confusion_matrix_Manual_MLP1')

    ################# Model trained on CNN ##################
    nn = Classifier(
        layers=[
            Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
            Layer("Softmax")],
        learning_rate=0.02,
        n_iter=20)
    
    nn.fit(X_scaled,y_train)
    y_pred = nn.predict((scaler.transform(X_valid)))
    print(y_pred)
    cnf_matrix = confusion_matrix(y_valid, y_pred)    
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                          title='Confusion matrix: CNN classifier on Validation Set')
    # plt.show()
    plt.savefig(save_dir+'/confusion_matrix_CNN')
    
    
    # ***************** Feature importances study of the forest *****************#
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                  random_state=0)
    forest = RandomForestClassifier(n_estimators=n_estimators,
                                  random_state=0)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature: %s \t %d (%f)" % (f + 1, features[indices[f]], indices[f], importances[indices[f]]))

    #***************** Plot the feature importances of the forest *****************#
    indices = np.argsort(importances)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    sorted_features =  [features[i] for i in indices]
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, importances[indices], xerr=std[indices], align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    # ax.invert_yaxis()  # labels read top-to-bottom
    # Green bars are the feature importances of the forest and black indicate their inter-trees variability (Stddev)
    ax.set_xlabel('Feature Importances')
    ax.set_ylabel('Features')

    ax.set_title('Feature Importances of Random Forest Classifier')
    plt.subplots_adjust(left=0.4)
    plt.ylim([-1, X_train.shape[1]])
    # plt.show()
    plt.savefig(save_dir+'/Important_features')

    #************************* Cross-Validation Study ****************************#
    # Perform Cross_Validation: With Splits of 5
    print ('Doing cross-validation for Ensemble')
    # Cross-validation on full training_set
    train_df, train_targets = encode_target(load_dataframe(full_training, shuffle=False), 'GROUP', heart_disease_label_map)
    # Select the features:
    features = list(train_df.columns[np.r_[START_COL:END_COL]])
    # print("* features:", features, sep="\n")
    X = train_df[features]
    y = train_df['GROUP']
    # Scale the feature vectors using standard scaler
    scaler = StandardScaler() 
    scaler.fit(X) 
    X_scaled = scaler.transform(X)

    # Range of classifiers experimented with
    RF_clf = RandomForestClassifier(n_estimators=n_estimators)
    XG_clf = XGBClassifier(n_estimators=n_estimators)
    MLP_clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000)
    LR_clf = LogisticRegression()
    GNB_clf = GaussianNB()
    SVM_clf = svm.SVC(kernel='rbf', probability=True)
    KNN_clf = KNeighborsClassifier(n_neighbors=5)
    nn = Classifier(
        layers=[
            Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
            Layer("Softmax")],
        learning_rate=0.02,
        n_iter=20)

    dic ={'Logistic Regression':LR_clf,'Random Forest':RF_clf, 'Naive Bayes':GNB_clf, 'XG_boost':XG_clf, 'SVM':SVM_clf, 'MLP':MLP_clf, 'KNN':KNN_clf,"CNN":nn}
    acc={}
    ##交叉验证过程
    for clf, label in zip([LR_clf, RF_clf, GNB_clf, XG_clf, SVM_clf, MLP_clf, KNN_clf,nn], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'XG_boost', 'SVM', 'MLP', 'KNN',"CNN"]):
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        acc[label]=[]
        print (scores)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        predictions = cross_val_predict(clf, X_scaled, y, cv=5)
        incorrect_idx = np.nonzero(predictions != y)[0]
        print (predictions)
        print (incorrect_idx)
        print (train_df['Name'][incorrect_idx])
        Truth = [class_names[x] for x in y[incorrect_idx]]       #错误样本的真值
        Predict =  [class_names[x] for x in predictions[incorrect_idx]] #错误样本预测值
        print ('Truth', Truth)
        print ('Predicted', Predict)
        num_cost_1 = 0    #一类错误样本数
        num_cost_2 = 0    #二类错误样本数
        for predict in Predict:
            if predict =="NOR":   #证明一个病人被误诊为了正常
                num_cost_1+=1       
            else:
                num_cost_2+=1     #其余错误情况
        cnf_matrix = confusion_matrix(y, predictions)
        np.set_printoptions(precision=2)
        accuracy = getevalution(cnf_matrix,label)
        acc[label].append(accuracy)
        acc[label].append(num_cost_1)
        acc[label].append(num_cost_2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                              title='Confusion matrix: {} classifier'.format(label))
        # plt.show()
        plt.savefig(save_dir+'/confusion_matrix_Manual_{}'.format(label))
    #********************* Evaluate Ensemble classifier model on Validation Set******************************#
    tagweight = getelf(acc)
    estimators = []
    weight=[]
    for clf in tagweight.keys():
        estimators.append((clf,dic[clf]))
        weight.append(tagweight[clf])

    E_clf = VotingClassifier(estimators=estimators, voting='soft',weights=weight)   #集成分类器
    EN_clf = E_clf.fit(X_scaled, y)
    label="EN_clf"
    acc[label]=[]
##以下为检验EN_clf集成分类器的性能
    scores = cross_val_score(EN_clf, X_scaled, y, cv=5, scoring='accuracy')
    print (scores)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    predictions = cross_val_predict(E_clf, X_scaled, y, cv=5)
    incorrect_idx = np.nonzero(predictions != y)[0]
    print (predictions)
    print (incorrect_idx)
    print (train_df['Name'][incorrect_idx])
    Truth = [class_names[x] for x in y[incorrect_idx]]       #错误样本的真值
    Predict =  [class_names[x] for x in predictions[incorrect_idx]] #错误样本预测值
    print ('Truth', Truth)
    print ('Predicted', Predict)
    num_cost_1 = 0    #一类错误样本数
    num_cost_2 = 0    #二类错误样本数
    for predict in Predict:
        if predict =="NOR":   #证明一个病人被误诊为了正常
            num_cost_1+=1       
        else:
            num_cost_2+=1     #其余错误情况    
    cnf_matrix = confusion_matrix(y, predictions)
    np.set_printoptions(precision=2)
    accuracy = getevalution(cnf_matrix,label)
    acc[label].append(accuracy)
    acc[label].append(num_cost_1)
    acc[label].append(num_cost_2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
                              title='Confusion matrix: {} classifier'.format(label))
        # plt.show()
    plt.savefig(save_dir+'/confusion_matrix_Manual_{}'.format(label))
    print(tagweight)
    print("estimator:",estimators)      #分类器
    print("weight:",weight)     #权重
    print(acc)
    ##评估所有分类器的性能
    evaluate(acc)
    # TODO: If preferred Naive Bayes over Ensemble 
    # EN_clf = GNB_clf.fit(X_scaled, y)
    # EN_clf = MLP_clf.fit(X_scaled, y)
    # EN_clf = SVM_clf.fit(X_scaled, y)
    # EN_clf = RF_clf.fit(X_scaled, y)···
    ##################### Automated Caridiac Diagnosis on Final Test data ########################### 
    # No Group/label is available in final test dataset 
    # CardiacDiagnosisModelTester(EN_clf, test_on_prediction, name='EnsembleOnFinalTestSet', scaler=scaler, save_dir=save_dir, label_available=False,
    #                             prediction_csv=test_on_prediction)
