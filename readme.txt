直接运行dignosis.py,有以下问题

from sknn.mlp import Classifier, Convolution, Layer这个引用可能存在报错需要修改一些文件，参考CSDN


"C:\Users\25075\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 772, in <genexpr>
clone(estimator) --> copy.deepcopy(estimator)


"C:\Users\25075\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 244, in <genexpr>
clone(estimator) --> copy.deepcopy(estimator)

整个文件import copy

File "C:\Users\25075\Anaconda3\lib\site-packages\sklearn\ensemble\_voting.py", line 75,
clone(clf) --> copy.deepcopy(clf)


Anaconda3\lib\site-packages\sknn\mlp.py  

368行 fit函数：
添加
X = X.reshape(shape(X)[0],4,5)

429行  predict_proba函数
添加：
X = X.reshape(shape(X)[0],4,5)

466行 predict函数：
return numpy.concatenate(ys, axis=1)改为：
return numpy.concatenate(ys, axis=1).reshape(-1)