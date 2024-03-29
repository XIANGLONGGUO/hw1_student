# 训练结果和说明
21051028-郭相隆
## 1
第一次实验使用学习了lr=0.01，epoch=100，隐藏层维度：25
{'name': 'SGD', 'lr': 0.01, 'clip_norm': 1.0, 'momentum': 0.9}
结果：
训练,验证集最佳：
Epoch 98 Training Loss: 0.0554 Training Accuracy: 0.984 Val Loss: 0.0558 Val Accuracy: 1.0
测试集结果：
Test Loss: 0.3021 Test Accuracy: 0.88
## 2
lr=0.005,epoch=200,维度：25
{'name': 'SGD', 'lr': 0.005, 'clip_norm': 1.0, 'momentum': 0.9}
训练，验证集最佳：
epoch 183 Training Loss: 0.0522 Training Accuracy: 0.992 Val Loss: 0.019 Val Accuracy: 1.0
测试集结果：
Test Loss: 0.1393 Test Accuracy: 0.96
## 3 
lr=0.02,epoch=70,维度：25
{'name': 'SGD', 'lr': 0.02, 'clip_norm': 1.0, 'momentum': 0.9}
训练，验证集最佳：
Epoch 66 Training Loss: 0.0514 Training Accuracy: 0.992 Val Loss: 0.0606 Val Accuracy: 1.0
测试集结果：
Test Loss: 0.1257 Test Accuracy: 0.96
## 4 
lr=0.02 ,epoch=100,维度：30
{'name': 'SGD', 'lr': 0.02, 'clip_norm': 1.0, 'momentum': 0.9}
训练，验证集最佳：
Epoch 82 Training Loss: 0.044 Training Accuracy: 0.984 Val Loss: 0.0048 Val Accuracy: 1.0
测试集结果：
Test Loss: 0.277 Test Accuracy: 0.96
## 5 
lr=0.01,epoch=150,维度：30
{'name': 'SGD', 'lr': 0.01, 'clip_norm': 1.0, 'momentum': 0.9}
训练，验证集最佳：
Epoch 141 Training Loss: 0.0389 Training Accuracy: 1.0 Val Loss: 0.0047 Val Accuracy: 1.0
测试集结果：
Test Loss: 0.123 Test Accuracy: 0.96
## 说明
### 学习率
我认为第一次实验中表现不佳的原因是由于训练epoch过少，未完全收敛，对于第二次(lr=0.005)也有相同的问题，他收敛的更好，但是他的效果没有第三次（lr=0.02)好代表了他的epoch也是较少的。
### 维度
从后面两个我们可以看到，在第四次训练中出现了一定的过拟合现象。其表现不佳，我们暂时得到以下的结论来说明现象
更大的隐藏层维度（例如维度=30）可能会增加模型的容量，使其更能够拟合训练数据，但也可能导致过拟合。
更小的隐藏层维度（例如维度=25）可能会限制模型的容量，可能导致欠拟合，但有助于防止过拟合。