import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
iris=pd.read_csv('/Users/Gobind/TransAtlantic/1TF_dataScience/Datasets/IRIS.csv')
#print(iris)
iris.species=iris.species.replace(to_replace=['Iris-setosa',
                                              'Iris-versicolor',
                                              'Iris-virginica'],
                                  value=[0,1,2])
#int64 has been substituted here
train,test=train_test_split(iris,test_size=0.2)

#print(len(train))
#print(len(test))
#print(train)

train_x=train.drop(['species'],axis=1).values
print(train_x)
train_y=train['species'].values
test_x=test.drop(['species'],axis=1).values
test_y=test.species.values
#print(train_y)
print("fdddsvjfdjsjvdslj")
#print(x[:20])
#train_index=np.random.choice(len(x),int(len(x)*0.8), replace=False)

#max=np.max(train_x)
#print(max)
#def ggg(data):
#    return (data+1)
#
#print("gdfgsbsbsbfgnb")
#print(ggg(train_x))

def normalize_data(data):
    col_max=np.max(data,axis=0)
    col_min=np.min(data,axis=0)
    return ((data-col_min)/(col_max-col_min))

normalize_train_x=normalize_data(train_x)
normalize_train_y=pd.get_dummies(train_y, 
                     drop_first=False).values
normalize_test_x=normalize_data(test_x)
normalize_test_y=pd.get_dummies(test_y, 
                     drop_first=False).values

#ohe=OneHotEncoder(sparse=False,categories='auto')
#encoded_columns=ohe.fit_transform(iris['species'])
#
#
#
#
#
#
#
#
print(pd.get_dummies(train_y, drop_first=False).values)
print(train_y)
print(normalize_train_y)
print(normalize_test_x.shape)
#
#
#
#
#
#
weights=tf.Variable(tf.random_normal([4,3],mean=0,stddev=0.1))
bias=tf.Variable(tf.random_normal([1,3],mean=0,stddev=0.1))
print("the shape of bias is: ")
shape=bias.get_shape()
print(shape)
#model_eq=tf.matmul(train_x,weights)+bias
#activation_op=tf.nn.sigmoid(model_eq)
#cost_op=tf.nn.l2_loss(activation_op-train_y)
#optimizer=tf.train.GradientDescentOptimizer(0.05)
#training_op=optimizer.minimize(cost_op)
placeholder_x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
placeholder_y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

model_eq=tf.matmul(placeholder_x,weights)+bias

cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=model_eq,
        labels=placeholder_y))


learningRate = tf.train.exponential_decay(learning_rate=0.005,
                                          global_step= 1,
                                          decay_steps=normalize_train_x.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)


learning_rate=0.003
batch_size=30
iter_num=1200

optimizer=tf.train.GradientDescentOptimizer(learning_rate)
training_op=optimizer.minimize(cost)




#prediction=tf.round(tf.sigmoid(model_eq-placeholder_y))
correct=tf.cast(tf.equal(tf.argmax(model_eq,axis=1),
                         tf.argmax(placeholder_y,axis=1)),dtype=tf.float32)
accuracy=tf.reduce_mean(correct)

# Start training model
# Define the variable that stores the result
loss_trace = []
train_acc = []
test_acc = []


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(iter_num):
    # Generate random batch index
#    batch_index = np.random.choice(len(normalize_train_x), 
#                                   size=batch_size)
#    batch_train_X = normalize_train_x[batch_index]
#    batch_train_y = normalize_train_y[batch_index]
    sess.run(training_op, feed_dict={placeholder_x: normalize_train_x, 
                                     placeholder_y: normalize_train_y})
    temp_loss = sess.run(cost, feed_dict={placeholder_x: normalize_train_x, 
                                          placeholder_y: normalize_train_y})
    

    
    
# convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, 
                              feed_dict= {
                                      placeholder_x: normalize_train_x, 
                                      placeholder_y: normalize_train_y})
    temp_test_acc = sess.run(accuracy, 
                             feed_dict={
                                     placeholder_x: normalize_test_x, 
                                     placeholder_y: normalize_test_y})
    # recode the result
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    # output
    print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))

# Visualization of the results
# loss function
plt.plot(loss_trace,label='loss')
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')

plt.show()

# accuracy
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()
