
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import accuracy_score as acc
#get_ipython().magic('matplotlib inline')


# In[109]:


file = open("dummy.csv", "r")
file1 = open("data_nn.csv", "w")
file1.write('"Ping Local" "Ping LocalApp-CorpApp" "Ping LocalApp-CorpDB" "Load Average LocalApp" "Load Average LocalDB" "Load Average CORPApp" "TNS ping LocalApp-LocalDB" "TNS ping LocalApp-CorpDB" "Class0" "Class1" "Class2" "Class3"\n')
for line in file:
    l = len(line)
    dat = list(line.split())
    if (dat[1] == 'x' or dat[5] == 'x'):
        line = ""
        for da in dat:
            if (da == 'x'):
                da = '9999999999'
            line+=da+" "
        line += "0 1 0 0\n"
    elif (dat[2] == 'x' or dat[7] == 'x'):
        line = ""
        for da in dat:
            if (da == 'x'):
                da = '9999999999'
            line+=da+" "
        line += "0 0 1 0\n"
    elif (dat[3] == 'x' or dat[4] == 'x' or dat[6] == 'x'):
        line = ""
        for da in dat:
            if (da == 'x'):
                da = '9999999999'
            line+=da+" "
        line += "0 0 0 1\n"
    else:
        line = line[:-1] + " 1 0 0 0" + line[-1]
    file1.write(line)
file.close()
file1.close()


# In[110]:


data = pd.read_csv("data_nn.csv",delimiter = ' ')
x = data.ix[:,:8]
y = data.ix[:,8:]
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=1)


# In[111]:


inputX = x_train.as_matrix()
inputY = y_train.as_matrix()


# In[112]:


inputX


# In[113]:


learning_rate = 0.08
training_epochs = 200000
display_step = 1000
n_samples = inputY.size


# In[114]:


#computation graph
x = tf.placeholder(tf.float32,[None,8])
#weights
W = np.array([[-0.87823284, 1.09219551, 2.0600667, -2.10759234][-0.69028187,  1.00607896, -0.53123796,  0.26810378]
    [1.0145067, 0.38401297, -1.34741151,  0.04980758][0.33991617, -1.39486635,  0.07186026, -0.28025275]
    [1.02296495, 1.54218566, 0.35818812, -0.43832418][0.30262086, -1.86795545, -0.82092249,  0.60501009]
    [-1.34071505 , 0.17339064,  0.88795203 ,0.75498962][-0.17814085, -2.91234589, -1.61736178, -1.92689192]])

#biases
b = np.array([-0.54782993,-0.30752248,1.33939207,0.24376555])

#calculations
y_values = tf.add(tf.matmul(x,W),b)

y = tf.nn.softmax(y_values)

y_ = tf.placeholder(tf.float32,[None,4])


# In[115]:


cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[116]:


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[117]:


#print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)) #, \"W=", sess.run(W), "b=", sess.run(b)

print ("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


# In[118]:


test_x = x_test.as_matrix()
result = sess.run(y, feed_dict={x: x_test})
A = np.asarray(result).reshape(-1)
test_y = y_test.as_matrix()
B = np.asarray(test_y).reshape(-1)
print (str(((A==B).sum()/A.size)*100)+"% accuracy ")
