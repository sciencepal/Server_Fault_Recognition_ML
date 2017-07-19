
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import accuracy_score as acc
#get_ipython().magic('matplotlib inline')


# In[52]:


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


# In[61]:


data = pd.read_csv("data_nn.csv",delimiter = ' ')
x = data.ix[:,:8]
y = data.ix[:,8:]
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=1)


# In[62]:


inputX = x_train.as_matrix()
inputY = y_train.as_matrix()


# In[55]:


learning_rate = 0.08
training_epochs = 200000
display_step = 1000
n_samples = inputY.size


# In[56]:


#computation graph
x = tf.placeholder(tf.float32,[None,8])
#weights
W = tf.Variable([[0.15463702, -1.21701765, 0.52780521, -1.4738549],
                [0.42833516, -0.04660674, 0.82068664, 0.09615696],
                [0.62039953, 0.93298858, -0.19275554, -1.06650257],
                [-0.05696351, -1.84619224, -0.93848866, -0.38569927],
                [-0.67261547, 0.19050605, -0.29760489, -0.62254256],
                [-1.1083765, -2.41506457, 0.25277814, -2.16972232],
                [0.35110912, 0.16467018, 2.88074422, -0.33394173],
                [1.98250139, -0.09985847, -0.04889834, -0.10028272]])
#W2 = tf.Variable(tf.random_normal([16,4]))
#biases
b = tf.Variable([0.13422392, -0.29879299, 0.13994822, 1.51326454])
#b2 = tf.Variable(tf.random_normal([4]))

#calculations
y_values = tf.add(tf.matmul(x,W),b)

y = tf.nn.softmax(y_values)

y_ = tf.placeholder(tf.float32,[None,4])


# In[57]:


cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[58]:


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[59]:


#print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)) #, \"W=", sess.run(W), "b=", sess.run(b)

print ("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


# In[64]:


test_x = x_test.as_matrix()
result = sess.run(y, feed_dict={x: x_test})
A = np.asarray(result).reshape(-1)
test_y = y_test.as_matrix()
B = np.asarray(test_y).reshape(-1)
print (str(((A==B).sum()/A.size)*100)+"% accuracy ")
np.savetxt("output_nn.csv", result , delimiter=" ",fmt= "%0.3f")
