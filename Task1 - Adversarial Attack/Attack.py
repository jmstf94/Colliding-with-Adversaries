import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the entire model
model = load_model('Data/Task1/best_model.hdf5')
def distance(a,b):
  return np.abs(a-b).mean(axis=1)

def accuracy(a,b):
  return (a.round()==b.round()).mean()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU


X= pd.read_feather('Data/Task1/Clean_Test_Public.feather').drop('Label',axis=1)
Y= pd.read_feather('Data/Task1/Clean_Test_Public.feather')['Label']


import tensorflow as tf
import numpy as np
def thresholded_bce(y_true, y_pred):
   # return  tf.square(y_pred-0.5)
    # Compute standard binary cross-entropy (with logits=False, assuming sigmoid probs)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Define threshold (log(2))
    #threshold = tf.math.log(2.0001)  # ≈ 0.6931
    threshold = tf.math.log(2.0) +0.1  # ≈ 0.6931
    bce =  bce - threshold
    # Return bce if bce > log(2), else 0
    #bce = tf.where(bce < -0.01, bce, bce*0.1)
    return tf.where(bce < 0, bce, 0)

def attack_model_tf(model, X_a,X_base, y_true,  alpha=0.01, beta=0.00001, steps=40,red_freq=5000,red_rate=0.5):
    X_base = tf.convert_to_tensor(X_base, dtype=tf.float32)
    X_adv = tf.convert_to_tensor(X_a, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_true = tf.reshape(y_true, (-1, 1))

    random_lr = tf.random.uniform(shape=tf.shape(y_true), minval=0.1, maxval=1.0)



    X_orig  = tf.identity(X_base)
    #X_adv = tf.identity(delta)

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            X_adv = X_adv
            logits = model(X_adv, training=False)
            probs = logits#tf.sigmoid(logits)

            pred_class = tf.cast(probs >= 0.5, tf.float32)
            flip_mask = tf.cast(pred_class != y_true, tf.float32)

            fooled = pred_class != y_true


            bce = tf.keras.losses.binary_crossentropy(y_true, probs)
            bce = thresholded_bce(y_true, probs)
            loss_bce = tf.reduce_mean(bce)

            d = tf.abs(X_adv - X_orig)
            D = tf.reduce_sum(d*flip_mask )

            loss =   loss_bce -D*beta

        grads = tape.gradient(loss, X_adv)
        X_adv = X_adv + alpha *grads*random_lr
        if  step%red_freq==(red_freq-1):
          alpha = alpha*red_rate


       # X_adv = tf.clip_by_value(X_adv, tf.reshape(min_vals, [1, -1]), tf.reshape(max_vals, [1, -1]))





    return X_adv.numpy()




def evaluate_score_tf(model, x_clean, x_adv, y_true):

    probs_clean = (model(x_clean))
    adv_pred = model.predict(x_adv,batch_size=100000,verbose=0)
    adv_pred = (np.squeeze(adv_pred) >= 0.5).astype(int)


    FR = 1 - accuracy_score(y_true, adv_pred)

    D = np.mean(np.mean(np.abs(x_clean - x_adv), axis=1))

    score = FR * np.exp(-20 * D)




    print(D)
    return {"FR": FR, "D": np.round(D,5),"ED":score, "S": np.round(score,5)}





MAX = X.max().values.tolist()
MIN = X.min().values.tolist()
min_vals = tf.constant(MIN, dtype=tf.float32)
max_vals = tf.constant(MAX, dtype=tf.float32)







def pairwise_l1_distance(A, B):
    return np.mean(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
D =  pd.DataFrame( pairwise_l1_distance(X.values,X.values ),index = X.index,columns=X.index) + 1-np.abs(Y.values-Y.values.reshape(-1,1))



def Up(x,n=4):
  if type(x)==list:
    out=pd.concat(x)
  else:
    out=pd.concat([x]*n)
  out=out.reset_index().drop('index',axis=1)
  return out
def  Mix(A,B):
  r= np.random.uniform(0,1,(A.shape[0],1))
  return A*r+B*(1-r)
def  Shift(A,B,Min=-1,Max=1):
  r= np.random.uniform(Min,Max,(A.shape[0],1))
  return A*(1-np.abs(r))+B*r
def  RanSelect(A,B,p=0.5):
  r= np.random.choice([0,1],(A.shape[0],1),p=[p,1-p] )
  return A*r+B*(1-r)
def  AddNoise(X,s=0.0001):
  return X+np.random.normal(0,s,X.shape)

class ToKnown:
  def __init__(self,D,X):
    self.D = D# pd.DataFrame( pairwise_l1_distance(X.values,X.values ),index = X.index,columns=X.index) + 1-np.abs(Y.values-Y.values.reshape(-1,1))
    self.X = X
  def __call__(self,X):
    n=len(X)//len(self.X)
    DT  =  self.D+np.random.normal(0,0.02,D.shape)
    if n>0:
      mylist = []
      for i in range(n):

        self.X.loc[DT.idxmin(axis=1).values].values
        mylist.append(self.X*0+self.X.loc[DT.idxmin(axis=1).values].values)



      return Mix(Up(mylist,n=n),X)
    else:
            corr =  self.X.loc[DT.idxmin(axis=1).values].values
            corr = corr[X.index]
            return Mix(corr,X)



def IsBetter(X,Y,A,B):
  n=len(B)//len(A)
  size = len(A)
  PA  =  model.predict(A,batch_size=100000,verbose=0).round().reshape(-1)
  PB  =  model.predict(B,batch_size=100000,verbose=0).round().reshape(-1)
  PX  = model.predict(X,batch_size=100000,verbose=0).round().reshape(-1)

  for i in range(n):


    I  = Y [   ( (PB[i*size:(i+1)*size] !=Y.values.round()) &  (distance(A,X)>=distance(B.loc[i*size:(i+1)*size-1].values,X)) ) | (  (PB[i*size:(i+1)*size] !=Y.values.round()) &  (PB[i*size:(i+1)*size]!=PA) )              ].index
    A.loc[I]=B.loc[I+i*size].values*1
    PA  =  model.predict(A,batch_size=100000,verbose=0).round().reshape(-1)
  return A

def Clean(X_b,X,Y):
  X_b =np.where(X_b<X.max(),X_b,X.max())+X_b*0
  X_b =np.where(X_b>X.min(),X_b,X.min())+X_b*0
  return X_b
tk =ToKnown(D,X)


X_a =  X*1
X_b = Up(X,20)*1

print(1,np.round(1,3),'FR: ',evaluate_score_tf(model, X.values,X_a.values, Y)['FR'],'S: ',evaluate_score_tf(model, X.values, X_a.values, Y)['S'])


for i in range(150):

  alpha  = 512*2** -(i//5)






  X1  =   X   * 1.0
  X2  =   X_a * 1.0
  X3  =   Up ([ Mix  ( X1 , X_a  ) for i in range(4)])
  X4  =   AddNoise( Up ([ Mix  ( X1 , X_a  ) for i in range(4)]),s=0.001*(i%5) )


  X1  = Up([X1,X2,X3,X4])
  X2  = tk(X1)


  X_b =  RanSelect  ( X_b , Up([X1,X2]),0.5 )


  X_b = attack_model_tf(model,X_b, Up(X,20), Up(Y,20), alpha=alpha, steps=2500,red_freq=2500) +Up(X,20)*0

  #polish
  X_b = attack_model_tf(model,X_b, Up(X,20), Up(Y,20), alpha=0.01, steps=250,red_freq=250,beta=0) +Up(X,20)*0

  #X_b = Clean(X_b,X,Y)
  X_d=IsBetter(X,Y,X_a,X_b)

  ScoreA= evaluate_score_tf(model, X.values,X_a.values, Y)
  ScoreD= evaluate_score_tf(model, X.values,X_d.values, Y)
  if ScoreD['S']>ScoreA['S']:
    X_a=X_d
    X_a.to_csv('AttackU.csv')
  else:
    X_a=X_a


  print(alpha,np.round(alpha,3),'FR: ',evaluate_score_tf(model, X.values,X_a.values, Y)['FR'],'S: ',evaluate_score_tf(model, X.values, X_a.values, Y)['S'])
