import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#load core data
Train = pd.read_feather('Data/Train.feather')
Val   = pd.read_feather('Data/Val.feather')
Test  = pd.read_feather('Data/Test.feather')



#load and concat all data in attempt folder
def get_artificialdata(batch):

  StefData = []
  files = os.listdir('Data/attempt{}'.format(batch))
  for file in files:

    data=pd.read_feather('Data/attempt{}/{}'.format(batch,file))
    StefData.append(data)
  return pd.concat(StefData,ignore_index=True)


ArtData1 = get_artificialdata(1)
ArtData2 = get_artificialdata(2)



Columns=pd.DataFrame(Train.columns)
Phi = Columns[Columns[0].str.contains('ConstPhi')].index
Eta = Columns[Columns[0].str.contains('ConstEta')].index
PT = Columns[Columns[0].str.contains('ConstPT')].index

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential




# SPlit data in 3 streams based on types
def F(selector, input_dim=87):
    selector = list(selector)
    output_dim = len(selector)

    # Create fixed weight matrix
    weights = np.zeros((input_dim, output_dim), dtype=np.float32)
    for i, col_idx in enumerate(selector):
        weights[col_idx, i] = 1.0

    # Sanitize name (optional: truncate or hash if long)
    sanitized_name = "FixedSelector_" + "_".join(map(str, selector))

    # Dense layer
    layer = Dense(
        units=output_dim,
        use_bias=False,
        trainable=False,
        name=sanitized_name  # safe name
    )
    layer.build((None, input_dim))
    layer.set_weights([weights])

    return layer






#embedding
def embedding(num_features,embed_size1,embed_size2,indrop):

    x_in = Input(shape=(num_features,))
    x = x_in
    x = Dropout(indrop)(x)
    x = GaussianNoise(0.01)(x)
    x = Reshape((num_features, 1))(x)
    x = Conv1D(embed_size1, kernel_size=1, activation='linear')(x)
    x = GaussianNoise(0.01)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(embed_size2, kernel_size=1)(x)
  #  x = BatchNormalization()(x)
    out=x
    return Model(x_in, out)



#build model
def build_model(num_features, embed_size1, embed_size2, indrop):
  x_in = Input(shape=(87,))
  x = x_in

  F_phi = F(Phi.tolist())
  F_eta = F(Eta.tolist())
  F_pt = F(PT.tolist())


  x1= Flatten()(embedding(len(Phi.tolist()), embed_size1, embed_size2,indrop)( F_phi(x)))
  x2= Flatten()(embedding(len(Eta.tolist()), embed_size1, embed_size2,indrop)(F_eta(x)))
  x3= Flatten()(embedding(len(PT.tolist()), embed_size1, embed_size2,indrop)(F_pt(x)))




  x = Concatenate()([x1,x2,x3])
  x = Flatten()(x)
  x = Dropout(0.2)(x)
  x = Dense(256, activation='linear')(x)
  x  = GaussianNoise(0.01)(x)
  x  = Activation('tanh')(x)
  out = Dense(1,activation='sigmoid')(x)


  return  Model(x_in, out)





from copy import deepcopy

Models = []


#Model1

X_train = pd.concat([Train,Test,Val,ArtData1],ignore_index=True).drop('Label',axis=1).values
Y_train = pd.concat([Train,Test,Val,ArtData1],ignore_index=True).Label.values


classifier = build_model(87,16,8,indrop=0.075)
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
classifier.fit(
          X_train, Y_train,
          epochs=1,
          batch_size=512,
                   verbose=1,

      )
Models.append(deepcopy(classifier))


#Model2

X_train = pd.concat([Train,Test,Val,ArtData1,ArtData2],ignore_index=True).drop('Label',axis=1).values
Y_train = pd.concat([Train,Test,Val,ArtData1,ArtData2],ignore_index=True).Label.values


classifier = build_model(87,16,8,indrop=0.075)
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
classifier.fit(
          X_train, Y_train,
          epochs=1,
          batch_size=512,
                   verbose=1,

      )
Models.append(deepcopy(classifier))





#Model3

X_train = pd.concat([Train,Test,Val,ArtData1],ignore_index=True).drop('Label',axis=1).values
Y_train = pd.concat([Train,Test,Val,ArtData1],ignore_index=True).Label.values


classifier = build_model(87,16,8,indrop=0.125)
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
classifier.fit(
          X_train, Y_train,
          epochs=1,
          batch_size=512,
                 verbose=1,

      )
Models.append(deepcopy(classifier))


#Model4

X_train = pd.concat([Train,Test,Val,ArtData1,ArtData2],ignore_index=True).drop('Label',axis=1).values
Y_train = pd.concat([Train,Test,Val,ArtData1,ArtData2],ignore_index=True).Label.values


classifier = build_model(87,16,8,indrop=0.125)
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
classifier.fit(
          X_train, Y_train,
          epochs=1,
          batch_size=512,
                verbose=1,

      )
Models.append(deepcopy(classifier))





x_in = Input(shape=(87,))
x    = x_in
out  = 0

for model in Models:
      out = out+model(x)/len(Models)

enselble = Model(x_in, out)




enselble.save('model.hdf5')




