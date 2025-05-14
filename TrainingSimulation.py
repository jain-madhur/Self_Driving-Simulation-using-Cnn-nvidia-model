print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utlis import *
from sklearn.model_selection import train_test_split

#### Step 1
path= 'myData'
data=importDatainfo(path)

#### Step 2
data = balanceData(data,display= False)

#### Step 3
imagesPath , Steerings =  loadData(path,data)
#print(imagesPath[0],Steerings[0])

#### Step 4
xTrain , xVal , yTrain ,yVal = train_test_split(imagesPath,Steerings,test_size=0.2,random_state=5)
print('Total Training Images:', len(xTrain))
print('Total validation Images:' , len(xVal))

#### Step 5
#DATA Augmentation

#### Step 6
# BATCH GENRATOR

#### Step 7
# Image perprocessing

#### Step 8
model= createModel()
model.summary()

### STEP 9
history = model.fit(batchGen(xTrain,yTrain,100,1), steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batchGen(xVal, yVal, 100, 0),
                                  validation_steps=200)

#### Step 10
model.save('model.h5')
print ('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim()
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
