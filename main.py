import matplotlib.pyplot as plt

print('Starting Self Car')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utlis import *
from sklearn.model_selection import train_test_split


###STEP 1
path='mydata'
data=importDatainfo(path)

###STEP 2

data=balanceData(data,display=True)

# ###STEP 3
imagespath,steering=loaddata(path,data)
# print(imagepath[0],steering[0])

###STEP 4(TRAINING AND VALIDATION)
xtrain,xval,ytrain,yval=train_test_split(imagespath,steering,test_size=0.2,random_state=20)
print('total training images',len(xtrain))
print('total validation images',len(xval))


####STEP 5(AUGMENTATION)


###STEP 6(PREPROCESSING)


##STEP 7(BATCH GENERATOR)


###STEP 8(CREATING A MODEL)
model = createmodel()
model.summary()

##STEP 9 (TRAINING DATA)

history=model.fit(batchgen(xtrain,ytrain,200,1),steps_per_epoch=200,epochs=15,
                  validation_data=batchgen(xval,yval,100,0),validation_steps=200)

###STEP 10
model.save('model.h5') #saves the architecture of the model
print('model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.ylim([0,1])
plt.title('loss')
plt.xlabel('epoch')
plt.show()
#

