import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random

from tensorflow import keras
from keras import Sequential
from keras import layers
from keras.layers import Dense ,Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())
words = []
classes = []
document = []
ignore_pattern = ['.' , '?' , '!' , ',']

print("intents:", intents)
for ele in intents['intents']:

    for pattern in ele['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        document.append((word_list ,ele['tag']))
        if ele['tag'] not in classes:
            classes.append(ele['tag'])

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_pattern]
words = sorted(set(words))

classes= sorted(set(classes))


pickle.dump(words , open('words.pkl' , 'wb'))
pickle.dump(classes , open('classes.pkl' , 'wb'))

training = []
output_empty = [0]*len(classes)


print("documnt::", document)
for doc in document:
    bag = []
    word_pattern = doc[0]
    # word_pattern = [lemmatizer.lemmatize(word) for word in word_pattern ]

    for word in words:
        bag.append(1) if word  in word_pattern else bag.append(0)
    
    output_row = list(output_empty)
    print("doc::", doc)
    output_row[classes.index(doc[1])]= 1

    training.append([bag , output_row])


random.shuffle(training)


train_x = list(training[: , 0])
train_y = list(training[:, 1])


model = Sequential()
model.add(Dense(128 ,  input_shape = (len(train_x[0]),) , activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64 , activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]) , activation= 'softmax') )


sgd = SGD(learning_rate= 0.05 , weight_decay= 1e-6 , momentum= 0.9 , nesterov= True)
model.compile(loss = 'categorial_crossentropy' , optimizer= sgd , metrics= ['accuracy'])
model.fit(np.array(train_x) , np.array(train_y) , epochs= 200 , batch_size= 5 )
model.save('ChatBot.model')

print("----------Model Saved-------------")

