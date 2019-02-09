# utf-8


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Embedding


# obtain data
train = pd.read_csv('train.tsv', sep='\t', header=0, index_col='PhraseId')
test = pd.read_csv('test.tsv', sep='\t', header=0, index_col='PhraseId')
sub = pd.read_csv('sampleSubmission.csv', sep=",", header=0)

# label
Y_train = np.array(train.loc[:, ['Sentiment']])
Y_train = to_categorical(Y_train)

# word to vector
full_text = list(train['Phrase'].values)+list(test['Phrase'].values)
tk = Tokenizer()
tk.fit_on_texts(full_text)   # 生成词典
train_tokenized = tk.texts_to_sequences(train['Phrase'])
test_tokenized = tk.texts_to_sequences(test['Phrase'])
max_len = 50
X_train = pad_sequences(train_tokenized, maxlen=max_len)
X_test = pad_sequences(test_tokenized, maxlen=max_len)   # 最大长度为50 不足50的前方补零
word_dict = tk.word_index

# split data
validation_split = 0.1
nb_validation_samples = int(validation_split * train.shape[0])
x_train = X_train[:-nb_validation_samples]
y_train = Y_train[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
y_val = Y_train[-nb_validation_samples:]

# embedding matrix
embeddings_index = {}
with open('glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(word_dict) + 1, 300))
for word, i in word_dict.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# model
model = Sequential()
model.add(Embedding(len(word_dict) + 1,
                            300,
                            weights =[embedding_matrix],
                            input_length =X_train.shape[1],
                            trainable = False,
                            mask_zero = True))
model.add(LSTM(300,activation='relu', return_sequences=True, input_shape=(50, 300)))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
model.save_weights("final_1.model")
# predictions
test_Predict = model.predict_classes(X_test,verbose=1)

sub.Sentiment = test_Predict
sub.to_csv('sub1.csv',index=False)
sub.head()