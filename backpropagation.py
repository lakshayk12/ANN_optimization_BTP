from keras.models import Sequential
from keras.layers import Dense


def get_model(no_of_hidden_neurons, no_of_features):
    model = Sequential()
    model.add(Dense(output_dim=no_of_features, init='uniform', activation='relu', input_dim=no_of_features))
    model.add(Dense(output_dim=no_of_hidden_neurons, init='uniform', activation='relu'))
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid',
                    # kernel_regularizer=regularizers.l2(0.2)
                    ))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_nn(x_train, x_test, y_train, y_test, no_of_hidden_neurons):
    model = get_model(no_of_hidden_neurons, len(x_train[0]))
    model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=500, verbose=0,
              # validation_split=0.1
              )
    return model


def model(x_train, x_test, y_train, y_test, no_of_hidden_neurons):
    # training with backprop
    model = train_nn(x_train, x_test, y_train, y_test, no_of_hidden_neurons)
    ypre = model.predict_classes(x_test)
    return ypre
