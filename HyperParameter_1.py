# Use scikit-learn to grid search over Keras model hyperparams
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Define some classification model hyper params to tune
hidden_layers = [
    (1),
    (0.5),
    (2),
    (1, 1),
    (0.5, 0.5),
    (2, 2),
    (1, 1, 1),
    (1, 0.5, 0.5),
    (0.5, 1, 1),
    (1, 0.5, 0.25),
    (1, 2, 1),
    (1, 1, 1, 1),
    (1, 0.66, 0.33, 0.1),
    (1, 2, 2, 1)
]

batch_size = [16, 32, 64, 128, 256, 512]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation = ['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']
epochs = [100, 200, 500, 1000]
lr = [0.01, 0.001, 0.0001, 0.00001]
momentum = [0.0, 0.3, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
weight_constraint = [1, 3, 5]
dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]

# Define these into a dictionary
model_params = dict(hidden_layers=hidden_layers, batch_size=batch_size, optimizer=optimizer, activation=activation, epochs=epochs, learning_rate=lr,
                    momentum=momentum, init_mode=init_mode, weight_constraint=weight_constraint, dropout_rate=dropout_rate)



# Define a function that creates our model structure for the current param set
def make_deep_learning_classifier(hidden_layers=None, batch_size=32, num_cols=None, learning_rate=0.001, optimizer='Adadelta', momentum=0.0, dropout_rate=0.2, 
                                  weight_constraint=0, final_activation='sigmoid', feature_learning=False, activation='elu', init_mode='normal',
                                  epochs=epochs):

    if hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with, it is simply values of 0.5, 1, and 2, 
    # which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in hidden_layers:
        scaled_layers.append(min(int(num_cols * layer), 10))

    model = Sequential()

    # There are times we will want the output from our penultimate layer, not the final layer, so give it a name that makes the penultimate layer easy to find
    # later (if I even use it. This has to do with feature_learning which I'm omitting in this example)
    model.add(Dense(scaled_layers[0], input_dim=num_cols, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        model.add(get_activation_layer(activation))

    model.add(Dense(scaled_layers[-1], kernel_initializer=kernel_initializer, name='penultimate_layer', kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    model.add(Dense(1, kernel_initializer=kernel_initializer, activation=final_activation))
    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(optimizer), metrics=['accuracy', 'poisson'])
    print(model.summary())
    return model



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# How many columns (nodes) to pass to our first layer
num_cols = X.shape[1]

model = KerasClassifier(build_fn=make_deep_learning_classifier, num_cols=num_cols,  **model_params) # feature_learning=self.feature_learning,

grid = GridSearchCV(estimator=model, param_grid=model_params, n_jobs=-2, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))