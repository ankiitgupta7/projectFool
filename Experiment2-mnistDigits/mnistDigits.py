import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

# Parse command-line arguments
model_name = sys.argv[1]
target_digit = int(sys.argv[2])
generation_interval = int(sys.argv[3])
replicate = int(sys.argv[4])  # New argument for the replicate number

# Define the models
models = {
    'SVM': SVC(gamma='scale', probability=True),
    'RF': RandomForestClassifier(),
    'GBM': GradientBoostingClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# Define CNN and RNN models
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(50, return_sequences=True),
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Train the models and save train/test accuracies and model parameters
model_params = {}
if model_name in models:
    model = models[model_name]
    model.fit(X_train_flat, np.argmax(y_train, axis=1))
    train_accuracy = model.score(X_train_flat, np.argmax(y_train, axis=1))
    test_accuracy = model.score(X_test_flat, np.argmax(y_test, axis=1))
    model_params = model.get_params()
    trained_model = model
elif model_name == "CNN":
    model = create_cnn_model((28, 28, 1))
    model.fit(X_train[..., np.newaxis], y_train, epochs=10, batch_size=32, verbose=0)
    train_accuracy = model.evaluate(X_train[..., np.newaxis], y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test[..., np.newaxis], y_test, verbose=0)[1]
    model_params = model.count_params()
    trained_model = model
elif model_name == "RNN":
    model = create_rnn_model((28, 28))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    model_params = model.count_params()
    trained_model = model
else:
    raise ValueError(f"Unknown model name: {model_name}")

# Save accuracies and model parameters to CSV
output_dir = os.path.join('fit_mnist_multiple_models', model_name)
os.makedirs(output_dir, exist_ok=True)
csv_file_path = os.path.join(output_dir, f'model_{model_name}_test_train_accuracy_and_parameters.csv')

df = pd.DataFrame({
    'model_name': [model_name],
    'train_accuracy': [train_accuracy],
    'test_accuracy': [test_accuracy],
    'model_params': [model_params]
})
df.to_csv(csv_file_path, index=False)

# Evolutionary algorithm setup
input_shape = X_train[0].shape

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, input_shape[0] * input_shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# def evaluate_model(individual, model, input_shape, target_digit):
#     image = np.array(individual).reshape(1, -1)
#     if isinstance(model, tf.keras.Model):
#         image = np.array(individual).reshape(1, *input_shape, 1) if len(input_shape) == 2 else np.array(individual).reshape(1, *input_shape)
#         probabilities = model.predict(image)
#     else:
#         probabilities = model.predict_proba(image)
#     return probabilities[0][target_digit],

def evaluate_model(individual, model, input_shape, target_digit):
    # Convert the individual (flattened image) into a numpy array and reshape it to match the model's input format
    # Initially reshape as a flat array (1 sample, many features)
    image = np.array(individual).reshape(1, -1)
    
    # Check if the model is a TensorFlow Keras model (e.g., CNN or RNN)
    if isinstance(model, tf.keras.Model):
        # If input shape is 2D (e.g., for CNN), reshape into 4D (batch, height, width, channels)
        if len(input_shape) == 2:
            image = np.array(individual).reshape(1, *input_shape, 1)  # Adding the channel dimension for grayscale
        else:
            # Otherwise, reshape to the provided 3D shape (used for RNNs or other cases)
            image = np.array(individual).reshape(1, *input_shape)
        
        # Predict the probabilities for each digit (0-9) using the Keras model
        probabilities = model.predict(image)
    else:
        # For scikit-learn models, use predict_proba to get the class probabilities
        probabilities = model.predict_proba(image)
    
    # Return the probability for the target digit (0-9) as a tuple (required by DEAP)
    return probabilities[0][target_digit],


def run_evolution(toolbox, ngen, model, input_shape, target_digit, output_subdir, generation_interval):
    os.makedirs(output_subdir, exist_ok=True)
    population = toolbox.population(n=100)  # population size 100
    generation_images = []
    generation_accuracies = []

    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        best_image = np.array(best_ind).reshape(input_shape)
        best_accuracy = toolbox.evaluate(best_ind)[0]

        if (gen) % generation_interval == 0:
            generation_images.append(best_image)
            generation_accuracies.append(best_accuracy)

            if len(generation_images) % 100 == 0:
                fig, axs = plt.subplots(10, 10, figsize=(25, 25))
                fig.subplots_adjust(hspace=0.5)  # Add vertical space between images
                for i in range(100):
                    row = i // 10
                    col = i % 10
                    idx = len(generation_images) - 100 + i
                    axs[row, col].imshow(generation_images[idx], cmap='gray')
                    axs[row, col].set_title(f'Gen {(idx)*generation_interval}\nConf: {generation_accuracies[idx]:.4f}', fontsize=15)
                    axs[row, col].axis('off')
                plt.savefig(os.path.join(output_subdir, f'digit_{target_digit}_rep{replicate}_gen_{gen}.png'))
                plt.close(fig)

    return generation_images, generation_accuracies

toolbox.register("evaluate", lambda ind: evaluate_model(ind, trained_model, input_shape, target_digit))
model_dir = os.path.join('fit_mnist_multiple_models', model_name)
digit_dir = os.path.join(model_dir, f'digit_{target_digit}', f'replicate_{replicate}')
run_evolution(toolbox, ngen=200000, model=trained_model, input_shape=input_shape, target_digit=target_digit, output_subdir=digit_dir, generation_interval=generation_interval)
