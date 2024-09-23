import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from deap import base, creator, tools
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Parse command-line arguments
model_name = sys.argv[1]
target_category = int(sys.argv[2])
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
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape[0], input_shape[1] * input_shape[2])),
        tf.keras.layers.SimpleRNN(128, return_sequences=True),
        tf.keras.layers.SimpleRNN(128),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Train the models and save train/test accuracies and model parameters
model_params = {}
if model_name in models:
    model = models[model_name]
    model.fit(X_train_flat, y_train)
    train_accuracy = model.score(X_train_flat, y_train)
    test_accuracy = model.score(X_test_flat, y_test)
    model_params = model.get_params()
    trained_model = model
elif model_name == "CNN":
    model = create_cnn_model((32, 32, 3))
    model.fit(X_train, y_train_one_hot, epochs=10, batch_size=64, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train_one_hot, verbose=0)[1]
    test_accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)[1]
    model_params = model.count_params()
    trained_model = model
elif model_name == "RNN":
    # Flatten spatial dimensions for RNN
    X_train_rnn = X_train.reshape(-1, X_train.shape[1], X_train.shape[2] * X_train.shape[3])
    X_test_rnn = X_test.reshape(-1, X_test.shape[1], X_test.shape[2] * X_test.shape[3])
    model = create_rnn_model(X_train[0].shape)
    model.fit(X_train_rnn, y_train_one_hot, epochs=10, batch_size=64, verbose=0)
    train_accuracy = model.evaluate(X_train_rnn, y_train_one_hot, verbose=0)[1]
    test_accuracy = model.evaluate(X_test_rnn, y_test_one_hot, verbose=0)[1]
    model_params = model.count_params()
    trained_model = model
else:
    raise ValueError(f"Unknown model name: {model_name}")

# Save accuracies and model parameters to CSV
output_dir = os.path.join('fit_cifar10_multiple_models', model_name)
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
toolbox.register("attr_float", lambda: np.random.rand())
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, np.prod(input_shape))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
def valid_pixel_mutation(individual, indpb):
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            individual[i] += np.random.normal(0, 0.1)
            individual[i] = min(max(individual[i], 0), 1)  # Clamp between 0 and 1
toolbox.register("mutate", valid_pixel_mutation, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_model(individual, model, input_shape, target_category):
    image = np.array(individual).reshape(1, *input_shape)
    if isinstance(model, tf.keras.Model):
        probabilities = model.predict(image, verbose=0)
    else:
        image_flat = image.reshape(1, -1)
        probabilities = model.predict_proba(image_flat)
    return probabilities[0][target_category],

def run_evolution(toolbox, ngen, model, input_shape, target_category, output_subdir, generation_interval):
    os.makedirs(output_subdir, exist_ok=True)
    population = toolbox.population(n=50)  # Adjusted population size due to computational constraints
    generation_images = []
    generation_accuracies = []

    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update population
        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        best_image = np.array(best_ind).reshape(input_shape)
        best_accuracy = toolbox.evaluate(best_ind)[0]

        if gen % generation_interval == 0:
            generation_images.append(best_image)
            generation_accuracies.append(best_accuracy)

            # Save images every 100 intervals
            if len(generation_images) % 100 == 0:
                fig, axs = plt.subplots(10, 10, figsize=(25, 25))
                fig.subplots_adjust(hspace=0.5)
                for i in range(100):
                    row = i // 10
                    col = i % 10
                    idx = len(generation_images) - 100 + i
                    axs[row, col].imshow(generation_images[idx])
                    axs[row, col].set_title(f'Gen {(idx)*generation_interval}\nConf: {generation_accuracies[idx]:.4f}', fontsize=15)
                    axs[row, col].axis('off')
                plt.savefig(os.path.join(output_subdir, f'category_{target_category}_rep{replicate}_gen_{gen}.png'))
                plt.close(fig)

    return generation_images, generation_accuracies

# Register the evaluation function
toolbox.register("evaluate", lambda ind: evaluate_model(ind, trained_model, input_shape, target_category))

# Set up directories
model_dir = os.path.join('fit_cifar10_multiple_models', model_name)
category_dir = os.path.join(model_dir, f'category_{target_category}', f'replicate_{replicate}')

# Run the evolutionary algorithm
run_evolution(
    toolbox,
    ngen=1000000,
    model=trained_model,
    input_shape=input_shape,
    target_category=target_category,
    output_subdir=category_dir,
    generation_interval=generation_interval
)