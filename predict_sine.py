# Predict the sine function using pure JAX. 
# Date created: April 13, 2022
# Last updated: April 14, 2022 

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import hydra

def generate_sin(num_examples, key):
    '''Generates samples from sin function'''
    x_train = jax.random.uniform(key, shape=(num_examples, 1), minval=0, maxval=2*np.pi)
    y_train = jnp.sin(x_train)
    x_test = jax.random.uniform(key, shape=(num_examples//10, 1), minval=0, maxval=2*np.pi)
    y_test = jnp.sin(x_test)
    return x_train, y_train, x_test, y_test

def init_network(layer_sizes, init_key, scale=1e-2): 
    '''Initialize a multilayer perceptron'''
    params = [] 
    keys = jax.random.split(init_key, len(layer_sizes)-1)
    for input_size, output_size, key in zip(layer_sizes[:-1], layer_sizes[1:], keys): 
        weight_key, bias_key = jax.random.split(key)
        params.append([
            # scaling makes for a better initialization
            scale * jax.random.normal(weight_key, shape=(output_size, input_size)), 
            scale * jax.random.normal(bias_key, shape=(output_size,))
            ]
        )
    return params

def predict(params, input_data):
    '''prediction with a forward pass using a multilayer perceptron'''
    hidden_layers = params[:-1]
    activation = input_data
    # all layers minus the last layer
    for weights, bias in hidden_layers: 
        activation = jax.nn.relu(jnp.dot(weights, activation) + bias)
    
    # last layer
    final_weights, final_bias = params[-1]
    prediction = jnp.dot(final_weights, activation) + final_bias
    return prediction  

def loss(params, x_inputs, y_target_outputs):
    '''Loss function'''
    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    return jnp.mean((batched_predict(params, x_inputs) - y_target_outputs) **2) # MSE loss 

@jax.jit
def update(params, x_train, y_train, lr):
    '''Calculate gradients and update parameters'''
    grads = jax.grad(loss)(params, x_train, y_train)
    return jax.tree_multimap(lambda params_i, grads_i: params_i - lr*grads_i, params, grads) #SGD 

def training_loop(params, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size):
    '''Training loop'''
    num_examples = len(x_train) 
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch_start in range(0, num_examples, batch_size):
            if batch_start + batch_size < num_examples:
                params = update(
                    params, 
                    x_train[batch_start:batch_start+batch_size], 
                    y_train[batch_start:batch_start+batch_size], 
                    lr)
            else: 
                # last batch is smaller if batch size not divisable
                params = update(
                    params, 
                    x_train[batch_start:num_examples], 
                    y_train[batch_start:num_examples], 
                    lr)
        epoch_time = time.time() - start_time
        train_mse = loss(params, x_train, y_train)
        test_mse = loss(params,x_test, y_test)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set MSE {}".format(train_mse))
        print("Test set MSE {}".format(test_mse))
    return params

# TODO: write a save function 
def save_model(params):
    '''Save model parameters'''
    return

@hydra.main(config_path="conf", config_name="config")
def main(args): 
    key = jax.random.PRNGKey(args.seed)
    x_train, y_train, x_test, y_test = generate_sin(args.num_examples, key)
    layers = [args.layers.input_size, args.layers.layer1_size, args.layers.layer2_size, args.layers.output_size]
    network_params = init_network(layers, key)
    network_params = training_loop(
        network_params, 
        x_train, 
        y_train, 
        x_test, 
        y_test, 
        args.learning_rate, 
        args.epochs, 
        args.batch_size)
    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    plt.scatter(x_train, y_train, s=3, label="Training data")
    plt.scatter(x_test, batched_predict(network_params, x_test), s=3, label = "Model Prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main() 