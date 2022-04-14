# Use JAX to predict the sine function. 
# Date created: April 13, 2022
# Last updated: April 14, 2022 

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import hydra

def generate_sin(N, key):
    '''Generates samples from sin function'''
    xTr = jax.random.uniform(key, shape=(N, 1), minval=0, maxval=2*np.pi)
    yTr = jnp.sin(xTr)
    xTe = jax.random.uniform(key, shape=(N//10, 1), minval=0, maxval=2*np.pi)
    yTe = jnp.sin(xTe)
    return xTr, yTr, xTe, yTe

def init_MLP(layer_widths, init_key, scale=1e-2): 
    '''Initialize MLP'''
    params = [] 
    keys = jax.random.split(init_key, len(layer_widths)-1)
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys): 
        weight_key, bias_key = jax.random.split(key)
        params.append([
            # scaling makes for a better initialization
            scale * jax.random.normal(weight_key, shape=(out_width, in_width)), 
            scale * jax.random.normal(bias_key, shape=(out_width,))
            ]
        )
    return params

def predict_MLP(params, x):
    '''Predict using the MLP'''
    hidden_layers = params[:-1]
    activation = x
    # all layers minus the last layer
    for w,b in hidden_layers: 
        activation = jax.nn.relu(jnp.dot(w, activation) + b)
    
    # last layer
    final_w, final_b = params[-1]
    prediction = jnp.dot(final_w, activation) + final_b
    return prediction  

def loss(params, x, y):
    '''Loss function'''
    batched_predict_MLP = jax.vmap(predict_MLP, in_axes=(None, 0))
    return jnp.mean((batched_predict_MLP(params, x) - y) **2) # MSE loss 

@jax.jit
def update(params, x, y, lr):
    '''Calculate gradients and update parameters'''
    grads = jax.grad(loss)(params, x, y)
    return jax.tree_multimap(lambda p, g: p - lr*g, params, grads) #SGD 

def training_loop(params, x, y, x_test, y_test, lr, num_epochs, batch_size):
    '''Training loop'''
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(0, len(x), batch_size):
            if i + batch_size < len(x):
                params = update(params, x[i:i+batch_size], y[i:i+batch_size], lr)
            else: 
                # last batch is smaller if batch size not divisable
                params = update(params, x[i:len(x)], y[i:len(x)], lr)
        epoch_time = time.time() - start_time
        train_acc = loss(params, x, y)
        test_acc = loss(params,x_test, y_test)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
    return params

# TODO: write a save function 

@hydra.main(config_path="conf", config_name="config")
def main(args): 
    key = jax.random.PRNGKey(args.seed)
    xTr, yTr, xTe, yTe = generate_sin(args.num_examples, key)
    MLP_params = init_MLP([1, 128, 128, 1], key)
    batched_predict_MLP = jax.vmap(predict_MLP, in_axes=(None, 0))
    MLP_params = training_loop(MLP_params, xTr, yTr, xTe, yTe, args.lr, args.epochs, args.batch_size)
    plt.scatter(xTr, yTr, s=3, label="Training data")
    plt.scatter(xTe, batched_predict_MLP(MLP_params, xTe), s=3, label = "Model Prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main() 