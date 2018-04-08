import tensorflow as tf
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, rmse, rmsle, \
    l2_regularizer, build_submission_name, plot_model_cost
from dataset import mini_batches


def model(train_set, train_labels, validation_set, validation_labels, layers_dims, learning_rate=0.01, num_epochs=15001,
          print_cost=True, plot_cost=True, l2_beta=0., keep_prob=1.0, hidden_activation='relu', return_max_acc=False,
          minibatch_size=0, lr_decay=0):
    """
    Implements a n-layer tensorflow neural network: LINEAR->RELU*(n times)->LINEAR->SOFTMAX.
    :param train_set: training set
    :param train_labels: training labels
    :param validation_set: validation set
    :param validation_labels: validation labels
    :param layers_dims: array with the layer for the model
    :param learning_rate: learning rate of the optimization
    :param num_epochs: number of epochs of the optimization loop
    :param print_cost: True to print the cost every 500 epochs
    :param plot_cost: True to plot the train and validation cost
    :param l2_beta: beta parameter for the l2 regularization
    :param keep_prob: probability to keep each node of each hidden layer (dropout)
    :param hidden_activation: activation function to be used on the hidden layers
    :param return_max_acc: True to return the highest accuracy from all epochs
    :param minibatch_size: size of th mini batch
    :param lr_decay: if != 0, sets de learning rate decay on each epoch
    :return parameters: parameters learnt by the model. They can then be used to predict.
    :return submission_name: name for the trained model
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    input_size = layers_dims[0]
    output_size = layers_dims[-1]
    num_examples = train_set.shape[0]
    n_layers = len(layers_dims)

    train_costs = []
    validation_costs = []
    prediction = []
    best_iteration = [float('inf'), 0]
    best_acc_params = None
    if minibatch_size == 0:
        minibatch_size = num_examples
    num_minibatches = int(num_examples / minibatch_size)

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    parameters = initialize_parameters(layers_dims)

    fw_output_train = forward_propagation(x, parameters, keep_prob, hidden_activation)
    train_cost = compute_cost(fw_output_train, y)

    fw_output_valid = forward_propagation(tf_valid_dataset, parameters, keep_prob, hidden_activation)
    validation_cost = compute_cost(fw_output_valid, validation_labels)

    if l2_beta > 0:
        train_cost = l2_regularizer(train_cost, l2_beta, parameters, n_layers)
        validation_cost = l2_regularizer(validation_cost, l2_beta, parameters, n_layers)

    if lr_decay != 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step=global_step, decay_rate=lr_decay,
                                                    decay_steps=1)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost, global_step=global_step)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            train_epoch_cost = 0.
            validation_epoch_cost = 0.

            minibatches = mini_batches(train_set, train_labels, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                feed_dict = {x: minibatch_X, y: minibatch_Y}

                _, minibatch_train_cost, prediction, minibatch_validation_cost = sess.run(
                    [optimizer, train_cost, fw_output_train, validation_cost], feed_dict=feed_dict)

                train_epoch_cost += minibatch_train_cost / num_minibatches
                validation_epoch_cost += minibatch_validation_cost / num_minibatches

            if return_max_acc is True and validation_epoch_cost < best_iteration[0]:
                best_iteration[0] = validation_epoch_cost
                best_iteration[1] = epoch
                best_acc_params = sess.run(parameters)

            if print_cost is True and epoch % 500 == 0:
                print("Train cost after epoch %i: %f" % (epoch, train_epoch_cost))
                print("Validation cost after epoch %i: %f" % (epoch, validation_epoch_cost))

            if plot_cost is True and epoch % 10 == 0:
                train_costs.append(train_epoch_cost)
                validation_costs.append(validation_epoch_cost)

        if return_max_acc is True:
            parameters = best_acc_params
        else:
            parameters = sess.run(parameters)
        print("Parameters have been trained!")

        train_cost = rmsle(prediction, minibatch_Y)
        validation_cost = rmsle(fw_output_valid.eval(), validation_labels)

        print('Train rmse: {:.4f}'.format(rmse(prediction, minibatch_Y)))
        print('Validation rmse: {:.4f}'.format(rmse(fw_output_valid.eval(), validation_labels)))
        print('Train rmsle: {:.4f}'.format(train_cost))
        print('Validation rmsle: {:.4f}'.format(validation_cost))
        print('Lowest rmse: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))

        submission_name = build_submission_name(train_cost, validation_cost, layers_dims, num_epochs, lr_decay,
                                                learning_rate, l2_beta, keep_prob, minibatch_size, num_examples)

        if plot_cost is True:
            plot_model_cost(train_costs, validation_costs, submission_name)

        return parameters, submission_name
