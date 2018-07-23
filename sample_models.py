from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
#from keras.utils.training_utils import multi_gpu_model

#def parallelize(model, gpus=1):
#    return multi_gpu_model(model, gpus=gpus)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    assert recur_layers >= 1, "Invalid number of recurrent layers specified. Must be >=1"
    rnns = input_data
    for i in range(recur_layers):
        rnns = GRU(units, activation='relu', return_sequences=True, implementation=2, name="rnn_{}".format(i))(rnns)
    
    bn_rnns = BatchNormalization(name="bn")(rnns)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnns)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    rnns = input_data
    for i in range(recur_layers):
        rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name="rnn_{}".format(i))
        rnns = Bidirectional(rnn, name="brnn_{}".format(i), merge_mode='concat', weights=None)(rnns)
    
    bn_rnns = BatchNormalization(name="bn")(rnns)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnns)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def custom_rnn_model(input_dim,
                     conv_filters, conv_kernel_size, conv_stride, conv_border_mode,
                     recur_layers, recur_units, recur_cells,
                     output_dim=29):
    """ Build a custom model for ASR using options for CNNs and BRNNs
    """
    # Main acoustic input (input_data)
    layer = input_data = Input(name='the_input', shape=(None, input_dim))
    
    # TODO: Add convolutional layers (cnns):
    layer = cnn = Conv1D(conv_filters, conv_kernel_size, 
                          strides=conv_stride, 
                          padding=conv_border_mode,
                          activation='relu',
                          name='conv1d')(layer)
    # Add batch normalization
    layer = BatchNormalization(name='bn_cnn')(layer)
    
    # TODO: Add bidirectional recurrent layers
    rnns = layer
    if recur_layers > 0:
        for i in range(recur_layers):
            rnn = recur_cells[i](recur_units[i], activation='relu', return_sequences=True, implementation=2, name="rnn_{}".format(i))
            layer = rnns = Bidirectional(rnn, name="brnn_{}".format(i), merge_mode='concat', weights=None)(rnns)
        layer = rnns = BatchNormalization(name="bn_rnn")(layer)
    # -------------------------------

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    layer = time_dense = TimeDistributed(Dense(units=output_dim))(layer)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(layer)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, conv_kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model():
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    ...
    # TODO: Add softmax activation layer
    y_pred = ...
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = ...
    print(model.summary())
    return model