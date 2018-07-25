from train_utils import train_model
from sample_models import custom_rnn_model
from keras.layers import SimpleRNN, GRU, LSTM
import argparse
import sys
from os.path import join

if __name__ == '__main__':
    print ("###############################################")
    print ("#                 ASR Trainer                 #")
    print ("###############################################")

    parser = argparse.ArgumentParser(description='ASR Driver')
    parser.add_argument('-o', dest='output', required=True, type=str, help='Path to folder containing model data input/output (.hd5 and .pickle files).')
    parser.add_argument('-i', dest='id', required=True, type=int, help='Id or name of the model')
    parser.add_argument('-cf', dest='conv_filters', type=int, help='# of convolution filters')
    parser.add_argument('-ck', dest='kernel_size', type=int, help='Size of convolution kernel')
    parser.add_argument('-cs', dest='conv_stride', type=int, help='Convolutional stride')
    parser.add_argument('-cp', dest='conv_padding', type=str, help="Convolutional padding mode ('same', or 'valid')")
    parser.add_argument('-cd', dest='conv_dropout', type=int, help='Dropout for convolutional output (between 0.0 and 1.0)')
    parser.add_argument('-rl', dest='recur_layers', type=int, help='Number of recurrent layers')
    parser.add_argument('-ru', dest='recur_units', nargs='*', type=int, help="List of 'rc' recurrent unit sizes")
    parser.add_argument('-rc', dest='recur_cells', nargs='*', type=int, help="List of 'rc' recurrent cell types (0: SimpleRNN, 1: GRU, or 2: LSTM)")
    parser.add_argument('-rb', dest='recur_bidis', nargs='*', type=bool, help="List of 'rc' flags indicating whether the layer is bidirectional ('True', 'False')")
    parser.add_argument('-rd', dest='recur_dropouts', nargs='*', type=float, help="List of 'rc' dropouts (between 0.0 and 1.0)")
    parser.add_argument('-dd', dest='dense_dropout', type=float, help="Dropout for fully connected output layer")
    
    args = parser.parse_args()
    args.recur_cells = map(lambda x: SimpleRNN if x is 0 else GRU if x is 1 else LSTM, args.recur_cells)
    
    model_weights_path = join(args.output, "model_{}.hd5".format(args.id))
    model_hist_path = join(args.output, "model_{}.pickle".format(args.id))
    
    # --------
    model_5 = custom_rnn_model(input_dim=13, # change to 13 if you would like to use MFCC features
                               conv_filters=args.conv_filters, conv_kernel_size=args.kernel_size, conv_stride=args.conv_stride, conv_border_mode=args.conv_padding, conv_batch_mode=True, conv_dropout=args.conv_dropout, \
                               recur_layers=args.recur_layers, recur_units=args.recur_units, recur_cells=args.recur_cells, recur_bidis=args.recur_bidis, recur_batchnorms=[True]*args.recur_layers, recur_dropouts=args.recur_dropouts, \
                               output_dropout=args.dense_dropout, output_dim=29)
    
    train_model(input_to_softmax=model_5, 
            pickle_path='model_5.pickle', 
            save_model_path='model_5.h5', 
            spectrogram=False) # change to False if you would like to use MFCC features
    
    print ("Training complete!")
    print ("\tModel weights stored in: {}".format(model_weights_path))
    print ("\tModel hist stored in: {}".format(model_hist_path))
    print ("# Thank you! #")