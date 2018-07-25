################################################################
# 			Training Iterator			#
#################################################################

# Start with small onvolutional kernel and SimpleRNN
python trainer.py -o results -i 1 -cf 20 -ck 5 -cs 2 -cp same -cd 0.25 -rl 1 -ru 100 -rb False -rd 0.25 -rc 0 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Doubling size of convolutional kernel (all else the same)
python trainer.py -o results -i 2 -cf 20 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 100 -rb False -rd 0.25 -rc 0 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Doubling # of convolutional filters (all else the same)
python trainer.py -o results -i 3 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 100 -rb False -rd 0.25 -rc 0 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Doubling units in recurrent layer (all else the same)
python trainer.py -o results -i 4 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb False -rd 0.25 -rc 0 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# GRU (all else the same)
python trainer.py -o results -i 5 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb False -rd 0.25 -rc 1 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# LSTM (all else the same)
python trainer.py -o results -i 6 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb False -rd 0.25 -rc 2 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Bidi-SimpleRNN (all else the same)
python trainer.py -o results -i 7 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb True -rd 0.25 -rc 0 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Bidi GRU (all else the same)
python trainer.py -o results -i 8 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb True -rd 0.25 -rc 1 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Bidi-LSTM (all else the same)
python trainer.py -o results -i 9 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 1 -ru 200 -rb True -rd 0.25 -rc 2 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Bidi-GRU + 1 non-bidi GRU-layer (all else the same)
python trainer.py -o results -i 10 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 200 200 -rb True False -rd 0.25 0.25 -rc 1 1 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Bidi-LSTM + 1 non-bidi LSTM layer (all else the same)
python trainer.py -o results -i 12 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 200 200 -rb True False -rd 0.25 0.25 -rc 2 2 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# 2x Bidi-LSTM-layer (all else the same)
python trainer.py -o results -i 14 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 200 200 -rb True True -rd 0.25 0.25 -rc 2 2 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# 2x Bidi-GRU-layer (all else the same)
python trainer.py -o results -i 13 -cf 40 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 200 200 -rb True True -rd 0.25 0.25 -rc 1 1 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# 2x Bidi-GRU-layer + Double # of conv kernels again
python trainer.py -o results -i 15 -cf 80 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 200 200 -rb True True -rd 0.25 0.25 -rc 1 1 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Add 1 Bidi-GRU
python trainer.py -o results -i 16 -cf 80 -ck 10 -cs 2 -cp same -cd 0.25 -rl 3 -ru 200 200 200 -rb True True True -rd 0.25 0.25 0.25 -rc 1 1 1 -dd 0.25 &>> out.txt


echo "Training iterations complete!"
