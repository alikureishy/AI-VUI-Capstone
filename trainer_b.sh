################################################################
# 			Training Iterator			#
#################################################################

# Increase # of epochs:
# This achieves a training loss of 65 units and validation loss of about 87 units in 35 epochs. The model starts overfitting near the 26th epoch. This suggests that 
# a higher dropout could be explored.
# The training loss could be further optimized with more training and  a higher dropout would also allow the two to converge closer.
# So, decreasing dropout will not be appropriate for this exercise.
#python trainer.py -o results -i 20 -cf 200 -ck 20 -cs 1 -cp same -cd 0.50 -rl 2 -ru 250 250 -rb 1 1 -rd 0.50 0.50 -rc 2 2 -dd 0.50 -e 45 &>> out.txt
#BACK_PID=$!
#while kill -0 $BACK_PID ; do
#    echo "Waiting for #1 to complete..."
#    sleep 10
#    # You can add a timeout here if you want
#done

# 3 recur layers, 250 units each, change to LSTM, longer training
python trainer.py -o results -i 21 -cf 80 -ck 10 -cs 2 -cp same -cd 0.25 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.25 0.25 0.25 -rc 2 2 2 -dd 0.25 -e 40 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Increase # of conv filters
python trainer.py -o results -i 22 -cf 200 -ck 10 -cs 2 -cp same -cd 0.25 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.25 0.25 0.25 -rc 2 2 2 -dd 0.25 -e 40 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Increase size of conv kernel
python trainer.py -o results -i 23 -cf 200 -ck 20 -cs 2 -cp same -cd 0.25 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.25 0.25 0.25 -rc 2 2 2 -dd 0.25 -e 40 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Decrease stride
python trainer.py -o results -i 24 -cf 200 -ck 20 -cs 1 -cp same -cd 0.25 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.25 0.25 0.25 -rc 2 2 2 -dd 0.25 -e 40 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

echo "Training iterations complete!"
