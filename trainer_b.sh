################################################################
# 			Training Iterator			#
#################################################################

# Increase # of epochs:
python trainer.py -o results -i 18 -cf 80 -ck 10 -cs 2 -cp same -cd 0.25 -rl 3 -ru 200 200 200 -rb 1 1 1 -rd 0.25 0.25 0.25 -rc 1 1 1 -dd 0.25 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Decrease dropout
python trainer.py -o results -i 19 -cf 80 -ck 10 -cs 2 -cp same -cd 0.15 -rl 3 -ru 200 200 200 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 1 1 1 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Change to LSTM
python trainer.py -o results -i 20 -cf 80 -ck 10 -cs 2 -cp same -cd 0.15 -rl 3 -ru 200 200 200 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 2 2 2 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Increase # of recurrent units
python trainer.py -o results -i 21 -cf 80 -ck 10 -cs 2 -cp same -cd 0.15 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 2 2 2 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Increase # of conv filters
python trainer.py -o results -i 22 -cf 200 -ck 10 -cs 2 -cp same -cd 0.15 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 2 2 2 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Increase size of conv kernel
python trainer.py -o results -i 23 -cf 200 -ck 20 -cs 2 -cp same -cd 0.15 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 2 2 2 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

# Decrease stride
python trainer.py -o results -i 24 -cf 200 -ck 20 -cs 1 -cp same -cd 0.15 -rl 3 -ru 250 250 250 -rb 1 1 1 -rd 0.15 0.15 0.15 -rc 2 2 2 -dd 0.15 -e 35 &>> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done

echo "Training iterations complete!"
