python trainer.py -o results -i 10 -cf 20 -ck 10 -cs 2 -cp same -cd 0.25 -rl 2 -ru 100 100 -rb True False -rd 0.25 0.25 -rc 1 2 -dd 0.25 &> out.txt
BACK_PID=$!
while kill -0 $BACK_PID ; do
    echo "Waiting for #1 to complete..."
    sleep 10
    # You can add a timeout here if you want
done
python trainer.py -o results -i 11 -cf 30 -ck 20 -cs 2 -cp same -cd 0.25 -rl 3 -ru 100 100 100 -rb True False False -rd 0.25 0.25 0.25 -rc 1 2 2 -dd 0.25 &>> out.txt
