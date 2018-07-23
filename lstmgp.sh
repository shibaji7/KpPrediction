#!/bin/bash
counter=1
while [ $counter -le  240 ]
do
    echo "python jobutil.py 7 deepGP 14 2004 $counter"
    python jobutil.py 7 deepGP 14 2004 $counter
    ((counter++))
done

echo All done
