#!/bin/bash

# select the filename and log_filename, along with the appropriate command based on the needs.
filename="53K"

log_filename="logfile_${filename}.log"
echo "Train model with classification threshold 0.5, threshold1 0.6, threshold2 1.0 on dataset ${filename}"
python3 lbp.py "${filename}" word2vec rbf sim False True 0.5 0.6 1.0 >> "$log_filename"
# normal, with cycles
# echo "With prior probability, with cycle - t1"
# python3 lbp.py "${filename}" word2vec rbf t1 False True 0.5 >> "$log_filename"

# echo "With prior probability, with cycle - sim_only"
# python3 lbp.py "${filename}" word2vec rbf sim_only False True 0.5 >> "$log_filename"

# echo "With prior probability, with cycle - sim with default thresholds"
# python3 lbp.py "${filename}" word2vec rbf sim False True 0.5 >> "$log_filename"


# delete cycles
#echo "With prior probability, delete cycle - sim"
#python3 lbp.py "${filename}" word2vec rbf t1 True True 0.5 >> "$log_filename"
#
#echo "With prior probability, delete cycle - sim"
#python3 lbp.py "${filename}" word2vec rbf sim_only True True 0.5 >> "$log_filename"
#
#echo "With prior probability, delete cycle - sim"
#python3 lbp.py "${filename}" word2vec rbf sim True True 0.5 >> "$log_filename" 0.6 0.4


## without probability
#
## echo "Without prior probability, with cycle - t1"
## python3 lbp.py "${filename}" word2vec rbf t1 False False 0.5 >> "$log_filename" >> "$log_filename"
#
#echo "Without prior probability, with cycle - sim_only"
#python3 lbp.py "${filename}" word2vec rbf sim_only False False 0.5 >> "$log_filename"
#
#echo "Without prior probability, with cycle - sim"
#python3 lbp.py "${filename}" word2vec rbf sim False False 0.5 >> "$log_filename"
#
#
#echo "Without prior probability, delete cycle - t1"
#python3 lbp.py "${filename}" word2vec rbf t1 True False 0.5 >> "$log_filename"
#
#echo "Without prior probability, delete cycle - sim_only"
#python3 lbp.py "${filename}" word2vec rbf sim_only False True 0.5 >> "$log_filename"
#
#echo "Without prior probability, delete cycle - sim"
#python3 lbp.py "${filename}" word2vec rbf sim True False 0.5 >> "$log_filename"

## find classification threshold, iterate over different threshold
# for ((t=10; t<=100; t+=10)); do
#         t_float=$(awk "BEGIN {print $t / 100}")
#         echo "Train Model with classification threshold = $t_float"
#         python3 lbp.py "${filename}" word2vec rbf sim True True "${t_float}" 0.6 1.0 >> "$log_filename"
# done

## Iterate over edge potential threshold1 and threshold2 combinations
#for ((t1=10; t1<=100; t1+=10)); do
#    for ((t2=100; t2>=10; t2-=10)); do
#        # Convert the current value of t1 and t2 to floating-point numbers
#        t1_float=$(awk "BEGIN {print $t1 / 100}")
#        t2_float=$(awk "BEGIN {print $t2 / 100}")
#
#        echo "Train Model with threshold1 = $t1_float, threshold2 = $t2_float"
#        python3 lbp.py "${filename}" word2vec rbf sim True True 0.5 $t1_float $t2_float >> "$log_filename"
#    done
#done

