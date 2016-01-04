#!/bin/sh

while [ $# -gt 1 ]; do
    case "$1" in
        "-task") task=$2;
             shift
             ;;        
        "-tree") tree=$2;
             shift
             ;;
        "-config") config=$2;
             shift
             ;;             
        "-seed") seed=$2;
             shift
             ;;
        "-bias") bias=$2;
             shift
             ;;
            *) shift
             ;;
    esac
done
echo "================================================================================"
echo "Running Grid Search on configuration:"
echo "TASK: $task, TREE: $tree, CONFIG: $config, BIAS: $bias, SEED=$seed"
echo "================================================================================"

echo "" | awk 'BEGIN{
                seed="'"$seed"'"
                bias="'"$bias"'"
                task="'"$task"'"
                tree="'"$tree"'"
                config="'"$config"'"
                biasstep=0.1
                biasMax=2.5} {
           while (bias < biasMax) {
                if (config == "binary")
                    cmd="th ./"task"/main.lua -m "tree" --binary -x "bias" -s "seed
                else if (config == "fineGrained")
                    cmd="th ./"task"/main.lua -m "tree" -x "bias" -s "seed
                else
                {
                    print "Invalid configuration "config
                    exit -1
                }
                print "Executing Command: "cmd
                system(cmd)
                bias+=biasstep
           }
       }'