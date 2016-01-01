#!/bin/sh

echo "" | awk 'BEGIN{
                  randomSeed=41
                  bias=0.7
                  biasstep=0.1
                  biasMax=2.5} {
		   while (bias < biasMax) {
			   print "Random Seed: ", randomSeed,  "Runing with Bias:", bias
			   print "Executing Command:  th ./relatedness/main.lua -m dependency_gru -x "bias" -s "randomSeed
			   system("th ./relatedness/main.lua -m dependency_gru -x "bias" -s "randomSeed)
			   bias+=biasstep
		   }
	   }'

               

