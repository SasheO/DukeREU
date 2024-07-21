#!/usr/bin/env bash
# run.sh
# Author: Kevin Chu
# Last Modified: 04/12/2020

cd objective_intelligibility
matlab -nodisplay -nosplash -nodesktop -r "run('calculateSrmrCiSingleCondition.m');exit;"

#cd feature_extraction
#matlab -nodisplay -nosplash -nodesktop -r "run('extractFeaturesTrainingData.m');exit;"
