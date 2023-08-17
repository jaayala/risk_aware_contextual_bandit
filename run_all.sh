#!/bin/bash

pythonver="python3"
script="eval_learning_alg_syn.py"
script_bayes="eval_learning_bayes_alg_syn.py"
script_plot="plot.py"

timestamp=$(date +"%Y%m%d%H%M%S")
sigmas="0.01 0.05 0.1 0.2 0.3"
betas="90 15 10 3.5 2"
reps=10
constr_val='[0.3, 0.3]'

# RANCB configurations
conf1='{"A_exec" : [0.995], "A_train" : [0.995], "A_explo" : [0.995]}'
conf3='{"A_exec" : [0.999], "A_train" : [0.999], "A_explo" : [0.999]}'
conf4='{"A_exec" : [0.99], "A_train" : [0.99], "A_explo" : [0.99]}'
conf5='{"A_exec" : [0.9], "A_train" : [0.9], "A_explo" : [0.9]}'
conf6='{"A_exec" : [0.8], "A_train" : [0.8], "A_explo" : [0.8]}'
conf9='{"A_exec" : [0.5], "A_train" : [0.5], "A_explo" : [0.5]}'

qc='{"quantile_critic" : [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999]}'

for r in $(seq 1 1 $reps) 
do
    for sigma in $sigmas; do
        tag="syn_var$sigma-rep$r"
        
        algo="RANCB"
        tag_c="$tag-conf1"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf1 --constr_val $constr_val --quant_critic $qc
                
       
        # Different alpha evaluation
        algo="RANCB"
        tag_c="$tag-conf3"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf3 --constr_val $constr_val --quant_critic $qc
        
        algo="RANCB"
        tag_c="$tag-conf4"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf4 --constr_val $constr_val --quant_critic $qc

        algo="RANCB"
        tag_c="$tag-conf5"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf5 --constr_val $constr_val --quant_critic $qc
        
        algo="RANCB"
        tag_c="$tag-conf6"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf6 --constr_val $constr_val --quant_critic $qc
        
        algo="RANCB"
        tag_c="$tag-conf9"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag_c -a $conf9 --constr_val $constr_val --quant_critic $qc
        
        
        # benchmarks
        algo="NCB"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag  --constr_val $constr_val
        
        algo="SC_DNCB"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag  --constr_val $constr_val
        
        algo="MC_NCB"
        $pythonver $script --algo $algo -T $timestamp --env_noise_var $sigma -t $tag --constr_val $constr_val 
        
        
        for beta in $betas; do
            tag="syn_var$sigma-rep$r-beta_$beta"
            $pythonver $script_bayes --beta $beta -T $timestamp --env_noise_var $sigma -t $tag --constr_val $constr_val
        done
        
    done
done


cd plot
$pythonver $script_plot --timestamp $timestamp --nreps $reps


