insize=$1;
ntrials=$2;

epochs_per_input=10;

for i in $(seq 0 $ntrials); do
    python3 rl_DQN_dev_dummy_simple_bits.py $i $insize $epochs_per_input; 
    python3 evaluate_rl_DQN_dev_dummy_simple_bits.py $i $insize $epochs_per_input; 
    python3 full_random_DQN_dev_dummy_simple_bits.py $i $insize $epochs_per_input; 
    echo "Done batch $i of $ntrials"
done

python3 determ_DQN_dev_dummy_simple_bits.py 0 $insize; 
python3 determ_DQN_dev_dummy_simple_bits.py 1 $insize; 

python3 combine_trial_data.py $insize $ntrials;
