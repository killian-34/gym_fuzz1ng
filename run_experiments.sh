insize=$1;
ntrials=$2;

epochs_per_input=10;

for i in $(seq 0 $ntrials); do
    python3 test/train_rlfuzz_2ladder.py $i $insize $epochs_per_input; 
    python3 test/test_rlfuzz_2ladder.py $i $insize $epochs_per_input; 
    python3 test/randomfuzz_2ladder.py $i $insize $epochs_per_input; 
    echo "Done batch $i of $ntrials"
done

python3 test/detfuzz_2ladder.py 0 $insize; 
python3 test/detfuzz_2ladder.py 1 $insize; 

python3 test/combine_trial_data.py $insize $ntrials;
