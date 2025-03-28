n_city=20
heuristic=dqn # dual, dqn, zero, greedy
policy_name=none # "none", "ppo"
solver_name=CABS # CABS, CAASDy, ACPS, APPS

model=kuroiwa # kuroiwa, cappart
softmax_temperature=20
no_policy_accumulation=0 # Use accumulated policy for policy-guidance

# Instance parameters
max_tw_gap=100
max_tw_size=1000

# Experiment settings
time=3600
memory_limit=8192

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)
            n_city="$2"
            shift 2
            ;;
        --heuristic)
            heuristic="$2"
            shift 2
            ;;
        --policy-name)
            policy_name="$2"
            shift 2
            ;;
        --solver-name)
            solver_name="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Iterate over 0.txt to 19.txt
for i in {0..19}
do

train_n_city=${n_city}
train_max_tw_gap=${max_tw_gap}
train_max_tw_size=${max_tw_size}

python3 tsp.py --n-city ${n_city} --train-n-city ${train_n_city} \
                    --heuristic ${heuristic} --policy-name ${policy_name} \
                    --solver-name ${solver_name} --model ${model} --num-instance 100 \
                     --time-out ${time} \
                    --grid-size 100 --max-tw-size ${max_tw_size} --max-tw-gap ${max_tw_gap} --seed 0 \
                    --softmax-temperature ${softmax_temperature} \
                    --no-policy-accumulation ${no_policy_accumulation} \
                    --file ${i}.txt

done