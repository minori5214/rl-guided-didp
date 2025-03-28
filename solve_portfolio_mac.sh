n_item=20
heuristic=dqn # dual, dqn, zero, greedy
policy_name=none # "none", "ppo"
solver_name=CABS # CABS, CAASDy, ACPS, APPS

model=narita # narita
dual_bound=2
greedy_heuristic_type=4
softmax_temperature=20
no_policy_accumulation=0 # Use accumulated policy for policy-guidance


# Experiment settings
time=3600
memory_limit=8192

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)
            n_item="$2"
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

train_n_item=${n_item}

python3 portfolio.py --n-item ${n_item} --train-n-item ${train_n_item} \
                    --heuristic ${heuristic} --policy-name ${policy_name} \
                    --solver-name ${solver_name} --model ${model} --num-instance 100 \
                     --time ${time} \
                    --lb 0 --ub 100 --capacity-ratio 0.5 --lambdas-0 1 --lambdas-1 5 \
                    --lambdas-2 5 --lambdas-3 5 --seed 0 \
                    --dual-bound ${dual_bound}\
                    --no-policy-accumulation ${no_policy_accumulation} \
                    --greedy-heuristic-type ${greedy_heuristic_type} \
                    --file ${i}.txt

done
