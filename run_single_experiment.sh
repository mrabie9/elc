python main.py --exp-name "exp-noisy-radchar-dcheck-redo-fulldata" \
    --arch "bayes_rfnet" \
    --base-path "radar/" \
    --task-folder "tasks-noisy-radchar" \
    --input-size 1024 \
    --classes 12 \
    --tasks 2 \
    --adaptive-ratio 0.9 \
    --epochs 60 \
    --epochs-prune 10 --rho-num 4\
    --epochs-mask-retrain 30 \
    --adaptive-mask true \
    --config-shrink 0.5 \
    --disjoint_classifier false \

# python main.py --exp-name "mixed-bresnet-tasks" \
#     --arch "bayes_rfnet" \
#     --base-path "mixed/" \
#     --task-folder "tasks" \
#     --classes 15 \
#     --tasks 3 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4\
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.33 \
#     --config-setting 0.25,0.4,0.33 \
#     --disjoint_classifier false \
#     # --load-model-pruned "mixed/exp-mixed-2/task0/retrained.pt"

# python main.py --exp-name "mixed-bresnet-sm" \
#     --arch "bayes_rfnet" \
#     --base-path "mixed/" \
#     --task-folder "tasks-sm" \
#     --classes 15 \
#     --tasks 1 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4\
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.33 \
#     --config-setting "1" \
#     --disjoint_classifier true \

# python main.py --exp-name "radar-bresnet-sm" \
#     --arch "bayes_rfnet" \
#     --base-path "radar/" \
#     --task-folder "tasks-sm" \
#     --classes 11 \
#     --tasks 1 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4\
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.5 \
#     --disjoint_classifier true \

# python main.py --exp-name "usrp-bresnet-sm" \
#     --arch "bayes_rfnet" \
#     --base-path "usrp/" \
#     --task-folder "tasks-sm" \
#     --classes 18 \
#     --tasks 1 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4\
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.33 \
#     --disjoint_classifier true \

# python main.py --exp-name "rfmls-bresnet-sm" \
#     --arch "bayes_rfnet" \
#     --base-path "rfmls/" \
#     --task-folder "tasks-sm" \
#     --classes 10 \
#     --tasks 1 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4 \
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.4 \
#     --disjoint_classifier true \

# python main.py --exp-name "drc-bresnet-sm" \
#     --arch "bayes_rfnet" \
#     --base-path "dronerc/" \
#     --task-folder "tasks-sm" \
#     --classes 15 \
#     --tasks 1 \
#     --adaptive-ratio 0.9 \
#     --epochs 100 \
#     --epochs-prune 30 --rho-num 4\
#     --epochs-mask-retrain 30 \
#     --adaptive-mask true \
#     --config-shrink 0.35 \
#     --disjoint_classifier true \