task=cheetah_run
seed=1
dir="./exp_local_adaptive_rr/${task}/aug_True/seed${seed}"

python train_adapt_rr.py \
	task=${task} \
    use_aug=True \
    seed=${seed}\
    hydra.run.dir=${dir}
