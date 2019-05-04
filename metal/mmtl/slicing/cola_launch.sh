python launch.py --seed 1 --tasks COLA --model_type hard_param --n_epochs 50 --model_weights /dfs/scratch0/mccreery/repos/metal/metal/mmtl/aws/output/2019_04_04_12_00_46/2/logdir/2019_04_04/COLA_19_27_11/best_model.pth --lr_scheduler linear --lr 1e-06 --optimizer adam --l2 1e-6 --min_lr 1e-6 --slice_dict '{"COLA": ["has_temporal_preposition", "BASE"]}'

