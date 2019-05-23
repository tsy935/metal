echo "Args are"
echo $1
echo $2
CUDA_VISIBLE_DEVICES=$1 python launch.py --model_type $2 --validate_on_slices 1 --n_epochs 10 --batch_size 32 --lr 0.00001 --lr_scheduler reduce_on_plateau --patience 2 --factor 0.5 --checkpoint_metric MapillaryClassificationTask/Payload1_valid/labelset_gold/accuracy --checkpoint_metric_mode max --seed 124 --pretrained_model /home/ankitmathur/metal/logs/model_checkpoint_50.002.pth --slices object--vehicle--on-rails object--junction-box object--bench nature--water animal--ground-animal construction--flat--pedestrian-area
