####Angiography
model_name="cunet"
subject_dir="demo_imgs"
check_path="EDMT_CUNET.pth"
CUDA_VISIBLE_DEVICES=3 python angiography.py --mode "blood" \
                                                    --check_name "results_angiography" \
                                                    --model_name $model_name \
                                                    --check_path_blood $check_path \
                                                    --check_path_choroid $check_path \
                                                    --subject_dir $subject_dir \
                                                    --out_dir "result_angiography"

####Validation/Test
target="RF"
model_name="cunet"
check_path="EDMT_CUNET.pth"
data_path="datas"
CUDA_VISIBLE_DEVICES=3 python validation.py --batch_size 32 \
                                            --data_path $data_path \
                                            --target $target \
                                            --check_path_blood $check_path \
                                            --check_path_choroid $check_path \
                                            --work_dir "test_logs" \
                                            --model_name $model_name


####Training
source="AF"
target="RF"
unlabel="RF_unlabel"
batch_size=4
model_name="cunet"
mode="multi-task"
data_path="datas"
CUDA_VISIBLE_DEVICES=7 python training.py --batch_size $batch_size \
                                                --data_path $data_path \
                                                --source $source \
                                                --target $target \
                                                --mode $mode \
                                                --use_resize 0 \
                                                --work_dir "train_logs" \
                                                --val_interval 1 \
                                                --model_name $model_name \
                                                --thresh_semi 0.85 \
                                                --semi_start 10 \
                                                --rate_semi 1.5 \
                                                --rate_da 0.2 \
                                                --thresh_da 0.9 \
                                                --max_epochs 250 \
                                                --initial_lr 6e-4 \
                                                --initial_lr_d 6e-5 \
                                                --rate_multi 0.6