python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 20 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_006nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_006nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 20 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_007nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_007nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 20 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_008nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_008nd"