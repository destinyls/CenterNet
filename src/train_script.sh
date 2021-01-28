python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_010nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_010nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_011nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_011nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_012nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Merge_Head_012nd"