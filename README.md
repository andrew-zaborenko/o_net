# o_net
Install required packages with `pip install -r requirements.txt`

Launch dataset preparation with `python3 preprocess_datasets.py --menpo_train './landmarks_task/Menpo/train/' --w_train './landmarks_task/300W/train/' --menpo_test './landmarks_task/Menpo/test/' --w_test './landmarks_task/300W/test/'`, change paths to the extracted landmarks_task.tgz

Launch training with `python3 train_net.py --train_root_dir './cropped_faces' --train_landmarks_file 'landmarks.txt' --test_root_dir './cropped_faces_test' --test_landmarks_file 'landmarks_test.txt' --batch_size 32 --batch_size_test 1024 --learning_rate 0.0003 --num_epochs 100`

Launch evaluation with `python3 get_metrics.py --root_dir './landmarks_task/300W/test/' --model_dir '.' --threshold 0.08`
