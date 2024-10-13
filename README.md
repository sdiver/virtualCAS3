
    启动脚本
    根据GPU数量决定是否使用多进程启动训练。 
    1\ face: python -m train.train_diffusion --save_dir checkpoints/diffusion/c1_face_test --data_root ./dataset/PXB184/ --batch_size 4 --dataset social --data_format face --layers 8 --heads 8 --timestep_respacing '' --max_seq_length 600
    2\ pose: python -m train.train_diffusion --save_dir checkpoints/diffusion/c1_pose_test --data_root ./dataset/PXB184/ --lambda_vel 2.0 --batch_size 4 --dataset social --add_frame_cond 1 --data_format pose --layers 6 --heads 8 --timestep_respacing '' --max_seq_length 600



pip install git+https://github.com/Tps-F/fairseq.git@main
pip install pysoundfile


python -m sample.generate 
    --model_path <path/to/model> 
    --resume_trans <path/to/guide/model> 
    --num_samples <xsamples> 
    --num_repetitions <xreps> 
    --timestep_respacing ddim500 
    --guidance_param 2.0c