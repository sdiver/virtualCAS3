
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


### 3) Body VQ VAE
To train a vq encoder-decoder, you will need to run the following script:
```
python -m train.train_vq 
    --out_dir <path/to/out/dir> 
    --data_root <path/to/data/root>
    --batch_size <bs>
    --lr 1e-3 
    --code_dim 1024 
    --output_emb_width 64 
    --depth 4 
    --dataname social 
    --loss_vel 0.0 
    --add_frame_cond 1 
    --data_format pose 
    --max_seq_length 600
```
:point_down: For person `PXB184`, it would be:
```
python -m train.train_vq --out_dir checkpoints/vq/c1_vq_test --data_root ./dataset/PXB184/ --lr 1e-3 --code_dim 1024 --output_emb_width 64 --depth 4 --dataname social --loss_vel 0.0 --data_format pose --batch_size 4 --add_frame_cond 1 --max_seq_length 600
```

### 4) Body guide transformer
Once you have the vq trained from 3) you can then pass it in to train the body guide pose transformer:
```
python -m train.train_guide 
    --out_dir <path/to/out/dir>
    --data_root <path/to/data/root>
    --batch_size <bs>
    --resume_pth <path/to/vq/model>
    --add_frame_cond 1 
    --layers 6 
    --lr 2e-4 
    --gn 
    --dim 64 
```
:point_down: For person `PXB184`, it would be:
```
python -m train.train_guide --out_dir checkpoints/guide/c1_trans_test --data_root ./dataset/PXB184/ --batch_size 4 --resume_pth checkpoints/vq/c1_vq_test/net_iter300000.pth --add_frame_cond 1 --layers 6 --lr 2e-4 --gn --dim 64
```