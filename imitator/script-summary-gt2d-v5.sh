trainging code:

============================================================================
========================  helix 0  =========================================
============================================================================
I: Loop starting
----------------------------------------------------------------------------------------------------------------------------
>> generate random trajectory
python pose_imitation/data_process/gen_random_traj.py --scale-start 4 --scale-end 8 --num-take 600 --curve-type circle --mocap-folder './checkpoint/exp_h36m_gt2d_v5/helix_0'
>> take a look at the generated trajectory
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m_helix_0 --vis-model humanoid_h36m_v4 --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_0/datasets/traj_dict/traj_dict.pkl
>> generate expert file from the reference trajectory
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_0
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_0 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_0
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_0 --data train --num-threads 52 --iter 900 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_0

>> take a look during policy training.
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_0 --iter 900 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_0
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m_helix_0 --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_test_naivefs.p
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_0 --iter 900 --save-gif --gif-ds 3 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_0

============================================================================
saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs.p

E: use the RL result to train the DL estimator 
----------------------------------------------------------------------------------------------------------------------------
>> train estimator with GAN projection, trained with two setting because sometime NAN.
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_0 --add_random_cam False -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_1 --add_random_cam False -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs
>> inferece the estimator result from the H36M 2D pose, to generate pseudo 3D reference motion data, save at traj_dict.pkl. with temp bvh file.
#CUDA_VISIBLE_DEVICES=0 python posegan_evaluate.py --note vp_0  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs/vp_0/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_1/datasets/traj_dict/traj_dict.pkl
CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_1  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_0/results/posemimic/subject_h36m_helix_0/results/iter_0900_train_naivefs/vp_1/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_1/datasets/traj_dict/traj_dict.pkl
>> check if the take_599 Hip flip, if flip, maybe not a good initialization.

============================================================================
========================  helix 1  =========================================
============================================================================

I: convert the bvh to qpos then save in a expert file
----------------------------------------------------------------------------------------------------------------------------
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.06 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_1
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_1/datasets/traj_dict/traj_dict.pkl
>> qpos to expert:
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_1

============================================================================

----------------------------------------------------------------------------------------------------------------------------
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_1 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_1
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_1 --data train --num-threads 52 --iter 4000 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_1
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_1 --iter 900 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_1
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_0900_test_naivefs.p

============================================================================

E: use the RL result to train the DL estimator 
----------------------------------------------------------------------------------------------------------------------------
saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs.p
>> train estimator with GAN projection, trained with two setting because sometime NAN.
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_0 --add_random_cam False -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_1 --add_random_cam False -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs
>> inferece the estimator result from the H36M 2D pose, to generate pseudo 3D reference motion data, save at traj_dict.pkl. with temp bvh file.
CUDA_VISIBLE_DEVICES=0 python posegan_evaluate.py --note vp_0  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs/vp_0/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_2/datasets/traj_dict/traj_dict.pkl
#CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_1  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_1/results/posemimic/subject_h36m_helix_1/results/iter_4000_train_naivefs/vp_1/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_2/datasets/traj_dict/traj_dict.pkl



============================================================================
========================  helix 2  =========================================
============================================================================
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.06 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_2/datasets/traj_dict/traj_dict.pkl
>> generate expert file from the reference trajectory
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
----
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_2 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
>> inference the RL result using trained model
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_2 --data train --num-threads 52 --iter 2400 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
#python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_2 --data train --num-threads 52 --iter 4000 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_2 --iter 5900 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_2
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_5900_test_naivefs.p
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs.p


=====
saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs.p
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_0 --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_1 --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs
>> inferece the estimator result from the H36M 2D pose, to generate pseudo 3D reference motion data, save at traj_dict.pkl. with temp bvh file.
#CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_0  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs/vp_0/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/datasets/traj_dict/traj_dict.pkl
CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_1  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_2/results/posemimic/subject_h36m_helix_2/results/iter_2400_train_naivefs/vp_1/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/datasets/traj_dict/traj_dict.pkl


============================================================================
========================  helix 3  =========================================
============================================================================
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.06 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_3
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_3/datasets/traj_dict/traj_dict.pkl
>> generate expert file from the reference trajectory
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_3
----
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_3 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_3
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_3 --data train --num-threads 52 --iter 6000 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_3
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_3 --iter 6000 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_3
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_5000_train_naivefs.p

=====

=> complete multiprocessing spends 714.67 seconds
num reset: 140
saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p


==== rib ==== 

============================================================================
H: pose rib
>> use the pretrained policy model to save time
>> convert the RL motion file to 22-joint bvh, which meet the lafan setting (in fact it would be better all EIH use one setting)
python rlpose2bvh.py --pkl_path ../../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p
# train
CUDA_VISIBLE_DEVICES=0 python train.py --cfg train-base-helix_3.yaml
# test
python test-randomfuture-v1.py --cfg test-base-helix_3.yaml
# convert the RIB motion to RL motion format
python ribpose2bvh.py --traj_save_path ../../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_3/datasets/traj_bvh

==== rl ====

I: do refinement on RIB result
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.0 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36mrib --traj-dict-path ./checkpoint/exp_h36mrib_gt2d_v5/helix_3/datasets/traj_dict/traj_dict.pkl
>> qpos to expert:
python ./pose_imitation/data_process/gen_expert.py --num-threads 52 --meta-id meta_subject_h36mrib --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3
>> train policy with pretrained model
# error meet:https://qastack.cn/programming/1367373/python-subprocess-popen-oserror-errno-12-cannot-allocate-memory
# solution: sudo echo 1 > /proc/sys/vm/overcommit_memory
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3
cp checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/models/iter_6000.p  checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/models/iter_0100.p
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --iter 100 --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --data train --num-threads 52 --iter 1200 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --iter 1200 --render --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_3

---
=> complete multiprocessing spends 1670.48 seconds
num reset: 319
saved results to ./checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p


==== videopose === 
E: use the result from I, H to train E
RL: checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p
RIB: checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
----------------------------------------------------------------------------------------------------------------------------
# projection setting A
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_4_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_5_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
# projection setting B
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_6_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_7_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_3/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p

# use default setting #5 if it dose not crash
CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_5_wrib  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_3/results/posemimic/subject_h36m_helix_3/results/iter_6000_train_naivefs/vp_5_wrib/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/datasets/traj_dict/traj_dict.pkl



============================================================================
========================  helix 4  =========================================
============================================================================
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.06 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_4
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_4/datasets/traj_dict/traj_dict.pkl
>> generate expert file from the reference trajectory
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_4
----
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_4 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_4
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_4 --data train --num-threads 52 --iter 6000 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_4
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_4 --iter 6000 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_4
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_5000_train_naivefs.p

=====

#saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p


==== rib ==== 

============================================================================
H: pose rib
>> use the pretrained policy model to save time
>> convert the RL motion file to 22-joint bvh, which meet the lafan setting (in fact it would be better all EIH use one setting)
python rlpose2bvh.py --pkl_path ../../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p
# train
CUDA_VISIBLE_DEVICES=0 python train.py --cfg train-base-helix_4.yaml
# test
python test-randomfuture-v1.py --cfg test-base-helix_4.yaml
# convert the RIB motion to RL motion format
python ribpose2bvh.py --traj_save_path ../../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_4/datasets/traj_bvh

==== rl ====

I: do refinement on RIB result
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.0 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36mrib --traj-dict-path ./checkpoint/exp_h36mrib_gt2d_v5/helix_4/datasets/traj_dict/traj_dict.pkl
>> qpos to expert:
python ./pose_imitation/data_process/gen_expert.py --num-threads 52 --meta-id meta_subject_h36mrib --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4
>> train policy with pretrained model
# error meet:https://qastack.cn/programming/1367373/python-subprocess-popen-oserror-errno-12-cannot-allocate-memory
# solution: sudo echo 1 > /proc/sys/vm/overcommit_memory
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4
cp checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/models/iter_6000.p  checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/models/iter_0100.p
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --iter 100 --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --data train --num-threads 52 --iter 1200 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --iter 1200 --render --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_4

---
#saved results to ./checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p


==== videopose === 
E: use the result from I, H to train E
RL: checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p
RIB: checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
----------------------------------------------------------------------------------------------------------------------------
# projection setting A
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_4_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_5_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
# projection setting B
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_6_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_7_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_4/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p

# use default setting #5 if it dose not crash
CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_5_wrib  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_4/results/posemimic/subject_h36m_helix_4/results/iter_6000_train_naivefs/vp_5_wrib/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/datasets/traj_dict/traj_dict.pkl



============================================================================
========================  helix 5  =========================================
============================================================================
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.06 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_5
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_5/datasets/traj_dict/traj_dict.pkl
>> generate expert file from the reference trajectory
python ./pose_imitation/data_process/gen_expert.py --meta-id meta_subject_h36m --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_5
----
>> train the policy
python pose_imitation/pose_mimic.py --cfg subject_h36m_helix_5 --num-threads 52 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_5
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_5 --data train --num-threads 52 --iter 6000 --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_5
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36m_helix_5 --iter 6000 --render --mocap-folder ./checkpoint/exp_h36m_gt2d_v5/helix_5
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36m --vis-model humanoid_h36m_v4_vis --data test --vis-model-num double --qpos-key traj_pred --eqpos-key traj_orig --traj-dict-path ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_5000_train_naivefs.p

=====

#saved results to ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p


==== rib ==== 

============================================================================
H: pose rib
>> use the pretrained policy model to save time
>> convert the RL motion file to 22-joint bvh, which meet the lafan setting (in fact it would be better all EIH use one setting)
python rlpose2bvh.py --pkl_path ../../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p
# train
CUDA_VISIBLE_DEVICES=0 python train.py --cfg train-base-helix_5.yaml
# test
python test-randomfuture-v1.py --cfg test-base-helix_5.yaml
# convert the RIB motion to RL motion format
python ribpose2bvh.py --traj_save_path ../../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_5/datasets/traj_bvh

==== rl ====

I: do refinement on RIB result
>> bvh to qpos:
python ./pose_imitation/data_process/convert_clip_multiprocess.py --offset-z 0.0 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5
>> take a look at the motion reference
python pose_imitation/vis_pose.py --posemimic-cfg subject_h36mrib --traj-dict-path ./checkpoint/exp_h36mrib_gt2d_v5/helix_5/datasets/traj_dict/traj_dict.pkl
>> qpos to expert:
python ./pose_imitation/data_process/gen_expert.py --num-threads 52 --meta-id meta_subject_h36mrib --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5
>> train policy with pretrained model
# error meet:https://qastack.cn/programming/1367373/python-subprocess-popen-oserror-errno-12-cannot-allocate-memory
# solution: sudo echo 1 > /proc/sys/vm/overcommit_memory
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5
cp checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/models/iter_6000.p  checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/models/iter_0100.p
python pose_imitation/pose_mimic.py --cfg subject_h36mrib --iter 100 --num-threads 52 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5
>> inference the RL result using trained model 
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --data train --num-threads 52 --iter 1200 --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5
>> vis if want to check
python pose_imitation/pose_mimic_eval.py --cfg subject_h36mrib --iter 1200 --render --mocap-folder ./checkpoint/exp_h36mrib_gt2d_v5/helix_5

---
#saved results to ./checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p


==== videopose === 
E: use the result from I, H to train E
RL: checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p
RIB: checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
----------------------------------------------------------------------------------------------------------------------------
# projection setting A
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_4_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=0 python posegan_train.py --note vp_5_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 0.5 --cam_t_range 3. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
# projection setting B
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_6_wrib --add_random_cam True -e 50 -b 512 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p
CUDA_VISIBLE_DEVICES=1 python posegan_train.py --note vp_7_wrib --add_random_cam True -e 50 -b 1024 -lr 3e-4 -lrgcam 1e-4 -lrdcam 1e-4 --df 3 -k gt --dcam_choice dcam_pa1 --cam_r_range 3.14 --cam_t_range 4. --expert_dict_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs.p --checkpoint ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs --extra_expert_dict_path ../imitator/checkpoint/exp_h36mrib_gt2d_v5/helix_5/results/posemimic/subject_h36mrib/results/iter_1200_train_naivefs.p

# use default setting #5 if it dose not crash
CUDA_VISIBLE_DEVICES=1 python posegan_evaluate.py --note vp_5_wrib  -arc 3,3,3 -ch 1024 --pretrain True --evaluate ./checkpoint/exp_h36m_gt2d_v5/helix_5/results/posemimic/subject_h36m_helix_5/results/iter_6000_train_naivefs/vp_5_wrib/ckpt/ckpt_ep_045.bin --traj_save_path ../imitator/checkpoint/exp_h36m_gt2d_v5/helix_6/datasets/traj_dict/traj_dict.pkl


---- end
