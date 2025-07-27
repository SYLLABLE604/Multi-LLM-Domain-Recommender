# export CUDA_VISIBLE_DEVICES=1,2,3,5,6,7
# export PYTHONPATH=${PYTHONPATH}:$(pwd)

# python src/transfer_matrix/cal_and_save_transfer_matrix.py \
#     /root/work/Crowdsourcing/MLDR/MLDR/represent_matrix \
#     /root/model/mistral \


export CUDA_VISIBLE_DEVICES=2,5,6,7
res_path=./res/Self_defin/3_Mistral
mkdir -vp ${res_path}
python src/test.py \
  --config confs/Self_defin/3_Mistral.json \
  -lpm based_on_probility_transfer_logits_fp32_naive_processor \
  -dp cuda:1 -d1 cuda:2  \
  -rsd ${res_path}\