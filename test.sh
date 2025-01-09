#!/bin/bash

export PYTHONUNBUFFERED=1

###################################################################################
# prune
# tiny 
output_name="checkpoints/finetune_meta-network/prune/tiny"
ppo_path="checkpoints/ppo/prune/tiny"

# ppo="acc_73_flops_1_param_6_tk_0.6"
ppo="acc_72_flops_1_param_7_tk_0.1"

cfg="structure_of_meta-network/supernet-T.yaml"
batch_size=64
token_mode="prune"

# # small
# # pruning
# output_name="checkpoints/finetune_meta-network/prune/small"
# ppo_path="checkpoints/ppo/prune/small"

# # ppo="acc_80_flops_3_param_21_tk_0.36"
# ppo="acc_79.5_flops_2.7_param_20_tk_0.13"

# cfg="structure_of_meta-network/supernet-S.yaml"
# batch_size=64
# token_mode="prune"

# # base
# output_name="checkpoints/finetune_meta-network/prune/base"
# ppo_path="checkpoints/ppo/prune/base"

# # ppo="acc_81_flops_11_param_63_tk_0.56"
# ppo="acc_81.4_flops_10.5_param_58_tk_0.51"

# cfg="structure_of_meta-network/supernet-B.yaml"
# batch_size=64
# token_mode="prune"

###################################################################################
# # merge
# # tiny
# output_name="checkpoints/finetune_meta-network/merge/tiny"
# ppo_path="checkpoints/ppo/merge/tiny"

# # ppo="acc_70_flops_1_param_6_tk_0.35" 
# ppo="acc_72.5_flops_1_param_6_tk_0.5"

# cfg="structure_of_meta-network/supernet-T.yaml"
# batch_size=64
# token_mode="merge"

# # small
# output_name="checkpoints/finetune_meta-network/merge/small"
# ppo_path="checkpoints/ppo/merge/small"

# # ppo="acc_79.5_flops_2.4_param_21_tk_0.18"
# ppo="acc_80_flops_2.8_param_24_tk_0.38"

# cfg="structure_of_meta-network/supernet-S.yaml"
# batch_size=64
# token_mode="merge"

# # base
# output_name="checkpoints/finetune_meta-network/merge/base"
# ppo_path="checkpoints/ppo/merge/base"

# ppo="acc_81.5_flops_11_param_72_tk_0.65"
# # ppo="acc_81.6_flops_10.7_param_74_tk_0.65"

# cfg="structure_of_meta-network/supernet-B.yaml"
# batch_size=64
# token_mode="merge"

###################################################################################
# Prune-Merge
# # tiny 
# output_name="checkpoints/finetune_meta-network/prune-then-merge/tiny"
# ppo_path="checkpoints/ppo/prune-then-merge/tiny"

# ppo="acc_73.8_flops_1_param_6.8_tk_0.4"
# # ppo="acc_73.8_flops_1_param_7_tk_0.3"

# cfg="structure_of_meta-network/supernet-T.yaml"
# batch_size=64
# token_mode="prune_merge"

# # small
# output_name="checkpoints/finetune_meta-network/prune-then-merge/small"
# ppo_path="checkpoints/ppo/prune-then-merge/small"

# ppo="acc_79.5_flops_2.2_param_23_tk_0.29"
# # ppo="acc_79.5_flops_2.7_param_22_tk_0.33"

# cfg="structure_of_meta-network/supernet-S.yaml"
# batch_size=64
# token_mode="prune_merge"

# # base
# output_name="checkpoints/finetune_meta-network/prune-then-merge/base"
# ppo_path="checkpoints/ppo/prune-then-merge/base"

# ppo="acc_80_flops_7_param_77_tk_0.27"
# # ppo="acc_81.5_flops_11.5_param_60_tk_0.73"

# cfg="structure_of_meta-network/supernet-B.yaml"
# batch_size=64
# token_mode="prune_merge"
###################################################################################

python -u test_model.py --output_dir $output_name --ppo_path $ppo_path --ppo_name $ppo --cfg $cfg --token_mode $token_mode --num_workers 8 --batch-size $batch_size
