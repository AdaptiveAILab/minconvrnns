# Training run commands

####################
### GEOPOTENTIAL ###
####################

# ConvGRU on geopotential, sequence length 24, teacher forcing seps 20
python scripts/train.py seed=1234 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=gp_cgru_v0 device=cuda:0
python scripts/train.py seed=1235 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=gp_cgru_v1 device=cuda:0
python scripts/train.py seed=1236 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=gp_cgru_v2 device=cuda:0
python scripts/train.py seed=1237 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=gp_cgru_v3 device=cuda:0
python scripts/train.py seed=1238 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=gp_cgru_v4 device=cuda:0

# ConvLSTM on geopotential, sequence length 24, teacher forcing seps 20
python scripts/train.py seed=1234 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=gp_clstm_v0 device=cuda:0
python scripts/train.py seed=1235 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=gp_clstm_v1 device=cuda:0
python scripts/train.py seed=1236 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=gp_clstm_v2 device=cuda:0
python scripts/train.py seed=1237 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=gp_clstm_v3 device=cuda:0
python scripts/train.py seed=1238 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=gp_clstm_v4 device=cuda:0

# MinConvGRU on geopotential, sequence length 24, teacher forcing seps 20
python scripts/train.py seed=1234 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=gp_mcgru_v0 device=cuda:0
python scripts/train.py seed=1235 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=gp_mcgru_v1 device=cuda:0
python scripts/train.py seed=1236 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=gp_mcgru_v2 device=cuda:0
python scripts/train.py seed=1237 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=gp_mcgru_v3 device=cuda:0
python scripts/train.py seed=1238 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=gp_mcgru_v4 device=cuda:0

# MinConvLSTM on geopotential, sequence length 24, teacher forcing seps 20
python scripts/train.py seed=1234 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=gp_mclstm_v0 device=cuda:0
python scripts/train.py seed=1235 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=gp_mclstm_v1 device=cuda:0
python scripts/train.py seed=1236 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=gp_mclstm_v2 device=cuda:0
python scripts/train.py seed=1237 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=gp_mclstm_v3 device=cuda:0
python scripts/train.py seed=1238 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=gp_mclstm_v4 device=cuda:0

# MinConvExpLSTM on geopotential, sequence length 24, teacher forcing seps 20
python scripts/train.py seed=1234 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v0 device=cuda:0
python scripts/train.py seed=1235 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v1 device=cuda:0
python scripts/train.py seed=1236 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v2 device=cuda:0
python scripts/train.py seed=1237 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v3 device=cuda:0
python scripts/train.py seed=1238 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v4 device=cuda:0


# Model evaluation
python scripts/evaluate.py -c outputs/gp_* -d cuda:0
