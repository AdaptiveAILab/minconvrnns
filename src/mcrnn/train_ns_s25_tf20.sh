# Training run commands

#####################
### NAVIER-STOKES ###
#####################

# ConvGRU on Navier-Stokes, sequence length 25, teacher forcing steps 20
python scripts/train.py seed=1234 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1235 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1236 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1237 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1238 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0

# ConvLSTM on Navier-Stokes, sequence length 25, teacher forcing steps 20
python scripts/train.py seed=1234 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=ns_clstm_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1235 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=ns_clstm_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1236 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=ns_clstm_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1237 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=ns_clstm_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1238 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=ns_clstm_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0

# MinConvGRU on Navier-Stokes, sequence length 25, teacher forcing steps 20
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0

# MinConvLSTM on Navier-Stokes, sequence length 25, teacher forcing steps 20
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0

# MinConvLSTM on Navier-Stokes, sequence length 25, teacher forcing steps 20
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda:0

# Model evaluation
python scripts/evaluate.py -c outputs/ns_*s25_tf20_v* -d cuda:0