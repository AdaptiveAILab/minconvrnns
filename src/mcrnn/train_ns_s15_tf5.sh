# Training run commands

#####################
### NAVIER-STOKES ###
#####################

# ConvGRU on Navier-Stokes, sequence length 15, teacher forcing steps 5
python scripts/train.py seed=1234 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s15_tf5_v0 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1235 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s15_tf5_v1 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1236 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s15_tf5_v2 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1237 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s15_tf5_v3 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1238 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=ns_cgru_s15_tf5_v4 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1

# ConvLSTM on Navier-Stokes, sequence length 15, teacher forcing seps 5
python scripts/train.py seed=1234 +experiment=navier-stokes model=convlstm model.hidden_sizes[25,25,25,25] model.name=ns_clstm_s15_tf5_v0 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1235 +experiment=navier-stokes model=convlstm model.hidden_sizes[25,25,25,25] model.name=ns_clstm_s15_tf5_v1 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1236 +experiment=navier-stokes model=convlstm model.hidden_sizes[25,25,25,25] model.name=ns_clstm_s15_tf5_v2 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1237 +experiment=navier-stokes model=convlstm model.hidden_sizes[25,25,25,25] model.name=ns_clstm_s15_tf5_v3 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1238 +experiment=navier-stokes model=convlstm model.hidden_sizes[25,25,25,25] model.name=ns_clstm_s15_tf5_v4 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1

# MinConvGRU on Navier-Stokes, sequence length 15, teacher forcing steps 5
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s15_tf5_v0 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s15_tf5_v1 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s15_tf5_v2 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s15_tf5_v3 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=ns_mcgru_s15_tf5_v4 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1

# MinConvLSTM on Navier-Stokes, sequence length 15, teacher forcing steps 5
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s15_tf5_v0 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s15_tf5_v1 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s15_tf5_v2 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s15_tf5_v3 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=ns_mclstm_s15_tf5_v4 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1

# MinConvExpLSTM on Navier-Stokes, sequence length 15, teacher forcing steps 5
python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s15_tf5_v0 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s15_tf5_v1 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s15_tf5_v2 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s15_tf5_v3 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1
python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s15_tf5_v4 data.sequence_length=15 training.teacher_forcing_steps=5 validation.teacher_forcing_steps=5 device=cuda:1

# Model evaluation
python scripts/evaluate_navier-stokes.py -c outputs/ns_*s15_tf5_v* -d cuda:1