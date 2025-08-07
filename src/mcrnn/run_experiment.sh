#!/bin/bash


####### Runtime experiments #######
# Geopotential experiments
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=convgru model.hidden_sizes=[14,14,14] model.name=test device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=convlstm model.hidden_sizes=[12,12,12] model.name=test device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=minconvgru model.hidden_sizes=[24,24,24] model.name=test device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=minconvlstm model.hidden_sizes=[20,20,20] model.name=test device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=test device=cuda training.epochs=6


# Navier-Stokes experiments
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=convgru model.hidden_sizes=[28,28,28,28] model.name=test data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=convlstm model.hidden_sizes=[25,25,25,25] model.name=test data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvgru model.hidden_sizes=[49,49,49,49] model.name=test data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvlstm model.hidden_sizes=[40,40,40,40] model.name=test data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda training.epochs=6
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=test data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda training.epochs=6




####### Train models #######
# MinConvExpLSTM on geopotential, sequence length 24, teacher forcing seps 20
# sbatch train_job.sh python scripts/train.py seed=1234 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v0 device=cuda
# sbatch train_job.sh python scripts/train.py seed=1235 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v1 device=cuda
# sbatch train_job.sh python scripts/train.py seed=1236 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v2 device=cuda
# sbatch train_job.sh python scripts/train.py seed=1237 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v3 device=cuda
# sbatch train_job.sh python scripts/train.py seed=1238 +experiment=geopotential model=minconvexplstm model.hidden_sizes=[20,20,20] model.name=gp_mcexplstm_v4 device=cuda


# MinConvExpLSTM on Navier-Stokes, sequence length 25, teacher forcing steps 20
sbatch train_job.sh python scripts/train.py seed=1234 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v0 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda
sbatch train_job.sh python scripts/train.py seed=1235 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v1 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda
sbatch train_job.sh python scripts/train.py seed=1236 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v2 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda
sbatch train_job.sh python scripts/train.py seed=1237 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v3 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda
sbatch train_job.sh python scripts/train.py seed=1238 +experiment=navier-stokes model=minconvexplstm model.hidden_sizes=[40,40,40,40] model.name=ns_mcexplstm_s25_tf20_v4 data.sequence_length=25 training.teacher_forcing_steps=20 validation.teacher_forcing_steps=20 device=cuda
