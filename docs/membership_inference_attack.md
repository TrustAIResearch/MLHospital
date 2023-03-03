# Membership inference attacks

# Step 0: train target/shadow models
```
cd MLHospital/mlh/examples;
python train_target_models.py --mode target;
python train_target_models.py --mode shadow;
```
Note that you can also specify the `--training_type` with different defense mechanisms, e.g., `Normal`, `LabelSmoothing`, `AdvReg`, `DP`, `MixupMMD`, and `PATE`.

# Step 1: perform membership inference attacks
```
python mia_example.py 
```
Note that you can also specify the `--attack_type` with different attacks, e.g., `black-box`, `black-box-sorted`, `black-box-top3`, `metric-based`, and `label-only`.
"black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only".