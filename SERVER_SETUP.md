# MANUAL SETUP INSTRUCTIONS FOR KALPANA SERVER ACCESS

## Quick Setup (30 seconds)

### Step 1: Copy SSH key to server
Open a terminal and run:

```bash
ssh-copy-id yash.sarang@kalpana.minds.iitb.ac.in
```

When prompted, enter password: `yash.sarang`

### Step 2: Test connection
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in
```

Should connect without password now.

---

## Alternative: Manual Setup

If `ssh-copy-id` doesn't work, do this manually:

```bash
# 1. Display your public key
cat ~/.ssh/id_rsa.pub

# 2. SSH to server (with password)
ssh yash.sarang@kalpana.minds.iitb.ac.in

# 3. On the server, add the key
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
# Paste the public key, save and exit

# 4. Set permissions
chmod 600 ~/.ssh/authorized_keys

# 5. Exit and test
exit
ssh yash.sarang@kalpana.minds.iitb.ac.in  # Should work without password
```

---

## Once Connected: Run Control Experiment

```bash
# 1. Navigate to repo
cd ~/Hybrid-Pairs-Trading-Ensemble

# 2. Pull latest changes
git pull origin main

# 3. Check server instructions
cat Implementation/CLUSTER_MONITORING.md  # or similar file

# 4. Activate environment
source .venv/bin/activate

# 5. Navigate to experimental folder
cd Implementation/experimental-ablation

# 6. Submit SLURM job (DO NOT RUN ON LOGIN NODE)
# Create job script or use existing ones in jobs/ directory

# OR if running interactively (get compute node first):
srun --partition=your_partition --pty bash
source /path/to/.venv/bin/activate
cd /path/to/experimental-ablation
python run_control_experiment_direct.py
```

---

## Your Public Key (for reference):
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC+DqJ7WC6ZiqbY590lFi8TwqOWnMfsX4+uTVVzlIsW1mNxCBlS/D9j6otyj1LATzOZHh2/RAgxA8Y8g5KBgc2FETeO9cFTrXvY/7qe+hxNLWY2rutP9Q0XxGeak0aV4qwkEeHeOgEef/T1OoDbxPdo57CYXixJDA6Aiqnl8DEFKuZy6T1KimC1/ox1NUUrGtVF3xcv2JnCG/fFNy/E5CqOGZRGkMpwIrmnHbEsCvBc2yUMuRW2Fu30QNKe0f0AQ5RSkh0aUH7glry3Zmncg2Lg1eZjaSKTcRr/n17OUGpxeUrzkCQF6ug5bK2TLwG2ukhpuISohNlvbGnx+mIpmUI1kcbaNtLce40brxJA2jbCP1FNjhc4U5NbFZnRARGrLaxpB/HX/91ZPHSHF+zDrCKg1KN01PALlORkuTSQdvu5x/iSTJ+psYoWT89CsIIbU+oBHgcnJP3QCAy0kFt5c/kmokkol0I1UfAEuqeLR1aNMkhTtsk4Okf4eP3NQoln37EHVWIG2v2Gh7cHm0XCjmhpYS0QTDAKP7FbA2pU2E1sBFoxlUGpnVlc6xvTtGwUqMP4uNpw3HmoEKwfRe2V/ny/7M42XdNRMmExSNW0eStcT9BfkZY3d1Ib7wfbIbb/8l+ArHE1F3N9Ev/1zoukzaJWIzOlHegmIK9Rkf3rDCqSNQ== hermes-agent@INTXVI-INT5715-L
```
