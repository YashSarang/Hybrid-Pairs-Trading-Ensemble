

CMInDS Kalpana Cluster
User Guide & Usage Policies
Welcome to the CMInDS High-Performance Computing (HPC) Cluster.
This guide explains:
How to access the cluster
How to select the correct hardware
Mandatory job submission policies
Resource limits and best practices
Please read carefully. Policies are strictly enforced by the scheduler.

1. Accessing the Cluster
1.1 Login Host
All users must login using the main gateway:
ssh <username>@kalpana.minds.iitb.ac.in
There are no direct user logins to compute nodes.
1.2 Home Directory
Your home directory:
/users/student/<group>/<user>
Shared across login and compute nodes
Files created anywhere are instantly visible everywhere

1.3 SSH Connection Instructions
Open a terminal on your local machine
Run:
ssh <ldapusername>@kalpana.minds.iitb.ac.in
Enter your LDAP password when prompted

1.4 LDAP Authentication
You must have a CMInDS LDAP account to access the cluster.
If you do not have one, raise a request at CMInDS Helpdesk.

1.5 Network Access Requirements
1.5.1 Network Access
On IITB Network: You must be connected to the IITB (Indian Institute of Technology Bombay) network to access the GPU server directly.


From Outside IITB Network: If you wish to use the server from outside the IITB network, you need to use a VPN. You can download the VPN software and follow the relevant installation instructions for your operating system from the IITB website:


Visit IITB VPN Documentation > Go to How To’s > VPN for detailed steps.

1.5.2 Internet Connectivity on the Server
We have set up a cron job to automatically establish an internet connection when required. If the Bochner server is still not connected to the internet. To gain internet access while on the server, you must first run a specific command. Please refer to the following document for instructions:
Internet Access Without GUI



2. Hardware & Partitions
The cluster is divided into partitions, each mapped to specific node and research groups.
You must submit jobs only to the partition assigned to your group.
Partition
Node Name
GPU Resources
Intended Usage
cn3_l40s
anandi
8× L40S (48GB VRAM)
Department-wide usage, large models, high-end training


3. Environment Management (Strongly Recommended)
To ensure a clean, reproducible workflow and avoid dependency conflicts, you should create and manage your own virtual environments. This is essential when working with multiple projects or packages that may require different versions of libraries.

3.1 Available Environment Managers
You may use any environment manager:
- Miniconda – Widely used for managing Python environments and packages
- Mamba – A faster drop-in replacement for Conda
- uv – A new, fast Python package and virtual environment manager
- venv – Built-in Python tool for lightweight environment creation
- pipx – Ideal for installing and running Python CLI tools in isolated environments

3.2 Setup Best Practices
1. Create environment in your project directory (not home directory) to save quota
2. Always activate environment in your SLURM script before running code
3. Use requirements.txt or environment.yml for reproducibility
4. Test environment locally before submitting large jobs
5. Document your environment setup in your project README

3.3 Example Setup in SLURM Script
```bash
# Option 1: Using venv (lightweight)
source /path/to/project/.venv/bin/activate

# Option 2: Using Conda
source ~/.bashrc
conda activate my_env

# Option 3: Using Mamba
source ~/.bashrc
mamba activate my_env
```

You are responsible for installing and configuring your environments according to your workflow. Most tools require internet access to download packages.

4. Mandatory Job Policies (Strictly Enforced)
Jobs that violate these rules will be automatically rejected.
1. Account Flag (Required)
You must specify the correct project account.
For department-wide usage on cn3_l40s:
#SBATCH --account=cminds_anandi

2. Interactive Jobs Disabled
Interactive modes are completely disabled:
srun
salloc
Only sbatch scripts are supported.

3. Time Limit (Required)
You must define walltime:
#SBATCH --time=04:00:00
If unspecified → default is 30 minutes.

4. Memory Request (Required)
You must explicitly request memory:
#SBATCH --mem=16G

5. CPU-GPU Ratio Limit
To prevent CPU hoarding:
Maximum 8 CPUs per 1 GPU
Example:
If requesting:
#SBATCH --gres=gpu:1
Do not request more than:
#SBATCH --cpus-per-task=8

5. Resource Limits (QoS Policies)
Strict limits ensure fair resource sharing.

Partition: cn3_l40s (Department-wide)
Max GPUs per job: 2
Maximum runtime: 24 hours (strict limit)
Max running jobs per user: 2
Max total jobs per user: 4 (2 running + 2 pending)
Total queue capacity: 16 jobs (cluster-wide)
If full → new submissions are blocked

6. Job Submission 
Standard Batch Script (Recommended & Only Supported Method)
Save as job.sh:
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --account=cminds_anandi       # Mention correct account
#SBATCH --partition=cn3_l40s          # Target L40S node
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --ntasks=1                     # No. of tasks 
#SBATCH --cpus-per-task=4             # Max 8 CPUs per GPU
#SBATCH --mem=16G                     # RAM request
#SBATCH --time=04:00:00               # Runtime (HH:MM:SS)
#SBATCH --output=logs/%x_%j.out       # Stdout log
#SBATCH --error=logs/%x_%j.err        # Stderr log

# 1. Load environment
source ~/.bashrc
conda activate my_env

# 2. Move to working directory
cd /users/student/pg/pg24/yash.sarang/project

# 3. Run your script
python train_model.py
Submit using:
sbatch job.sh

7. Best Practices
Do NOT Run Heavy Tasks on Login Nodes
Login nodes are only for:
Editing files
Submitting jobs
Light commands
Running training or heavy computation here will result in your process being killed.

Monitoring Jobs
Check job status:
squeue -u <your_username>
Monitor log output:
tail -f logs/my_training_<job_id>.out




8. Data Management & Storage Policy
To ensure fair and efficient use of shared storage resources, the following policies are strictly enforced.

8.1 Home Directory Storage Quota
Each user has a 200 GB storage quota on their home directory.
Once you reach the 200 GB limit:
You will not be able to write new files
Jobs may fail due to insufficient disk space
Writing will resume only after you reduce usage
This quota ensures equitable access to shared infrastructure.
Important Notice on Dataset Storage
A large collection of commonly used datasets is already available at:
/janaki/common/Datasets
Please check this directory before downloading or copying any dataset.
To avoid unnecessary duplication and storage pressure:
Do not store large datasets in your home directory
Do not download datasets that are already available in /janaki/common/Datasets
Use the shared datasets directly from the NAS whenever possible
Storing duplicate datasets in home directories is one of the primary causes of quota exhaustion and inefficient storage usage.

Best Practices
Please follow these guidelines to maintain server health:
1. Clean Up Regularly
After completing experiments:
Delete temporary outputs
Remove large logs
Remove unused virtual environments
Delete intermediate checkpoints you no longer need

2. Monitor Your Usage
Check your disk usage:
du -sh ~
Check your quota:
quota -u <your-username>
Do not wait until you hit the limit — free space proactively.

3. Backup Critical Data to Janaki NAS
All important models, checkpoints, and datasets should be backed up to Janaki NAS.

8.2 Janaki NAS Storage
The NAS storage is mounted at:
/janaki
It contains two main directories:
/janaki/common
/janaki/backup

/janaki/common
Accessible to all users
Designed for collaboration
Suitable for shared datasets
Users can move folders here and inform collaborators
Recommended Usage
Store datasets required by multiple users
Move large datasets here when not actively using them
Free up home directory space by relocating inactive data

/janaki/backup
Private directory for each user
Not accessible to other users
Intended for:
Trained models
Checkpoints
Private research data
Each user has a 500 GB quota for their backup folder.
Use this space judiciously.

Accessing Your Personal Backup Folder
Go to your home directory:
cd ~
Get your absolute home directory path:
pwd
Your backup path will be:
/janaki/backup/<your_home_directory_path>
Example
If your home directory is:
/home/username
Your backup folder will be:
/janaki/backup/home/username

8.3 Important Rules
Do NOT Change Permissions on Janaki
Do not modify directory permissions
Do not use chmod -R on NAS directories
Any data loss or system errors caused by permission changes will not be supported by system administrators.

8.4 Shared Responsibility
This is shared infrastructure used by the entire department.
Please:
Monitor your storage
Clean up regularly
Use NAS appropriately
Be considerate of other users
Responsible storage management ensures smooth operation for everyone.

9. Troubleshooting and Support
9.1 Common Issues
SSH Connection Issues: Ensure you’re using the correct SSH credentials and the server address is up-to-date.
Package Issues: If any required software packages are missing, please request Professor Arjun for installation.
9.2 Getting Help
If you encounter any technical issues or have questions, please raise a support ticket at: CMInDS Helpdesk
9.3 Best  Practices
Environment Management: Use Conda environments to isolate different projects and avoid dependency conflicts.
Data Backup: Regularly back up your data to avoid data loss.
Fair Usage: Respect the guidelines regarding GPU usage and ensure that your work does not interfere with other users.
Use the server wisely – we all depend on it for our research, so let’s keep it running smoothly and responsibly. Remember, your contribution to keeping the server clean and efficient is appreciated by all users!

10. Troubleshooting Common Errors
Error Message
Meaning
Fix
--account is mandatory
Account flag missing
Add #SBATCH --account=cminds_anandi
Interactive jobs disabled
Used srun or salloc
Use sbatch only
Too many CPUs for requested GPUs
>8 CPUs per GPU requested
Reduce --cpus-per-task
Job violates accounting/QOS policy
>2 GPUs requested on L40S
Use --gres=gpu:1 or gpu:2
QOSGrpSubmitJobLimit
Queue full (16 jobs total)
Wait before submitting


Summary Checklist Before Submitting
 Correct partition selected
 Account specified
 GPU requested
 CPUs ≤ 8 per GPU
 Memory specified
 Time specified
 Using sbatch (not interactive mode)





FAQ: Ensuring You’re Using the Correct PyTorch (GPU) Version on the Cluster

 Why is my PyTorch code not using the GPU?
One of the most common issues is accidentally installing the CPU-only version of PyTorch instead of the CUDA-enabled (GPU) version.
If you install torch without specifying the correct CUDA build, it may default to a CPU build — meaning your code will run only on CPU, even if a GPU is allocated.

 How can I check if my environment is using GPU-enabled PyTorch?
You can run a quick SLURM test job to verify whether CUDA is available and which device PyTorch is using.
Below is a minimal test script you can submit to the cluster:
#!/bin/bash

#SBATCH --job-name=testing_cuda
#SBATCH --account=cminds_anandi       # Use the correct account
#SBATCH --partition=cn3_l40s          # Target L40S node
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8             
#SBATCH --mem=100G                    
#SBATCH --time=00:05:00               
#SBATCH --output=logs/%x_%j.out       
#SBATCH --error=logs/%x_%j.err        

# 1. Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <your env>

# 2. Move to working directory
cd /users/student/<your directory>

# 3. Run a quick PyTorch device check
python - <<'EOF'
import torch

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

 What output should I expect if everything is correct?
If your environment is properly configured with GPU-enabled PyTorch, you should see:
CUDA available: True
Using device: cuda
GPU name: NVIDIA L40S
If you instead see:
CUDA available: False
Using device: cpu
then your environment is using the CPU-only version of PyTorch.

 How do I fix a CPU-only PyTorch installation?
Reinstall PyTorch with CUDA support inside your conda environment.
For example (adjust CUDA version as needed):
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
Or using conda:
conda install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia
Always verify the installation again using the test script above.
 Why does this matter?
If you accidentally use CPU-only PyTorch:
Your jobs will run much slower
GPU resources will be allocated but unused
Training deep learning models may take hours instead of minutes
Running this quick check before large jobs can save significant time and compute resources.

Recommended Best Practice
Before launching long training jobs:
Activate your conda environment
Run the CUDA test job
Confirm CUDA available: True
Then launch your actual training script
This small verification step can prevent major delays and wasted GPU allocations.

