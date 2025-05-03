# AIC CTU System at FEVER 8
Quick and dirty setup suitable for AMI (Conda):

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash

cd /opt/dlami/nvme
git clone https://github.com/heruberuto/FEVER-8-Shared-Task
cd FEVER-8-Shared-Task
./download_data.sh
./installation.sh
conda activate aic
./run_system.sh
```

For any debugging or precomputed CSVS, please get in touch with me at FEVER Slack or via email at ullriher@fal.cvut.cz


