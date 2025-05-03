# AIC CTU System at FEVER 8
Quick and dirty setup:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash

cd /opt/dlami/nvme/
git clone https://github.com/heruberuto/FEVER-8-Shared-Task
cd fever-8-shared-task

download_data.sh
installation.sh
conda activate hero
run_system.sh
```



## Evaluation of Shared Task Systems

The shared task evaluates fact-checking systems on AVeriTeC across three dimensions:


### Reproducibility

Every system participating in the shared task is expected to produce reproducible code. Subsequently, all participating systems will be evaluated during inference on the test set by the shared task organisers on a virtual machine. 

The virtual machine is a [g5.2xlarge EC2 instance](https://aws.amazon.com/de/ec2/instance-types/g5/) on AWS. The configuration of the virtual machine is:
- GPU: Nvidia A10G, with 23GB memory
- CPU: 8 vCPUs
- RAM: 32GB
- Storage: 450GB (including the AVeriTeC knowledge base)


 We provide two methods for participants to make sure that their code will run on the virtual machine:

 #### Docker Image

 We provide a Docker image with the exact configuration as the EC2 virtual machine. You can download the following image:
 - An empty docker image, preconfigured like the virtual machine, [here](https://drive.google.com/file/d/1-AiMrgjWUmcSPFehCF7wI13HJerT3MlO/view?usp=sharing). The docker was created using the setup in `docker_setup` and running `docker build -t averitec .`.

 To load the docker, execute `gunzip -c averitec.tar.gz | docker load` and to start the docker with the image, run `docker run --gpus all -it averitec`.


 #### Amazon AMI

To fully ensure that your system will run on the virtual machine, you can configure an identical VM instance yourself on AWS. The Image name is: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250107` and the AMI-ID is `ami-0940d6cb015d3bae4`, configured with 120 GB of root storage. 

The commands to set up Conda on that instance are:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash
```

The commands to set up Docker (for running systems submitted via Docker) are:

```bash
vim /etc/docker/daemon.json
```

and adding `"data-root":"/opt/dlami/nvme/docker"` into the json.

```bash
sudo systemctl restart docker
```

All systems will be executed on the EC2 instance storage `/opt/dlami/nvme`.


### Efficiency

At inference time, a submitted system must verify a single claim on average within *at most* 1 minute. A valid system submission must run on the aforementioned VM within the specified time constraints.


### Prediction performance
The AVeriTec scoring is built on the [FEVER scorer](https://github.com/sheffieldnlp/fever-scorer), with a few key changes:
- Claims in Fact-Checking datasets are typically supported or refuted by evidence, or there is not enough evidence. We add a fourth class: conflicting evidence/cherry-picking. This covers both conflicting evidence, and technically true claims that mislead by excluding important context, i.e., the claim has both supporting and refuting evidence.
- Unlike FEVER, using a closed source of evidence such as Wikipedia, AVERITEC is intended for use with evidence retrieved from the open web. Since the same evidence may be found in different sources, we cannot rely on exact matching to score retrieved evidence. As such, we instead rely on approximate matching. Specifically, we use the [Ev2R score](https://arxiv.org/abs/2411.05375) to find an optimal matching of provided evidence to annotated evidence. The Ev2R score thus replaces the Hungarian Meteor score used in the [previous shared task](https://fever.ai/2024/task.html).
- The shared task uses `Llama 3.3 70B` as the grader.

The evaluation script is `averitec_evaluate.py` and is called automatically by `system_inference.sh`. The script requires the use of an API key to run the Llama model. The script takes substential time to run, with about 4 hours on the dev set. Note that the evaluation script is identical to the one executed on the Huggingface Leaderboard. Therefore, the input format of both predictions and gold answers have to be in .csv format. The script `prepare_leaderboard_submission.py` can be used for the conversion into the right format.


## Preparing your submission
This repository also serves as a valid system submission, executable on our virtual machine. Every system submission must contain the following two files:

0. DO NOT MODIFY: A `download_data.sh` which downloads all relevant AVeriTec data (knowledge store and test data). AVeriTeC test data will be downloaded into `data_store/averitec/test.json` and the knowledge store will be downloaded into `knowledge_store/test/`
1. MODIFY: An `installation.sh` which sets up a conda environment for your system and installs all relevant packages, and other dependencies (e.g. Java, any models that need to be preloaded etc.)
2. MODIFY: A configured `system_inference.sh`, which runs the inference pipeline of the system, finishing with the system's final output. 

The pipeline for running a submitted system on the virtual machine will consist of the following calls:

```bash
download_data.sh
installation.sh
conda activate [YOUR ENVIRONMENT]
run_system.sh
```

The script `run_system.sh` calls `system_inference.sh` while measuring the total time the system takes.

A system can be submitted either:

1. As an URL to a Docker image which will be downloaded on the VM and executed.
2. As a ZIP file, which will be unzipped on the VM and executed.


## Baseline 

The baseline (located in `baseline/`) for the AVeriTeC Shared Task 2025 is a computationally optimized version of the [HerO system](https://github.com/ssu-humane/HerO), proposed by the SSU-Humane Team for the AVeriTeC 2024 Shared Task. The baseline largely follows their code, and only optimizes the computations of the HerO pipeline. To run the baseline, execute the commands above:

```bash
download_data.sh
installation.sh
conda activate hero
run_system.sh
```


