<div align="center">
    <h1 style="font-size:300%;">Gradually Updated Neural Networks for Large-Scale Image Recognition</h1>
    <h3 style="font-size:100%;">Aishwarya Radhakrishnan <br>
    March 14, 2020</h3>
</div>
  
<hr>



## Abstract / Introduction

*Convolutional Neural Networks(CNNs) decreases the computation cost for Computer Vision problems than the Deep Neural Networks(DNNs). Increasing the depth leads to increase in parameters of the CNN and hence, increase in computation cost. But as depth plays a vital role in Computer Vision problems, increasing the depth without increasing the computation cost can lead to increased learning. This is achieved by introducing
computation orderings to the channels within convolutional layers.*

*Gradually Updated Neural Networks (GUNN) as opposed to the default Simultaneously Updated Convolutional Network (SUNN / CNN), gradually updates the feature representations against the
traditional convolutional network that computes its output
simultaneously.  This is achieved by updating one channel at a time and using the newly computed parameter of this channel and old parameter of other channels to get another channels in a single convolutional layer. This is repeated for all the channels untill all the old parameters of a single convolutional layer are updated to new values. Thus a single convolutional layer is broken down into multiple gradually updated convolutional layer.*



## Prerequisites

Installing pip, tensorflow

## CIFAR-10 Dataset

Number of Training examples = 50000 <br>
Number of Test examples = 10000 <br>
X_train shape: (50000, 32, 32, 3) <br>
Y_train shape: (50000, 10) <br>
X_test shape: (10000, 32, 32, 3) <br>
Y_test shape: (10000, 10)


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/CIFAR10.png' /><br>
<figcaption>Figure 1: CIFAR 10 dataset with 10 classes that are randomly chosen</figcaption></div>
</figure>



## GUNN Layer implementation (Keras Custom Layer)

Keras Custom Layer needs to be used because GUNN layer consist of convolutional network which decomposes
the original computation into multiple sequential channel-wise convolutional operations as opposed to single convolutional operations in Conv2D.


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/SUNNvsGUNN.png' /><br>
<figcaption>Figure 2: SUNN vs GUNN. SUNN architecture is our usually used Convolution Layer while GUNN is the channel-wise decomposed architecture.</figcaption></div>
</figure>


The expansion rate is used to know how many channels are simultaneously updated when decomposing the channels. The kernel channel is the same as the expansion rate. The kernel size is given by fxf.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnLayer.png' /><br>
<figcaption>Figure 3: Gunn2D Keras custom layer. The Gunn2D layer expands channel at K rate and also uses Residual Network identity block implementation. </figcaption></div>
</figure>

<br>

The Keras custom layer needs forward propagation definition only. 

Note: If you want to build a keras model with a custom layer that performs a custom operation and has a custom gradient, you should use @tf.custom_gradient.


## Building GUNN-15 Model in Keras for 10 classes of CIFAR-10 dataset

Using Keras Convolutional Neural Networks building blocks and custom implemented Gunn2D layer created GUNN-15 Model.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnModel.png' /><br>
<figcaption>Figure 4: Gunn Model. The Gunn Model has Gunn2D layer implemented at the layers marked as * i.e. Conv2*, Conv3* and Conv4* layers. </figcaption></div>
</figure>



## Training

* Ocelote includes a large memory node with 2TB of RAM available on 48 cores.  More details on the Large Memory Node



             total       used       free     shared    buffers     cached
Mem:          3832       2538       1293          0        164       1944
-/+ buffers/cache:        429       3402
Swap:         3275          0       3275


             total       used       free     shared    buffers     cached
Mem:             3          2          1          0          0          1
-/+ buffers/cache:          0          3
Swap:            3          0          3



[aishwarya34@gatekeeper ~]$ elgato va
The authenticity of host 'login.elgato.hpc.arizona.edu (10.140.86.3)' can't be established.
RSA key fingerprint is 61:d7:a3:46:0e:8c:52:6a:ff:b7:00:41:79:6f:56:57.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'login.elgato.hpc.arizona.edu,10.140.86.3' (RSA) to the list of known hosts.
aishwarya34 current allocations:
------------------------------
Group           	Type                 	Available Time
-----           	----                 	--------------
cscheid         	Standard             	36000:00
cscheid         	ElGato Standard      	7000:00


[aishwarya34@gatekeeper ~]$ ocelote free -g
             total       used       free     shared    buffers     cached
Mem:           188        181          7          0          0        177
-/+ buffers/cache:          2        185
Swap:           15          0         15
[aishwarya34@gatekeeper ~]$ elgato free -g
              total        used        free      shared  buff/cache   available
Mem:             62           0           1           0          59          61
Swap:             7           0           7



 HPC systems use a queueing system (PBS) to manage compute resources and schedule jobs. 
 
 Jobs are submitted to the batch system using PBS scripts that specify the jobs required resources such as number of nodes, cpus, memory, group, wallclock time.  
 
Does TensorFlow view all CPUs of one machine as ONE device?
 By default all CPUs available to the process are aggregated under cpu:0 device.


Scheduler Options
1.  Select
The basic select statement is:

#PBS -l select=X:ncpus=Y:mem=Z

X = the number of nodes or units of the resources required

Y = the number of cores (individual processors) required on each node

Z = the amount of memory (in mb or gb) required on each node

For Ocelote, all of the standard nodes have 6GB per core. "pcmem=6gb" can be added to the line or left off and it will default to 6gb.  The large memory node has 42GB per core so "pcmem=42gb" must be added to use the large memory node.  The following select statement would request one complete node:

#PBS -l select=4:ncpus=28:mem=168gb




 
module load python/3.6

source venv/bin/activate

pip install tensorflow-gpu

module load cuda10.1 # will load cuda10.1/toolkit/10.1.168

module load cuda101/neuralnet  # You may also need module load cuda101/neuralnet in case you need the cuDNN libraries for Tensorflow


module load tensorrt 



 python3
Python 3.6.5 (default, Apr 11 2018, 13:45:41)
[GCC 6.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from tensorflow.python.client import device_lib

>>> print(device_lib.list_local_devices())




6/ Run the job using PBS command qsub.

Now run submit this script to the queue to schedule your job.

qsub tensorflow_gunn_model.pbs
You receive a line with the jobid when it is successfully submitted like:

698413.head1.cm.cluster
 When the job ends you should have one or two output files from PBS. They start with the job name and end with the jobid.  In the middle is 'o' for output information or 'e' for error information.


mpi_hello_world.o698413
mpi_hello_world.e698413



--


(venv) [aishwarya34@login2 ~]$ qsub tensorflow_gunn_model.pbs
3146145.head1.cm.cluster


qstat 3146145
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
3146145.head1     tensorflow_gunn  aishwarya34              0 Q oc_windfall

qstat 3146716
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
3146716.head1     gunnmodel_small  aishwarya34              0 Q oc_windfall


>> qsub smalljob.pbs
3147030.head1.cm.cluster
(venv) [aishwarya34@login2 ~]$ qstat 3147030
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
3147030.head1     gunnmodel_small  aishwarya34              0 Q oc_standard




<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GUNN-15_model.png' /><br>
<figcaption>Figure 5: GUNN-15 model having custon Gunn2D layer. </figcaption></div>
</figure>


## Evaluation

## Results / Conclusion

## Reference

[1] GUNN is used to replace convolutional layers which is stated in paper:  Qiao, Siyuan et al. “Gradually Updated Neural Networks for Large-Scale Image Recognition.” ICML (2017).

[2]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
learning for image recognition. IEEE Conference on
Computer Vision and Pattern Recognition, 2016a.

[3]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings
in deep residual networks. ECCV, 2016b.

[4]  Batch Normalization will be implemented. Ioffe, S. and Szegedy, C. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In International Conference on Machine Learning, 2015.




