# Using AWS with Keras/Tensorflow

## Launch an instance

First create an instance from the AWS management console. You'll want
the Ubuntu 13.0 Deep Learning AMI and a GPU capable instance (the
smallest is p2.xlarge). Make sure you have the key you launch the
instance with - I'll assume it's called ~/.ssh/key.pem

## Login and setup an instance

After the instance launches, you can log in with

    ssh -i ~/.ssh/key.pem ubuntu@<IP>

where <IP> is the IP address you can find under the description of
your instance in the management console. Once logged in you can
activate your desired environment like

    source activate tensorflow_p36

which activates the tensorflow/keras/python3 environment.

## Transfer files

Files can be copied from your local machine to the instance like

    scp -i ~/ssh/key.pem mnist.py ubuntu@<IP>:~

which copies the mnist.py example file in this directory to the
instance.

## Run

Back in the instance, we can run the file with

    python3 mnist.py

and check performance. To verify the GPU is working you can use

    nvidia-smi

to see the GPU info or

    watch -n 1 nvidia-smi

in another terminal when running the file to see current GPU usage.
