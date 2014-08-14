---
layout: post
title: awsConnect tutorial - part 1
tags: [Cloud Computing, AWS, tutorial]
categories: R
author: Lalas
---

{% include JB/setup %}

### Topic Covered:

-   [Download and install the package](#topic1)
-   [Configuring and using the package](#topic2)
-   [Creating and Managing Key-Pairs for EC2](#topic3)
-   [Launch, Stop, and Terminate On-Demand EC2 cluster in a specific region with default security group](#topic4)
-   [Defining default region using enviroment variables](#topic5)
-   [Launch Spot Instances](#topic6)

In this R-tutorial, we explain how to use the `awsConnect` package to launch, terminate EC2 instances on Amazon's Cloud service.

### <a name="topic1"></a>Download and install the package

1.  Make sure the amazon's AWS CLI is installed on the machine you want to use this package on. For detailed information on how to do so, consult AWS CLI [setup guide](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-set-up.html).
2.  Then using the `devtools` R-package, run the following command to install `awsConnect`

``` {.r}
devtools::install("https://github.com/lalas/awsConnect/")
```

1.  Finally load the library, with

``` {.r}
library(awsConnect)
```

### <a name="topic2"></a>Configuring and using the package

As mentioned elsewhere in the package documentation, the first thing to do once the package is installed is to setup the enviroment variables. If you are using RStudio, then enviroment variables defined in the `.bash` or `.profile` file are not available within RStudio. For instance

``` {.r}
Sys.getenv("PATH")
```

    ## [1] "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin:/usr/local/git/bin:/usr/texbin"

does not contains the path the the `aws bin`; and other enviroment variables such as `aws_access_key_id`; the `aws_secret_access_key`; and the default `region` are empty. To fix this problem; we issue the following commands:

``` {.r}
add.aws.path("~/.local/lib/aws/bin")
Sys.setenv(AWS_ACCESS_KEY_ID = aws_access_key)
Sys.setenv(AWS_SECRET_ACCESS_KEY = aws_secret_key)
```

The above 3-lines of code assumes that the path to aws binaries is `~/.local/lib/aws/bin`, that your aws access key and secret keys are stored in R-variable called `aws_access_key` and `aws_secret_key` respectively.

### <a name="topic3"></a>Creating and Managing Key-Pairs for EC2

Amazon EC2 Key Pairs (commonly called an "SSH key pair"), is need to be able to Launch and connect to an instance on AWS cloud. If no Key pairs is used when launching the instances, the user will not be able to ssh login into his/her instance.

Amazon offers 2 options: 1. The possibility of using it's service to create a key-pair for you; where you download the private key to your local machine 2. Uploading your own public key to be used.

##### Creating and Deleting key-pairs for a specific region

Assuming the name of the key we want to create is *MyEC2* and we want to be able to use it with instances launched in US West (N. California) region, we will issue this command:

``` {.r}
create.ec2.key(key.name = "MyEC2", region = "us-west-1")
```

This will create a file called `MyEC2.pem` in the `~/.ssh` directory. Simiarly, in order to delete an existing key, named *MyEC2* from a specific region, we use

``` {.r}
delete.ec2.key(key.name = "MyEC2", region = "us-west-1")
```

and the file `~/.ssh/MyEC2.pem` will be deleted

##### Uploading and using your own public key

The second method that Amazon web services allows, is to use your own public/private key pairs. Assuming that the user have used `ssh-keygen` command (or something similar) to generate the public/private key pairs; he/she can then upload their public key to a specific region using:

``` {.r}
upload.key(path_public_key = "~/.ssh/id_rsa.pub", key_name = "MyEC2", region = "us-east-1")
```

where `id_rsa.pub` is the file that contains the protocol version 2 RSA public key for authentication.

### <a name="topic4"></a>Launch, Stop, and Terminate On-Demand EC2 cluster in a specific region with default security group

Now that we have created (or upload) the required key to AWS Cloud, we are ready to launch EC2 instances. We start by launching a cluster (of 2 nodes); in the US East (N. Virginia) region.

``` {.r}
cl <- startCluster(ami = "ami-7376691a",instance.count = 2,instance.type = "t1.micro", key.name = "MyEC2", region = "us-east-1")
```

using the default security group, since we didn't specify the `security.grp` parameter. As of the time of writting this tutorial, `ami-7376691a` is an Amazon Machine Instance that run of `Ubuntu-13.10` (64bit), and has `R version 3.1.0` and `RStudio-0.98.501` installed on it. To stop this cluster, we use

``` {.r}
stopCluster(cluster = cl, region = "us-east-1")
```

and to terminate this cluster, we use:

``` {.r}
terminateCluster(cluster = cl, region = "us-east-1")
```

### <a name="topic5"></a>Defining default region using enviroment variables

Instead of always specifying the `region` parameters when calling various function in the `awsConnect` package; the user can use the enviroment variable `AWS_DEFAULT_REGION` to set his/her default region. For example, to set up the default region to be US East (N. Virginia); we use

``` {.r}
Sys.setenv(AWS_DEFAULT_REGION = "us-east-1")
```

where, `us-east-1` is the ID for a specific region. To find out the IDs' of the different regions; use the following command:

``` {.r}
regions <- describe.regions()
```

And to find out the availability zone and their status for a specif regions, find out its ID and use the following command:

``` {.r}
describe.avail.zone(regions$ID[3])
```

### <a name="topic6"></a>Launch Spot Instances

Spot Instances allow you to name your own price for Amazon EC2 computing capacity. You simply bid on spare Amazon EC2 instances and run them whenever your bid exceeds the current Spot Price, which varies in real-time based on supply and demand.

Notes: \* Spot Instances perform exactly like other Amazon EC2 instances while running. They only differ in their pricing model and the possibility of being interrupted when the Spot price exceeds your max bid.

-   You will never pay more than your maximum bid price per hour.

-   If your Spot instance is interrupted by Amazon EC2, you will not be charged for any partial hour of usage. For example, if your Spot instance is interrupted 59 minutes after it starts, we will not charge you for that 59 minutes. However, if you terminate your instance, you will pay for any partial hour of usage as you would for On-Demand Instances

-   You should always be prepared for the possibility that your Spot Instance may be interrupted. A high max bid price may reduce the probability that your Spot Instance will be interrupted, but cannot prevent interruption.

With this in mind; in order to launch a spot instance, we would:

1.  To find out the most recent spot price of a machine of a specific type (e.g. cr1.8xlarge) in a particular region we use the following command:

``` {.r}
get.spot.price(instance.type = "cr1.8xlarge", region = "us-east-1")
```

Note: if we had defined the enviroment variable `AWS_DEFAULT_REGION`; we could have omitted the `region` parameters in the previous command to find out the most recent spot price history for the *default* region.

1.  Once we obtain the most recent spot price, we can bid for a spot instance, by providing a price higher than or equal to the spot price returned in the previous step. For instance, assuming that the spot price for the `t1.micro` is \$0.0031/hr; we would use the following command:

``` {.r}
spot.instances <- req.spot.instances(spot.price = 0.0032, ami = "ami-7376691a", instance.count = 1, instance.type = "t1.micro",key.name = "MyEC2")
```

to bid for a `t1.micro` instance in our default region.

1.  Obtain the request IDs for the instances using:

``` {.r}
Spot.Instance.Req.IDs <- spot.instances[,"SpotInstanceReqID"]
```

1.  Find out the status of the bidding request, using the IDs obtained in the previous step

``` {.r}
describe.spot.instanceReq(Spot.Instance.Req.IDs)
```

1.  Assuming the *statusCode* returned from the previous command is *fulfilled*; then we can obtained the *instanceIDs* that were launched as a result of our bidding request using the following command:

``` {.r}
Instance.IDs <- describe.spot.instanceReq(Spot.Instance.Req.IDs)[,"InstanceID"]
```

1.  (OPTIONAL STEP) By default, Amazon EBS-backed instance root volumes have the DeleteOnTermination flag set to true, which causes the volume to be deleted upon instance termination. This might be problematic, since a spot instance could be terminated at anytime, as stated above. In order to fix this default behaviour, assuming we have the Instance IDs of the spot instances which we launched (as we did in the previous step), we can use the following command

``` {.r}
rootVol.DeleteOnTermination(Instance.IDs[1])
```

in order not to delete the root volume of the first instance, once it's terminated.

Finally, to get information about running instances, running in the default region, we would use the following command:

``` {.r}
describe.instances()
```
