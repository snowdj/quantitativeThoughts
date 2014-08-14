
aws_access_key <- 'put_aws_access_key_here'
aws_secret_key <- 'put_aws_secret_access_key_here'
devtools::install("https://github.com/lalas/awsConnect/")
library(awsConnect)


# Checking enviroment Variables
Sys.getenv("PATH")
Sys.getenv("AWS_DEFAULT_REGION")
Sys.getenv("AWS_DEFAULT_OUTPUT")
Sys.getenv("AWS_ACCESS_KEY_ID")

# 0.1 Adding AWS Binaries path
add.aws.path("~/.local/lib/aws/bin")

# 0.2 Adding AWS ACCESS Key (public and private)
Sys.setenv(AWS_ACCESS_KEY_ID = aws_access_key)
Sys.setenv(AWS_SECRET_ACCESS_KEY = aws_secret_key)

# 1. Launch On-Demand EC2 instances in a specific region with default security group
cl <- startCluster(ami = "ami-7376691a",instance.count = 1,instance.type = "t1.micro", key.name = "R_EC2", region = "us-east-1")

# 1.2 We can stop the cluster with the following command
stopCluster(cluster = cl, region = "us-east-1")
# 1.3 Or terminate the cluster with the following command
terminateCluster(cluster = cl, region = "us-east-1")

# 0.3 Specifying the "default" region for further usage
Sys.setenv(AWS_DEFAULT_REGION = "us-east-1")

## 2. EC2 key-Pairs
# 2.1.1 Describe Region Available
regions <- describe.regions()

# 2.1.2 Describe Zone Availability in a particular region so that you can launch an EC2 instance into it
describe.avail.zone(regions$ID[3])

# 2.2.1 Create Key-pairs for a specific location
create.ec2.key(key.name = "MyEC2", region = "us-west-1")

# 2.2.2 upload own key to default region (since it's not specified). 
# The public key was generated with the system command "ssh-keygen"
upload.key(path_public_key = "~/.ssh/id_rsa.pub", key_name = "MyEC2")

# 2.2.3 Delete key-paris from a specific region
delete.ec2.key(key.name = "MyEC2", region = "us-west-1")


## 3 Launching Spot Instances

# 3.1.1 Finding out the most recent spot price of a machine type "cr1.8xlarge" in a particular region, 
get.spot.price(instance.type = "cr1.8xlarge", region = "us-east-1")

# 3.1.2 Finding out the most recent spot price of a machine type "t1.micro" in a the default region (defined previously)
get.spot.price(instance.type = "t1.micro")

# 3.2 To Launch spot instance request in the default region:

# 3.2.1 Make spot instance request
spot.instances <- req.spot.instances(spot.price = 0.0032, ami = "ami-7376691a", instance.count = 1, instance.type = "t1.micro",key.name = "MyEC2")

# Get the IDs of the spot instances request
Spot.Instance.Req.IDs <- spot.instances[,"SpotInstanceReqID"]

# 3.2.2 To find out the status of the spot request
describe.spot.instanceReq(Spot.Instance.Req.IDs)

# Get the launched instance IDs (if any, assuming the StatusCode returned from the previous command is "fulfilled")
Instance.IDs <- describe.spot.instanceReq(Spot.Instance.Req.IDs)[,"InstanceID"]

# 3.3 (Optional) Make the root Volume of an instance (first instance in this example) persist after Termination
rootVol.DeleteOnTermination(Instance.IDs[1])

# 3.4 (Optional) Get Information about instances in default region
describe.instances()

# 3.3 Terminate an EC2 instance
ec2terminate.instances(Instance.IDs[1])
