from sagemaker import RandomCutForest
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import boto3
import botocore
import sagemaker
import sys
import matplotlib.pyplot as plt


bucket = (sagemaker.Session().default_bucket())
prefix = 'sagemaker/rcf-benchmarks'
execution_role = sagemaker.get_execution_role()
region = boto3.Session().region_name

def check_bucket_permission(bucket):
    # check if the bucket exists
    permission = False
    try:
        boto3.Session().client("s3").head_bucket(Bucket=bucket)
    except botocore.exceptions.ParamValidationError as e:
        print(
            "Hey! You either forgot to specify your S3 bucket"
            " or you gave your bucket an invalid name!"
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "403":
            print(f"Hey! You don't have permission to access the bucket, {bucket}.")
        elif e.response["Error"]["Code"] == "404":
            print(f"Hey! Your bucket, {bucket}, doesn't exist!")
        else:
            raise
    else:
        permission = True
    return permission


if check_bucket_permission(bucket):
    print(f"Training input/output will be stored in: s3://{bucket}/{prefix}")

print("bucket info : {}".format(bucket))
print("prefix info : {}".format(prefix))
print("execution_role info : {}".format(execution_role))
print("region info : {}".format(region))

import pandas as pd
csv_path = "Pseudo_data_biggest.csv"
df = pd.read_csv(csv_path)

# print(df.describe())
# print(df.info())

df_target = df[['Feature_0']]


rcf = RandomCutForest(
    role=execution_role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    data_location=f"s3://{bucket}/{prefix}/",
    output_path=f"s3://{bucket}/{prefix}/output",
    num_samples_per_tree=512,
    num_trees=50,
)

rcf.fit(rcf.record_set(df_target.Feature_0.to_numpy().reshape(-1, 1)))

print("Training job name : {}".format(rcf.latest_training_job.job_name))

rcf_inference = rcf.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

rcf_inference.serializer = CSVSerializer()
rcf_inference.deserializer = JSONDeserializer()

test_df = df.Feature_0.to_numpy().reshape(-1,1)
print(test_df[:10])

inference_result = rcf_inference.predict(test_df)
scores = [datum["score"] for datum in inference_result["scores"]]

df_target["score"] = pd.Series(scores, index=df_target.index)
print(df_target.head())


score_mean = df_target['score'].mean()
score_std = df_target['score'].std()
score_based_threshold = score_mean + 3 * score_std

over_threshold = df_target['score'] > score_based_threshold

print(type(over_threshold))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(df_target["Feature_0"], color="blue", alpha=0.8, label='Pseudo Data')
ax2.plot(df_target["score"], color="C1",alpha=0.5, label='RCF Score')
ax1.scatter(df_target.index[over_threshold], df_target['Feature_0'][over_threshold], color='red', label='Anomaly')
ax2.axhline(y=over_threshold, color = 'C1', linestyle='--', label = 'Score threshold', alpha=0.8)

ax1.grid(which="major", axis="both")

ax1.set_ylabel("Pseudo Data", color="C0")
ax2.set_ylabel("Anomaly Score", color="C1")

ax1.tick_params("y", colors="C0")
ax2.tick_params("y", colors="C1")

ax1.set_ylim(0, 100)
ax2.set_ylim(min(scores), 1.4 * max(scores))
fig.set_figwidth(10)
fig.legend(loc='upper center')



