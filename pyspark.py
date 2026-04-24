from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, when, rand, struct, array, to_json
import zipfile
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("EdgeIIoT_To_LLM_Chatbot") \
    .getOrCreate()

# ==========================================
# Method 1: If Spark can read CSV directly from ZIP (some versions support this)
# ==========================================

# Try direct read from ZIP (Spark might handle it automatically)
s3_input_path = "s3://20596360-raw-data/dataset.zip"

print("Reading data from ZIP file...")

# Spark can read .zip files if the internal file is CSV
# The path format: s3://bucket/dataset.zip!/
try:
    df = spark.read.csv("s3://20596360-raw-data/dataset.zip", header=True, inferSchema=True)
    print("Successfully read ZIP directly!")
except:
    print("Direct ZIP read failed, using alternative method...")
    
    # ==========================================
    # Method 2: Download and unzip on EMR (works always)
    # ==========================================
    import boto3
    import zipfile
    
    # Download ZIP from S3 to local EMR storage
    s3 = boto3.client('s3')
    s3.download_file('20596360-raw-data', 'dataset.zip', '/tmp/dataset.zip')
    
    print("Downloaded ZIP file to /tmp/dataset.zip")
    
    # Unzip
    with zipfile.ZipFile('/tmp/dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('/tmp/')
    
    print("Unzipped contents:")
    os.listdir('/tmp/')
    
    # Find the CSV file (it might have a different name)
    csv_files = [f for f in os.listdir('/tmp/') if f.endswith('.csv')]
    csv_path = f'/tmp/{csv_files[0]}'
    print(f"Found CSV: {csv_path}")
    
    # Read the CSV
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

# ==========================================
# Continue with preprocessing (same as before)
# ==========================================

print("Data cleaning...")
df_clean = df.dropDuplicates()

# Create conversation format
user_prompt = concat(
    lit("Analyze this IoT network traffic record and determine if it is malicious. Details -> "),
    lit("Protocol: "), col("arp.opcode"), 
    lit(", TCP Flags: "), col("tcp.flags"), 
    lit(", Packet Size: "), col("frame.len"), 
    lit(". Is this an attack?")
)

assistant_response = when(
    col("Attack_label") == 1,
    concat(lit("Yes, this traffic is malicious. It is a "), col("Attack_type"), lit(" attack."))
).otherwise(
    lit("No, this network traffic appears to be normal and benign.")
)

df_chat = df_clean.withColumn("user_content", user_prompt) \
                  .withColumn("assistant_content", assistant_response)

# Format as JSONL
df_formatted = df_chat.select(
    struct(
        array(
            struct(lit("user").alias("role"), col("user_content").alias("content")),
            struct(lit("assistant").alias("role"), col("assistant_content").alias("content"))
        ).alias("messages")
    ).alias("chat_record")
)

# Sample 100,000 records
print("Sampling 100,000 records...")
df_sample = df_formatted.orderBy(rand()).limit(100000)

# Convert to JSON
df_final_json = df_sample.select(to_json(col("chat_record")).alias("json_string"))

# Save to S3
s3_output_path = "s3://20596360-processed-data/fine_tuning_dataset.jsonl"

print("Writing JSONL to S3...")
df_final_json.write.text(s3_output_path, compression="none")

print("Data processing complete!")