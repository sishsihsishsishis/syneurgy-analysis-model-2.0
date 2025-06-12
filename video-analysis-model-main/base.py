import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import uuid

import boto3
import requests

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)

# SQS SETUP
aws_access_key_id = "AKIATRKQV5SYQTJ7DBGF"
aws_secret_access_key = "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"
aws_region = "us-east-2"

# SQS queue URL
on_off_queue_url = "https://sqs.us-east-2.amazonaws.com/243371732145/sync-main.fifo"
meetings_queue_url = (
    "https://sqs.us-east-2.amazonaws.com/243371732145/MeetingsQueue.fifo"
)
cloudfront_base_url = "https://d2n2ldezfv2tlg.cloudfront.net"

# S3
s3_endpoint = "https://s3.us-east-2.amazonaws.com"

# Initialize SQS client with AWS credentials
boto = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

# Initialize AWS Services with AWS credentials
sqs = boto.client("sqs")
dynamodb = boto.resource("dynamodb")
s3 = boto.client("s3", endpoint_url=s3_endpoint)


# Receive messages from the SQS queue
def fetch_sqs_messages(group):
    try:
        logging.info("Fetching messages from SQS queue...")
        response = sqs.receive_message(
            QueueUrl=meetings_queue_url,
            MaxNumberOfMessages=10,  # Adjust as needed
            WaitTimeSeconds=20,  # Poll every 20 seconds
            MessageAttributeNames=["All"],  # Include all message attributes
        )
        messages = response.get("Messages", [])
        logging.info(f"Found {len(messages)} messages")
        # logging.info(str(messages))
        # Filter messages by the specified group
        filtered_messages = []
        for message in messages:
            body = json.loads(message.get("Body", "{}"))
            if body.get("type") == group:
                filtered_messages.append(message)

        if filtered_messages:
            logging.info(f"Found {len(filtered_messages)} messages for group: {group}")
            for message in filtered_messages:
                logging.info(f"Message: {message}")
        else:
            logging.info(f"No messages found for group: {group}")

        return filtered_messages

    except Exception as e:
        logging.error(f"Error fetching messages: {e}")
        return []


# Delete message from SQS queue
def delete_message_from_queue(message):
    try:
        # Delete the message from the queue
        sqs.delete_message(
            QueueUrl=meetings_queue_url, ReceiptHandle=message["ReceiptHandle"]
        )
        logging.info("Message deleted from the queue.")
    except Exception as e:
        logging.error(f"Error deleting message from the queue: {e}")


# Stop instance request SQS entry
def request_stop_instance(group):
    try:
        # Send a request to stop the instance
        logging.info(f"Requesting stop of {group}")

        message_body = {"action": "stop", "group": group}

        # Send the message to the on_off_queue
        response = sqs.send_message(
            QueueUrl=on_off_queue_url,
            MessageBody=str(message_body),
            MessageGroupId="stop-"
            + group,  # Ensuring messages are grouped appropriately
            MessageDeduplicationId="stop-"
            + group
            + "-"
            + str(int(time.time())),  # Unique ID to avoid duplication
        )

        logging.info(
            f"Stop request sent to {on_off_queue_url} with response: {response}"
        )

    except Exception as e:
        logging.error(f"Error requesting stop of instance: {e}")


# Download videos from s3
def download_video(resource_path, base_cache_dir, bucket_name="sync5"):
    try:
        logging.info(f"downloading... {resource_path}")
        start = time.time()

        # init cache dir
        if os.path.exists(base_cache_dir):
            if os.path.isdir(base_cache_dir):
                shutil.rmtree(base_cache_dir, ignore_errors=True)
            elif os.path.isfile(base_cache_dir):
                os.remove(base_cache_dir)
        cache_dir = os.path.join(base_cache_dir)

        os.makedirs(cache_dir)

        # Generate a unique file name
        file_name = uuid.uuid4().hex
        cached_path = os.path.join(cache_dir, f"{file_name}.mp4")

        # Construct the full URL to the resource on CloudFront
        url = f"{cloudfront_base_url}/{resource_path}"

        # Download the file from CloudFront
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Write the downloaded content to a local file
        with open(cached_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        finish = time.time()
        logging.info(f"Downloaded in {finish - start}s")
        return file_name

    except:
        log = traceback.format_exc()
        logging.error(log)


# Download resource from s3
def download_resource(key, filename, bucket_name="syneurgy-prod"):
    logging.info("downloading")
    start = time.time()

    # Download the file from S3
    s3.download_file(bucket_name, key, filename)

    finish = time.time()
    print("downloaded in ", finish - start, "s")
    return filename


# Upload resource to s3
def upload_resource(filename, key, bucket_name="syneurgy-prod"):
    logging.info("uploading")
    start = time.time()

    # Upload the file to S3
    s3.upload_file(filename, bucket_name, key)

    finish = time.time()
    print("uploaded in ", finish - start, "s")
    return filename
