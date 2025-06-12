from base import meetings_queue_url
from helpers import send_message_to_sqs

if __name__ == '__main__':
    # Define the message parameters
    message = {
        "bucket": "syneurgy-prod",
        "key": "meetings/e5c41a63-611d-4c6e-9d9e-5637fc8bed17/no_trust.mp4",
        "meetingId": "e5c41a63-611d-4c6e-9d9e-5637fc8bed17",
        "type": "analysis"
    }

    # message = {
    #     "bucket": "syneurgy-prod",
    #     "key": "meetings/e5a1d664-2c0e-47aa-9459-260a7eedcc38/Syuneurgy stand up - 16 July 2024.mp4",
    #     "meetingId": "e5a1d664-2c0e-47aa-9459-260a7eedcc38",
    #     "type": "analysis"
    # }

    # message = {
    #     "bucket": "syneurgy-prod",
    #     "key": "meetings/b9ff8c4f-a5a2-46f6-9d32-220b10247fef/Syneurgy - Saturday Review call - Active Speaker Detection _ Platform _ Pdf Report _ Mirror Site - 13 July 2024.mp4",
    #     "meetingId": "b9ff8c4f-a5a2-46f6-9d32-220b10247fef",
    #     "type": "analysis"
    # }

    # Call the function with the provided parameters
    send_message_to_sqs(
        object_key=message['key'],
        meeting_id=message['meetingId'],
        message_type=message['type'],
        bucket_name=message['bucket'],
        queue_url=meetings_queue_url
    )
