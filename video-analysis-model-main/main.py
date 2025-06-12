import sys

print(">>> sys.path in main.py:")
for p in sys.path:
    print("  ", p)
import json
import logging

# import redis
import os
import shutil
import time
import traceback
from decimal import Decimal

import base  # basic functions
import dialogue  # model
import helpers  # helper functions
import rppg  # model
from boto3.dynamodb.conditions import Key
from emotion import go_emotion
from speaker import diarization

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    NO_MESSAGE_COUNT = 0
    MAX_NO_MESSSAGE_COUNT = 5
    GROUP = "analysis"

    while True:
        try:
            logging.info("Fetching messages from SQS...")
            messages = base.fetch_sqs_messages(GROUP)

            if messages:
                NO_MESSAGE_COUNT = 0  # Reset the counter
                cache_dir = "./data"
                asr_path = "./data/transcription.txt"
                diarization_path = "./data/diarization.txt"
                local_cached_filename = ""
                start_time = time.time()
                # Process received messages
                for message in messages:
                    try:
                        logging.info("Received message: %s", message["Body"])
                        # Delete the message from the queue
                        base.delete_message_from_queue(message)

                        # Reset cache
                        logging.info("Resetting cache...")
                        if os.path.exists(cache_dir):
                            shutil.rmtree(cache_dir)

                        # Extract details from SQS event
                        body = json.loads(message.get("Body", "{}"))
                        download_url = body.get("key")
                        meeting_id = body.get("meetingId")

                        try:
                            helpers.update_key(meeting_id, "started", True)
                            helpers.update_key(meeting_id, "queued", False)
                            helpers.update_key(
                                meeting_id, "start", Decimal(str(start_time))
                            )
                            helpers.update_status(meeting_id, "extracting")

                            # Get TeamID from MeetingTable using meetingId
                            meetingTable = base.dynamodb.Table("MeetingTable")
                            meeting_response = meetingTable.get_item(
                                Key={"id": meeting_id}, ProjectionExpression="teamId"
                            )
                            if "Item" not in meeting_response:
                                logging.error(
                                    f"Meeting with ID {meeting_id} not found."
                                )
                            team_id = meeting_response["Item"]["teamId"]
                        except Exception as e:
                            logging.error(
                                f"Error extracting details for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "extracting")

                        try:
                            helpers.update_status(meeting_id, "downloading")

                            # Download from s3
                            logging.info(
                                f"Downloading resource from S3: {download_url}"
                            )
                            local_cached_filename = base.download_video(
                                download_url, "./raw"
                            )
                            video_path = f"./raw/{local_cached_filename}.mp4"
                            process_start_time = time.time()
                        except Exception as e:
                            logging.error(
                                f"Error downloading resource from S3 for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "downloading")

                        try:
                            helpers.update_status(meeting_id, "resampling")

                            # Change FPS
                            logging.info(f"Resampling video: {local_cached_filename}")
                            resample_path = f"./data/{local_cached_filename}.mp4"
                            output_data_path, duration = helpers.change_fps(
                                video_path, resample_path, 25
                            )
                            print(duration)
                            helpers.update_key(
                                meeting_id, "duration", Decimal(str(duration))
                            )

                        except Exception as e:
                            logging.error(
                                f"Error resampling video for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "resampling")

                        try:
                            helpers.update_status(meeting_id, "rppg")

                            # Run rppg model
                            logging.info(f"Running rppg model: {local_cached_filename}")
                            rppg.go_rppg(video_path, cache_dir, 0.7)
                        except Exception as e:
                            logging.error(
                                f"Error running rppg model for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "rppg")

                        try:
                            # Find the number of speakers in the video
                            logging.info(f"Counting speakers in the video...")
                            speaker_count, jpg_files = helpers.get_speaker_count("data")
                            logging.info(f"Speaker count: {speaker_count}")

                            # Upload rppg files
                            logging.info(f"Uploading detected avatars...")
                            helpers.upload_detected_avatars(jpg_files, meeting_id)

                            helpers.update_status(meeting_id, "scores")
                            scores = helpers.upload_scores(meeting_id)
                            helpers.update_team_avg_scores(team_id, scores)
                        except Exception as e:
                            logging.error(
                                f"Error processing speaker data for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "scores")

                        try:
                            helpers.update_status(meeting_id, "heatmap")

                            # process and save heatmap
                            helpers.process_heatmap(meeting_id)
                        except Exception as e:
                            logging.error(
                                f"Error processing heatmap for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "heatmap")

                        try:
                            helpers.update_status(meeting_id, "audio")

                            # Extract audio
                            logging.info(f"Extracting audio...")
                            wav_file = helpers.extract_audio(resample_path)
                        except Exception as e:
                            logging.error(
                                f"Error extracting audio for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "audio")

                        try:
                            helpers.update_status(meeting_id, "transcript")

                            # Run NLP model
                            logging.info(f"Generating transcript...")
                            transcription_segments = helpers.asr_transcribe(wav_file)

                            logging.info(f"Saving transcript to file...")
                            helpers.save_transcription_to_file(
                                transcription_segments, asr_path
                            )
                        except Exception as e:
                            logging.error(
                                f"Error generating transcript for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "transcript")

                        try:
                            helpers.update_status(meeting_id, "nlp")

                            # Run diarization model
                            logging.info(f"Running diarization model...")
                            speaker_chunks = diarization(wav_file, int(speaker_count))

                            logging.info(
                                f"Matching speakers with ASR and combining speaker chunks..."
                            )
                            result_path = helpers.match_speakers_asr(
                                speaker_chunks, meeting_id
                            )
                        except Exception as e:
                            logging.error(
                                f"Error running diarization model for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "nlp")

                        try:
                            helpers.update_status(meeting_id, "speaker")

                            # Upload NLP text data
                            logging.info(f"Uploading NLP text data to S3...")
                            s3_nlp_upload_path = f"out/{meeting_id}/transcript.txt"
                            base.upload_resource(diarization_path, s3_nlp_upload_path)
                            # TODO: send message to SQS once we start using speaker model.
                            # helpers.send_message_to_sqs(download_url, meeting_id, 'speaker')
                        except Exception as e:
                            helpers.update_errors(meeting_id, "speaker")
                            logging.error(
                                f"Error uploading NLP text data for meeting ID {meeting_id}: {e}"
                            )

                        try:
                            helpers.update_status(meeting_id, "emotion")

                            # Run emotion model
                            logging.info(f"Running emotion model...")
                            diarization_result, emotion_data = (
                                helpers.process_diarization_and_emotion(
                                    diarization_path
                                )
                            )
                            emotion_labels = go_emotion(emotion_data)
                            emotion_path = helpers.save_emotion_results(
                                emotion_labels, diarization_result, meeting_id
                            )
                        except Exception as e:
                            logging.error(
                                f"Error running emotion model for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "emotion")

                        try:
                            helpers.update_status(meeting_id, "participation")
                            # Dialogue model
                            logging.info(f"Running dialogue model...")
                            emotion_texts, text = helpers.load_emotion_data()
                            dialogue_act_labels = dialogue.go_dialogue_act(text)
                            unique_speakers = helpers.save_dialogue_act_labels(
                                dialogue_act_labels, emotion_texts, meeting_id
                            )
                            helpers.upload_participation_emotion(
                                "./data/a_results.csv",
                                "./data/v_results.csv",
                                unique_speakers,
                                meeting_id,
                            )
                        except Exception as e:
                            helpers.update_errors(meeting_id, "participation")
                            logging.error(
                                f"Error running dialogue model for meeting ID {meeting_id}: {e}"
                            )

                        try:
                            helpers.update_status(meeting_id, "completed")
                            helpers.update_key(meeting_id, "finished", True)
                            finish_time = time.time()
                            helpers.update_key(
                                meeting_id, "finish", Decimal(str(finish_time))
                            )
                        except Exception as e:
                            logging.error(
                                f"Error updating status to 'completed' for meeting ID {meeting_id}: {e}"
                            )
                            helpers.update_errors(meeting_id, "completed")
                    except Exception as e:
                        logging.error(f"Error processing message: {e}")
                        traceback.print_exc()

            else:
                NO_MESSAGE_COUNT += (
                    1  # Increment the counter if no messages are received
                )
                if NO_MESSAGE_COUNT >= MAX_NO_MESSSAGE_COUNT:
                    logging.info(f"Requesting stop for the instance in group: {GROUP}")
                    base.request_stop_instance(GROUP)
                    NO_MESSAGE_COUNT = 0  # Reset the counter after requesting stop

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            traceback.print_exc()
        logging.info(f"Message Count: {NO_MESSAGE_COUNT}")
        time.sleep(20)  # Wait for 20 seconds before polling again
