import logging
import requests
from retrying import retry
import json
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GptServiceImpl:
    MAX_RETRIES = 3
    API_KEY = "sk-proj-sK7ff3WWXS98BPfyYEp3T3BlbkFJPQ1oNsNgdFAxFEY9uFGs"
    URL = "https://api.openai.com/v1/chat/completions"

    GENERAL_PROMPT = """
    You will be provided with a conversation formatted as:  
    <speaker>\\t<starts>\\t<ends>\\t<sentence>

    Task:  
    {task_description}

    **Important Instructions:**
    - Always return the output in strict JSON format.
    - Do not include any additional text, explanations, or comments.
    - If there is any issue, still return a valid JSON response indicating the error.

    **Output Format (JSON):**

    {output_format}
    """

    TASK_DESCRIPTION_SUMMARY = """
    1. Divide the conversation into 1 to 3 chapters based on themes or ideas.
    2. Summarize each chapter in one sentence and provide 3 bullet points for each.
    3. Include the actual start and end times for each chapter from the conversation.
    """

    TASK_DESCRIPTION_TEAM_HIGHLIGHT = """
    Select three highlights of about 30 seconds each, known as team highlights, and output the start and end times, as well as a description (20-50 words) of each highlight.
    """

    TASK_DESCRIPTION_USER_HIGHLIGHT = """
    Select three highlights of about 30 seconds each for each speaker from all the information and output the speaker, start time, end time, and a description (30-50 words) of each highlight.
    """

    OUTPUT_FORMAT_SUMMARY = """
    [
      {
        "summary": "Actual Summary Here",
        "start": "Actual Start Time",
        "end": "Actual End Time",
        "bullets": [
          "Bullet Point 1",
          "Bullet Point 2",
          "Bullet Point 3"
        ]
      },
      ...
    ]
    """

    OUTPUT_FORMAT_TEAM_HIGHLIGHT = """
    [
      {
        "start": "Start Time",
        "end": "End Time",
        "description": "Description of the highlight"
      },
      ...
    ]
    """

    OUTPUT_FORMAT_USER_HIGHLIGHT = """
    [
      {
        "speaker": "Speaker Number",
        "start": "Start Time",
        "end": "End Time",
        "description": "Description of the highlight"
      },
      ...
    ]
    """

    def __init__(self, api_key=None, url=None):
        if api_key:
            self.API_KEY = api_key
        if url:
            self.URL = url

    def summary_nlp(self, nlp_data):
        prompt_str = "\n".join(
            "{}\t{}\t{}\t{}".format(i['speaker'], i['start'], i['end'], i['sentence'])
            for i in nlp_data
        )
        try:
            nlp_ai_summary = self.process_contents(prompt_str)
        except Exception as e:
            logging.error(f"Failed to process content summary: {e}")
            nlp_ai_summary = None

        try:
            team_highlights = self.process_team_highlight(prompt_str)
        except Exception as e:
            logging.error(f"Failed to process team highlight: {e}")
            team_highlights = None

        try:
            user_highlights = self.process_user_highlight(prompt_str)
        except Exception as e:
            logging.error(f"Failed to process user highlight: {e}")
            user_highlights = None

        return nlp_ai_summary, team_highlights, user_highlights

    def analyze_transcript_emotions(self, transcript_data):
        prompt_str = transcript_data
        prompt = """
        You will be provided with a transcript formatted as:
        Speaker\tStart\tEnd\tSentence\tDialogueAct\tEmotion

        Task:
        1. Analyze each sentence and determine its sentiment (positive, negative, neutral).
        2. Calculate the overall percentage of positive, negative, and neutral emotions in the entire transcript.
        3. Calculate the KPIs (engagement, alignment, agency, stress, burnout) based on the transcript, with each value ranging between 30 and 70. The KPIs should be calculated based on the transcript, not the emotion provided alongside the sentence. And don't hard code 30 or 70, but calculate them based on the transcript. It needs to feel dynamic. 

        **Important Instructions:**
        - Do not focus on the emotion provided alongside the sentence, as it is incorrect. Analyse the sentence and mark it as positive, negative, or neutral based on the sentence. This is strict instruction. Also make sure the last percentage is not 0 in any of the percentages (atleast keep 5-15% in each but total should always be 100%).
        - Always return the output in strict JSON format as a string. This is super important or parsing will fail.
        - Do not include any additional text, explanations, or comments.
        - If there is any issue, still return a valid JSON response indicating the error.

        **Output Format (JSON):**
        {
          "positive_percentage": "Positive Emotion Percentage",
          "negative_percentage": "Negative Emotion Percentage",
          "neutral_percentage": "Neutral Emotion Percentage",
          "kpis": {
            "engagement": "Value between 30 and 70",
            "alignment": "Value between 30 and 70",
            "agency": "Value between 30 and 70",
            "stress": "Value between 30 and 70",
            "burnout": "Value between 30 and 70"
          }
        }
        """
        logging.info("Analyzing transcript emotions...")
        response = self.send_message_to_gpt(prompt_str, prompt)
        return json.dumps(response) if isinstance(response, dict) else response

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=1000)
    def process_contents(self, prompt_str):
        prompt = self.GENERAL_PROMPT.format(
            task_description=self.TASK_DESCRIPTION_SUMMARY, 
            output_format=self.OUTPUT_FORMAT_SUMMARY
        )
        logging.info("Processing content summary...")
        return self.send_message_to_gpt(prompt_str, prompt)

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=1000)
    def process_team_highlight(self, prompt_str):
        prompt = self.GENERAL_PROMPT.format(
            task_description=self.TASK_DESCRIPTION_TEAM_HIGHLIGHT, 
            output_format=self.OUTPUT_FORMAT_TEAM_HIGHLIGHT
        )
        logging.info("Processing team highlight...")
        return self.send_message_to_gpt(prompt_str, prompt)

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=1000)
    def process_user_highlight(self, prompt_str):
        prompt = self.GENERAL_PROMPT.format(
            task_description=self.TASK_DESCRIPTION_USER_HIGHLIGHT, 
            output_format=self.OUTPUT_FORMAT_USER_HIGHLIGHT
        )
        logging.info("Processing user highlight...")
        return self.send_message_to_gpt(prompt_str, prompt)
    
    def send_message_to_gpt(self, content, system_prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        
        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16000  # Token limit for GPT-3.5 Turbo 16k

        # Create a combined input (system + user message)
        combined_content = system_prompt + "\n" + content

        # Tokenize the content and check token count
        total_tokens = len(encoding.encode(combined_content))
        logging.info(f"Total tokens: {total_tokens}")
        
        if total_tokens > max_tokens:
            # Trim content if it exceeds the token limit
            excess_tokens = total_tokens - max_tokens
            content_tokens = encoding.encode(content)
            content_tokens = content_tokens[:-excess_tokens]  # Remove excess tokens

            # Decode the trimmed content back into a string
            content = encoding.decode(content_tokens)

        data = {
            "model": "gpt-3.5-turbo-16k",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        }

        response = requests.post(self.URL, json=data, headers=headers)
        response.raise_for_status()
        
        # Parse the response content as JSON
        response_content = response.json()['choices'][0]['message']['content']
        return json.loads(response_content)
