import os
import io
from enum import Enum
import csv
import json
import datetime
import time
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# use constants defined in .env file
load_dotenv()
provider = os.getenv('API_PROVIDER')
max_requests = 40000


class BatchTaskType(str, Enum):
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    EXTRACTION = "key_attributes_extraction"


class ApiProvider(Enum):
    MICROSOFT_AZURE = "MICROSOFT_AZURE"
    OPENAI = "OPENAI"


def latest_file(files, prefix):
    files_by_type = [filename for filename in files if prefix in filename]
    return max(files_by_type, key=lambda x: x.split('_')[-1].split('.')[0])


def create_chunks(file_content):
    chunks = []
    for i in range(0, len(file_content), max_requests):
        chunks.append(file_content[i:i + max_requests])
    return chunks


class BatchResult:
    def __init__(self):
        self.directory = "output_data"
        self.sentiment_type_prefix = f"sentiment_score_avg_result_{BatchTaskType.SENTIMENT.value}"
        self.summary_type_prefix = f"join_result_{BatchTaskType.SUMMARIZATION.value}"
        self.extraction_type_prefix = f"join_result_{BatchTaskType.EXTRACTION.value}"
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.join_final_result_data_path = self.output_data(f"join_final_result_{time_str}.csv")

    def output_data(self, filename):
        return f"{self.directory}/{filename}"

    def merge(self):
        file_list = []
        for filename in os.listdir(self.directory):
            f = os.path.join(self.directory, filename)
            if os.path.isfile(f):
                file_list.append(filename)
        latest_sentiment_file = latest_file(file_list, self.sentiment_type_prefix)
        latest_summary_file = latest_file(file_list, self.summary_type_prefix)
        latest_extraction_file = latest_file(file_list, self.extraction_type_prefix)
        print(latest_sentiment_file)
        print(latest_summary_file)
        print(latest_extraction_file)
        self.join_final_result_data(
            self.output_data(latest_sentiment_file),
            self.output_data(latest_summary_file),
            self.output_data(latest_extraction_file),
        )

    def join_final_result_data(self, latest_sentiment_file, latest_summary_file, latest_extraction_file):
        sentiment_output_data = pd.read_csv(latest_sentiment_file)
        summarization_data = pd.read_csv(latest_summary_file)[['club_id', 'course_id', 'result', 'comment']].rename(
            columns={'result': 'summarization_in_80_words_paragraph', 'comment': 'last_20_comments'})
        extraction_data = pd.read_csv(latest_extraction_file)[['club_id', 'course_id', 'result']].rename(
            columns={'result': 'summary_in_short_phrases'})
        course_names = pd.read_csv("input_data/golf_course_names.csv")
        merged_df = pd.merge(sentiment_output_data, summarization_data, on=['club_id', 'course_id'], how='inner')
        merged_df = pd.merge(merged_df, extraction_data, on=['club_id', 'course_id'], how='inner')
        merged_df = pd.merge(merged_df, course_names, on=['club_id', 'course_id'], how='inner')
        merged_df.to_csv(self.join_final_result_data_path)


class BatchTask:

    def __init__(self,
                 task_type: BatchTaskType,
                 input_data_path,
                 encoding="utf-8",
                 comment_col_name="comment",
                 ):
        self.task_type = task_type
        self.input_data_path = input_data_path
        self.input_data_concatenated_path = input_data_path.replace(".csv", "_concatenated.csv")
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.llm_result_data_path = f"output_data/llm_result_{task_type.value}_{time_str}.csv"
        self.join_result_data_path = f"output_data/join_result_{task_type.value}_{time_str}.csv"
        self.sentiment_score_avg_result_data_path = f"output_data/sentiment_score_avg_result_{task_type.value}_{time_str}.csv"
        self.encoding = encoding
        self.comment_col_name = comment_col_name
        self.batch_ids = []
        self.num_of_comments = 20
        if provider == ApiProvider.MICROSOFT_AZURE.value:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.batch_endpoint = "/chat/completions"
            self.model = "course-review"
        else:
            OpenAI.api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI()
            self.batch_endpoint = "/v1/chat/completions"
            self.model = "gpt-4o-mini"

    def get_prompt(self) -> str:
        if self.task_type == BatchTaskType.SENTIMENT:
            return """
                The following is a review of a golf course from a golfer who had played there.
                Please classify the sentiment of this review as either "positive" or "negative" or "neutral". 
                Translate non-English content before answering, ignore unicodes and unrecognized words. 
                Respond with one word only.
                The review text is:
            """
        elif self.task_type == BatchTaskType.SUMMARIZATION:
            return """
                The following are reviews of a golf course from multiple golfers who had played there recently.
                individual review is started with a new line.
                Please summarize the reviews into a single paragraph, in 40~80 words, 
                so that new golfers can quickly know about the golf course. 
                Translate non-English content before answering, ignore unicodes and unrecognized words.
                The text of the reviews is:
            """
        elif self.task_type == BatchTaskType.EXTRACTION:
            return """
                The following are reviews of a golf course from multiple golfers who had played there recently.
                individual review is started with a new line.
                Please summarize the reviews into 1 to 3 concise phrases, separated by semicolons, without using 
                bullet points, so that new golfers can quickly know about the golf course. 
                Translate non-English content before answering, ignore unicodes and unrecognized words.
                The text of the reviews is:
            """
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def run(self):
        # see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?pivots=programming-language-python
        # https://platform.openai.com/docs/guides/batch
        # https://platform.openai.com/docs/api-reference/batch
        self.batch_ids = self.upload_and_create_chunked_batches()
        if len(self.batch_ids) > 0:
            llm_success = self.track_and_save_batch_result()
            if not llm_success:
                print("Something went wrong with the LLM batch task.")
            if self.task_type == BatchTaskType.SENTIMENT:
                self.join_llm_result_with_input_data(self.input_data_path)
                self.save_calculate_sentiment_score_avg()
            elif self.task_type == BatchTaskType.SUMMARIZATION:
                self.join_llm_result_with_input_data(self.input_data_concatenated_path)
            elif self.task_type == BatchTaskType.EXTRACTION:
                self.join_llm_result_with_input_data(self.input_data_concatenated_path)

    def upload_and_create_chunked_batches(self):
        jsonl_file_path = self.convert_csv_to_jsonl()
        with open(jsonl_file_path, "r") as file:
            file_content = [json.loads(line) for line in file]
        batch_ids = []
        for chunk in create_chunks(file_content):
            file_id = self.upload_file(chunk)
            batch_id = self.create_batch(file_id)
            batch_ids.append(batch_id)
            print(f"Created batch with ID {batch_id} for chunk with File ID {file_id}")
        return batch_ids

    def convert_csv_to_jsonl(self):
        if self.task_type == BatchTaskType.SUMMARIZATION or self.task_type == BatchTaskType.EXTRACTION:
            # concatenate comments for each golf course
            df = pd.read_csv(self.input_data_path).sort_values(by=["comment_time"], ascending=False)
            requests = (df.groupby(["club_id", "course_id"])["comment"]
                        .apply(lambda x: "\n".join(x.head(self.num_of_comments).fillna('')))
                        .reset_index())
            df = pd.DataFrame(requests)
            df.to_csv(self.input_data_concatenated_path, index=False, quoting=csv.QUOTE_ALL)
        else:
            df = pd.read_csv(self.input_data_path, encoding=self.encoding)
        prompt = self.get_prompt()
        jsonl_output = []
        for idx, row in df.iterrows():
            json_obj = {
                "custom_id": str(idx),  # custom_id is the row number(starts from 0) of the input csv file
                "method": "POST",
                "url": self.batch_endpoint,
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": f"{prompt} {str(row[self.comment_col_name])}"}
                    ]
                }
            }
            jsonl_output.append(json.dumps(json_obj))
        output_file_path = f"input_data/requests_{self.task_type.value}.jsonl"
        with open(output_file_path, "w") as file:
            for item in jsonl_output:
                file.write(item + "\n")
        return output_file_path

    def upload_file(self, chunk):
        json_data = '\n'.join(json.dumps(data) for data in chunk).encode(self.encoding)
        file = self.client.files.create(
            file=io.BytesIO(json_data),
            purpose="batch"
        )
        return file.id

    def create_batch(self, file_id: str) -> str:
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint=self.batch_endpoint,
            completion_window="24h",
        )
        print(batch_response.model_dump_json(indent=2))
        return batch_response.id

    def track_and_save_batch_result(self) -> bool:
        output_file_ids = self.wait_for_batches_to_complete(self.batch_ids)
        if len(output_file_ids) > 0:
            # merge output files
            merged_results = []
            for file_id in output_file_ids:
                result = self.retrieve_batch_result(file_id)
                merged_results.extend(result)
            # save data
            with open(self.llm_result_data_path, "w", newline="", encoding=self.encoding, errors="ignore") as csv_file:
                writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                header = ["custom_id", "result"]
                writer.writerow(header)
                merged_results = sorted(merged_results, key=lambda x: int(x["custom_id"]))
                for data in merged_results:
                    custom_id = data.get("custom_id")
                    result = data.get("result")
                    try:
                        writer.writerow([custom_id, result])
                    except Exception as e:
                        print(f"Error in track_and_save_batch_result: {e}")
                        print(custom_id, "=========", result)
            print(f"Data has been exported to {self.llm_result_data_path}")
            return True
        else:
            print("The batch is canceled.")
            return False

    def wait_for_batches_to_complete(self, batch_ids):
        output_file_ids = []
        for batch_id in batch_ids:
            status = "pending"
            while status not in ("completed", "failed", "canceled"):
                time.sleep(60)
                batch_response = self.client.batches.retrieve(batch_id)
                status = batch_response.status
                print(f"{datetime.datetime.now()} Batch ID: {batch_id}, Status: {status}")
                if status == "failed":
                    if batch_response:
                        for error in batch_response.errors.data:
                            print(f"Error code {error.code} Message {error.message}")
                    return []
                elif status == "completed" and batch_response.output_file_id:
                    output_file_ids.append(batch_response.output_file_id)
                    print(f"Output file: {batch_response.output_file_id}")
        return output_file_ids

    def retrieve_batch_result(self, output_file_id):
        file_response = self.client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split("\n")
        json_data = []
        result = []
        for raw_response in raw_responses:
            json_response = json.loads(raw_response)  # Convert raw JSON string to Python dict
            json_data.append(json_response)
        if len(json_data) > 0:
            json_data = sorted(json_data, key=lambda x: int(x["custom_id"]))
            for entry in json_data:
                custom_id = entry.get("custom_id")
                try:
                    result.append({
                        'custom_id': custom_id,
                        'result': entry["response"]["body"]["choices"][0]["message"]["content"].capitalize()
                    })
                except Exception as e:
                    print(f"Error in retrieve_batch_result: {e}")
                    print(custom_id, "=========", result)
        return result

    def join_llm_result_with_input_data(self, input_data_path):
        df_input_data = pd.read_csv(input_data_path)
        df_input_data["custom_id"] = range(0, len(df_input_data))
        df_llm_result = pd.read_csv(self.llm_result_data_path)
        df_merged = pd.merge(df_input_data, df_llm_result, on="custom_id").drop(["custom_id"], axis=1)
        df_merged.to_csv(self.join_result_data_path, encoding=self.encoding, index=False, quoting=csv.QUOTE_ALL)

    def save_calculate_sentiment_score_avg(self):
        sentiment_map = {
            'Positive': 1,
            'Negative': -1,
            'Neutral': 0,
            'Mixed': 0,
        }
        df = pd.read_csv(self.join_result_data_path)
        df['sentiment_score'] = df.iloc[:, -1].map(sentiment_map)
        aggregated = (df.sort_values(by='comment_time', ascending=False)
                      .groupby(['club_id', 'course_id'], group_keys=False)
                      .apply(lambda x: x.head(self.num_of_comments))
                      .groupby(['club_id', 'course_id']).agg(score_sum=('sentiment_score', 'sum'),
                                                             review_count=('sentiment_score', 'count'))
                      .reset_index())
        aggregated['sentiment_score_avg'] = aggregated['score_sum'] / aggregated['review_count']
        scored_course_review_df = aggregated[['club_id', 'course_id', 'sentiment_score_avg']]
        scored_course_review_df.to_csv(self.sentiment_score_avg_result_data_path,
                                       encoding='utf-8',
                                       index=False,
                                       quoting=csv.QUOTE_ALL)
