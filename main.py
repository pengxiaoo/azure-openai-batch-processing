import sys
from batch_task import BatchTask, BatchTaskType, BatchResult


def run_batch_tasks():
    extraction_task = BatchTask(
        task_type=BatchTaskType.EXTRACTION,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    extraction_task.run()
    sentiment_task = BatchTask(
        task_type=BatchTaskType.SENTIMENT,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    sentiment_task.run()
    summarization_task = BatchTask(
        task_type=BatchTaskType.SUMMARIZATION,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    summarization_task.run()


def merge():
    result = BatchResult()
    result.merge()


if __name__ == "__main__":
    running_mode = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if running_mode == "batch":
        run_batch_tasks()
        merge()
    elif running_mode == "merge":
        merge()
