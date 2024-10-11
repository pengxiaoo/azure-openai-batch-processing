from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":
    sentiment_task = BatchTask(
        task_type=BatchTaskType.SENTIMENT,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    sentiment_task.run()

    summary_task = BatchTask(
        task_type=BatchTaskType.SUMMARIZATION,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    summary_task.run()

    extraction_task = BatchTask(
        task_type=BatchTaskType.EXTRACTION,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    extraction_task.run()
