from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":
    sentiment_task = BatchTask(
        task_type=BatchTaskType.SENTIMENT,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    sentiment_output_data = sentiment_task.run()

    summarization_task = BatchTask(
        task_type=BatchTaskType.SUMMARIZATION,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    summarization_data = summarization_task.run()

    extraction_task = BatchTask(
        task_type=BatchTaskType.EXTRACTION,
        input_data_path="input_data/golf_course_reviews_sample.csv",
    )
    extraction_data = extraction_task.run()

    extraction_task.merge(sentiment_output_data, summarization_data, extraction_data)
