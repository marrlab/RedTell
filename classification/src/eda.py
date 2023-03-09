import enum

from dataclasses import dataclass

import data_ingest


class TaskType(enum.Enum):
    binary = 'binary'
    multiclass = 'multiclass'


@dataclass(frozen=True)
class DataSetEDA:
    unique_labels: set[str]

    @property
    def num_unique_labels(self) -> int:
        return len(self.unique_labels)

    @property
    def task_type(self) -> TaskType:
        if self.num_unique_labels == 2:
            return TaskType.binary
        else:
            return TaskType.multiclass


def perform_eda(data_ingest_result: data_ingest.DataIngestResult) -> DataSetEDA:
    unique_labels = {str(value) for value in data_ingest_result.all_target_labels}
    return DataSetEDA(unique_labels=unique_labels)
