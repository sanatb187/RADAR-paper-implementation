from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from radar_bench.schemas import EvaluationRecord, RADARSchema


class ResponseMatrix(RADARSchema):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    values: NDArray[np.int8]
    configuration_ids: tuple[str, ...]
    query_ids: tuple[str, ...]

    @classmethod
    def from_records(
        cls, evaluation_records: Sequence[EvaluationRecord]
    ) -> "ResponseMatrix":
        if not evaluation_records:
            raise ValueError("evaluation_records cannot be empty")

        configuration_ids = tuple(
            sorted(
                {record.generation.configuration_id for record in evaluation_records}
            )
        )

        query_ids = tuple(
            sorted({record.generation.query_id for record in evaluation_records})
        )

        configuration_index = {
            configuration_id: index
            for index, configuration_id in enumerate(configuration_ids)
        }

        query_index = {query_id: index for index, query_id in enumerate(query_ids)}

        # -1 means that a configuration-query pair has not been filled.
        values = np.full(
            shape=(len(configuration_ids), len(query_ids)),
            fill_value=-1,
            dtype=np.int8,
        )

        for record in evaluation_records:
            configuration_id = record.generation.configuration_id
            query_id = record.generation.query_id

            row = configuration_index[configuration_id]
            column = query_index[query_id]

            if values[row, column] != -1:
                raise ValueError(
                    "Duplicate evaluation record for "
                    f"configuration={configuration_id!r}, query={query_id!r}"
                )

            values[row, column] = int(record.correct)

        missing_locations = np.argwhere(values == -1)

        if missing_locations.size > 0:
            missing_pairs = [
                (
                    configuration_ids[row],
                    query_ids[column],
                )
                for row, column in missing_locations
            ]

            raise ValueError(f"Missing evaluation records for pairs: {missing_pairs}")

        return cls(
            values=values,
            configuration_ids=configuration_ids,
            query_ids=query_ids,
        )
