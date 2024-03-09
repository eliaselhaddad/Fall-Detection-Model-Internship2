import pandas as pd
import numpy as np
import pandas as pd

from src.helper_functions.model_helper_functions import ModelHelpingFunctions


class SplitSequences:
    def __init__(self, filepath: str, output_directory: str):
        data_to_split = pd.read_csv(filepath)
        self.data = data_to_split[data_to_split["fall_state"].isin([0, 1])]
        self.output_directory = output_directory
        self.in_fall_sequence = False
        self.end_index = None
        self.start_index = 0
        self.counter = 0
        self.model_helper = ModelHelpingFunctions()

    def random_value_between_20_and_40(self) -> int:
        return np.random.randint(20, 40)

    def split_csv(self) -> None:
        try:
            self.model_helper.log_info(
                "Splitting CSV files into sequences of falls and non-falls"
            )

            for i in range(1, len(self.data)):
                current_state = self.data.iloc[i]["fall_state"]
                previous_state = self.data.iloc[i - 1]["fall_state"]

                self.process_sequence(current_state, previous_state, i)

            if self.start_index < len(self.data):
                self.save_sequence(
                    self.start_index, len(self.data), self.in_fall_sequence
                )

        except Exception as e:
            self.model_helper.log_exception(e)

    def process_sequence(self, current_state: int, previous_state: int, i: int) -> None:
        try:
            self.model_helper.log_info(f"Processing sequence at index {i}")
            if current_state != previous_state:
                self.end_index = i
                if previous_state == 1:
                    self.save_sequence(self.start_index, self.end_index, True)
                    self.in_fall_sequence = False
                else:
                    self.save_sequence(self.start_index, self.end_index, False)
                    self.in_fall_sequence = False
                self.start_index = i

            if current_state == 1 and not self.in_fall_sequence:
                self.in_fall_sequence = True
                self.start_index = i
        except Exception as e:
            self.model_helper.log_error(e)

    def save_sequence(self, start_index: int, end_index: int, is_fall: bool) -> None:
        try:
            self.model_helper.log_info("Saving sequence")
            sequence = self.extract_sequence(start_index, end_index, is_fall)
            if sequence is not None:
                self.write_sequence_to_file(sequence, is_fall)

        except Exception as e:
            self.model_helper.log_exception(e)

    def extract_sequence(
        self, start_index: int, end_index: int, is_fall: bool
    ) -> pd.DataFrame:
        if is_fall:
            return self.extract_fall_sequence(start_index, end_index)
        else:
            return self.extract_non_fall_sequence(start_index, end_index)

    def extract_fall_sequence(self, start_index: int, end_index: int) -> pd.DataFrame:
        try:
            fall_sequence = self.data.iloc[start_index:end_index]
            fall_indices = fall_sequence["fall_state"] == 1

            if not fall_indices.any():
                self.model_helper.log_error(
                    "No fall indices found in fall sequence. Skipping..."
                )
                return None

            fall_start_index = fall_sequence[fall_indices].index[0]
            fall_end_index = fall_sequence[fall_indices].index[-1]

            random_pre_fall = self.random_value_between_20_and_40()
            pre_fall_start_index = max(fall_start_index - random_pre_fall, 0)
            pre_fall_sequence = self.data.iloc[pre_fall_start_index:fall_start_index]
            pre_fall_sequence = pre_fall_sequence[pre_fall_sequence["fall_state"] == 0]

            post_fall_start_index = fall_end_index + 1
            max_post_fall_length = 100 - (len(pre_fall_sequence) + len(fall_sequence))
            post_fall_end_index = min(
                post_fall_start_index + max_post_fall_length, len(self.data)
            )
            post_fall_sequence = self.data.iloc[
                post_fall_start_index:post_fall_end_index
            ]
            post_fall_sequence = post_fall_sequence[
                post_fall_sequence["fall_state"] == 0
            ]

            return pd.concat([pre_fall_sequence, fall_sequence, post_fall_sequence])
        except Exception as e:
            self.model_helper.log_exception(e)

    def extract_non_fall_sequence(
        self, start_index: int, end_index: int
    ) -> pd.DataFrame:
        try:
            non_fall_end_index = min(end_index, start_index + 99, len(self.data))
            self.model_helper.log_info(
                f"Extracting non-fall sequence from {start_index} to {non_fall_end_index}"
            )
            return self.data.iloc[start_index:non_fall_end_index]
        except Exception as e:
            self.model_helper.log_exception(e)

    def write_sequence_to_file(self, sequence: pd.DataFrame, is_fall: bool) -> None:
        try:
            file_name = f"{self.output_directory}{'fall' if is_fall else 'non_fall'}_{self.counter}.csv"
            sequence.to_csv(file_name, index=False)
            self.counter += 1
            self.model_helper.log_info(f"Sequence saved to {file_name}")
        except Exception as e:
            self.model_helper.log_exception(e)

    def handle_last_sequence(self):
        try:
            self.model_helper.log_info(
                f"Handling last sequence at index {len(self.data)}"
            )
            last_state = self.data.iloc[-1]["fall_state"]
            self.save_sequence(self.start_index, len(self.data), last_state == 1)
        except Exception as e:
            self.model_helper.log_exception(e)


def main():
    filepath = "data/cleaned/merged_data.csv"
    split_sequences = SplitSequences(filepath, "data/seq/")
    split_sequences.split_csv()


if __name__ == "__main__":
    main()
