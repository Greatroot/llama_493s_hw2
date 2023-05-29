import json
import jsonlines
import random

"""
    A script that splits a large .jsonl file into training and testing .jsonl files following a ratio
    defined below.
"""

# Path to the large JSON file
input_file = 'data/subset_data.jsonl'

# Path to the new JSON file
train_file = 'data/train.jsonl'
test_file = 'data/test.jsonl'

# Define the percentage of lines to read (40% in this case)
train_ratio = 0.85

# Open the large JSON file for reading
with jsonlines.open(input_file, 'r') as input_reader:
    # Open the new JSON file for writing
    with jsonlines.open(train_file, 'w') as train_writer, jsonlines.open(test_file, 'w') as test_writer:
        # Iterate over each line in the large JSON file
        for obj in input_reader:
            # 85% of the data will go into the training .jsonl file
            if random.random() <= train_ratio:
                # Write the selected JSON data to the new JSON file
                train_writer.write(obj)
            else:
                # The other 15% will go into the testing .jsonl file
                test_writer.write(obj)
