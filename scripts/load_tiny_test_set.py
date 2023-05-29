import json
import jsonlines
import random

"""
    A script that splits a large .jsonl file into one that is "split_ratio" times smaller.
"""

# Path to the large JSON file
input_file = 'data/subset_data.jsonl'

# Path to the new JSON file
output_file = 'data/tiny_test_set.jsonl'

num_lines = 10
lines_read = 0

# Open the large JSON file for reading
with jsonlines.open(input_file, 'r') as input_reader:
    # Open the new JSON file for writing
    with jsonlines.open(output_file, 'w') as output_writer:
        # Iterate over each line in the large JSON file
        for obj in input_reader:
            # Write the selected JSON data to the new JSON file
            output_writer.write(obj)
            lines_read += 1

            if num_lines < lines_read:
                break

