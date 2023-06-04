import jsonlines

"""
    A script that stores the first 10,000 examples in Pile's validation data in a seperate .jsonl file.
"""

# Path to the large JSON file
input_file = 'data/val.jsonl'

# Path to the new JSON file
output_file = 'data/val_10000.jsonl'

num_lines = 10000
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

