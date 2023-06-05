import jsonlines

"""
    A script that creates smaller datasets of size 
    10,000, 100,000, and 1,000,000 training examples from the Pile 00.jsonl dataset.
"""

# Path to the large JSON file
# input_file = 'data/subset_data.jsonl'
input_file = 'data/train_1000000.jsonl'
# data_sizes = [10000, 100000, 1000000]
data_sizes = [50000]


lines_read = 0

for size in data_sizes:
    output_file = f'data/train_{size}.jsonl'
    # Open the large JSON file for reading
    with jsonlines.open(input_file, 'r') as input_reader:
        # Open the new JSON file for writing
        with jsonlines.open(output_file, 'w') as output_writer:
            # Iterate over each line in the large JSON file
            for obj in input_reader:
                # Write the selected JSON data to the new JSON file
                output_writer.write(obj)
                lines_read += 1

                if size < lines_read:
                    break

