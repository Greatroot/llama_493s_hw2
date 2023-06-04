import jsonlines

"""
    A script that counts the number of examples in a .jsonl training data file.
"""

# Path to the large JSON file
input_file = 'data/subset_data.jsonl'

num_examples = 0

# Open the large JSON file for reading
with jsonlines.open(input_file, 'r') as input_reader:
    # Iterate over each line in the large JSON file
    for obj in input_reader:
        num_examples += 1

print(f"The number of examples in {input_file} is {num_examples}")
