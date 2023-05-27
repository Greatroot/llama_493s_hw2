import zstandard as zstd
import json
import shutil

# def load_data(file_path):
#     """
    
#     """
#     # Open the .zst file in binary mode
#     with open(file_path, 'rb') as compressed_file:
#         # print(f"did this get opened?")
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(compressed_file) as reader:
#             # Read the uncompressed data
#             data = reader.read()

#     # Process the uncompressed data
#     # (e.g., convert it to string or parse it as JSON)
#     print(json.load(data))

# def split_zst_file(input_file, output_file, split_ratio):
#     with open(input_file, 'rb') as compressed_file:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(compressed_file) as reader:
#             # Create the first destination file
#             with open(output_file, 'wb') as dest_file_1:
#                 with dctx.stream_writer(dest_file_1) as writer_1:
#                     # Read and process the data line by line
#                     for line in reader:
#                         # Parse the JSON data
#                         data = json.loads(line)

#                         # Process the data as needed
#                         # (e.g., convert it to tensors)

#                         # Write the line to the first destination file
#                         writer_1.write(line)

#                         # Break the loop when the desired split size is reached
#                         if reader.current_offset > reader.total_read * split_ratio:
#                             break


def split_zst_file(input_file, output_file, split_ratio):
    # Open the compressed input file in read mode
    with open(input_file, 'rb') as infile:

        # Create a ZstdDecompressor object to decompress the file
        dctx = zstd.ZstdDecompressor()

        # Create a ZstdDecompressedReader object to read the decompressed data
        reader = dctx.stream_reader(infile)

        # Open the output file in write mode
        with open(output_file, 'wb') as outfile:

            # Create a ZstdCompressor object to compress the data
            cctx = zstd.ZstdCompressor()

            # Create a ZstdCompressedWriter object to write the compressed data
            writer = cctx.stream_writer(outfile)

            # Iterate over the lines in the reader (streaming the JSON data)
            for line in reader:
                # Parse each line as JSON
                data = json.loads(line)

                json.dump(data, writer)
                writer.write('\n')  # Add a newline character between JSON objects

                # Break the loop when the desired split size is reached
                if reader.current_offset > reader.total_read * split_ratio:
                    break



def split_train_test(source_file, train_file, test_file, split_ratio):
    with open(source_file, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            # Determine the number of lines in the file
            line_count = sum(1 for _ in reader)

            # Calculate the number of lines to be included in each split
            split_size = int(line_count * split_ratio)

            # Rewind to the beginning of the file
            compressed_file.seek(0)

            # Create the first destination file
            with open(train_file, 'wb') as dest_file_1:
                with dctx.stream_writer(dest_file_1) as writer_1:
                    # Copy the first split_size lines to the first destination file
                    for _ in range(split_size):
                        line = reader.readline()
                        writer_1.write(line)

            # Create the second destination file
            with open(test_file, 'wb') as dest_file_2:
                with dctx.stream_writer(dest_file_2) as writer_2:
                    # Copy the remaining lines to the second destination file
                    shutil.copyfileobj(reader, writer_2)
    

if __name__ == '__main__':
    # Load a subset of the Pile dataset from a .zst file

    # Paths to the input .zst file and output split files
    input_file = '00.jsonl.zst'
    output_file1 = 'subset_data.jsonl.zst'

    # Split the .zst file into two separate files
    split_zst_file(input_file, output_file1, split_ratio=0.4)

    # # Specify the split ratio (0.85 for the first file, 0.15 for the second file)
    # split_ratio = 0.85

    # # Split the .zst file into two separate files
    # split_train_test(source_file_path, dest_file_path_1, dest_file_path_2, split_ratio)
