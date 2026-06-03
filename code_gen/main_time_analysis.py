import os
import sys

import pattern_utiles.pattern_dictionary as pdict
from spmm.spmm_time_analysis import (kernel_generator_nnz_base,
                                             kernel_generator_pattern_base,
                                             codelet_generator_nnz_base,
                                             codelet_generator_pattern_base,
                                             class_file_generator, base_file)

import argparse


def write_text_to_file(text, file_path):
    """
    Writes the given text to a file specified by the file path.

    Args:
        text (str): The text to write to the file.
        file_path (str): The path to the file where the text should be written.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(text)
    except Exception as e:
        print(f"An error occurred: {e}")


def main_format_compact(m, n, path, pattern_type_dictionary):
    bcol_unroll_factor = 1

    thread_block_size = 1

    second_grid_dim = 1

    kernel = kernel_generator_nnz_base(m, n, pattern_type_dictionary)

    codelet = codelet_generator_nnz_base(m, n, 1, pattern_type_dictionary)

    # print("codelet", codelet)

    class_text = ""

    text = base_file(codelet, kernel, class_text, m, n)

    write_text_to_file(text, path)


def main_format(m, n, path, pattern_type_dictionary):
    bcol_unroll_factor = 1

    thread_block_size = 1

    second_grid_dim = 1

    kernel = kernel_generator_pattern_base(m, n, pattern_type_dictionary)

    codelet = codelet_generator_pattern_base(m, n, pattern_type_dictionary)

    # print("codelet", codelet)

    class_text = ""

    text = base_file(codelet, kernel, class_text, m, n)

    write_text_to_file(text, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that demonstrates argument parsing.")

    # Add arguments
    parser.add_argument("path", type=str, default="spmm_demo_gpu_utils.h")
    parser.add_argument('m', type=int, default=4)
    parser.add_argument("n", type=int, default=4)
    parser.add_argument("compact", type=int, default=1)

    args = parser.parse_args()

    path = args.path
    m = args.m
    n = args.n
    compact = args.compact

    pattern_type_dictionary = pdict.generate_pattern_dictionary(m)

    if compact:
        main_format_compact(m, n, path, pattern_type_dictionary)
    else:
        main_format(m, n, path, pattern_type_dictionary)