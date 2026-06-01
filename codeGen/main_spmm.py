import os
import sys

import pattern_utiles.pattern_dictionary as pdict
from spmm.spmm_format_code_generator import (kernel_generator_nnz_base,
                                             codelet_generator_nnz_base, class_file_generator, base_file)
from spmm.spmm_dense_code_generator import (dense_kernel_generator_nnz_base,
                                            dense_codelet_generator_nnz_base, dense_class_file_generator,
                                            dense_base_file)
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
        print(f"Text successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main_format(m, n, bcol_unroll_factor, warp_num, path, second_grid_dim, pattern_type_dictionary):
    bcol_unroll_factor = int(bcol_unroll_factor)

    thread_block_size = warp_num

    kernel = kernel_generator_nnz_base(m, n, pattern_type_dictionary)

    codelet = codelet_generator_nnz_base(m, n, bcol_unroll_factor, pattern_type_dictionary)

    class_text = class_file_generator(m, n, thread_block_size, second_grid_dim, pattern_type_dictionary)

    text = base_file(codelet, kernel, class_text, m, n, pattern_type_dictionary)

    write_text_to_file(text, path)


def main_dense(m, n, bcol_unroll_factor, warp_num, path, loop_order, pattern_type_dictionary):
    bcol_unroll_factor = int(bcol_unroll_factor)

    thread_block_size = warp_num

    kernel = dense_kernel_generator_nnz_base(m, n, pattern_type_dictionary)

    codelet = dense_codelet_generator_nnz_base(m, n, bcol_unroll_factor, loop_order)

    class_text = dense_class_file_generator(m, n, thread_block_size)

    text = dense_base_file(codelet, kernel, class_text)

    write_text_to_file(text, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that demonstrates argument parsing.")

    # Add arguments
    parser.add_argument("path", type=str, default="spmm_demo_gpu_utils.h")
    parser.add_argument("matrix_type", type=str, default="format")
    parser.add_argument('m', type=int, default=4)
    parser.add_argument("n", type=int, default=4)
    parser.add_argument("bcol_unroll_factor", type=int, default=32)
    parser.add_argument("warp_num", type=int, default=1)
    parser.add_argument("loop_order", type=str, default="i-j")
    parser.add_argument("--pattern_dictionary", type=str, default="false")
    parser.add_argument("--second_grid_dim", type=int, default=1)

    args = parser.parse_args()

    path = args.path
    matrix_type = args.matrix_type
    m = args.m
    n = args.n
    bcol_unroll_factor = args.bcol_unroll_factor
    warp_num = args.warp_num
    loop_order = args.loop_order
    pattern_dict = args.pattern_dictionary
    second_grid_dim = args.second_grid_dim

    if pattern_dict != "false":
        pattern_type_dictionary = pdict.read_pattern_dictionary(pattern_dict)
    else:
        pattern_type_dictionary = pdict.generate_pattern_dictionary(m, n)

    if matrix_type == "dense":
        main_dense(m, n, bcol_unroll_factor, warp_num, path, loop_order, pattern_type_dictionary)
    else:
        main_format(m, n, bcol_unroll_factor, warp_num, path, second_grid_dim, pattern_type_dictionary)
