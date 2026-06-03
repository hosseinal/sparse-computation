import argparse
import pattern_utiles.pattern_dictionary as pdict
from spmm_small_b_col.spmm_small_b_col_format_code_generator import *


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


def main(m, number_of_warps, path, b_col, pattern_type_dictionary):
    # the kernel generator code will be here
    kernel_code = kernel_code_generator(pattern_type_dictionary)
    #
    # the codelet generator code will be here
    codelet_code = codelet_code_generator(m, b_col)
    # the class file generator code will be here
    class_code = class_code_generator(m, number_of_warps, pattern_type_dictionary)
    # the base file generator code will be here
    base_code = base_file(codelet_code, kernel_code, class_code, m, pattern_type_dictionary)
    # write text to file code will be here
    write_text_to_file(base_code, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that demonstrates argument parsing.")

    # Add arguments
    parser.add_argument("path", type=str, default="spmm_small_b_col_demo_gpu_utils.h")
    parser.add_argument('m', type=int, default=4)
    parser.add_argument('number_of_warps', type=int, default=1)
    parser.add_argument('b_col', type=int, default=1)
    parser.add_argument("--pattern_dictionary", type=str, default="false")

    args = parser.parse_args()

    path = args.path
    b_col = args.b_col
    m = args.m
    number_of_warps = args.number_of_warps
    pattern_dict = args.pattern_dictionary

    if pattern_dict != "false":
        pattern_type_dictionary = pdict.read_pattern_dictionary(pattern_dict)
    else:
        pattern_type_dictionary = pdict.generate_pattern_dictionary(m, 1)

    main(m, number_of_warps, path, b_col, pattern_type_dictionary)
