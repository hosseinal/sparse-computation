import argparse

import pattern_utiles.pattern_dictionary as pdict
from spmv.spmv_format_code_generator import (
    base_code_generator,
    class_code_generator,
    codelet_code_generator,
    kernel_code_generator,
)


def write_text_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)
    print(f"Text successfully written to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="example_cuda/spmv_demo_gpu_utils.h")
    parser.add_argument("m", type=int, default=4)
    parser.add_argument("number_of_warps", type=int, default=1)
    parser.add_argument("--pattern_dictionary", type=str, default="false")
    args = parser.parse_args()

    if args.pattern_dictionary != "false":
        pattern_type_dictionary = pdict.read_pattern_dictionary(args.pattern_dictionary)
    else:
        pattern_type_dictionary = pdict.generate_pattern_dictionary(args.m, 1)

    kernel_file = kernel_code_generator(pattern_type_dictionary)
    codelet_file = codelet_code_generator(args.m)
    class_file = class_code_generator(args.m, args.number_of_warps)
    text = base_code_generator(codelet_file, kernel_file, class_file)
    write_text_to_file(text, args.path)
