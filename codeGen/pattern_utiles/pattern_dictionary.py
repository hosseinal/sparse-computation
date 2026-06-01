class PatternTypeDictionary:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.id_to_patten_bin = {}
        self.pattern_bin_to_id = {}
        self.generated_patterns_ids = []
        self.pattern_id_to_density = {}

    def add_pattern(self, pattern_number, pattern_binary):
        self.id_to_patten_bin[pattern_number] = pattern_binary
        self.pattern_bin_to_id[pattern_binary] = pattern_number

    def append_generated_pattern(self, pattern_number):
        self.generated_patterns_ids.append(pattern_number)

    def is_pattern_generated(self, pattern_number):
        return pattern_number in self.generated_patterns_ids

    def pattern_id_to_number_of_nnz(self, pattern_number):
        return self.id_to_patten_bin[pattern_number].count('1')

    def export_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in self.id_to_patten_bin.items():
                f.write(f'{key} {value}\n')


def build_patterns_map(kernel_m: int, kernel_n: int) -> dict:
    """All non-zero kernel masks, same order as generate_pattern_dictionary."""
    patterns = {}
    bits = kernel_m * kernel_n
    for i in range(1, 2**bits):
        x = 2**bits - i
        arr = tuple(c == "1" for c in bin(x)[2:].zfill(bits))
        patterns[arr] = i - 1
    return patterns


def generate_pattern_dictionary(m, n=1):
    pattern_type_dictionary = PatternTypeDictionary(m, n)
    bits = m * n
    for i in range(1, 2**bits):
        x = 2**bits - i
        pattern_type_dictionary.add_pattern(i - 1, bin(x)[2:].zfill(bits))
    return pattern_type_dictionary


def read_pattern_dictionary(file_path):
    with open(file_path, 'r') as f:
        # get the first line
        m, n, number_of_patterns = f.readline().split()
        pattern_type_dictionary = PatternTypeDictionary(m, n)

        # loop over the rest of the lines
        for line in f:
            value, key = line.split()
            pattern_type_dictionary.add_pattern(int(key), value)

    return pattern_type_dictionary
