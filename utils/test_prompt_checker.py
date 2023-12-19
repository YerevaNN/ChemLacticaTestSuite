import unittest
from prompt_checker import check_syntax, get_start2end_tags_map
import numpy as np
import random
import string


def random_str(length: int, num_of_dis_spaces: int=0, exclude: list=[]):
    vocab = list(string.printable[:-6] + string.digits)
    for char in exclude:
        vocab.remove(char)

    char_list = random.choices(vocab, k=length)
    if num_of_dis_spaces > 1:
        raise NotImplemented
    elif num_of_dis_spaces == 1:
        ind = random.randint(1, len(char_list) - 2)
        char_list[ind] = ' '
    return ''.join(char_list)


def generate_correct_prompts(start2end: dict, num_of_cases: int=200):
    tags_array = np.array(tuple(start2end.items()))

    test_cases = []
    for _ in range(num_of_cases):
        rand_indices = np.random.permutation(len(tags_array))
        random_tags = tags_array[rand_indices]
        similar_index = np.where(random_tags[:, 0] == "[SIMILAR]")[0][0]
        random_values = np.array([
            random_str(random.randint(10, 25), num_of_dis_spaces=1) 
            if i == similar_index else random_str(random.randint(10, 25))
            for i in range(len(tags_array))
        ])
        test_cases.append(''.join(np.char.add(random_tags[:, 0], np.char.add(random_values, random_tags[:, 1]))))

    return test_cases



class TestSyntax(unittest.TestCase):

    def test_correct(self):
        tokenizer_path = "src/tokenizer/ChemLacticaTokenizer_50066"
        start2end = get_start2end_tags_map(tokenizer_path)
        test_prompts = generate_correct_prompts(start2end, 1000)
        for prompt in test_prompts:
            check_syntax(prompt, start2end)

    def test_correct_full(self):
        tokenizer_path = "src/tokenizer/ChemLacticaTokenizer_50066"
        start2end = get_start2end_tags_map(tokenizer_path)
        prompts = []
        prompts.append("[")
        prompts.append("[CLO")
        prompts.append("[CLOGP")
        prompts.append("[CLOGP]")
        prompts.append("[CLOGP]1")
        prompts.append("[CLOGP]1.0129387980")
        prompts.append("[CLOGP]300[")
        prompts.append("[CLOGP]1.0129387980[/")
        prompts.append("[CLOGP]1123[/CLO")
        prompts.append("[CLOGP]1123[/CLOGP")
        prompts.append("[SIMILAR]112")
        prompts.append("[SIMILAR]11 2[")
        prompts.append("[SIMILAR]112 ")
        prompts.append("[SIMILAR]0.123 CCC[/SIM")
        test_prompts = generate_correct_prompts(start2end, len(prompts))
        for i in range(len(prompts)):
            test_prompts[i] += prompts[i]
        for prompt in test_prompts:
            check_syntax(prompt, start2end)

    def test_incorrect_full(self):
        tokenizer_path = "src/tokenizer/ChemLacticaTokenizer_50066"
        start2end = get_start2end_tags_map(tokenizer_path)
        prompts = []
        prompts.append("[CL OGP]1.01[/CLOGP]")
        prompts.append("[CLOGP]1.01 asd[/CLOGP]")
        prompts.append("[CLOGP] 300[/CLOGP]")
        prompts.append("[SIMILAR]112asd[/SIMILAR]")
        prompts.append("[SIMILAR]11 2[/SIM ILAR]")
        prompts.append("[ ")
        prompts.append("[C LO")
        prompts.append("[CLO GP")
        prompts.append("[CLOGP ]")
        prompts.append("[CLOGP] 1")
        prompts.append("[CLOGP]1.01 29387980")
        prompts.append("[CLOGP]300[ ")
        prompts.append("[CLOGP]1.0129387980[ /")
        prompts.append("[CLOGP]1123[/CL O")
        prompts.append("[CLOGP]1123[/CLO GP")
        prompts.append("[SIMILAR] 112")
        prompts.append("[SIMILAR]1 12 ")
        prompts.append("[SIMILAR]11 2[ ")
        prompts.append("[SIMILAR]11 2[/SIM ")
        test_prompts = generate_correct_prompts(start2end, len(prompts))
        for i in range(len(prompts)):
            test_prompts[i] += prompts[i]
        for prompt in test_prompts:
            try:
                check_syntax(prompt, start2end)
            except Exception as e:
                pass
            else:
                raise Exception(f"{prompt} should not have passed.")


if __name__ == "__main__":
    unittest.main()