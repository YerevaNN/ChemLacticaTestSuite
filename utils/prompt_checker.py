import json
import os
from functools import cache


def assert_spaces(text: str, num_of_spaces: int):
    space_count = text.count(" ")
    if space_count > num_of_spaces:
        raise Exception(f"Too many spaces in entry '{text}'")
    elif space_count < num_of_spaces:
        raise Exception(f"Too few spaces in entry '{text}'")


@cache
def get_start2end_tags_map(tokenizer_path: str):
    with open(os.path.join(tokenizer_path, "special_tokens_map.json"), "r") as _f:
        special_tokens_map = json.load(_f)
    additional_tokens = special_tokens_map["additional_special_tokens"]
    n = len(additional_tokens)
    assert (n & 1) == 0 # should be even
    return {
        additional_tokens[i]: additional_tokens[n // 2 + i] for i in range(n // 2)
    } | {"[START_SMILES]": "[END_SMILES]"}


def check_syntax(prompt: str, start2end: dict):
    processing_str = ""
    start_tag = end_tag = ""
    state = 0
    for i in range(len(prompt)):
        processing_str += prompt[i]
        if not start_tag and prompt[i] == "]":
            start_tag = processing_str
            if not start2end.get(start_tag):
                raise Exception(f"Not valid start tag '{start_tag}'")
            state = 1
        
        if start_tag:
            for etag in start2end.values():
                if processing_str[-len(etag):] == etag:
                    end_tag = etag
                    state = 2
                    break

        if end_tag:
            if start2end[start_tag] != end_tag:
                raise Exception(f"Start and end tags don't match: '{start_tag}', '{end_tag}'")
            prop_value = processing_str[len(start_tag):-len(end_tag)]
            assert prop_value[0] != ' ' and prop_value[-1] != ' '
            assert_spaces(prop_value, 0 + int(start_tag == "[SIMILAR]"))
            processing_str = ""
            start_tag = end_tag = ""
            state = 0

    if processing_str: print("proc_str", processing_str)
    """
        3 possible states.
            0: some part of (or full) the start tag
            1: start tag + some part of (or full) the value
            2: start tag + value + some part of the end tag
    """
    if state == 0:
        for stag in start2end.keys():
            if stag.startswith(processing_str):
                return
        start_tag = processing_str
        raise Exception(f"Not valid start tag '{start_tag}'")
    else:
        prop_value = processing_str[len(start_tag):len(processing_str)-len(end_tag)]
        if prop_value == "":
            return
        assert prop_value[0] != ' '
        if start_tag != "[SIMILAR]":
            assert_spaces(prop_value, 0)
        else:
            assert prop_value.count(' ') < 2

    if state == 2:
        for etag in start2end.values():
            if etag.startswith(processing_str):
                return
        end_tag = processing_str[len(start_tag) + len(prop_value):]
        raise Exception(f"Not valid end tag '{end_tag}'")


oldsyntax_property_tags = {
    "[START_SMILES]": "[END_SMILES]",
    "[CID ": "]",
    "[SAS ": "]",
    "[WEIGHT ": "]",
    "[TPSA ": "]",
    "[CLOGP ": "]",
    "[QED ": "]",
    "[NUMHDONORS ": "]",
    "[NUMHACCEPTORS ": "]",
    "[NUMHETEROATOMS ": "]",
    "[NUMROTATABLEBONDS ": "]",
    "[NOCOUNT ": "]",
    "[NHOHCOUNT ": "]",
    "[RINGCOUNT ": "]",
    "[HEAVYATOMCOUNT ": "]",
    "[FRACTIONCSP3 ": "]",
    "[NUMAROMATICRINGS ": "]",
    "[NUMSATURATEDRINGS ": "]",
    "[NUMAROMATICHETEROCYCLES ": "]",
    "[NUMAROMATICCARBOCYCLES ": "]",
    "[NUMSATURATEDHETEROCYCLES ": "]",
    "[NUMSATURATEDCARBOCYCLES ": "]",
    "[NUMALIPHATICRINGS ": "]",
    "[NUMALIPHATICHETEROCYCLES ": "]",
    "[NUMALIPHATICCARBOCYCLES ": "]",
    "[IUPAC ": "]",
    "[SIMILARITY ": "]"
}


def check_nonassay_prompt_oldsyntax(prompt: str):
    SMILES_START_TAG = "[START_SMILES]"
    SMILES_END_TAG = "[END_SMILES]"
    SIMILARITY_TAG = "[SIMILARITY "
    processing_str = ""
    for i in range(len(prompt)):
        processing_str += prompt[i]
        if prompt[i] == "]":
            if processing_str == SMILES_START_TAG:
                continue
            if processing_str[-len(SMILES_END_TAG):] == SMILES_END_TAG:
                start_tag = processing_str[:len(SMILES_START_TAG)]
                value = processing_str[len(SMILES_START_TAG):-len(SMILES_END_TAG)]
                end_tag = processing_str[-len(SMILES_END_TAG):]
                if value.find(" ") !=  -1:
                    raise Exception(f"'{processing_str}' should not have spaces.")
            else:
                if processing_str.startswith(SIMILARITY_TAG):
                    assert_spaces(processing_str, 3)
                else:
                    assert_spaces(processing_str, 2)
                index = processing_str.find(" ") + 1
                start_tag, ending = processing_str[:index], processing_str[index:]
                value = ending[:-1]
                end_tag = ending[-1:]
            
            # print(start_tag, value, end_tag)
            if not oldsyntax_property_tags.get(start_tag):
                raise Exception(f"'{start_tag}' is not a valid starting tag.")
            if oldsyntax_property_tags[start_tag] != end_tag:
                raise Exception(f"'{end_tag}' is not a valid starting tag for '{start_tag}'.")
            processing_str = ""

    if processing_str != "":
        ok = False
        for tag in oldsyntax_property_tags.keys():
            if tag.startswith(processing_str) or processing_str.startswith(tag):
                if processing_str.startswith(tag):
                    value = processing_str[len(tag):]
                    if tag == SIMILARITY_TAG:
                        if len(value.split(" ")) > 2:
                            raise Exception(f"'{value}' has too many spaces.")
                    elif value.find(" ") !=  -1:
                        raise Exception(f"'{value}' should not have spaces.")
                ok = True
                break
        if not ok:
            raise Exception(f"'{processing_str}' is not a valid start.")