all_possible_tags = {
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

def assert_spaces(text: str, num_of_spaces: int):
    entries = text.split(" ")
    if len(entries) > num_of_spaces:
        raise Exception(f"Too many spaces in entry '{text}'")
    elif len(entries) < num_of_spaces:
        raise Exception(f"No spaces in entry '{text}'")


def check_prompt(prompt: str):
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
            if not all_possible_tags.get(start_tag):
                raise Exception(f"'{start_tag}' is not a valid starting tag.")
            if all_possible_tags[start_tag] != end_tag:
                raise Exception(f"'{end_tag}' is not a valid starting tag for '{start_tag}'.")
            
            # print(processing_str)
            processing_str = ""

    if processing_str != "":
        ok = False
        for tag in all_possible_tags.keys():
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


if __name__ == "__main__":
    prompt = "[CLOGP 3.0][START_SMILES]lkhasdlkjasldlhasd[END_SMILES][SIMILARITY 0.8SMILES asd"
    check_prompt(prompt)