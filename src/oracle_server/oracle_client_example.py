import requests
import json


def score_with_oracle(input_strings,oracle_url):
    input_data_json = json.dumps(input_strings)
    response = requests.post(oracle_url, data=input_data_json)
    return dict(response.json())

if __name__ == '__main__':
    oracle_url = 'http://172.26.26.101:5006/oracles/vina/drd2'
    #oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/drd2'
    input_strings = ["CCO"]
    print(score_with_oracle(input_strings,oracle_url))
    # oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/ache'
    # print(score_with_oracle(input_strings,oracle_url))
    # oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/mk2'
    # print(score_with_oracle(input_strings,oracle_url))
    # oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/drd2'
    # print(score_with_oracle(input_strings,oracle_url))
    # oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/gsk3b'
    # print(score_with_oracle(input_strings,oracle_url))
    # input_strings = "CCCCCCC"
    # oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/jnk3'
    # print(score_with_oracle(input_strings,oracle_url))
