import requests
import json


def score_with_oracle(input_strings,oracle_url):
    input_data_json = json.dumps(input_strings)
    response = requests.post(oracle_url, data=input_data_json)
    return dict(response.json())

if __name__ == '__main__':
    oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/drd2'

    input_strings = [
        'O=C1CC=C(c2ccc(N3CC(c4ccccc4)CC3=O)cc2)CN1',
        'Cc1cc(N2CCCC2)ccc1CNC(=O)c1cc2c(c(C)n1)CCCC2',
        'O=C(NC1C2CC3CC(C2)CC1C3)C1=CN=C(N2CCC3=CC=CC=C3C2)N=N1'
        ]
    print(score_with_oracle(input_strings,oracle_url))
    oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/ache'
    print(score_with_oracle(input_strings,oracle_url))
    oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/mk2'
    print(score_with_oracle(input_strings,oracle_url))
    oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/drd2'
    print(score_with_oracle(input_strings,oracle_url))
    oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/gsk3b'
    print(score_with_oracle(input_strings,oracle_url))
    input_strings = "CCCCCCC"
    oracle_url =  'http://ap.yerevann.com:5006/oracles/tdc/jnk3'
    print(score_with_oracle(input_strings,oracle_url))
    input_strings = ["CCO","CCCCN"]
    oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/drd2'
    print(score_with_oracle(input_strings,oracle_url))
