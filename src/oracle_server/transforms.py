import numpy as np
from typing import List
import math
from scipy.interpolate import interp1d


def reverse_sigmoid_transformation(low, predictions: list) -> np.array:
    _low = low
    _high = -1
    _k = 0.25
    def _reverse_sigmoid_formula(value, low, high, k) -> float:
        try:
            return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
    transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in predictions]
    return np.array(transformed, dtype=np.float32)


def _double_sigmoid_formula(value, low, high, coef_div=500., coef_si=250., coef_se=250.):
    A = 10 ** (coef_se * (value / coef_div))
    B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
    C = (10 ** (coef_si * (value / coef_div)) / (10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
    return (A / B) - C

def double_sigmoid(predictions: list) -> np.array:
    _low = 0.00
    _high = 500.00
    _coef_div = 500.00
    _coef_si = 250.00
    _coef_se = 250.00


    transformed = [_double_sigmoid_formula(pred_val, _low, _high, _coef_div, _coef_si, _coef_se) for pred_val in predictions]
    return np.array(transformed, dtype=np.float32)

def _calculate_pow(values, weight):
        y = [math.pow(value, weight) for value in values]
        return np.array(y, dtype=np.float32)

def compute_non_penalty_components(weights,values,smiles: List[str]):
        product = np.full(len(smiles), 1, dtype=np.float32)
        all_weights = sum(weights)

        for smvalue,weight in zip(values,weights):
            comp_pow = _calculate_pow(smvalue, weight/ all_weights)
            product = product * comp_pow

        return product







