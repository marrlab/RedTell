import pandas as pd

from configuration import RAW_NAN_REPRESENTATIONS, RAW_PINF_REPRESENTATIONS, RAW_NINF_REPRESENTATIONS, \
    NAN_REPRESENTATION, PINF_REPRESENTATION, NINF_REPRESENTATION


def generate_nan_replacement():
    return {nan_value: NAN_REPRESENTATION for nan_value in RAW_NAN_REPRESENTATIONS}


def generate_pinf_replacement():
    return {pinf_value: PINF_REPRESENTATION for pinf_value in RAW_PINF_REPRESENTATIONS}


def generate_ninf_replacement():
    return {ninf_value: NINF_REPRESENTATION for ninf_value in RAW_NINF_REPRESENTATIONS}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace all NaN representations by np.nan
    # Replace all infinity representations by np.inf
    replacement = generate_nan_replacement() | generate_pinf_replacement() | generate_ninf_replacement()
    df = df.replace(replacement)
    return df
