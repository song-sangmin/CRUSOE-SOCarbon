
import mod_loading as loader

import pandas as pd


def main():
    # Load data
    coreDS, coreINDEX = loader.import_data(type=['core'])