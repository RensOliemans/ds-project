import sys

import pandas as pd
import consts


class DataReader:

    def __init__(self):
        self.base_url = consts.BASE_URL
        self.files = consts.FILES
        self._setup()

    def _setup(self):
        for set, filename in self.files.items():
            try:
                df = pd.read_csv(f'{self.base_url}/{filename}.csv')
            except FileNotFoundError as e:
                print(f"Could not find file {e}\n" +
                      "Call `python converter.py` first")
                sys.exit(1)

            setattr(self, set, df)
