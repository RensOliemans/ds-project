import pandas as pd
import consts


class DataReader:

    def __init__(self):
        self.base_url = consts.BASE_URL
        self.files = consts.FILES
        self._setup()

    def _setup(self):
        for set, filename in self.files.items():
            df = pd.read_csv(f'{self.base_url}/{filename}.csv')
            setattr(self, set, df)



if __name__ == '__main__':
    d = DataReader()
    print(d)
    print(d.train)
    print(d.test)
    print(d.ground_truth)
    x = d.train.tweets
    y = d.train.subjects
    id = d.train.id

    print(x.shape)
    print(y.shape)
    print(id.shape)
