import pandas as pd

from models import Tweet
from consts import BASE_URL, FILES


def get_tweets(dataset_name):
    with open(dataset_name) as f:
        contents = [x.strip() for x in f.readlines()]

    for line in contents:
        parts = line.split('\t')
        id = int(parts[0])
        if len(parts) == 3:
            subjects = parse_subjects(parts[1])
            tweettext = parts[2]
        elif len(parts) == 2:
            subjects = None
            tweettext = parts[1]
        else:
            raise ValueError(f"Error in line {line}")
        yield Tweet(id, subjects, tweettext)


def parse_subjects(part):
    subjects = part.split(';')
    subjects = [s for s in subjects if s != '']

    return subjects

def convert_txt_to_csv(filename, base_url='resources'):
    tweets = get_tweets(f'{base_url}/{filename}.txt')
    d = {'id': [], 'subjects': [], 'tweets': []}
    for tweet in tweets:
        d['id'].append(tweet.id)
        d['subjects'].append(tweet.subjects)
        d['tweets'].append(tweet.tweettext)

    df = pd.DataFrame(data=d)
    df.to_csv(f'{base_url}/{filename}.csv', index=False)



def convert_all():
    for f in FILES.values():
        convert_txt_to_csv(f, BASE_URL)


if __name__ == '__main__':
    convert_all()
