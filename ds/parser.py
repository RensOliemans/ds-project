import pandas as pd

from models import Tweet


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
    subjects = [s.split('/')[1] for s in subjects]

    return subjects

def tweets_to_dataframe(tweets):
    print(tweets)


def main():
    tweets = get_tweets('TweetsTrainset.txt')
    tweets_to_dataframe(tweets)


if __name__ == '__main__':
    main()
