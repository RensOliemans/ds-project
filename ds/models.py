class Tweet:
    def __init__(self, id, subjects, tweettext):
        self.id = id
        self.subjects = subjects
        self.tweettext = tweettext

    def __str__(self):
        return f"id: {self.id}, s: {self.subjects}, t: {self.tweettext}"

    def __repr__(self):
        return f"Tweet: {str(self)}"
