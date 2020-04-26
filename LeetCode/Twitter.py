class Tweet:

    def __init__(self, tweet_id, user_id):
        self.tweet_id = tweet_id
        self.user_id = user_id


class Twitter:
    """
    355. 设计推特
    :see https://leetcode-cn.com/problems/design-twitter/
    """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 推文列表
        self.tweet_list = []
        # 用户关注列表
        self.user_follower_dict = {}

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        self.tweet_list.append(Tweet(tweetId, userId))

    def getNewsFeed(self, userId: int) -> list:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        followee_set = self.user_follower_dict.get(userId, set())

        latest_tweet_list = []
        for tweet in self.tweet_list[::-1]:
            if tweet.user_id in followee_set or tweet.user_id == userId:
                latest_tweet_list.append(tweet.tweet_id)
                if len(latest_tweet_list) > 9:
                    break
        return latest_tweet_list

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId not in self.user_follower_dict:
            self.user_follower_dict[followerId] = set()
        self.user_follower_dict[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId in self.user_follower_dict and followeeId in self.user_follower_dict[followerId]:
            self.user_follower_dict[followerId].remove(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)

if __name__ == '__main__':
    obj = Twitter()
    # obj.postTweet(1, 5)
    # print(obj.getNewsFeed(1))
    # obj.follow(1, 2)
    # obj.postTweet(2, 6)
    # print(obj.getNewsFeed(1))
    obj.follow(1, 3)
    obj.unfollow(1, 2)
    # print(obj.getNewsFeed(1))
