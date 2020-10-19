import pandas as pd


def get_pandas_df(action_path, song_path):
    """
    data set information: https://tianchi.aliyun.com/competition/entrance/231531/information

    load Ali music DataFrame

    :param action_path: data set path of user action
    :param song_path: data set path of song information
    :returns
    action: user action DataFrame
    song: song information DataFrame
    """
    """
    
    :param path: the local path of DataSet
    :return data:
    """
    action_header = ['user_id',
                     'song_id',
                     'gmt_create',
                     'action_type',
                     'Ds']

    song_header = ['song_id',
                   'artist_id',
                   'publish_time',
                   'song_init_plays',
                   'language',
                   'gender']
    action = pd.read_csv(action_path, names=action_header, sep=',')
    song = pd.read_csv(song_path, names=song_header, sep=',')
    return action, song
