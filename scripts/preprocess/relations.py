"""
Module that creates structural relationships
between data points.
"""
import re

import pandas as pd


def create_text_relation(train_df, val_df, test_df, verbose=0):
    """
    Create a hub ID for each text message.
    """

    if verbose > 0:
        print('concatenating train, val, and test sets..')
    df = pd.concat([train_df, val_df, test_df])
    df = df[['com_id', 'text']]

    # assign each text an text id
    if verbose > 0:
        print('assigning each text an ID...')
    df['text'] = df['text'].fillna('')
    gf = df.groupby('text').size().reset_index()
    gf = gf[gf[0] > 1]
    gf = gf[gf['text'] != '']

    gf['text_id'] = range(len(gf))
    gf['text_id'] = gf['text_id'].astype(int)
    del gf[0]

    # merge messages and text ids
    if verbose > 0:
        print('merging text IDs with messages...')
    rf = df.merge(gf, on='text', how='left')
    rf = rf[~pd.isnull(rf['text_id'])]

    return rf


def create_user_relation(train_df, val_df, test_df, verbose=0):
    """
    Creates a hub ID for each user.
    """
    if verbose > 0:
        print('concatenating train, val, and test sets..')
    df = pd.concat([train_df, val_df, test_df])
    df = df[['com_id', 'user_id']]

    # filter out users with only 1 message
    if verbose > 0:
        print('filtering out users with only one message...')
    gf = df.groupby('user_id').size().reset_index()
    gf = gf[gf[0] > 1]
    rf = df[df['user_id'].isin(gf['user_id'])]
    return rf


def create_link_relation(train_df, val_df, test_df, sets=False, verbose=0):
    """
    Create a hub ID for each link or set of links in a message.
    """

    if verbose > 0:
        print('concatenating train, val, and test sets..')
    df = pd.concat([train_df, val_df, test_df])
    df = df[['com_id', 'text']]
    df['text'] = df['text'].fillna('')

    # link hubs
    if verbose > 0:
        print('extracting links...')

    all_links = []
    all_messages = []
    for row in df.itertuples(index=False):
        links = re.findall(r'(http[^\s]+)', row[1])

        if len(links) > 0:
            if sets:
                links = [' '.join(sorted(set(links)))]

            all_links += links
            all_messages += [row[0]] * len(links)

    # assign each link a link id
    if verbose > 0:
        print('assigning link ids...')
    link_df = pd.DataFrame(list(zip(all_links, all_messages)),
                           columns=['link', 'com_id'])
    unique_links = link_df.groupby('link').size().reset_index()
    del unique_links[0]
    unique_links['link_id'] = range(len(unique_links))

    # merge link ids and message ids
    if verbose > 0:
        print('merging link ids and message ids...')
    rf = link_df.merge(unique_links, on='link')
    rf = rf[~pd.isnull(link_df['link'])]
    rf = rf[link_df['link'] != ''].sort_values('link_id')
    rf = rf.drop_duplicates()

    # filter out links posted by only one user
    if verbose > 0:
        print('filtering out links posted once...')
    keep_link_ids = []
    for link_id, qf in rf.groupby('link_id'):
        if len(set(qf['com_id'])) > 1:
            keep_link_ids.append(link_id)

    rf = rf[rf['link_id'].isin(keep_link_ids)]
    return rf


def create_hashuser_relation(train_df, val_df, test_df, sets=False, verbose=0):
    """
    Create a hub ID for each hashtag-user combination or set of
    hashtags used by a user in a message.
    """
    if verbose > 0:
        print('concatenating train, val, and test sets..')
    df = pd.concat([train_df, val_df, test_df])
    df = df[['com_id', 'user_id', 'text']]
    df['text'] = df['text'].fillna('')

    # link hubs
    if verbose > 0:
        print('extracting user hashtags...')
    all_hashes = []
    all_users = []
    all_messages = []
    for i, (user_id, user_df) in enumerate(df.groupby('user_id')):

        for row in user_df.itertuples(index=False):
            hashes = re.findall(r'(#\w+)', row[2])

            if len(hashes) > 0:

                if sets:
                    hashes = [' '.join(sorted(set(hashes)))]

                all_hashes += hashes
                all_users += [user_id] * len(hashes)
                all_messages += [row[0]] * len(hashes)

    # assign each hashuser an ID
    if verbose > 0:
        print('assigning hashuser IDs...')
    hashuser_df = pd.DataFrame(list(zip(all_hashes, all_users, all_messages)),
                               columns=['hash', 'user_id', 'com_id'])
    unique_hashusers = hashuser_df.groupby(['hash', 'user_id']).size().reset_index()
    unique_hashusers = unique_hashusers[unique_hashusers[0] > 1]
    del unique_hashusers[0]
    unique_hashusers['hashuser_id'] = range(len(unique_hashusers))

    rf = hashuser_df.merge(unique_hashusers, on=['hash', 'user_id'], how='left')
    rf = rf[~pd.isnull(rf['hashuser_id'])]

    return rf
