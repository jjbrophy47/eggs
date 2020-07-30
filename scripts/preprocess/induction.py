"""
This module identifies and filters out the test
instances that have connections to training.
"""


def get_inductive_indices(relations, train_df, test_df):
    """
    Return test messages that have a connection to the training set.
    """
    test_ids = set()
    for relation, rf in relations.items():

        # filter out messages with no relational connections
        temp_train_df = train_df.merge(rf)
        temp_test_df = test_df.merge(rf)

        # find hubs that crossover from training to test
        train_hubs = temp_train_df['%s_id' % relation]
        test_hubs = temp_test_df['%s_id' % relation]
        hub_overlap = set(train_hubs).intersection(test_hubs)

        # keep test messages with connections in the training set
        test_ids.update(temp_test_df[temp_test_df['%s_id' % relation].isin(hub_overlap)]['com_id'])

    # filter out messages with connections to training
    new_test_df = test_df[~test_df['com_id'].isin(test_ids)]
    indices = new_test_df['com_id'].to_numpy()

    return indices
