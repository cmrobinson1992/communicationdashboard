def split_data(data, split_ratio = 0.8):
    
    '''
    ToDo: Split the dataset into train and test data using the split_ratio.
    Do not import any additional libraries for this task, doing so will cause the autograder to fail.
    For Autograder purposes, there is no need to shuffle the dataset.

    Input:
        data: dataframe containing the dataset. 
        split_ratio: desired ratio of the train and test splits.
        
    Output:
        train: train split of the data
        test: test split of the data
    '''
    nrows = data.shape[0]
    ntrain = int(nrows*split_ratio)
    train = data[0:ntrain]
    test = data[ntrain:]
    return train, test
#    raise NotImplementedError