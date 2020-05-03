import numpy as np
import pandas as pd
from time import time

'''
# transcript_offer: dateset_from_persons

# set the target_dataset_list structure
target_dataset_list = pd.DataFrame(columns=['person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed','amount_with_offer','label_effective_offer'])

# def get_dateset_from_unique_person(transcript_offer):
    person_ids = transcript_offer.person.unique()
    for person_id in person_ids:
        unique_person = transcript_offer[transcript_offer['person']==person_id]
        transactions = unique_person[unique_person.event=='transaction']

        groupby_offer_id = unique_person.groupby(['offer_id'])
        offer_id_list = unique_person.offer_id.unique()

        get_dateset_from_unique_offer_id(groupby_offer_id, offer_id_list, transactions)

    return None
'''

class PreprocessingData:
    '''Module to preprocess data'''

    def __init__(self, target_dataset_list, transcript_offer):
        self.target_dataset_list = target_dataset_list
        self.transcript_offer = transcript_offer

    def get_dateset_from_unique_offer_id(self, groupby_offer_id, offer_id_list, transactions):
        '''
        DESCRIPTION:
        Based on unique_offer_id of unique_person update the following dataset:

        Update transactions with related offer_id in the original dataset transcript_offer.

        Update target_dataset_list - (list), the structure of target_dataset_list element is a dict with keys
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed','amount_with_offer','label_effective_offer'

        INPUT:
            groupby_offer_id - (pandas groupby object) groupby offer_id of get_dateset_from_unique_person
            offer_id_list - (list) offer_id list with unique values of unique person
            transactions - (DataFrame) all transactions of a unique person

        UTPUT: None
        '''
        start = time()

        valid_offer_id = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for offer_id in offer_id_list:
            # offer_id has 10 valid values, except -1 represent nan values of offer_id
            if offer_id not in valid_offer_id:
                continue
            unique_offer_id = groupby_offer_id.get_group(offer_id)
            df_units, units_count = self.cut_unique_offer_id_2_units(unique_offer_id)

            # informational offer_type without 'offer completed'
            if unique_offer_id.offer_type.unique()[0] == 'informational':
                self.get_dateset_from_informational_offer(df_units, units_count, transactions)
            else:
            #unique_offer_id.offer_type.unique()[0] in ['bogo', 'discount']:
                self.get_dateset_from_other_offer(df_units, units_count, transactions)
            #print("data for unique offer_id wrangled, time is {}" .format(time()-start))
        print("data for unique person wrangled, time is {}" .format(time()-start))



    def fill_offer_id_4_transaction(self, original_id, updated_id):
        '''Filling the offer_id for transactions, if they are related with an offer.
        Final result is '-1' or a list: [offer_id_1, offer_id_2,...]

        INPUT:
            original_id - (str) original is '-1';after updated is a str (because there will be more than one transaction related with an offer)
            updated_id - (int) represent the related offer_id

        OUTPUT:
            new_value - (str) updated offer_id infomation of transactions
        '''
        if original_id == '-1':  #obejct(astype('str')): str '-1'
            new_value = updated_id
        else:
            # there is already at least one valid transaction related with an offer
            new_value = original_id+','+updated_id
        return new_value



    def get_dateset_from_informational_offer(self, df_units, units_count, transactions):
        # target_dataset_list and transcript_offer are global variables.
        '''
        DESCRIPTION:
        For the offer_id == 2 | 7 (informational_offer)

        Update transactions with related offer_id in the original dataset transcript_offer.

        Update target_dataset_list - (list), the structure of target_dataset_list is list of dicts with following keys:
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed','amount_with_offer','label_effective_offer'

        INPUT:
            df_units - (list of DataFrame) transaction units from a unique_offer_id of unique_person
            units_count - (int) number of transaction units in df_units
            transactions - (DataFrame) 'event' is 'transaction' for a unique_person

        OUTPUT: None
        '''
        #!!!REMEMBER: it's already units, with 'offer received' or not
        #!!!REMEMBER: units_count can't be 0, since there is no nan in 'event'
        person   = df_units[0].person.unique()[0]  #.unique() returns a numpy.array
        offer_id = df_units[0].offer_id.unique()[0]
        # different offer_id has a different valid duration
        duration = df_units[0].duration.unique()[0]

        for i in range(units_count):
            df_unit = df_units[i]

            time_received = df_unit[df_unit.event=='offer received'].time.min()
            time_viewed = df_unit[df_unit.event=='offer viewed'].time.min()
            time_completed = df_unit[df_unit.event=='offer completed'].time.min()

            # init the transaction time
            # (after a valid transaction, the offer is finished, so there will be at most one transaction time)
            time_transaction = -1
            # init the amount related to an offer
            amount_with_offer = 0
            # init the label of effective_offer
            label_effective_offer = -1

            # FLAG of 'offer received'
            is_received = (df_unit[df_unit.event=='offer received'].shape[0]!=0)

            if is_received:
                # at least one transaction exist
                if transactions.shape[0] != 0:
                    transaction_time = np.array(transactions.time)
                    time_begin = time_received
                    time_end = time_received + duration

                    is_valid_duration = (transaction_time >= time_begin) & (transaction_time <= time_end)
                    valid_transactions = transactions[is_valid_duration]

                    # update the 1st transaction, get the label_effective_offer
                    if valid_transactions.shape[0] != 0:
                        # the 1st transaction is the valid transaction related with an offer
                        valid_transactions.head(1).loc[:, 'offer_id'] = offer_id
                        time_transaction = valid_transactions.head(1).time.min()

                        # get the data in original dataset transcript_offer, to update the offer_id of transaction with the related offer_id

                        valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]

                        # update the offer_id of transaction in transcript_offer
                        valid_transactions_2b_labeled['offer_id'] = valid_transactions_2b_labeled['offer_id'].apply(self.fill_offer_id_4_transaction, args=(offer_id,))
                        self.transcript_offer.update(valid_transactions_2b_labeled)

                        label_effective_offer = 1
                        amount_with_offer = valid_transactions.head(1).amount.sum()

                # received but without transaction
                else:
                    label_effective_offer = 0

            # update the target_dataset_list
            target_dict = {
                        "person":   person,
                        "offer_id": offer_id,
                        "time_received": time_received,
                        "time_viewed": time_viewed,
                        "time_transaction": time_transaction,
                        "time_completed": time_completed,
                        "amount_with_offer": amount_with_offer,
                        "label_effective_offer": label_effective_offer}
            self.target_dataset_list.append(target_dict)



    def get_dateset_from_other_offer(self, df_units, units_count, transactions):
        '''
        DESCRIPTION:
        For the offer_id != (2 & 7)
        (REMEMBER to exclude offer_id == -1)

        Update transactions with related offer_id in the original dataset transcript_offer.

        Update target_dataset_list - (list), the structure of target_dataset_list is list of dicts with following keys:
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed','amount_with_offer','label_effective_offer'

        INPUT:
            df_units - (list of DataFrame) transaction units from a unique_offer_id of unique_person
            units_count - (int) number of transaction units in df_units
            transactions - (DataFrame) 'event' is 'transaction' for a unique_person

        OUTPUT: None
        '''
        #!!!REMEMBER: it's already units, with 'offer received' or not
        #!!!REMEMBER: units_count can't be 0, since there is no nan in 'event'
        person   = df_units[0].person.unique()[0]  #.unique() returns a numpy.array
        offer_id = df_units[0].offer_id.unique()[0]
        # different offer_id has a different valid duration
        duration = df_units[0].duration.unique()[0]

        for i in range(units_count):
            df_unit = df_units[i]

            time_received = df_unit[df_unit.event=='offer received'].time.min()
            time_viewed = df_unit[df_unit.event=='offer viewed'].time.min()
            time_completed = df_unit[df_unit.event=='offer completed'].time.min()

            # init the transaction time with a empty list
            # (there will be more than one valid transaction time)
            time_transaction = ""
            # init the amount related to an offer
            amount_with_offer = 0
            # init the label of effective_offer
            label_effective_offer = -1

            # FLAG of 'offer received'
            is_received = (df_unit[df_unit.event=='offer received'].shape[0]!=0)
            # FLAG of 'offer completed'
            is_completed = (df_unit[df_unit.event=='offer completed'].shape[0]!=0)

            if is_received:
                if is_completed:
                    #REMEMBER: to be completed, there must be transaction(s)
                    transaction_time = np.array(transactions.time)

                    #valid transaction(s) exist between 'offer received' and 'offer completed'
                    is_valid_duration = (transaction_time >= time_received) & (transaction_time <= time_completed)
                    valid_transactions = transactions[is_valid_duration]

                    valid_transactions.loc[:, 'offer_id'] = offer_id

                    # get the index of valid transactions, to update offer_id of transactions in the original dataset transcript_offer
                    valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]
                    valid_transactions_2b_labeled['offer_id']=valid_transactions_2b_labeled['offer_id'].apply(self.fill_offer_id_4_transaction, args=(offer_id,))
                    self.transcript_offer.update(valid_transactions_2b_labeled)

                    # update the label of effective_offer
                    label_effective_offer = 1
                    amount_with_offer = valid_transactions.amount.sum()
                    # there may be more than one valid transaction
                    for time in valid_transactions.time.values.tolist():
                        time_transaction = time_transaction+','+str(time)

                else:
                    # without 'offer completed'
                    transaction_time = np.array(transactions.time)
                    time_begin = time_received
                    time_end = time_received + duration
                    # transaction(s) in valid duration should be regarded as 'tried transaction(s)'
                    is_valid_duration = (transaction_time >= time_begin) & (transaction_time <= time_end)
                    valid_transactions = transactions[is_valid_duration]

                    # transaction(s) in valid duration should be updated with the related offer_id in the dataset transcript_offer
                    valid_transactions.loc[:, 'offer_id'] = offer_id
                    valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]
                    valid_transactions_2b_labeled['offer_id']=valid_transactions_2b_labeled['offer_id'].apply(self.fill_offer_id_4_transaction, args=(offer_id,))
                    self.transcript_offer.update(valid_transactions_2b_labeled)

                    label_effective_offer = 0 # tried but not completed
                    amount_with_offer = valid_transactions.amount.sum()
                    for time in valid_transactions.time.values.tolist():
                        time_transaction = time_transaction+','+str(time)

            # update the target_dataset_list
            target_dict = {
                        "person":   person,
                        "offer_id": offer_id,
                        "time_received": time_received,
                        "time_viewed": time_viewed,
                        "time_transaction": time_transaction,
                        "time_completed": time_completed,
                        "amount_with_offer": amount_with_offer,
                        "label_effective_offer": label_effective_offer}
            self.target_dataset_list.append(target_dict)


    def cut_unique_offer_id_2_units(self, unique_offer_id):
        '''
        DESCRIPTION:
            The raw data is transcript of unique_offer_id in one unique_person. Since there may be more than one offer for this unique_offer_id. That's why we cut the transcript to independent pieces, and call it 'units'.

        INPUT:
            unique_offer_id - (DataFrame) transcript of unique_offer_id in one unique_person

        OUTPUT:
            df_units - (list of DataFrame) transaction units from a unique_offer_id of unique_person
            units_count - (int) number of transaction units in df_units
        '''
        events = unique_offer_id['event']
        index = events.index.values
        index_min = events.index.min()
        index_max = events.index.max()
        index_received = events[events=='offer received'].index

        df_units = []
        if len(index_received) == 0:
            units_count = 1
            df_unit = unique_offer_id
            df_units.append(df_unit)

        elif index_received[0] == index_min:
            units_count = len(index_received)
            #当units_count=1时？？
            if units_count == 1:
                df_unit = unique_offer_id
                df_units.append(df_unit)

            else:
                for i in range(units_count - 1):
                    df_unit = unique_offer_id[(index >= index_received[i]) & (index < index_received[i+1])]
                    df_units.append(df_unit)
                df_unit = unique_offer_id[(index >= index_received[i+1]) & (index <= index_max)]
                df_units.append(df_unit)

        else:
            units_count = len(index_received)+1
            df_unit = unique_offer_id[(index >= index_min) & (index < index_received[0])]
            df_units.append(df_unit)
            #当units_count=2时？？
            if units_count == 2:
                df_unit = unique_offer_id[(index >= index_received[0]) & (index <= index_max)]
                df_units.append(df_unit)
            for i in range(1, units_count - 1):
                df_unit = unique_offer_id[(index >= index_received[i]) & (index < index_received[i+1])]
                df_units.append(df_unit)
            df_unit = unique_offer_id[(index >= index_received[i+1]) & (index <= index_max)]
            df_units.append(df_unit)

        return df_units, units_count
