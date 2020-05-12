import numpy as np
import pandas as pd
from time import time


class PreprocessingData:
    '''Module to preprocess data'''

    def __init__(self, target_dataset_list, transcript_offer):
        self.target_dataset_list = target_dataset_list # append constructed data
        self.transcript_offer = transcript_offer # with transaction infos



    def construct_data_from_unique_offer_id(self, offer_id_Groupby, offer_id_list, transactions):
        '''
        DESCRIPTION:
        Based on unique_offer_id of unique_person to get the following dataset:

        1. Update transactions with related offer_id in the original dataset transcript_offer.

        2. Construct target_dataset_list, the structure of target_dataset_list element is a dict with keys:
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed',
            'amount_with_offer','label_effective_offer'

        INPUT:
        - offer_id_Groupby(pandas groupby object): groupby offer_id of get_dateset_from_unique_person
        - offer_id_list(list): offer_id list with unique values of unique person
        - transactions(DataFrame): all transactions of a unique person

        OUTPUT: None
        '''
        valid_offer_id = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for offer_id in offer_id_list:
            # offer_id has 10 valid values, except -1 represent NaN of offer_id

            # !!!Attention: when person haven't received any offer, will be ignored. (need to consider later independently)
            # But they may have transactions.
            if offer_id not in valid_offer_id:
                continue
            # one unique_offer_id
            unique_offer_id = offer_id_Groupby.get_group(offer_id)
            # get the transaction unit(s): that means all related transactions under a unique offer_id
            units_df_list, units_count = self.cut_unique_offer_id_2_units(unique_offer_id)

            #!!!REMEMBER: it's already units, with 'offer received' or not
            #!!!REMEMBER: units_count can't be 0, since there is no NaN in 'event'
            person = units_df_list[0].person.unique()[0]  #.unique() returns a numpy.array
            offer_id = units_df_list[0].offer_id.unique()[0]
            # different offer_id has a different valid duration
            duration = units_df_list[0].duration.unique()[0]

            # in each transaction unit, construct the target data
            for i in range(units_count):
                units_df = units_df_list[i]

                is_received = (units_df.event=='offer received')
                time_received = units_df[is_received].time.min()

                is_viewed = (units_df.event=='offer viewed')
                time_viewed = units_df[is_viewed].time.min()

                is_completed = (units_df.event=='offer completed')
                time_completed = units_df[is_completed].time.min()

                # FLAG of 'offer received' existing
                flag_received = (units_df[is_received].shape[0]!=0)

                # Situation 1: informational offer_type which without 'offer completed'
                if unique_offer_id.offer_type.unique()[0] == 'informational':
                    time_completed, time_transaction, amount_with_offer, label_effective_offer = self.construct_data_from_informational_offer(time_received, time_completed, duration, offer_id, flag_received, transactions)
                # Situation 2: other offer_type 'bogo', 'discount'
                else:
                    # FLAG of 'offer completed'
                    flag_completed = (units_df[is_completed].shape[0]!=0)

                    time_completed, time_transaction, amount_with_offer, label_effective_offer = self.construct_data_from_other_offer(time_received, time_completed, duration, offer_id, flag_received, flag_completed, transactions)

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



    def update_transaction_offer_id(self, original_id, updated_id):
        '''Update the offer_id for transactions, if they are related with an offer.
        Final result is:
            '-1': related with no offer
            or a list: [offer_id_1, offer_id_2,...] related with more than one offer

        INPUT:
        - original_id(str): original is '-1'. After updated is a str (because
            there will be more than one transaction related with an offer, set type to 'string')
        - updated_id(int): represent the related offer_id(s)

        OUTPUT:
        - new_value(str): updated offer_id(s) of transactions
        '''
        if original_id == '-1':  # str '-1'
            new_value = updated_id
        else:
            # there is already at least one valid transaction related with an offer
            new_value = original_id + ',' + updated_id

        return new_value



    def construct_data_from_informational_offer(self, time_received, time_completed, duration, offer_id, flag_received, transactions):
        # !!! Target_dataset_list and transcript_offer are global variables.
        '''
        DESCRIPTION:
        For the offer_id == (2 | 7) (informational offer)

        1. Update transactions with related offer_id in the original dataset transcript_offer.

        2. Construct target_dataset_list, the structure of target_dataset_list element is a dict with keys:
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed',
            'amount_with_offer','label_effective_offer'

        INPUT:
        - time_received(float): time when offer received, 'NaN' represents not      received
        - time_completed(float): time when offer completed, 'NaN' represents not completed
        - duration(float): the valid duration of an offer
        - offer_id(object of str): from '-1' to '9', '-1' represents no related info with offer
        - flag_received(bool): 1 if received
        - transactions(DataFrame): 'event' value is 'transaction' of a
        unique_person

        OUTPUT:
        - time_completed(float): updated time for offer compeleting
        - time_transaction(float): -1 is the initial value which means no related transaction, otherwise will be a float record the time of transaction
        - amount_with_offer(float): the transactions amount related with this offer
        - label_effective_offer(int)
        '''
        # init the transaction time
        # (after a valid transaction, the offer is finished, so there will be at most one transaction time)
        time_transaction = -1
        # init the amount related to an offer
        amount_with_offer = 0
        # init the label of effective_offer
        label_effective_offer = -1

        # received the offer
        if flag_received:
            # at least one transaction exists
            if transactions.shape[0] != 0:
                # if there is no transactions, we can't use `transactions.time`
                transaction_time = np.array(transactions.time)
                offer_begin = time_received
                offer_end = time_received + duration

                is_valid_duration = (transaction_time >= offer_begin) & (transaction_time <= offer_end)
                valid_transactions = transactions[is_valid_duration]

                # at least one valid transaction exists
                if valid_transactions.shape[0] != 0:
                    # update the 1st transaction, get the label_effective_offer
                    # the 1st transaction is the valid transaction related with the offer
                    valid_transactions.head(1).loc[:, 'offer_id'] = offer_id
                    time_transaction = valid_transactions.head(1).time.min()

                    # update time_completed with the 1st valid transaction
                    # !!!Attention: type of time_transaction is 'str'
                    time_completed = float(time_transaction)

                    # get the data in original transcript_offer dataset, to update the offer_id of transaction with the related offer_id
                    valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]

                    # update the offer_id of transaction in transcript_offer
                    valid_transactions_2b_labeled['offer_id'] = valid_transactions_2b_labeled['offer_id'].apply(self.update_transaction_offer_id, args=(offer_id,))
                    self.transcript_offer.update(valid_transactions_2b_labeled)

                    # update 'label' and 'amount'
                    label_effective_offer = 1
                    amount_with_offer = valid_transactions.head(1).amount.sum()

                # received, person has transactions, but has no valid transaction for this offer_id
                else:
                    label_effective_offer = 0
            # received but the person without any transactions
            else:
                label_effective_offer = 0

        return time_completed, time_transaction, amount_with_offer, label_effective_offer



    def construct_data_from_other_offer(self, time_received, time_completed, duration, offer_id, flag_received, flag_completed, transactions):
        '''
        DESCRIPTION:
        For the 'bogo' & 'discount' offers, offer_id is not 2 or 7. (REMEMBER to exclude offer_id == -1)

        1. Update transactions with related offer_id in the original dataset transcript_offer.

        2. Construct target_dataset_list, the structure of target_dataset_list element is a dict with keys:
            'person', 'offer_id', 'time_received', 'time_viewed', 'time_transaction','time_completed',
            'amount_with_offer','label_effective_offer'

        INPUT:
        - time_received(float): time when offer received, 'NaN' represents not      received
        - time_completed(float): time when offer completed, 'NaN' represents not completed
        - duration(float): the valid duration of an offer
        - offer_id(object of str): from '-1' to '9', '-1' represents no related info with offer
        - flag_received(bool): 1 if received
        - flag_completed(bool): 1 if completed
        - transactions(DataFrame): 'event' value is 'transaction' of a
        unique_person

        OUTPUT:
        - time_completed(float): updated time for offer compeleting
        - time_transaction(float): -1 is the initial value which means no related transaction, otherwise will be a float record the time of transaction
        - amount_with_offer(float): the transactions amount related with this offer
        - label_effective_offer(int)
        '''
        # init the transaction time with a empty list
        # (there will be more than one valid transaction time)
        time_transaction = ""
        # init the amount related to an offer
        amount_with_offer = 0
        # init the label of effective_offer
        label_effective_offer = -1

        # received the offer
        if flag_received:
            # at least one transaction exists
            if flag_completed:
                #REMEMBER: to be completed, there must be transaction(s) & amount >= difficulty
                transaction_time = np.array(transactions.time)

                #valid transaction(s) exist between 'offer received' and 'offer completed'
                is_valid_duration = (transaction_time >= time_received) & (transaction_time <= time_completed)
                valid_transactions = transactions[is_valid_duration]

                # update the info of 'offer_id' with related offer
                valid_transactions.loc[:, 'offer_id'] = offer_id

                # get the data in original transcript_offer dataset, to update the offer_id of transaction with the related offer_id
                valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]
                valid_transactions_2b_labeled['offer_id']=valid_transactions_2b_labeled['offer_id'].apply(self.update_transaction_offer_id, args=(offer_id,))
                self.transcript_offer.update(valid_transactions_2b_labeled)

                # update the label of effective_offer and valid 'amount'
                label_effective_offer = 1
                amount_with_offer = valid_transactions.amount.sum()
                # there may be more than one valid transaction
                for time in valid_transactions.time.values.tolist():
                    time_transaction = time_transaction+','+str(time)
            # without 'offer completed'
            else:
                transaction_time = np.array(transactions.time)
                offer_begin = time_received
                offer_end = time_received + duration
                # transaction(s) in valid duration should be regarded as 'tried transaction(s)'
                is_valid_duration = (transaction_time >= offer_begin) & (transaction_time <= offer_end)
                valid_transactions = transactions[is_valid_duration]

                # transaction(s) in valid duration should be updated with the related offer_id in the dataset transcript_offer
                valid_transactions.loc[:, 'offer_id'] = offer_id
                valid_transactions_2b_labeled = self.transcript_offer.loc[valid_transactions.index]
                valid_transactions_2b_labeled['offer_id']=valid_transactions_2b_labeled['offer_id'].apply(self.update_transaction_offer_id, args=(offer_id,))
                self.transcript_offer.update(valid_transactions_2b_labeled)

                label_effective_offer = 0 # tried but not completed
                amount_with_offer = valid_transactions.amount.sum()
                for time in valid_transactions.time.values.tolist():
                    time_transaction = time_transaction+','+str(time)

        return time_completed, time_transaction, amount_with_offer, label_effective_offer



    def cut_unique_offer_id_2_units(self, unique_offer_id):
        '''
        DESCRIPTION:
            The raw data is transcript of unique_offer_id in one unique_person. Since there may be more than one offer under this unique_offer_id. That's why we cut the transcript to independent pieces, and call it 'units'.

        INPUT:
            - unique_offer_id(DataFrame): transcript of unique_offer_id in one unique_person

        OUTPUT:
            - units_df_list(list of DataFrame): transaction units from a unique_offer_id of unique_person
            - units_count(int): number of transaction units in units_df_list
        '''
        # all the events under a unique_offer_id of a unique_person
        events = unique_offer_id['event']
        # the original transcript has already a time ascending order
        index = events.index.values
        index_min = events.index.min()
        index_max = events.index.max()
        # based on the 'received' event to cut to units
        index_received = events[events=='offer received'].index

        units_df_list = []
        # the person has only transactions without 'offer received'
        if len(index_received) == 0:
            units_count = 1
            # there are just transactions records
            units_df = unique_offer_id
            # only one DataFrame appended
            units_df_list.append(units_df)
        # transactions begin with 'offer received'
        elif index_received[0] == index_min:
            # the number of 'offer received' is the number of units
            units_count = len(index_received)
            # only one 'offer received'
            if units_count == 1:
                units_df = unique_offer_id
                units_df_list.append(units_df)
            # more than one 'offer received'
            else:
                for i in range(units_count - 1):
                    # cut the unit between two neighbor 'offer received'
                    units_df = unique_offer_id[(index >= index_received[i]) & (index < index_received[i+1])]
                    units_df_list.append(units_df)
                # the last unit
                units_df = unique_offer_id[(index >= index_received[i+1]) & (index <= index_max)]
                units_df_list.append(units_df)
        # transactions begin with another event, e.g. 'transaction'
        else:
            # the first unit until the 1st 'offer received'
            units_count = len(index_received)+1
            units_df = unique_offer_id[(index >= index_min) & (index < index_received[0])]
            units_df_list.append(units_df)
            # only one 'offer received'
            if units_count == 2:
                units_df = unique_offer_id[(index >= index_received[0]) & (index <= index_max)]
                units_df_list.append(units_df)
            # more than one 'offer received'
            for i in range(1, units_count - 1):
                # cut the unit between two neighbor 'offer received'
                units_df = unique_offer_id[(index >= index_received[i]) & (index < index_received[i+1])]
                units_df_list.append(units_df)
            # the last unit
            units_df = unique_offer_id[(index >= index_received[i+1]) & (index <= index_max)]
            units_df_list.append(units_df)

        return units_df_list, units_count
