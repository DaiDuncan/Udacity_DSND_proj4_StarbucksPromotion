import json
import plotly
import pandas as pd

from datetime import datetime
from collections import defaultdict

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals
import joblib
import pickle

import os


app = Flask(__name__)
images_folder = os.path.join('static', 'images')  #没用上 static足矣

# load in model
# model = joblib.load("../models/classifier.pkl")
with open('../model_sent_target(GradientBoosting).pckl', 'rb') as f:
    model = pickle.load(f)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index(): # cool visuals

    # extract data needed for visuals
    # load in data
    segment_num = pd.Series(range(0, 12))  # 12 segments
    segment_count_list = pd.Series([5806, 6465, 118, 4404, 10017, 3573, 4990, 13295, 7962, 1769, 5018, 3089])  # see in Notebook '2_heuristic_exploration'

    columns = ['age', 'income', 'member_days', 'gender_F',
            'gender_M', 'gender_O', 'offer_0', 'offer_1', 'offer_2', 'offer_3', 'offer_4', 'offer_5', 'offer_6', 'offer_7', 'offer_8', 'offer_9', 'amount_with_offer', 'amount_total', 'offer_received_cnt', 'time_received', 'time_viewed']

    feature_importances = pd.DataFrame(model.steps[1][1].feature_importances_,
                     index = columns, columns=['importance']).sort_values('importance',ascending=False)

    graphs = [
        {
            'data': [
                Bar(
                    x=segment_num,
                    y=segment_count_list
                )
            ],

            'layout': {
                'title': 'Distribution of 12 Segments',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Each Generation includes income: low medium high",
                    'tickvals': [1, 4, 7, 10],
                    'ticktext': ['Millenials', 'Gen X', 'Boomer', 'Silent']
                }
            }
        },

        {
            'data': [
                Bar(
                    x=feature_importances.index.tolist(),
                    y=feature_importances['importance'],
                )
            ],

            'layout': {
                'title': 'Distribution of feature_importances in model',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Different Features"
                }
            }
        }
    ]




    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    # get two Inputs: one is 'message', another is 'genre'
    age = int(request.args.get('age', ''))
    income = float(request.args.get('income', ''))
    enroll_date = request.args.get('date', '')
    gender = request.args.get('select_gender', '')
    offer_id = request.args.get('select_offer', '')

    amount_with_offer = float(request.args.get('amount_offer', ''))
    amount_total = float(request.args.get('amount_total', ''))
    offer_received_cnt = int(request.args.get('offer_number', ''))
    time_received = float(request.args.get('received_time', ''))
    time_viewed = float(request.args.get('viewed_time', ''))


    year, month, day = enroll_date.split('-')
    member_date = datetime(int(year), int(month), int(day))
    # CHECK type of 'enroll_date': is str 'XXXX-XX-XX'
    member_days = (datetime(2019,1,1) - member_date).days

    # init gender and offer_id
    gender_dict = {'female': 0, 'male': 0, 'other':0}
    offer_id_dict = defaultdict(int)
    for idx in range(0, 10):
        key_name = 'offer{}'.format(idx)
        offer_id_dict[key_name] = 0
    # update based on the web input value
    gender_dict[gender] = 1
    offer_id_dict[offer_id] = 1

    model_input = []
    model_input.extend([age, income, member_days])
    model_input.extend(gender_dict.values())  # inplace func, return None
    model_input.extend(offer_id_dict.values())
    other_features = [amount_with_offer, amount_total, offer_received_cnt, time_received, time_viewed]
    model_input.extend(other_features)

    # adjust to DataFrame: one row
    # Alternative: .predict(np.array(model_input).reshape(1, -1))[0]
    input_df = pd.DataFrame([model_input]) # with [] form one row, without is one col

    # use model to predict classification: 0 or 1
    # model input: 21 features as inputs
    pred_label = model.predict(input_df)[0] # output is an array [0] is the first value
    pred_prob = model.predict_proba(input_df)[0][1]

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query_age=age,
        query_income=income,
        query_enroll=enroll_date,
        query_gender=gender,
        query_offer_id=offer_id,
        query_amount_offer=amount_with_offer,
        query_amount_total=amount_total,
        query_offer_cnt=offer_received_cnt,
        query_time_received=time_received,
        query_time_viewed=time_viewed,
        pred_label= pred_label,
        pred_prob= round(pred_prob * 100, 2)
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
