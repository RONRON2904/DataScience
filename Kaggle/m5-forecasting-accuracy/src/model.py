import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.utils import shuffle

def wrmsse(pivoted_test_df, pivoted_preds_df):
    preds = np.transpose(pivoted_preds_df.iloc[:, -28:])
    targets = np.transpose(pivoted_test_df.iloc[:, -28:])
    weights = pivoted_test_df.weighted_sales.values.reshape(1, -1)
    return np.mean(mse(preds, targets, multioutput='raw_values'))
    #print('MSE: ', np.mean(mse(preds, targets, multioutput='raw_values')))
    #return np.sum(weights * (np.sqrt(mse(preds, targets, multioutput='raw_values') / pivoted_test_df['mean_sum_gap'].values)))

def get_train_test_dfs(preprocess_df, test_size=0.3):
    preprocess_df['diff_std_sell_price'] = preprocess_df.sell_price - preprocess_df.groupby(['store_id', 'cat_id', 'state_id', 'dept_id']).sell_price.transform('std')
    preprocess_df = preprocess_df.drop(['cat_id', 'state_id', 'month', 'year', 'yearly_week', 'event_type_1', 
                                        'event_name_1', 'event_type_2', 'event_name_2', 'max_sales_matching_wd',
                                        'min_sales_matching_wd'], axis=1)
    train_ids, test_ids = train_test_split(preprocess_df.id.unique().tolist(), test_size=test_size)
    train_df = preprocess_df[preprocess_df['id'].isin(train_ids)]
    test_df = preprocess_df[preprocess_df['id'].isin(test_ids)] 
    return train_df, test_df, train_df.sales

def get_trained_model(train_df, train_targets):
    lgb = LGBMRegressor(n_estimators=1400, learning_rate=0.1, max_depth=6, num_leaves=81, max_bin=100)
    model = lgb.fit(train_df, train_targets, eval_metric='mse')
    model.booster_.save_model("basic_model.txt")
    feature_importance = np.array(model.feature_importances_)
    cols = np.array(train_df.columns)
    print(np.column_stack((cols, feature_importance)))
    return model

def train_and_predict(preprocess_df):
    train_df, test_df, train_targets = get_train_test_dfs(preprocess_df)
    train_df = train_df.drop(['id', 'item_id', 'wm_yr_wk', 'sales', 'd', 'weighted_sales', 'total_sales'], axis=1)
    test_to_pred = test_df.drop(['id', 'item_id', 'wm_yr_wk', 'sales', 'd', 'weighted_sales', 'total_sales'], axis=1)   
    
    model = get_trained_model(train_df, train_targets)
    
    preds = model.predict(test_to_pred)
    preds[preds < 0] = 0
    test_df['preds'] = preds
    
    pivoted_preds_df = pd.DataFrame(test_df.pivot_table(index=['id', 'mean_sum_gap', 'weighted_sales'], columns='d', values='preds').to_records())
    pivoted_test_df = pd.DataFrame(test_df.pivot_table(index=['id', 'mean_sum_gap', 'weighted_sales'], columns='d', values='sales').to_records())
    
    pivoted_preds_df['category'] = pivoted_preds_df.id.apply(lambda x: x.split('_')[0])
    pivoted_test_df['category'] = pivoted_test_df.id.apply(lambda x: x.split('_')[0])

    foods_preds = pivoted_preds_df[pivoted_preds_df['category'] == 'FOODS']
    household_preds = pivoted_preds_df[pivoted_preds_df['category'] == 'HOUSEHOLD']
    hobbies_preds = pivoted_preds_df[pivoted_preds_df['category'] == 'HOBBIES']

    foods_test = pivoted_test_df[pivoted_test_df['category'] == 'FOODS']
    household_test = pivoted_test_df[pivoted_test_df['category'] == 'HOUSEHOLD']
    hobbies_test = pivoted_test_df[pivoted_test_df['category'] == 'HOBBIES']

    print('\nWRMSSE FOODS: ', wrmsse(foods_test.drop('category', axis=1), foods_preds.drop('category', axis=1)))
    print('\nWRMSSE HOUSEHOLD: ', wrmsse(household_test.drop('category', axis=1), household_preds.drop('category', axis=1)))
    print('\nWRMSSE HOBBIES: ', wrmsse(hobbies_test.drop('category', axis=1), hobbies_preds.drop('category', axis=1)))

    print('\nWRMSSE: ', wrmsse(pivoted_test_df.drop('category', axis=1), pivoted_preds_df.drop('category', axis=1)))

if __name__ == '__main__':
    preprocess_df = pd.read_csv('../data/basic_train_preprocess.csv')
    train_and_predict(preprocess_df)