import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from multiprocessing import Pool
import numpy as np
import warnings

warnings.filterwarnings('ignore')

train_first_day_to_pred = 'd_1886' 
test_first_day_to_pred = 'd_1914'

def preprocess_calendar(calendar_df):
    calendar_df['date'] = pd.to_datetime(calendar_df.date)
    calendar_df['yearly_day'] = calendar_df['date'].dt.dayofyear / 365.
    calendar_df['yearly_week'] = calendar_df['wm_yr_wk'].apply(lambda x: int(str(x)[-2:]) / 52)

    le = LabelEncoder()
    calendar_df = calendar_df.fillna('None')
    calendar_df['event'] = calendar_df.apply(lambda x: 1 if ((x.event_type_1 != 'None') or (x.event_type_2 != 'None')) else 0, axis=1)
    calendar_df['event_type_1'] = le.fit_transform(calendar_df.event_type_1)
    calendar_df['event_type_2'] = le.fit_transform(calendar_df.event_type_2)
    calendar_df['event_name_1'] = le.fit_transform(calendar_df.event_name_1)
    calendar_df['event_name_2'] = le.fit_transform(calendar_df.event_name_2)

    return calendar_df[['wm_yr_wk', 'd', 'yearly_day', 'month', 'year', 'yearly_week', 'wday', 'event_type_1', 
                     'event_name_1', 'event_type_2', 'event_name_2', 'event', 'snap_CA', 'snap_TX', 'snap_WI']]

def exp_smooth_forecast(sales_df, is_test):
    first_day_to_pred = test_first_day_to_pred if is_test else train_first_day_to_pred
    first_day_idx = sales_df.columns.get_loc(first_day_to_pred)
    history = 56
    def get_preds(x):
        train = x.iloc[first_day_idx - history: first_day_idx]
        train.index = pd.to_datetime(calendar[calendar['d'].isin(train.index)].date)
        train = pd.DataFrame({'date': train.index, 'sales': train.values})
        model = ExponentialSmoothing(np.asarray(train['sales']))
        model._index = pd.to_datetime(train.index.values).astype(np.int64) // 10**9
        fit1 = model.fit(smoothing_level=.1, smoothing_slope=.1)
        return fit1.forecast(28)
    
    preds = sales_df.apply(lambda x: get_preds(x), axis=1)
    preds = np.concatenate(preds.values).reshape(-1, 28)
    for i in range(28):
        sales_df[f'exp_smooth_d_{i+int(first_day_to_pred[-4:])}'] = preds[:, i]
    return sales_df

def enrich_sales(sales_df, calendar_df, is_test):
    le = LabelEncoder()
    sales_copy_df = sales_df.copy()
    calendar_copy_df = calendar_df.copy()

    sales_df['dept_id'] = le.fit_transform(sales_df.dept_id) 
    sales_df['cat_id'] = le.fit_transform(sales_df.cat_id)
    sales_df['state_id'] = le.fit_transform(sales_df.state_id)

    history = 28
    first_day_to_pred = test_first_day_to_pred if is_test else train_first_day_to_pred
    first_day_idx = sales_df.columns.get_loc(first_day_to_pred)

    sales_df['mean_sales'] = sales_copy_df.iloc[:, first_day_idx - history: first_day_idx].mean(axis=1)
    sales_df['max_sales'] = sales_copy_df.iloc[:, first_day_idx - history: first_day_idx].max(axis=1)
    sales_df['std_sales'] = sales_copy_df.iloc[:, first_day_idx - history: first_day_idx].std(axis=1)
    sales_df['total_sales'] = sales_copy_df.iloc[:, first_day_idx:].sum(axis=1)

    sales_df['weighted_sales'] = sales_df.total_sales / sales_df.total_sales.sum()
    training_series = sales_copy_df.iloc[:, 6: first_day_idx]
    n = training_series.shape[1]
    sales_df['mean_sum_gap'] = 1/(n-1) * np.sum(np.diff(training_series)**2, axis=1).reshape(-1, 1)
    
    calendar = calendar_copy_df.iloc[-(56 + 2*history):-56] if is_test else calendar_copy_df.iloc[-(84 + 2*history):-84]
    for i in range(7):
        wday = calendar[calendar['wday'] == i+1].d.tolist()
        sales_df[f'min_sales_wd{i+1}'] = sales_copy_df.iloc[:, 6: first_day_idx][wday].min(axis=1)
        sales_df[f'mean_sales_wd{i+1}'] = sales_copy_df.iloc[:, 6: first_day_idx][wday].mean(axis=1)
        sales_df[f'std_sales_wd{i+1}'] = sales_copy_df.iloc[:, 6: first_day_idx][wday].std(axis=1)
        sales_df[f'max_sales_wd{i+1}'] = sales_copy_df.iloc[:, 6: first_day_idx][wday].max(axis=1)
    return sales_df 

def add_matching_weeklyday_cols(sales_df):
    sales_df['min_sales_matching_wd'] = sales_df.apply(lambda x: x[f"""min_sales_wd{x['wday']}"""], axis=1)
    sales_df['mean_sales_matching_wd'] = sales_df.apply(lambda x: x[f"""mean_sales_wd{x['wday']}"""], axis=1)
    sales_df['std_sales_matching_wd'] = sales_df.apply(lambda x: x[f"""std_sales_wd{x['wday']}"""], axis=1)
    sales_df['max_sales_matching_wd'] = sales_df.apply(lambda x: x[f"""max_sales_wd{x['wday']}"""], axis=1)
    for i in range(7):
        sales_df = sales_df.drop([f'mean_sales_wd{i+1}', f'std_sales_wd{i+1}', f'max_sales_wd{i+1}', f'min_sales_wd{i+1}' ], axis=1)
    
    sales_df['exp_smooth_fcst'] = sales_df.apply(lambda x: x[f"""exp_smooth_{x['d']}"""], axis=1)
    #sales_df['diff_exp_smooth_fcst'] = sales_df.exp_smooth_fcst - sales_df.groupby(['store_id', 'dept_id', 'cat_id']).exp_smooth_fcst.transform('mean')
    cols = list(filter(lambda x: not x.startswith('exp_smooth_d_'), sales_df.columns.tolist()))
    return sales_df[cols]

def preprocess_sales(sales_df, calendar_df, is_test):
    if is_test:
        first_day_to_pred = int(str(sales_df.columns[-1]).split('_')[1]) + 1
        cols_to_add = [f'd_{i+first_day_to_pred}' for i in range(28)] 
        cols_dict = dict.fromkeys(cols_to_add, 0)
        sales_df = sales_df.assign(**cols_dict)
    
    value_vars = sales_df.columns[-28:].tolist()
    sales_df = exp_smooth_forecast(sales_df, is_test)
    sales_df = enrich_sales(sales_df, calendar_df, is_test)
    id_vars = list(filter(lambda x: not x.startswith('d_'), sales_df.columns.tolist()))
    sales_df =  pd.melt(sales_df, id_vars=id_vars, value_vars=value_vars, var_name='d', value_name='sales')
    sales_df = sales_df.merge(calendar_df, on='d', how='left')   
    return add_matching_weeklyday_cols(sales_df) 

#Basic because only keeps prices for last 28 days to predict
def merge_with_sell(merged_df, sell_df):
    merged_df = merged_df.merge(sell_df, on=['item_id', 'wm_yr_wk', 'store_id'], how='left')
    le = LabelEncoder()
    merged_df['store_id'] = le.fit_transform(merged_df.store_id)
    return merged_df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

#Basic preprocess because uses basic preprocess functions above
def basic_preprocess(calendar_df, sales_df, sell_df, is_test):
    calendar =  preprocess_calendar(calendar_df)
    merged = preprocess_sales(sales_df, calendar, is_test)
    preprocess_df = merge_with_sell(merged, sell)
    preprocess_df = reduce_mem_usage(preprocess_df)
    file_type = 'test' if is_test else 'train'
    preprocess_df.to_csv(f'../data/basic_{file_type}_preprocess.csv', index=False)

if __name__ == '__main__':
    calendar = pd.read_csv('../data/calendar.csv')
    sales = pd.read_csv('../data/sales_train_validation.csv')
    sell = pd.read_csv('../data/sell_prices.csv')
    test_calendar = calendar.copy()
    test_sales = sales.copy()
    test_sell = sell.copy()

    with Pool(2) as p:
        p.starmap(basic_preprocess, zip([calendar, test_calendar], [sales, test_sales], [sell, test_sell], [False, True]))
    """
    is_test = True
    basic_preprocess(test_calendar, test_sales, test_sell, is_test)
    is_test = False
    basic_preprocess(calendar, sales, sell, is_test)
    """
