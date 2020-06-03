import pandas as pd
import lightgbm
import numpy as np

def predict(model, test_preprocess_df):
    test_preprocess_df['diff_std_sell_price'] = test_preprocess_df.sell_price - test_preprocess_df.groupby(['store_id', 'cat_id', 'state_id', 'dept_id']).sell_price.transform('std')
    test_preprocess_df = test_preprocess_df.drop(['cat_id', 'state_id', 'month', 'year', 'yearly_week', 'event_type_1', 
                                        'event_name_1', 'event_type_2', 'event_name_2', 'max_sales_matching_wd',
                                        'min_sales_matching_wd'], axis=1)

    df_to_pred = test_preprocess_df.drop(['id', 'item_id', 'wm_yr_wk', 'sales', 'd', 'weighted_sales', 'total_sales'], axis=1)
    preds = model.predict(df_to_pred)
    preds[preds < 0] = 0
    test_preprocess_df['sales_pred'] = preds #np.rint(preds).astype(int)
    submission_val = test_preprocess_df[['id', 'd', 'sales_pred']].pivot(index='id', columns='d', values='sales_pred')
    submission_val = pd.DataFrame(submission_val.to_records())
    submission_eval = submission_val.copy()
    submission_eval.id = submission_eval.id.apply(lambda x: x.replace('validation', 'evaluation'))
    cols = ['id'] + [f'F{i+1}' for i in range(28)]
    submission_df = pd.concat([submission_val, submission_eval])
    submission_df.columns = cols
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    model = lightgbm.Booster(model_file='basic_model.txt')
    test_preprocess_df = pd.read_csv('../data/basic_test_preprocess.csv')
    predict(model, test_preprocess_df)