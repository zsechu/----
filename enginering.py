# 数据预处理
import pandas as pd
import numpy as np
from tqdm import tqdm


def pre_process(sunshine_path,temp_path,wind_path):
    date = pd.date_range(start='2023-07-01 01:00:00',periods=7200,freq='H')
    radiation_data = pd.read_csv(sunshine_path).iloc[:7200]
    temp_data = pd.read_csv(temp_path)[:7200]
    wind_data = pd.read_csv(wind_path)[:7200]
    wind_data.loc[wind_data.Dir == 'Variable','Dir'] = np.nan
    all_data = temp_data.merge(wind_data,on=['Day','Hour'],how='left').merge(radiation_data,on=['Day','Hour'],how='left')
    all_data.insert(loc=0,column='date',value=date)


    all_data['Radiation'] = all_data['Radiation'].fillna(0) # Radiation 0填充
    # 缺失值前向填充避免数据穿越
    all_data['Temp'] = all_data['Temp'].fillna(method='ffill') 
    all_data['Spd'] = all_data['Spd'].fillna(method='ffill')  
    all_data['Dir'] = all_data['Dir'].fillna(method='ffill')
    all_data['Dir'] = all_data['Dir'].astype('float64')
    all_data['month'] = date.month
    
    # all_data['Dir'] = all_data['Dir'].apply(lambda x:np.float32(x) / 360.0) 
    all_data['sin_hour'] = np.sin(all_data['Hour'] / 24 * 2 * np.pi)
    all_data['cos_hour'] = np.cos(all_data['Hour'] / 24 * 2 * np.pi)

    all_data['sin_month'] = np.sin(all_data['month'] / 12 * 2 * np.pi)
    all_data['cos_month'] = np.cos(all_data['month'] / 12 * 2 * np.pi)

    # all_data['sin_Dir'] = np.sin(all_data['Dir'] / 360 * 2 * np.pi)
    # all_data['cos_Dir'] = np.cos(all_data['Dir'] / 360 * 2 * np.pi)
    # Spd和Temp的0值用第二小值替代
    all_data.loc[all_data['Spd'] == 0,'Spd'] = all_data["Spd"].value_counts().sort_index(ascending=True).index[1]
    all_data.loc[all_data['Temp'] == 0,'Temp'] = all_data["Temp"].value_counts().sort_index(ascending=True).index[1]
    # Radiation**0.5替代Radiation
    # all_data['Radiation'] = np.power(all_data['Radiation'], 0.5)
    return all_data.drop('Dir',axis=1)

# rolling特征
def fe_rolling_stat(data, time_col,time_varying_cols, window_size):
    """
    :param df: DataFrame原始数据
    :param time_varying_cols: time varying columns 需要rolling的列
    :param window_size: window size 窗口大小
    :return: DataFrame rolling
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col]},index=df[time_col].index)
    
    add_feas = []
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            for op in ['mean','std','median','max','min','kurt','skew']:
                name = f'{val}__{cur_ws}h__{op}'
                if op == 'mean':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).mean())
                    add_feas.append(name)
                if op == 'std':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).std())
                    add_feas.append(name)
                if op == 'median':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).median())
                    add_feas.append(name)
                if op == 'max':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).max())
                    add_feas.append(name)
                if op == 'min':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).min())
                    add_feas.append(name)
                if op == 'kurt' and cur_ws == 24:
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).kurt())
                    add_feas.append(name)
                if op == 'skew' and cur_ws == 24:
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).skew())
                    add_feas.append(name)
                # if op in ['0.05','0.25','0.75','0.95']:
                #     df[name] = df[val].transform(
                #         lambda x: x.rolling(window=cur_ws).quantile(float(op)))
                #     add_feas.append(name)
    return result.merge(df[[time_col,] + add_feas], on = [time_col], how = 'left')[add_feas]


def fe_lag(data, time_col,time_varying_cols, lags):
    """
    滞后特征
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for lag in tqdm(lags):
        for column in time_varying_cols:
            name = f'{column}_lag_{lag}'
            add_fetures.append(name)
            df[name] = df[column].shift(lag)
    
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

def fe_diff(data, time_col,time_varying_cols,diffs=[1,2,3]):
    """
    构造差分特征
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param diffs: 差分阶数(1,2,3)
    :return: DataFrame diff
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for i,diff in tqdm(enumerate(diffs)):
        for column in time_varying_cols:
            # 命名规则 列名_几小时_diff
            name = f'{column}_{diff}hour_diff'
            add_fetures.append(name)
            if i == 0: 
                df[name] = df[column] - df[column].shift(diff)
            else:
                df[name] = df[column].shift(diff-1) - df[column].shift(diff)

    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

def fe_div(data, time_col,time_varying_cols, divs=[1,2,3]):
    """
    构造比值差分特征
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param diff: 默认做1，2,3阶差分
    :return: DataFrame lag
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for i,div in tqdm(enumerate(divs)):
        for column in time_varying_cols:
            name = f'{column}_{div}hour_div'
            add_fetures.append(name)
            if i == 0: 
                df[name] = np.divide(df[column],df[column].shift(div))
            else:
                df[name] = np.divide(df[column].shift(div-1),df[column].shift(div))
        
    
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

def feature_combination(df_list):
    """
    特征融合
    :param df_list: DataFrame list
    :return: DataFrame
    """
    result = df_list[0]
    for df in tqdm(df_list[1:], total=len(df_list[1:])):
        if df is None or df.shape[0] == 0:
            continue

        assert (result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
        print(result.shape[0], df.shape[0])
    return result

if __name__ == '__main__':
    # 初始文件地址
    sunshine_path = '/home/zhangch/kaggle/competition/复赛_val/datasets/原始数据/sunshine.csv'
    temp_path = '/home/zhangch/kaggle/competition/复赛_val/datasets/原始数据/temp.csv'
    wind_path = '/home/zhangch/kaggle/competition/复赛_val/datasets/原始数据/wind.csv'


    df = pre_process(sunshine_path,temp_path,wind_path)
    df_rolling = fe_rolling_stat(
        data = df,
        time_col = 'date',
        time_varying_cols = ['Spd','Temp'],
        window_size = [3,6,24]
    )

    df_lag = fe_lag(
        data = df, 
        time_col = 'date', 
        time_varying_cols=['Spd','Temp'], 
        lags=[1,2,3,24]
    )

    df_diff = fe_diff(
        data = df,
        time_col = 'date',
        time_varying_cols = ['Spd','Temp'],
        diffs = [1,2,3]
    )

    df_div =fe_div(
        data = df,
        time_col = 'date',
        time_varying_cols = ['Spd','Temp'],
        divs = [1,2,3]
    )

    df_all = feature_combination([df,df_rolling, df_lag,df_diff,df_div])
    print(df_all.shape)

    rolling_diff_columns = ['Spd__3h__mean','Temp__3h__mean','Spd__6h__mean','Temp__6h__mean','Spd__24h__mean','Temp__24h__mean']
    df_rolling_diff = fe_diff(
        data = df_all,
        time_col = 'date',
        time_varying_cols = rolling_diff_columns,
        diffs = [1,2,3]
    )

    df_rolling_div = fe_div(
        data = df_all,
        time_col = 'date',
        time_varying_cols = rolling_diff_columns,
        divs = [1,2,3]
    )

    df_all = feature_combination([df_all, df_rolling_diff, df_rolling_div])
    print(df_all.shape)
    # use_columns = pd.read_csv('/home/tinky/天池/太阳辐射预测/复赛/data/processed_data/use_features.csv')['use_features']
    # df_all = df_all[use_columns].iloc[2*24:]
    # print(df_all.shape)
    df_all.to_csv('/home/zhangch/kaggle/competition/复赛_val/datasets/原始数据/data_1116_withna.csv',index=False)