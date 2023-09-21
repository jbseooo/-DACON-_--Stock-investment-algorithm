import pandas as pd
import prophet
from prophet import Prophet
from tqdm import tqdm

train=pd.read_csv("/content/train.csv")
train['일자']=pd.to_datetime(train['일자'], format='%Y%m%d')
unique_codes = train['종목코드'].unique()
len(unique_codes)
results_df = pd.DataFrame(columns=['종목코드', 'final_return'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()
results_df = pd.DataFrame(columns=['종목코드', 'final_return'])
# 각 종목코드에 대해서 모델 학습 및 추론 반복
for code in tqdm(unique_codes):

    # 학습 데이터 생성
    train_close=train[train['종목코드'] == code][['일자', '종가']].rename(columns ={ "종가":'y'})
    train_close['ds']=pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)


    m = Prophet()
    m.fit(train_close)
    future=m.make_future_dataframe(periods= 15)
    forecast=m.predict(future)
    forecast=forecast['yhat']
    final_return=(forecast.iloc[-1] - forecast.iloc[-15]) / forecast.iloc[-15]

    results_df=results_df.append({'종목코드': code, 'final_return': final_return}, ignore_index=True)


results_df['순위']=results_df['final_return'].rank(method='first').astype('int') # 각 순위를 중복없이 생성

sample_submission=pd.read_csv('/content/sample_submission.csv')


baseline_submission=sample_submission[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')


baseline_submission.to_csv('/content/baseline_submission_pro2.csv', index=False)