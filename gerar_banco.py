import panda as pd

link = 'https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv'
data = pd.read_csv(link)
data['RM'] = data['RM'].astype(int)
data = data.drop(columns=['TOWN', 'TRACT', 'LON', 'LAT', 'ZN', 'AGE', 'RAD', 'DIS', 'TAX'])

data.to_csv('data.csv', index=False)
