import pandas as pd
from sklearn.preprocessing import StandardScaler

class RFMAnalysis:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def calculate_rfm_values(self):
        # Recency
        recency = pd.DataFrame(index=self.dataset.index)
        recency['Recency'] = 10  # arbitrary recency value

        # Frequency
        frequency_columns = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']
        frequency = self.dataset[frequency_columns].sum(axis=1).to_frame('Frequency')

        # Monetary
        monetary_columns = ['PURCHASES', 'PAYMENTS']
        monetary = self.dataset[monetary_columns].sum(axis=1).to_frame('Monetary')

        # Merge dataframes
        rfm = pd.concat([recency, frequency, monetary], axis=1)

        # Calculate ranks
        rfm['R_rank'] = rfm['Recency'].rank(ascending=False)
        rfm['F_rank'] = rfm['Frequency'].rank(ascending=True)
        rfm['M_rank'] = rfm['Monetary'].rank(ascending=True)
        
        # Normalize ranks
        rfm['R_rank_norm'] = (rfm['R_rank']) / (rfm['R_rank'].max()) * 100
        rfm['F_rank_norm'] = (rfm['F_rank']) / (rfm['F_rank'].max()) * 100
        rfm['M_rank_norm'] = (rfm['M_rank']) / (rfm['M_rank'].max()) * 100

        # Drop the original rank columns
        rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)


        # # Normalize ranks
        # scaler = StandardScaler()
        # rfm[['R_rank', 'F_rank', 'M_rank']] = scaler.fit_transform(rfm[['R_rank', 'F_rank', 'M_rank']])

        # Calculate RFM score
        # rfm['RFMScore'] = rfm[['R_rank', 'F_rank', 'M_rank']].sum(axis=1)
        rfm['RFMScore'] = rfm[['R_rank_norm', 'F_rank_norm', 'M_rank_norm']].mean(axis=1)
        

        return rfm
