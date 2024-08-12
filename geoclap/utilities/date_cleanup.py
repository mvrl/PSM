# This script contains few basic functions to cleanup date string to get month and year

from dateutil import parser

def clean_datetime_str(date):
    t = date.lower().strip()
    try:
        if "am" in t or "pm" in t:
            if "am" in t:
                pattern = "am"
            elif 'pm' in t:
                pattern = "pm"
            i0 = t.find(pattern)
            t1 = t[:i0+2]
            date = t1.strip()
        elif "gmt" in t:
            pattern = "gmt"
            i0 = t.find(pattern)
            t1 = t[:i0]
            date = t1.strip()
        else:
            date = t
    except:
        date = None
    return date

def get_clean_date(original_date_str):
    date_str = clean_datetime_str(original_date_str)
    try:
        date = parser.parse(date_str)
    except:
        # print("Failed for:", original_date_str)
        date = None
    return date

if __name__ == '__main__':
    import os
    import pandas as pd
    from tqdm import tqdm

    data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data"
    train_df = pd.read_csv(os.path.join(data_path,"train_metadata.csv")).sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(os.path.join(data_path,"val_metadata.csv")).sample(frac=1).reset_index(drop=True)
    test_df = pd.read_csv(os.path.join(data_path,"test_metadata.csv")).sample(frac=1).reset_index(drop=True)
    
    def clean_date_column(df):
        dates_original = list(df.date)
        clean_dates = []
        failed_count = 0
        for date in tqdm(dates_original):
            clean_date = get_clean_date(date)
            if clean_date == None:
                failed_count += 1
            clean_dates.append(clean_date)
        df['clean_date'] = clean_dates
        print("date cleanup failed for:",failed_count)
        return df
    
    print("for train")
    train_df = clean_date_column(train_df) #date cleanup failed for: 1263

    print("for val")
    val_df = clean_date_column(val_df)    #date cleanup failed for: 15

    print("for test")                     #date cleanup failed for: 38
    test_df = clean_date_column(test_df)

    # import code;code.interact(local=dict(globals(), **locals()));

## Date formates across sources:
# >>> df[df['source']=='freesound'].iloc[0]['date']
# 'June 26th, 2010'
# >>> df[df['source']=='iNat'].iloc[0]['date']
# '2021/10/31 3:11 PM EDT'
# >>> df[df['source']=='yfcc'].iloc[0]['date']
# '2011-07-26 12:46:19.0'
# >>> df[df['source']=='aporee'].iloc[0]['date']
# '1/29/09 10:28' ##This is 24-hour format        




