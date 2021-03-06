import streamlit as st
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('wheat_sarima.pkl', 'rb'))


def handle_start_date(selected_start_date):
    temp_series = pd.Series(pd.to_datetime(selected_start_date))
    date_string = str(temp_series.dt.year[0]) + "-" + str(temp_series.dt.month[0]) + "-" + '01'
    return date_string


def handle_end_date(selected_end_date):
    temp_series = pd.Series(pd.to_datetime(selected_end_date))
    month = str(temp_series.dt.month[0] + 1) if temp_series.dt.month[0] < 12 else '01'
    date_string = str(temp_series.dt.year[0]) + "-" + month  + "-" + '01'
    return date_string


def model_predict(selected_start_date):
    selected_start_date = pd.to_datetime(selected_start_date)
    month_start_date = handle_start_date(selected_start_date)
    next_month_start_date = handle_end_date(selected_start_date)
    dummy_df = pd.DataFrame({'forecast':[np.nan, np.nan]}, index=pd.to_datetime([month_start_date, next_month_start_date]))
    dummy_df['forecast'] = model.predict(start=dummy_df.index[0], dynamic=True)
    dummy_df = dummy_df.resample('D').interpolate()
    series = dummy_df.loc[selected_start_date]
    return_df = pd.DataFrame({'Predicted (Kilograms)': np.round(series.values)}, index=series.index)
    return return_df


def model_predict(selected_start_date, selected_end_date):
    selected_start_date = pd.to_datetime(selected_start_date)
    selected_end_date = pd.to_datetime(selected_end_date)
    month_start_date = handle_start_date(selected_start_date)
    next_month_start_date = handle_end_date(selected_end_date)
    dummy_df = pd.DataFrame({'forecast':[np.nan, np.nan]}, index=pd.to_datetime([month_start_date, next_month_start_date]))
    dummy_df['forecast'] = model.predict(start=dummy_df.index[0], end=dummy_df.index[1], dynamic=True)
    dummy_df = dummy_df.resample('D').interpolate()
    series = dummy_df.loc[selected_start_date:selected_end_date]
    return_df = pd.DataFrame({'Predicted (Kilograms)': np.round(series['forecast'])}, index=series.index)
    return return_df


def main():

    warehouse = ['Warehouse-1 (Haryana)', 'Warehouse-2 (Rajasthan)']
    crop = ['Wheat', 'Rice']
    date_range = ['Date', 'Date Range']

    st.title("Inventory Management using Demand Analysis")
    st.subheader("Enter the selection parameters:")

    selected_warehouse = st.selectbox("Select a warehouse:", warehouse, index=0)
    selected_rice = st.radio("Select a crop:", crop, index=-0)

    is_range = st.radio("Predict for a date or a date range?", date_range, index=0)
    if (is_range == 'Date'):
        selected_start_date = st.date_input("Select a date:")
    else:
        selected_start_date = st.date_input("Select a date range:", [])
        if len(selected_start_date) == 2:
            selected_start_date, selected_end_date = selected_start_date

    if(st.button("Submit")):
        st.subheader("Output:")
        if is_range == 'Date':
            st.write(model_predict(selected_start_date))
        else:
            st.write(model_predict(selected_start_date, selected_end_date))





if __name__=='__main__':
    main()