from plotdata import plot_raw_data
import streamlit as st
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd


st.set_page_config(page_title='Stock Predictor')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


@st.cache
def load_symbols():
    symbol_data = pd.read_csv('nasdaqlisted.csv')
    symbols = symbol_data['Symbol'].values.tolist()
    return symbols


# Taking inputs
st.title('Stock Forecast App')

symbols = load_symbols()
selected_stock = st.selectbox('Select stock for prediction', symbols)

n_years = st.slider('Years of prediction:', min_value=1,
                    max_value=4, step=1)
period = n_years * 365

# Loading data


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Data load successful')


data_visualization = st.beta_container()

with data_visualization:
    st.subheader('Raw Data')
    st.write(data.tail())

    # Plotting raw data
    st.subheader(f'Stock behaviour of {selected_stock}')
    plot_raw_data(data, 'Open', 'stock_open', '#008046', 'Opening price')
    plot_raw_data(data, 'Close', 'stock_close', '#b50009', 'Closing Price')

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data['Date'],
                low=data['Low'],
                high=data['High'],
                close=data['Close'],
                open=data['Open'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )
        ]
    )

    fig.layout.update(title="Candlestick chart")

    st.plotly_chart(fig)

data_forecast = st.beta_container()
with data_forecast:
    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={
        "Date": "ds",
        "Close": "y"
    })

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write("Forecast Data")
    figure_1 = plot_plotly(model, forecast)
    st.plotly_chart(figure_1)

    st.write("Forecast Components")
    figure_2 = model.plot_components(forecast)
    st.write(figure_2)
