import streamlit as st
from plotly import graph_objs as go


def plot_raw_data(data, y_data, g_name, g_color, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data[y_data], name=g_name, line=dict(color=g_color)))

    fig.layout.update(xaxis_rangeslider_visible=True,
                      title_text=title)
    st.plotly_chart(fig)
