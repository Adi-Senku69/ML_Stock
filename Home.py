import streamlit as st
import backend as bk
import mpld3 as mp
import streamlit.components.v1 as components

global fig
flag = 0
st.set_page_config(layout="wide")

st.header("Stock Market Trend Analysis")

with st.form(key="ticker"):
    ticker = st.text_input("Enter a Company's Yahoo Finance Ticker\\"
                           "\nExample Tickers are infy.ns, adanient.ns, hdfcbank.bo")
    days = st.text_input("Enter Trend Analysis Duration(Max 30 days)")
    button = st.form_submit_button("Submit")
    st.write("Disclaimer: For mobile users please use in Landscape mode")
    new = ticker.split(".")
    ticker = new[0].upper()
    if button:
        # try:
        val = bk.news_data(ticker)
        fig_sentiment = bk.sentiment_analysis(val)
        fig_sentiment_html = mp.fig_to_html(fig_sentiment)
        stock_df = bk.stock_data()
        fig_stock = bk.analysis(stock_df, int(days))
        fig_stock_html = mp.fig_to_html(fig_stock)
        flag = 1

        # except:
        #     st.write("Enter a valid Ticker")

if flag:
    components.html(fig_sentiment_html, height=600, width=1000)
    components.html(fig_stock_html, height=800, width=2000)
