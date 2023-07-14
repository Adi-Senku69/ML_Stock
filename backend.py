import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import nltk
import math
import seaborn as sns
import warnings
import re
from sklearn import metrics
from datetime import date
from jugaad_data.nse import stock_df
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from scipy.fftpack import fft, ifft
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR

def init():
    global svr
    global year
    global month
    global day
    now = dt.datetime.now()
    now = now.strftime("%Y/%m/%d")
    now = now.split("/")
    now = [int(i) for i in now]
    year = now[0]
    month = now[1]
    day = now[2]
    warnings.filterwarnings('ignore')
    plt.style.use('fivethirtyeight')

    # Dictionary required for Sentiment Analysis
    nltk.download('vader_lexicon')
    svr = SVR(kernel='linear')

def configuration_func():
    init()
    global now
    global yesterday
    global user_agent
    global configuration
    now = dt.date.today()
    now = now.strftime('%m-%d-%Y')
    yesterday = dt.date.today() - dt.timedelta(days=1)
    yesterday = yesterday.strftime('%m-%d-%Y')

    nltk.download('punkt')
    user_agent = 'Mozilla/3.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Brave/1.27.109'
    configuration = Config()
    configuration.browser_user_agent = user_agent
    configuration.request_timeout = 10

def news_data(company_name_user):
    # As long as the company name is valid, not empty...
    configuration_func()
    global company_name
    company_name = company_name_user
    if company_name != '':
        print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')
        # Extract News with Google News
        googlenews = GoogleNews(period="24mo")
        googlenews.search(company_name)
        result = googlenews.result()
        # store the results
        df = pd.DataFrame(result)

        try:
            list = []  # creating an empty list
            for i in df.index:
                dict = {}  # creating an empty dictionary to append an article in every single iteration
                article = Article(df['link'][i], config=configuration)  # providing the link
                try:
                    article.download()  # downloading the article
                    article.parse()  # parsing the article
                    article.nlp()  # performing natural language processing (nlp)
                except:
                    pass
                    # storing results in our empty dictionary
                dict['Date'] = df['date'][i]
                dict['Media'] = df['media'][i]
                dict['Title'] = article.title
                dict['Article'] = article.text
                dict['Summary'] = article.summary
                dict['Key_words'] = article.keywords
                list.append(dict)
            check_empty = not any(list)
            # print(check_empty)
            if check_empty == False:
                news_df = pd.DataFrame(list)  # creating dataframe

        except Exception as e:
            # exception handling
            print("exception occurred:" + str(e))
            print(
                'Looks like, there is some error in retrieving the data, Please try again or try with a different '
                'ticker.')

        news_list = []
        for i in news_df['Title']:
            news_list.append(i)
        res = [ord(ele) for sub in news_list for ele in sub]
        ascii_df = pd.DataFrame(res)
        transformed_val = iffTransforms(ascii_df)
        return transformed_val
def sentiment_analysis(transformed_val):
    # Sentiment Analysis

    # Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    # Creating empty lists
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    # Iterating over the tweets in the dataframe
    for news in transformed_val:
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']

        if neg > pos:
            negative += 1.45  # increasing the count by 1
        elif pos > neg:
            positive += 1.45  # increasing the count by 1
        elif pos == neg:
            neutral += 0.45  # increasing the count by 1
            negative += 0.3

    whole = pos + neg + neu
    positive = percentage(pos, whole)  # percentage is the function defined above
    negative = percentage(neg, whole)
    neutral = 100 - (positive + negative)

    # Creating PieChart
    fig = plt.figure()
    labels = ['Positive [' + str(round(positive)) + '%]', 'Neutral [' + str(round(neutral)) + '%]',
              'Negative [' + str(round(negative)) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue', 'red']
    plt.pie(sizes, colors=colors, startangle=90, labels=labels)
    plt.style.use('default')
    plt.legend()
    plt.title("Sentiment Analysis Result for stock= " + company_name + "")
    plt.axis('equal')
    return fig

def iffTransforms(df_ascii):
    '''
       The code performs fft on the data , performing ifft at the same time
       Conversion of ifft to ascii , ascii to sentences is also performed
    '''
    result_sentences = []
    #for index, row in df_ascii.iterrows():
    #Performing ifft on the data obtained fft
    data_result_ifft = ifft(fft(df_ascii))
    #print(data_result_ifft)

    #Converting the ifft value back to sentences i.e ascii to sentences
    reconstructed_sentence = ''.join([chr(int(np.round(value.real))) for value in data_result_ifft])

    #Adding the sentences to a list
    result_sentences.append(reconstructed_sentence)

    return result_sentences

def percentage(part, whole):
    return 100 * float(part) / float(whole)

def stock_data():
    # define the ticker you will use
    df_stock = stock_df(symbol=company_name, from_date=date(2020, 1, 1),
                  to_date=date(year, month, day), series="EQ")

    return df_stock

def analysis(df_stock, days):
    df_stock['DATE'] = df_stock['DATE'].dt.strftime("%Y-%m-%d")
    x = df_stock[['OPEN', 'LOW']]
    y = df_stock['CLOSE']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=25, shuffle=False)
    svr.fit(X_train, Y_train)
    predicted_svr = svr.predict(X_test)
    dfr = pd.DataFrame({"Actual_Price": Y_test, "Predicted_Price": predicted_svr})
    dfr = dfr.tail(days)
    dfr['date'] = df_stock.tail(days).DATE
    fig_stock = plt.figure()
    plt.plot(dfr.date, dfr.Predicted_Price, color='red', label='Predicted Price')
    plt.title(f"{company_name} prediction chart")
    plt.xlabel("days")
    plt.ylabel("Amount")
    plt.grid()
    plt.legend()
    return fig_stock