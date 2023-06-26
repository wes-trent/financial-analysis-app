# Financial Analysis App

Check out the app [HERE](https://financial-analysis-app-wes-trent.streamlit.app/). Running this cloud version may run into API call limits or other errors.

This is a repository for the Financial Analysis App. The app uses the Alpha Vantage API to fetch and visualize various types of financial data. It provides various ratios and charts for the specified U.S. equity. Simply type in the stock ticker (uppercase or lowercase), and hit enter (example: AAPL).

This is an early version, and includes fairly basic information. It is built for clarity and easy customization for the addition of new ratios, charts, and features.
As the code has gone through many different iterations, some parts of the code may be arbitrary or unoptimized. This will be improved over time.
Some stocks may not be available or may lead to errors due to missing data. 

The app is meant to provide information for educational and research purposes. It is not intended to provide financial advice. It's an early version and might contain issues, errors or limitations. It has been tested with several U.S. equities but not with all of them.

## Features

- Company Overview
- Financial Health Ratios
- Ratios Over Time
- Stock Price Chart
- Revenue vs Net Income
- Cash Flow Analysis

## API Limitations

The app uses the free version of the Alpha Vantage API which has a limit of 5 API requests per minute and 500 requests per day. Loading data for each equity symbol may take some time due to this limitation. The code is built to manage this limitation, but is not built to manage concurrent usage. For frequent use or testing, run a local
version with your own API key. Rerunning modified code will not cause additional API requests if the stock is unchanged, allowing for efficient testing.

You can get your own free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) which takes less than 10 seconds. Replace the existing API logic in the code with your own to overcome this.

## Getting Started

Feel free to test out a live version of this [app](https://financial-analysis-app-wes-trent.streamlit.app/). Note that this app is using a free API, and therefore
will run into API call limits if there are concurrent users.

It is recommended to run the app locally, allowing for the addition of any ratios or charts you want to add to your own version.
Clone this repository and navigate to the directory. Install the required Python libraries using the following command:

```
pip install -r requirements.txt
```

Ensure that you add your own API key when running locally. Note that running the Python script and then running the local Streamlit app 
may run into API usage limitations. Avoid running simultaneous versions of the app.

## Contributing

If you experience any bugs not previously mentioned or wish to request new features, please open an issue.

---

**Enjoy the app!**
