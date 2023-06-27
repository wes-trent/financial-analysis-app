import numpy as np
import streamlit as st
from alpha_vantage.fundamentaldata import FundamentalData
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import yfinance as yf
import os
import requests

# Add a title to the Streamlit app
st.title("Financial Analysis App")

if 'ALPHA_VANTAGE_API_KEY' in os.environ:
    # Running locally, fetch from environment variables
    API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
elif 'api' in st.secrets:
    # Running on Streamlit, fetch from secrets
    API_KEY = st.secrets["api"]["key"]
else:
    raise Exception("API key not found")

@st.cache_resource
def get_fundamental_data(symbol):
    fundamental_data = FundamentalData(key=API_KEY)

    # Fetch annual and quarterly balance sheet, income statement, and cash flow statement data
    data = []
    for method in [fundamental_data.get_balance_sheet_annual, fundamental_data.get_income_statement_annual,
                   fundamental_data.get_cash_flow_annual,
                   fundamental_data.get_balance_sheet_quarterly, fundamental_data.get_income_statement_quarterly,
                   fundamental_data.get_cash_flow_quarterly]:
        try:
            data_, _ = method(symbol)
            data.append(data_)
            time.sleep(12)
        except ValueError:
            raise ValueError(f"No data found for the symbol: {symbol}. Please try another symbol.")
    return data

@st.cache_resource
def get_stock_price_data(symbol):
    stock = yf.Ticker(symbol)
    try:
        data = stock.history(period="5y")['Close']
        return data
    except ValueError:
        raise ValueError(f"No price data found for the symbol: {symbol}. Please try another symbol.")

@st.cache_resource
def get_company_overview(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}'
    r = requests.get(url)
    time.sleep(12)
    data = r.json()
    # Convert the data to a DataFrame
    data_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    data_df.replace(['None', '-'], 0, inplace=True)
    return data_df


# Format numbers in billions
def format_billions(x):
    if abs(x) >= 1_000_000_000:
        return f'{"-" if x < 0 else ""}{abs(x) // 1_000_000_000:.0f}(B)'
    elif abs(x) >= 1_000_000:
        return f'{"-" if x < 0 else ""}{abs(x) // 1_000_000:.0f}(M)'
    elif abs(x) >= 1_000:
        return f'{"-" if x < 0 else ""}{abs(x) // 1_000:.0f}(K)'
    else:
        return str(int(x))

# User input for stock symbol
symbol = st.text_input("Enter stock symbol:")
if symbol:
    symbol = symbol.upper()

    try:
        with st.spinner("Fetching data..."):
            data = get_fundamental_data(symbol)
            stock_price_data = get_stock_price_data(symbol)
            company_overview_data = get_company_overview(symbol)

        if data is not None and len(data) == 6 and stock_price_data is not None:
            annual_balance_sheet_df, annual_income_statement_df, annual_cash_flow_statement_df, quarterly_balance_sheet_df, quarterly_income_statement_df, quarterly_cash_flow_statement_df = data

            # Fill null and none values with 0
            annual_balance_sheet_df.fillna(0, inplace=True)
            annual_balance_sheet_df.replace('None', 0, inplace=True)

            annual_income_statement_df.fillna(0, inplace=True)
            annual_income_statement_df.replace('None', 0, inplace=True)

            annual_cash_flow_statement_df.fillna(0, inplace=True)
            annual_cash_flow_statement_df.replace('None', 0, inplace=True)

            quarterly_balance_sheet_df.fillna(0, inplace=True)
            quarterly_balance_sheet_df.replace('None', 0, inplace=True)

            quarterly_income_statement_df.fillna(0, inplace=True)
            quarterly_income_statement_df.replace('None', 0, inplace=True)

            quarterly_cash_flow_statement_df.fillna(0, inplace=True)
            quarterly_cash_flow_statement_df.replace('None', 0, inplace=True)


            # START VISUAL ANALYSIS

            def display_company_overview(overview_data, latest_price, symbol):
                st.markdown(f"<h2 style='text-align: center; margin-bottom: 1px;'>Overview of {symbol}</h2>",
                            unsafe_allow_html=True)

                col1, spacer, col2 = st.columns([1, 0.1, 1])  # Create a spacer column with 10% of the total width

                # Column 1
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Most Recent Price: </b></p><p style="font-size:18px; font-weight:bold; color:white">${latest_price:.2f}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Analyst Target Price: </b></p><p style="font-size:18px; font-weight:bold; color:white">${float(overview_data["AnalystTargetPrice"]):.2f}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Market Capitalization: </b></p><p style="font-size:18px; font-weight:bold; color:white">{format_billions(float(overview_data["MarketCapitalization"]))}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Dividend Yield: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["DividendYield"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Profit Margin: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["ProfitMargin"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col1.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>PEG Ratio: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["PEGRatio"]):.2f}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)

                # Column 2
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Quarterly Revenue Growth YOY: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["QuarterlyRevenueGrowthYOY"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Quarterly Earnings Growth YOY: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["QuarterlyEarningsGrowthYOY"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Trailing PE: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["TrailingPE"]):.2f}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Forward PE: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["ForwardPE"]):.2f}</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Operating Margin TTM: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["OperatingMarginTTM"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top:18px;"><p style="font-size:17px; color: lightgrey"><b>Return On Equity TTM: </b></p><p style="font-size:18px; font-weight:bold; color:white">{float(overview_data["ReturnOnEquityTTM"]) * 100:.2f}%</p></div><hr style="border:1px solid #C0C0C0; margin:0;">',
                    unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Display company overview
            latest_price = stock_price_data[-1]  # Get the most recent price
            display_company_overview(company_overview_data, latest_price, symbol)


        # CALCULATE FINANCIAL RATIOS
        def get_dividend_per_share(symbol, API_KEY):
            fd = FundamentalData(API_KEY)
            data, _ = fd.get_company_overview(symbol)
            dividend_per_share = data.get('DividendPerShare')
            if dividend_per_share is not None:
                return float(dividend_per_share)
            else:
                return 0

        def get_annual_dividend_yield(symbol, API_KEY):
            dividend_per_share = get_dividend_per_share(symbol, API_KEY)
            stock_price_data = get_stock_price_data(symbol)
            latest_close_price = stock_price_data.iloc[-1]

            # Calculate and return the annual dividend yield
            return dividend_per_share / latest_close_price

        # Current Ratio
        current_assets = float(quarterly_balance_sheet_df['totalCurrentAssets'][0])
        current_liabilities = float(quarterly_balance_sheet_df['totalCurrentLiabilities'][0])
        current_ratio = current_assets / current_liabilities

        # Net Profit Margin
        if 'totalRevenue' in quarterly_income_statement_df.columns and 'netIncome' in quarterly_income_statement_df.columns:
            revenue = float(quarterly_income_statement_df['totalRevenue'][0])
            net_income_quarterly = float(quarterly_income_statement_df['netIncome'][0])
            net_profit_margin = net_income_quarterly / revenue if revenue else np.nan
        else:
            print("Columns 'totalRevenue' and/or 'netIncome' not found in quarterly_income_statement_df.")

        # Calculate Return on Equity (ROE)
        net_income_annual = float(annual_income_statement_df['netIncome'].values[0])
        shareholder_equity = float(annual_balance_sheet_df['totalShareholderEquity'].values[0])
        roe = net_income_annual / shareholder_equity

        # Calculate Debt Ratio
        total_debt = float(annual_balance_sheet_df['totalLiabilities'].values[0])
        total_assets = float(annual_balance_sheet_df['totalAssets'].values[0])
        debt_ratio = total_debt / total_assets

        # Operating Cash Flow Ratio
        if 'operatingCashflow' in quarterly_cash_flow_statement_df.columns and 'totalCurrentLiabilities' in quarterly_balance_sheet_df.columns:
            operating_cash_flow = float(quarterly_cash_flow_statement_df['operatingCashflow'][0])
            current_liabilities = float(quarterly_balance_sheet_df['totalCurrentLiabilities'][0])
            operating_cash_flow_ratio = operating_cash_flow / current_liabilities if current_liabilities else np.nan
        else:
            print(
                "Column 'operatingCashflow' and/or 'totalCurrentLiabilities' not found in their respective dataframes.")

        # Calculate P/E ratio
        net_income = float(annual_income_statement_df['netIncome'].values[0])
        outstanding_shares = float(annual_balance_sheet_df['commonStockSharesOutstanding'].values[0])
        eps = net_income / outstanding_shares
        pe_ratio = stock_price_data[-1] / eps

        # Calculate Debt
        short_term_debt = float(quarterly_balance_sheet_df.get('shortTermDebt', [0])[0]) if \
        quarterly_balance_sheet_df.get('shortTermDebt', ['None'])[0] != 'None' else 0
        long_term_debt = float(quarterly_balance_sheet_df.get('longTermDebt', [0])[0]) if \
        quarterly_balance_sheet_df.get('longTermDebt', ['None'])[0] != 'None' else 0
        total_debt = short_term_debt + long_term_debt

        # Calculate quick ratio
        current_assets = float(quarterly_balance_sheet_df['totalCurrentAssets'][0])
        inventory = float(quarterly_balance_sheet_df['inventory'][0])
        current_liabilities = float(quarterly_balance_sheet_df['totalCurrentLiabilities'][0])
        quick_ratio = (current_assets - inventory) / current_liabilities

        # Debt-to-Equity Ratio
        total_equity = float(quarterly_balance_sheet_df['totalShareholderEquity'][0])
        debt_to_equity_ratio = total_debt / total_equity

        # Fetch data
        annual_dividend_yield = get_annual_dividend_yield(symbol, API_KEY)
        eps = net_income / outstanding_shares


        #DATAFRAMES

        # Create a DataFrame for financial health ratios
        ratios_df = pd.DataFrame({
            'Ratios': ['Quick Ratio', 'Current Ratio', 'Debt Ratio', 'Debt-to-Equity Ratio'],
            'Values': [f"{quick_ratio:.2f}", f"{current_ratio:.2f}",
                       f"{debt_ratio:.2f}", f"{debt_to_equity_ratio:.2f}"]
        }).set_index('Ratios')

        # Display the DataFrame as a table
        st.subheader("Financial Health Ratios")
        st.table(ratios_df)

        # Create a DataFrame for financial performance ratios
        # ratios2_df = pd.DataFrame({
        #    'Ratios': ['Net Profit Margin', 'Operating Cash Flow Ratio',
        #               'Earnings Per Share', 'P/E Ratio',  'Annual Dividend Yield'],
        #    'Values': [f"{net_profit_margin:.2f}", f"{operating_cash_flow_ratio:.2f}",
        #               f"{eps:.2f}", f"{pe_ratio:.2f}", f"{annual_dividend_yield*100:.2f}%"]
        #}).set_index('Ratios')

        # Display the DataFrame as a table
        #st.subheader("Financial Performance Ratios")
        #st.table(ratios2_df)

        # Sort the data by fiscalDateEnding in ascending order
        annual_income_statement_df = annual_income_statement_df.sort_values('fiscalDateEnding', ascending=True)


        #DATAFRAME OVER TIME

        def calculate_ratios(annual_balance_sheet_df, annual_income_statement_df, stock_price_data):
            ratios_dict = {
                'Year': [],
                'Current Ratio': [],
                'Debt Ratio': [],
            }

            for i in range(-1, -6, -1):  # Iterate from -1 to -6 to get the last 5 years

                # Current Ratio
                current_assets = float(annual_balance_sheet_df['totalCurrentAssets'].values[i])
                current_liabilities = float(annual_balance_sheet_df['totalCurrentLiabilities'].values[i])
                current_ratio = current_assets / current_liabilities if current_liabilities else np.nan

                # Debt Ratio
                total_debt = float(annual_balance_sheet_df['totalLiabilities'].values[i])
                total_assets = float(annual_balance_sheet_df['totalAssets'].values[i])
                debt_ratio = total_debt / total_assets if total_assets else np.nan

                # Append data to dictionary
                year = pd.to_datetime(annual_balance_sheet_df['fiscalDateEnding'].values[i]).year
                ratios_dict['Year'].append(year)
                ratios_dict['Current Ratio'].append(current_ratio)
                ratios_dict['Debt Ratio'].append(debt_ratio)

            return pd.DataFrame(ratios_dict)

        # Dataframe
        ratios_df = calculate_ratios(annual_balance_sheet_df, annual_income_statement_df, stock_price_data)

        # Transpose the DataFrame so that 'Year' becomes the column header
        ratios_df = ratios_df.set_index('Year').T

        # Convert the year values to integers to remove decimal points
        ratios_df.columns = ratios_df.columns.astype(int)

        # Display the DataFrame with float data shown to two decimal places
        st.subheader("Ratios over 5 years")
        st.dataframe(ratios_df.style.format("{:.2f}"))


        #DuPont Analysis
        def calculate_dupont(annual_balance_sheet_df, annual_income_statement_df):
            dupont_list = []

            num_years = min(5, len(annual_balance_sheet_df), len(annual_income_statement_df))

            for i in range(-num_years, 0, 1):
                year = pd.to_datetime(annual_balance_sheet_df['fiscalDateEnding'].values[i]).year

                # Profit Margin
                if 'totalRevenue' in annual_income_statement_df.columns and 'netIncome' in annual_income_statement_df.columns:
                    revenue = float(annual_income_statement_df['totalRevenue'].values[i])
                    net_income = float(annual_income_statement_df['netIncome'].values[i])
                    profit_margin = net_income / revenue if revenue else np.nan
                    dupont_list.append(['Profit Margin', year, profit_margin])

                # Total Asset Turnover
                total_assets = float(annual_balance_sheet_df['totalAssets'].values[i])
                asset_turnover = revenue / total_assets if total_assets else np.nan
                dupont_list.append(['Asset Turnover', year, asset_turnover])

                # Financial Leverage
                shareholder_equity = float(annual_balance_sheet_df['totalShareholderEquity'].values[i])
                financial_leverage = total_assets / shareholder_equity if shareholder_equity else np.nan
                dupont_list.append(['Financial Leverage', year, financial_leverage])

                # ROE
                roe = profit_margin * asset_turnover * financial_leverage
                dupont_list.append(['Return on Equity (ROE)', year, roe])

            return pd.DataFrame(dupont_list, columns=['Ratio', 'Year', 'Value'])


        # Call the calculate_dupont function
        dupont_df = calculate_dupont(annual_balance_sheet_df, annual_income_statement_df)

        # Reshape the DataFrame
        dupont_df = dupont_df.pivot(index='Ratio', columns='Year', values='Value')

        # Display the DataFrame with float data shown to two decimal places.
        st.subheader("DuPont Analysis over 5 years")
        st.dataframe(dupont_df.style.format("{:.2f}"))


        # VISUALISATIONS

        # Function to convert large numbers into strings with 'B' or 'M' suffixes
        def billions(x, pos):
            'The two args are the value and tick position'
            if abs(x) >= 1_000_000_000:
                if abs(x) % 1_000_000_000 == 0:
                    return f'{"-" if x < 0 else ""}{abs(x) // 1_000_000_000:.0f}(B)'
                else:
                    return f'{"-" if x < 0 else ""}{abs(x) / 1_000_000_000:.2f}(B)'
            elif abs(x) >= 1_000_000:
                if abs(x) % 1_000_000 == 0:
                    return f'{"-" if x < 0 else ""}{abs(x) // 1_000_000:.0f}(M)'
                else:
                    return f'{"-" if x < 0 else ""}{abs(x) / 1_000_000:.2f}(M)'
            elif abs(x) >= 1_000:
                if abs(x) % 1_000 == 0:
                    return f'{"-" if x < 0 else ""}{abs(x) // 1_000:.0f}(K)'
                else:
                    return f'{"-" if x < 0 else ""}{abs(x) / 1_000:.2f}(K)'
            else:
                return str(x)

        formatter = FuncFormatter(billions)

        plt.style.use('ggplot')

        # Price over 5 Years Chart
        # Create a seaborn lineplot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=stock_price_data)
        plt.title(f'{symbol} Stock Price Over the Last 5 Years', fontsize=20, pad=15)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Stock Price ($)', fontsize=14)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()

        st.pyplot(plt.gcf())  # Display the plot in the Streamlit app

        # Data for Total Revenue plot
        x = pd.to_datetime(annual_income_statement_df['fiscalDateEnding'])
        y1 = annual_income_statement_df['totalRevenue'].astype(float)

        # Data for Net Income plot
        y2 = annual_income_statement_df['netIncome'].astype(float)

        # Create a function to annotate data points
        def annotate_data_points(ax, df, y_col):
            for x, y in zip(df['fiscalDateEnding'], df[y_col]):
                ax.text(x, y + 0.05 * y, format_billions(y), color='black', ha='center', va='bottom')


        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Convert fiscalDateEnding to datetime
        annual_income_statement_df['fiscalDateEnding'] = pd.to_datetime(annual_income_statement_df['fiscalDateEnding'])

        # Plot for Total Revenue
        sns.lineplot(x='fiscalDateEnding',
                     y='totalRevenue',
                     data=annual_income_statement_df.astype({'totalRevenue': 'float'}),
                     ax=ax, marker='o', color='blue', label='Total Revenue')
        annotate_data_points(ax, annual_income_statement_df.astype({'totalRevenue': 'float'}), 'totalRevenue')

        # Plot for Net Income
        sns.lineplot(x='fiscalDateEnding',
                     y='netIncome',
                     data=annual_income_statement_df.astype({'netIncome': 'float'}),
                     ax=ax, marker='o', color='orange', label='Net Income')
        annotate_data_points(ax, annual_income_statement_df.astype({'netIncome': 'float'}), 'netIncome')

        # Set plot details
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Year')
        ax.set_ylabel('Value ($)')
        ax.set_title('Total Annual Revenue and Net Income', fontsize=20, pad=12)
        ax.set_ylim(bottom=None)
        ax.tick_params(axis='x')

        # Add legend
        ax.legend()

        # Show the plot
        st.subheader("Total Revenue and Net Income Over 5 Years")
        st.pyplot(fig)


        #QUARTERLY BAR CHART-REVENUE VS INCOME

        def format_y_axis(y, _):
            return format_billions(y)

        formatter = FuncFormatter(format_y_axis)

        # Define a function to annotate the data points
        def annotate_data_points2(ax, x_points, y_points, color, y_offset, x_offset=0, fontsize=10):
            for x, y in zip(x_points, y_points):
                va = 'bottom' if y >= 0 else 'top'  # Vertical alignment
                y_offset = abs(y_offset)  # Make sure y_offset is positive
                y_offset = y_offset if y >= 0 else -y_offset  # If y is negative, make the offset negative
                ax.text(x + x_offset, float(y) + y_offset, format_billions(int(y)), color=color, ha='center',
                        va=va, fontsize=fontsize)


        # Sort the dataframe in descending order
        quarterly_income_statement_df = quarterly_income_statement_df.sort_index(ascending=False)

        # Fetch quarters
        quarters = pd.to_datetime(quarterly_income_statement_df['fiscalDateEnding'].values[-8:])

        # Create quarter labels
        quarter_labels = quarters.to_period('Q').strftime("Q%q '%y")

        # Fetching the revenue and net income data for the last 8 quarters
        revenue = quarterly_income_statement_df['totalRevenue'].values[-8:].astype(float)
        net_income = quarterly_income_statement_df['netIncome'].values[-8:].astype(float)

        # Create a DataFrame for seaborn
        df_revenue = pd.DataFrame({
            'Quarter': range(8),
            'Amount': revenue,
            'Type': 'Revenue'
        })

        df_net_income = pd.DataFrame({
            'Quarter': range(8),
            'Amount': net_income,
            'Type': 'Net Income'
        })

        # Combining both dataframes
        df_combined = pd.concat([df_revenue, df_net_income])

        # Assign 'Revenue', 'Net Income - Positive' or 'Net Income - Negative' based on the value
        df_combined['Type_Color'] = np.where(df_combined['Amount'] >= 0, df_combined['Type'],
                                             df_combined['Type'] + ' - Negative')

        plt.figure(figsize=(12, 8))

        # Create a seaborn barplot
        barplot = sns.barplot(x='Quarter', y='Amount', hue='Type', data=df_combined)

        # Set bar color based on value and hue
        colors = {'Revenue': ['blue', 'red'], 'Net Income': ['green', 'red']}
        hues = df_combined['Type'].values
        for bar, hue in zip(barplot.patches, hues):
            color_index = 0 if bar.get_height() >= 0 else 1
            bar.set_color(colors[hue][color_index])

        # Remove original legend
        barplot.get_legend().remove()

        legend_elements = [Patch(facecolor='blue', label='Revenue'),
                           Patch(facecolor='green', label='Net Income - Positive'),
                           Patch(facecolor='red', label='Net Income - Negative')]
        plt.legend(handles=legend_elements, title='Type', loc='upper left')

        # Adding title and labels
        plt.title(f'Revenue vs Net Income for last 8 Quarters for {symbol}', fontsize=20, pad=15)
        plt.ylabel('Amount ($)', fontsize=15)
        plt.xlabel('Quarter', fontsize=15)

        # Set y-axis formatter
        plt.gca().yaxis.set_major_formatter(formatter)

        # Annotate data points for Revenue
        annotate_data_points2(plt.gca(), np.arange(len(quarters)) - 0.2, df_revenue['Amount'].values, 'black', 0.8, fontsize=10)

        # Annotate data points for Net Income
        annotate_data_points2(plt.gca(), np.arange(len(quarters)) + 0.25, df_net_income['Amount'].values, 'black', 0.8, fontsize=10)

        # Change x-axis labels to quarter format
        plt.xticks(range(8), quarter_labels)

        # Adjust y limit
        plt.ylim([min(min(revenue), min(net_income)) * 1.1,
                  max(max(revenue), max(net_income)) * 1.1])  # Adjusts for both top and bottom

        # Displaying the bar chart
        st.pyplot(plt.gcf())


        #CASHFLOW VISUALISATION

        # Sort the data by fiscalDateEnding in ascending order
        annual_cash_flow_statement_df.sort_values('fiscalDateEnding', inplace=True)

        # Convert the cash flow columns into float type and replace "None" with 0
        annual_cash_flow_statement_df['operatingCashflow'] = annual_cash_flow_statement_df['operatingCashflow'].replace(
            'None', '0').astype(float)
        annual_cash_flow_statement_df['cashflowFromInvestment'] = annual_cash_flow_statement_df[
            'cashflowFromInvestment'].replace('None', '0').astype(float)
        annual_cash_flow_statement_df['cashflowFromFinancing'] = annual_cash_flow_statement_df[
            'cashflowFromFinancing'].replace('None', '0').astype(float)


        # Define a function to annotate the data points
        def annotate_data_points(ax, x_points, y_points, color, y_offset, fontsize=12):
            bar_width = 0.8  # default bar width in seaborn
            for x, y in zip(x_points, y_points):
                if y < 0:  # check if the value is negative
                    ax.text(x - bar_width / 2, y - y_offset, format_billions(int(y)), color=color, ha='center',
                            va='top', fontsize=fontsize)
                else:
                    ax.text(x - bar_width / 2, y + y_offset, format_billions(int(y)), color=color, ha='center',
                            va='bottom', fontsize=fontsize)


        # Fetching the cashflow data for the last 5 years
        cashflows = annual_cash_flow_statement_df[
            ['fiscalDateEnding', 'operatingCashflow', 'cashflowFromInvestment', 'cashflowFromFinancing']].tail(5)

        # Setting the figure size
        plt.figure(figsize=(12, 8))

        # Create a DataFrame for seaborn
        df_operating = pd.DataFrame({
            'Year': cashflows['fiscalDateEnding'],
            'Cashflow': cashflows['operatingCashflow'],
            'Type': 'Operating'
        })

        df_investing = pd.DataFrame({
            'Year': cashflows['fiscalDateEnding'],
            'Cashflow': cashflows['cashflowFromInvestment'],
            'Type': 'Investing'
        })

        df_financing = pd.DataFrame({
            'Year': cashflows['fiscalDateEnding'],
            'Cashflow': cashflows['cashflowFromFinancing'],
            'Type': 'Financing'
        })

        # Combining all dataframes
        df_combined = pd.concat([df_operating, df_investing, df_financing])

        # Creating a Seaborn barplot
        sns.barplot(x='Year', y='Cashflow', hue='Type', data=df_combined, palette=['blue', 'orange', 'purple'])

        # Adding title and labels
        plt.title(f'Operating, Investing, and Financing Cashflows for Last 5 Years for {symbol}', fontsize=20, pad=15)
        plt.ylabel('Cashflow ($)', fontsize=15)
        plt.xlabel('Year', fontsize=15)
        plt.legend(title='Type')

        # Annotate data points for Operating Cashflow
        annotate_data_points(plt.gca(), np.arange(len(cashflows)) + 0.13, df_operating['Cashflow'].values, 'black', 0.2, fontsize=10)

        # Annotate data points for Investing Cashflow
        annotate_data_points(plt.gca(), np.arange(len(cashflows)) + 0.4, df_investing['Cashflow'].values, 'black', 0.2, fontsize=10)

        # Annotate data points for Financing Cashflow
        annotate_data_points(plt.gca(), np.arange(len(cashflows)) + 0.67, df_financing['Cashflow'].values, 'black', 0.2, fontsize=10)

        # Set Y-axis formatting
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_billions(x)))

        # Adjust y limit
        plt.ylim(
            [min(min(df_operating['Cashflow']), min(df_investing['Cashflow']), min(df_financing['Cashflow'])) * 1.1,
             max(max(df_operating['Cashflow']), max(df_investing['Cashflow']), max(df_financing['Cashflow'])) * 1.1])

        # Displaying the bar chart
        st.pyplot(plt.gcf())


    except ValueError as e:
        st.error(str(e))
else:
    st.write("(example: type 'AAPL' and press Enter)")