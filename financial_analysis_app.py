import numpy as np
import streamlit as st
from alpha_vantage.fundamentaldata import FundamentalData
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import yfinance as yf
import os

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

# User input for stock symbol
symbol = st.text_input("Enter stock symbol:")
if symbol:
    symbol = symbol.upper()

    try:
        with st.spinner("Fetching data..."):
            data = get_fundamental_data(symbol)
            stock_price_data = get_stock_price_data(symbol)

        if data is not None and len(data) == 6 and stock_price_data is not None:
            annual_balance_sheet_df, annual_income_statement_df, annual_cash_flow_statement_df, quarterly_balance_sheet_df, quarterly_income_statement_df, quarterly_cash_flow_statement_df = data

            # Fill null values with 0
            annual_balance_sheet_df.fillna(0, inplace=True)
            annual_income_statement_df.fillna(0, inplace=True)
            annual_cash_flow_statement_df.fillna(0, inplace=True)

            quarterly_balance_sheet_df.fillna(0, inplace=True)
            quarterly_income_statement_df.fillna(0, inplace=True)
            quarterly_cash_flow_statement_df.fillna(0, inplace=True)


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


        def calculate_ratios(annual_balance_sheet_df, annual_income_statement_df, quarterly_balance_sheet_df,
                             quarterly_income_statement_df, stock_price_data):
            ratios_dict = {
                'Year': [],
                'Current Ratio': [],
                'Debt Ratio': [],
                'Net Profit Margin': [],
                'P/E Ratio': [],
            }

            for i in range(-5, 0, 1):  # Iterate from -5 to 0 to get the last 5 years
                # Current Ratio
                current_assets = float(quarterly_balance_sheet_df['totalCurrentAssets'][i])
                current_liabilities = float(quarterly_balance_sheet_df['totalCurrentLiabilities'][i])
                current_ratio = current_assets / current_liabilities if current_liabilities else np.nan

                # Debt Ratio
                total_debt = float(annual_balance_sheet_df['totalLiabilities'].values[i])
                total_assets = float(annual_balance_sheet_df['totalAssets'].values[i])
                debt_ratio = total_debt / total_assets if total_assets else np.nan

                # Net Profit Margin
                if 'totalRevenue' in quarterly_income_statement_df.columns and 'netIncome' in quarterly_income_statement_df.columns:
                    revenue = float(quarterly_income_statement_df['totalRevenue'][i])
                    net_income = float(quarterly_income_statement_df['netIncome'][i])
                    net_profit_margin = net_income / revenue if revenue else np.nan

                # P/E Ratio
                net_income = float(annual_income_statement_df['netIncome'].values[i])
                outstanding_shares = float(annual_balance_sheet_df['commonStockSharesOutstanding'].values[i])
                eps = net_income / outstanding_shares if outstanding_shares else np.nan
                pe_ratio = stock_price_data[-1] / eps if eps else np.nan

                # Append data to dictionary
                year = pd.to_datetime(annual_balance_sheet_df['fiscalDateEnding'].values[i]).year
                ratios_dict['Year'].append(year)
                ratios_dict['Current Ratio'].append(current_ratio)
                ratios_dict['Debt Ratio'].append(debt_ratio)
                ratios_dict['Net Profit Margin'].append(net_profit_margin)
                ratios_dict['P/E Ratio'].append(pe_ratio)

            return pd.DataFrame(ratios_dict)


        #DATAFRAMES

        with st.spinner("Fetching stock price data..."):
            stock_price_data = get_stock_price_data(symbol)

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
        ratios2_df = pd.DataFrame({
            'Ratios': ['Net Profit Margin', 'Operating Cash Flow Ratio',
                       'Earnings Per Share', 'P/E Ratio',  'Annual Dividend Yield'],
            'Values': [f"{net_profit_margin:.2f}", f"{operating_cash_flow_ratio:.2f}",
                       f"{eps:.2f}", f"{pe_ratio:.2f}", f"{annual_dividend_yield*100:.2f}%"]
        }).set_index('Ratios')

        # Display the DataFrame as a table
        st.subheader("Financial Performance Ratios")
        st.table(ratios2_df)

        # Sort the data by fiscalDateEnding in ascending order
        annual_income_statement_df = annual_income_statement_df.sort_values('fiscalDateEnding', ascending=True)

        #DATAFRAME OVER TIME

        def calculate_ratios(annual_balance_sheet_df, annual_income_statement_df, quarterly_balance_sheet_df,
                             quarterly_income_statement_df, stock_price_data):
            ratios_list = []

            num_years = min(5, len(annual_balance_sheet_df))

            for i in range(-num_years, 0, 1):
                year = pd.to_datetime(annual_balance_sheet_df['fiscalDateEnding'].values[i]).year

                # Current Ratio
                current_assets = float(quarterly_balance_sheet_df['totalCurrentAssets'].values[i])
                current_liabilities = float(quarterly_balance_sheet_df['totalCurrentLiabilities'].values[i])
                current_ratio = current_assets / current_liabilities if current_liabilities else np.nan
                ratios_list.append(['Current Ratio', year, current_ratio])

                # Debt Ratio
                total_debt = float(annual_balance_sheet_df['totalLiabilities'].values[i])
                total_assets = float(annual_balance_sheet_df['totalAssets'].values[i])
                debt_ratio = total_debt / total_assets if total_assets else np.nan
                ratios_list.append(['Debt Ratio', year, debt_ratio])

                # Net Profit Margin
                if 'totalRevenue' in quarterly_income_statement_df.columns and 'netIncome' in quarterly_income_statement_df.columns:
                    revenue = float(quarterly_income_statement_df['totalRevenue'].values[i])
                    net_income = float(quarterly_income_statement_df['netIncome'].values[i])
                    net_profit_margin = net_income / revenue if revenue else np.nan
                    ratios_list.append(['Net Profit Margin', year, net_profit_margin])

                # P/E Ratio
                net_income = float(annual_income_statement_df['netIncome'].values[i])
                outstanding_shares = float(annual_balance_sheet_df['commonStockSharesOutstanding'].values[i])
                eps = net_income / outstanding_shares if outstanding_shares else np.nan
                pe_ratio = stock_price_data[-num_years] / eps if eps else np.nan
                ratios_list.append(['P/E Ratio', year, pe_ratio])

            return pd.DataFrame(ratios_list, columns=['Ratio', 'Year', 'Value'])


        #Dataframe
        # Call the calculate_ratios function
        ratios_df = calculate_ratios(annual_balance_sheet_df, annual_income_statement_df, quarterly_balance_sheet_df,
                                     quarterly_income_statement_df, stock_price_data)

        # Reshape the DataFrame
        ratios_df = ratios_df.pivot(index='Ratio', columns='Year', values='Value')

        # Display the DataFrame with float data shown to two decimal places.
        st.subheader("Ratios over 5 years")
        st.dataframe(ratios_df.style.format("{:.2f}"))


        # VISUALISATIONS

        plt.style.use('ggplot')

        # Data for Total Revenue plot
        x = pd.to_datetime(annual_income_statement_df['fiscalDateEnding'])
        y1 = annual_income_statement_df['totalRevenue'].astype(float)

        # Data for Net Income plot
        y2 = annual_income_statement_df['netIncome'].astype(float)


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


        # Create a function to annotate data points
        def annotate_data_points(ax, df, y_col):
            for x, y in zip(df['fiscalDateEnding'], df[y_col]):
                ax.text(x, y + 0.05 * y, format_billions(y), color='black', ha='center', va='bottom')

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), dpi=100)

        # Plot for Total Revenue
        sns.lineplot(x='fiscalDateEnding', y='totalRevenue',
                     data=annual_income_statement_df.astype({'totalRevenue': 'float'}), ax=axs[0], marker='o')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[0].yaxis.set_major_formatter(formatter)
        axs[0].set_xlabel('Year')
        axs[0].set_ylabel('Total Revenue ($)')
        axs[0].set_title('Total Annual Revenue', fontsize=20, pad=12)
        axs[0].set_ylim(bottom=0)
        axs[0].tick_params(axis='x')
        annotate_data_points(axs[0], annual_income_statement_df.astype({'totalRevenue': 'float'}),
                             'totalRevenue')

        # Plot for Net Income
        sns.lineplot(x='fiscalDateEnding', y='netIncome',
                     data=annual_income_statement_df.astype({'netIncome': 'float'}), ax=axs[1], marker='o')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[1].yaxis.set_major_formatter(formatter)
        axs[1].set_xlabel('Year')
        axs[1].set_ylabel('Net Income ($)')
        axs[1].set_title('Annual Net Income', fontsize=20, pad=12)
        axs[1].set_ylim(bottom=0)
        axs[1].tick_params(axis='x')
        annotate_data_points(axs[1], annual_income_statement_df.astype({'netIncome': 'float'}),
                             'netIncome')

        # Add space between subplots
        plt.subplots_adjust(hspace=0.3, top=1.0)

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

        # Show plots using Streamlit
        st.subheader("Total Revenue and Net Income Over 5 Years")
        st.pyplot(fig)


        #QUARTERLY BAR CHART-REVENUE VS INCOME

        def format_y_axis(y, _):
            return format_billions(y)

        formatter = FuncFormatter(format_y_axis)


        # Define a function to annotate the data points
        # Define a function to annotate the data points
        def annotate_data_points2(ax, x_points, y_points, color, y_offset, x_offset=0, fontsize=10):
            for x, y in zip(x_points, y_points):
                ax.text(x + x_offset, float(y) + y_offset, format_billions(int(y)), color=color, ha='center',
                        va='bottom', fontsize=fontsize)

        # Sort the dataframe in descending order
        quarterly_income_statement_df = quarterly_income_statement_df.sort_index(ascending=False)

        # Fetch quarters
        quarters = pd.to_datetime(quarterly_income_statement_df['fiscalDateEnding'].values[-8:])

        # Create quarter labels
        quarter_labels = quarters.to_period('Q').strftime("Q%q '%y")

        # Fetching the revenue and net income data for the last 8 quarters
        revenue = quarterly_income_statement_df['totalRevenue'].values[-8:].astype(float)
        net_income = quarterly_income_statement_df['netIncome'].values[-8:].astype(float)

        # Setting the figure size
        plt.figure(figsize=(12, 8))

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

        # Create a seaborn barplot
        sns.barplot(x='Quarter', y='Amount', hue='Type', data=df_combined, palette=['blue', 'green'])

        # Adding title and labels
        plt.title(f'Revenue vs Net Income for last 8 Quarters for {symbol}', fontsize=20, pad=15)
        plt.ylabel('Amount ($)', fontsize=15)
        plt.xlabel('Quarter', fontsize=15)
        plt.legend(title='Type')

        # Set y-axis formatter
        plt.gca().yaxis.set_major_formatter(formatter)

        # Annotate data points for Revenue
        annotate_data_points2(plt.gca(), np.arange(len(quarters)) - 0.2, df_revenue['Amount'].values, 'black', 0.8, fontsize=10)

        # Annotate data points for Net Income
        annotate_data_points2(plt.gca(), np.arange(len(quarters)) + 0.25, df_net_income['Amount'].values, 'black', 0.8, fontsize=10)

        # Change x-axis labels to quarter format
        plt.xticks(range(8), quarter_labels)

        # Adjust y limit
        plt.ylim([0, max(max(revenue), max(net_income)) * 1.1])  # gives 10% extra room at the top

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
    st.write('Please enter a stock symbol.')