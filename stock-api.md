# 증권 정보 조회 API

## yfinance

[Yahoo Finance에서 제공하는 API](https://finance.yahoo.com/) 입니다.

이를 위해 아래와 같이 yfiance를 미리 설치하여야 합니다.

```text
pip install yfinance
```

네이버와 같은 한국회사들은 "KS"를 접미어로 사용하고, 일본은 "T"입니다.

구현된 주식 tool은 아래와 같습니다.

```python
@tool
def stock_data_lookup(ticker, country):
    """
    Retrieve accurate stock trends for a given ticker.
    ticker: the ticker to retrieve price history for
    country: the english country name of the stock
    return: the information of ticker
    """ 
    com = re.compile('[a-zA-Z]') 
    alphabet = com.findall(ticker)
    print('alphabet: ', alphabet)

    print("country:", country)

    if len(alphabet)==0:
        if country == "South Korea":
            ticker += ".KS"
        elif country == "Japan":
            ticker += ".T"
    print("ticker:", ticker)
    
    stock = yf.Ticker(ticker)
    
    # get the price history for past 1 month
    history = stock.history(period="1mo")
    print('history: ', history)
    
    result = f"## Trading History\n{history}"
    #history.reset_index().to_json(orient="split", index=False, date_format="iso")    
    
    result += f"\n\n## Financials\n{stock.financials}"    
    print('financials: ', stock.financials)

    result += f"\n\n## Major Holders\n{stock.major_holders}"
    print('major_holders: ', stock.major_holders)

    print('result: ', result)

    return result
```
