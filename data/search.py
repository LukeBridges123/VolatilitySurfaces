import pandas as pd
import os

def search_options(option_type="E", strike_price=None, put_call="C", min_bid=None, max_ask=None, expiration_date=None):
    """
    Search for options based on specified criteria.
    
    Parameters:
    - option_type: The style of the option ('A' for American, 'E' for European)
    - strike_price: The strike price of the option
    - put_call: 'P' for put, 'C' for call
    - min_bid: Minimum bid price
    - max_ask: Maximum ask price
    - expiration_date: Expiration date of the option (format: MM/DD/YY)
    
    Returns:
    - DataFrame containing filtered option data
    """
    # Get the absolute path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Options_prices_US_Canada.csv')
    
    # Read the CSV file
    options_data = pd.read_csv(csv_path)
    
    # Apply filters based on parameters provided
    filtered_data = options_data.copy()
    
    if option_type is not None:
        filtered_data = filtered_data[filtered_data['style'] == option_type]
        
    if strike_price is not None:
        filtered_data = filtered_data[filtered_data['strike'] == strike_price]
        
    if put_call is not None:
        filtered_data = filtered_data[filtered_data['call/put'] == put_call]
        
    if min_bid is not None:
        filtered_data = filtered_data[filtered_data['bid'] >= min_bid]
        
    if max_ask is not None:
        filtered_data = filtered_data[filtered_data['ask'] <= max_ask]
    
    if expiration_date is not None:
        filtered_data = filtered_data[filtered_data['expiration'] == expiration_date]
    
    return filtered_data

# Example usage:
if __name__ == "__main__":
    # Example: Search for European-style call options with strike price of 400 expiring on 03/31/17
    results = search_options(option_type='E', put_call='C', expiration_date='03/31/17')
    if not results.empty:
        print(f"Found {len(results)} matching options:")
        print(results[['symbol', 'expiration', 'strike', 'call/put', 'style', 'bid', 'ask']])
    else:
        print("No options found matching the criteria.")
