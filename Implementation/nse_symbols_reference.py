"""
NSE Stock Symbol Reference
Curated list of major NSE stocks organized by sector
"""

NSE_STOCK_SYMBOLS = {
    "🏦 Banking & Financial Services": {
        "symbols": [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "PNB", "BANKBARODA",
            "IDFCFIRSTB", "AUBANK", "RBLBANK"
        ],
        "description": "Major banks and financial institutions"
    },
    
    "💻 Information Technology": {
        "symbols": [
            "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
            "LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "LTTS"
        ],
        "description": "IT services and software companies"
    },
    
    "Energy & Power": {
        "symbols": [
            "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO",
            "POWERGRID", "NTPC", "COALINDIA", "GAIL", "ADANIGREEN",
            "TATAPOWER", "ADANIPOWER", "ADANITRANS"
        ],
        "description": "Oil, gas, and power generation companies"
    },
    
    "🏭 Infrastructure & Capital Goods": {
        "symbols": [
            "LT", "ADANIENT", "ADANIPORTS", "ULTRAcemco", "GRASIM",
            "SHREECEM", "AMBUJACEM", "ACC", "SIEMENS", "ABB",
            "CROMPTON", "HAVELLS", "VOLTAS", "BHARATFORG"
        ],
        "description": "Construction, cement, and engineering"
    },
    
    "🚗 Automobile & Auto Components": {
        "symbols": [
            "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO",
            "EICHERMOT", "TVSMOTOR", "ASHOKLEY", "BALKRISIND",
            "MRF", "APOLLOTYRE", "CEAT", "MOTHERSON", "BOSCHLTD"
        ],
        "description": "Auto manufacturers and component makers"
    },
    
    "🏥 Pharma & Healthcare": {
        "symbols": [
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
            "LUPIN", "BIOCON", "TORNTPHARM", "ALKEM", "APOLLOHOSP",
            "LAURUSLABS", "IPCALAB", "GLAXO", "ABBOTINDIA"
        ],
        "description": "Pharmaceutical and healthcare companies"
    },
    
    "🛒 FMCG & Consumer": {
        "symbols": [
            "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
            "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "UBL",
            "MCDOWELL-N", "EMAMILTD", "VBL", "VARUN", "PGHH"
        ],
        "description": "Fast-moving consumer goods"
    },
    
    "📱 Telecom & Media": {
        "symbols": [
            "BHARTIARTL", "IDEA", "INDIAMART", "ZEEL", "NETWORK18",
            "TATACOMM", "HATHWAY", "DEN"
        ],
        "description": "Telecom operators and media companies"
    },
    
    "🏠 Realty & Housing": {
        "symbols": [
            "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
            "SOBHA", "PHOENIXLTD", "SUNTECK"
        ],
        "description": "Real estate developers"
    },
    
    "🎨 Consumer Durables & Retail": {
        "symbols": [
            "TITAN", "DMART", "TRENT", "BAJAJFINSV", "VOLTAS",
            "CROMPTON", "HAVELLS", "WHIRLPOOL", "SYMPHONY", "RELAXO"
        ],
        "description": "Retail chains and consumer durables"
    },
    
    "Travel & Hospitality": {
        "symbols": [
            "INDIGO", "SPICEJET", "IRCTC", "INDIANHUME", "LEMONTREE",
            "CHALET", "TBO"
        ],
        "description": "Airlines, hotels, and travel"
    },
    
    "🧪 Chemicals & Materials": {
        "symbols": [
            "UPL", "PIDILITIND", "ATUL", "DEEPAKNTR", "SRF",
            "AARTI", "GNFC", "BALRAMCHIN", "ALKYLAMINE", "FINEORG"
        ],
        "description": "Chemical manufacturers"
    },
    
    "🏛️ PSU & Government": {
        "symbols": [
            "SAIL", "NMDC", "COALINDIA", "ONGC", "NTPC",
            "POWERGRID", "GAIL", "IOC", "BPCL", "HINDPETRO",
            "HAL", "BEL", "BHEL", "IRFC", "IRCON"
        ],
        "description": "Public Sector Undertakings"
    },
    
    "💎 Metals & Mining": {
        "symbols": [
            "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
            "NMDC", "COALINDIA", "NATIONALUM", "JINDALSTEL", "RATNAMANI"
        ],
        "description": "Steel, aluminum, and mining companies"
    },
    
    "🎯 Top 30 Liquid Stocks (Nifty 50 Core)": {
        "symbols": [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
            "TITAN", "BAJFINANCE", "ULTRACEMCO", "NESTLEIND", "WIPRO",
            "M&M", "HCLTECH", "ADANIENT", "NTPC", "ONGC",
            "POWERGRID", "TATAMOTORS", "BAJAJ-AUTO", "COALINDIA", "TATASTEEL"
        ],
        "description": "Most liquid stocks for pairs trading (recommended)"
    }
}


def get_all_symbols():
    """Return a flat list of all unique symbols"""
    all_symbols = set()
    for sector_data in NSE_STOCK_SYMBOLS.values():
        all_symbols.update(sector_data["symbols"])
    return sorted(list(all_symbols))


def format_symbols_for_input(symbols):
    """Format list of symbols for text input (comma-separated)"""
    return ", ".join(symbols)


def get_sector_symbols(sector_name):
    """Get symbols for a specific sector"""
    if sector_name in NSE_STOCK_SYMBOLS:
        return NSE_STOCK_SYMBOLS[sector_name]["symbols"]
    return []
