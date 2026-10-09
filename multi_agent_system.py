import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
import json
import re
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " OR ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE ({where_clause})
        AND q.total_amount > 0
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# Load environment variables
dotenv.load_dotenv()

api_key = os.getenv("UDACITY_OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "UDACITY_OPENAI_API_KEY was not found. "
        "Confirm that .env is in the same folder as multi_agent_system.py."
    )

# Connect smolagents to the Udacity OpenAI-compatible endpoint
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=api_key,
)


# Create a lookup for the exact catalog names and prices
CATALOG = {
    item["item_name"]: item
    for item in paper_supplies
}


# Map common customer wording to exact database item names
ITEM_ALIASES = {
    "a4 paper": "A4 paper",
    "a4 white paper": "A4 paper",
    "white a4 paper": "A4 paper",
    "a4 printer paper": "A4 paper",
    "a4 printing paper": "A4 paper",
    "a4 copier paper": "A4 paper",
    "a4 copy paper": "A4 paper",

    "letter paper": "Letter-sized paper",
    "letter-sized paper": "Letter-sized paper",
    "letter size paper": "Letter-sized paper",

    "cardstock": "Cardstock",
    "card stock": "Cardstock",
    "white cardstock": "Cardstock",
    "colored cardstock": "Cardstock",
    "colourful cardstock": "Cardstock",
    "colorful cardstock": "Cardstock",
    "heavy cardstock": "Heavyweight paper",
    "heavyweight cardstock": "Heavyweight paper",
    "250 gsm cardstock": "250 gsm cardstock",

    "colored paper": "Colored paper",
    "colorful paper": "Colored paper",
    "bright colored paper": "Bright-colored paper",

    "glossy paper": "Glossy paper",
    "a4 glossy paper": "Glossy paper",
    "glossy a4 paper": "Glossy paper",
    "matte paper": "Matte paper",
    "a4 matte paper": "Matte paper",

    "printer paper": "Standard copy paper",
    "standard printer paper": "Standard copy paper",
    "standard printing paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "copier paper": "Standard copy paper",
    "plain paper": "Standard copy paper",

    "poster paper": "Poster paper",
    "poster board": "Poster paper",
    "poster boards": "Poster paper",
    "colorful poster paper": "Poster paper",
    "large poster paper": "Large poster paper (24x36 inches)",
    "poster boards (24 x 36)": "Large poster paper (24x36 inches)",
    "poster boards 24 x 36": "Large poster paper (24x36 inches)",
    "poster boards 24x36": "Large poster paper (24x36 inches)",
    "24 x 36 poster boards": "Large poster paper (24x36 inches)",
    "24x36 poster boards": "Large poster paper (24x36 inches)",

    "banner paper": "Banner paper",
    "banner rolls": "Rolls of banner paper (36-inch width)",
    "rolls of banner paper": "Rolls of banner paper (36-inch width)",

    "streamers": "Party streamers",
    "party streamers": "Party streamers",

    "washi tape": "Decorative adhesive tape (washi tape)",
    "decorative washi tape": "Decorative adhesive tape (washi tape)",
    "decorative adhesive tape": "Decorative adhesive tape (washi tape)",

    "napkins": "Paper napkins",
    "paper napkins": "Paper napkins",
    "table napkins": "Paper napkins",

    "paper cups": "Paper cups",
    "cups": "Paper cups",
    "disposable cups": "Disposable cups",

    "paper plates": "Paper plates",
    "plates": "Paper plates",

    "envelopes": "Envelopes",
    "flyers": "Flyers",
    "table covers": "Table covers",
    "sticky notes": "Sticky notes",
    "notepads": "Notepads",
    "invitation cards": "Invitation cards",
    "presentation folders": "Presentation folders",
    "name tags": "Name tags with lanyards",

    "recycled paper": "Recycled paper",
    "eco-friendly paper": "Eco-friendly paper",
    "kraft paper": "Kraft paper",
    "construction paper": "Construction paper",
    "wrapping paper": "Wrapping paper",
    "decorative paper": "Decorative paper",
    "glitter paper": "Glitter paper",
    "crepe paper": "Crepe paper",
    "photo paper": "Photo paper",
    "glossy photo paper": "Photo paper",
    "letterhead": "Letterhead paper",
    "letterhead paper": "Letterhead paper",
    "legal size paper": "Legal-size paper",
    "100 lb cover stock": "100 lb cover stock",
    "80 lb text paper": "80 lb text paper",
    "220 gsm poster paper": "220 gsm poster paper",
}


def normalize_text(value: str) -> str:
    """Normalize product wording for catalog matching."""
    value = value.lower().strip()
    value = value.replace('"', "")
    value = value.replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    return value


def resolve_item_name(customer_item: str) -> Union[str, None]:
    """Convert customer wording into an exact catalog item name."""
    normalized = normalize_text(customer_item)

    # Check exact catalog names first
    for exact_name in CATALOG:
        if normalize_text(exact_name) == normalized:
            return exact_name

    # Check exact aliases second
    if normalized in ITEM_ALIASES:
        return ITEM_ALIASES[normalized]

    # Check whether a known alias appears in a longer description
    for alias in sorted(ITEM_ALIASES, key=len, reverse=True):
        if alias in normalized:
            return ITEM_ALIASES[alias]

    return None


def get_bulk_discount(total_units: int) -> float:
    """Return a deterministic bulk discount based on total units."""
    if total_units >= 10000:
        return 0.15
    if total_units >= 5000:
        return 0.12
    if total_units >= 1000:
        return 0.10
    if total_units >= 500:
        return 0.05
    return 0.0

# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def check_inventory(item_name: str, request_date: str) -> str:
    """
    Check inventory for a specific item.

    Args:
        item_name: Exact inventory item name.
        request_date: Date to check inventory against in YYYY-MM-DD format.

    Returns:
        Inventory information as JSON.
    """
    stock = get_stock_level(item_name, request_date)

    if stock.empty:
        return f"{item_name} not found."

    return stock.to_json(orient="records")


@tool
def check_all_inventory(request_date: str) -> str:
    """
    Return all available inventory.

    Args:
        request_date: Date used to calculate inventory levels.

    Returns:
        All inventory levels as JSON.
    """
    inventory = get_all_inventory(request_date)
    return json.dumps(inventory, indent=2)


@tool
def check_supplier_timeline(
    request_date: str,
    shortage_quantity: int
) -> str:
    """
    Determine supplier delivery date.

    Args:
        request_date: Requested order date.
        shortage_quantity: Quantity that must be restocked.

    Returns:
        Supplier delivery information as JSON.
    """
    delivery_date = get_supplier_delivery_date(
        request_date,
        shortage_quantity
    )

    return json.dumps(
        {
            "request_date": request_date,
            "shortage_quantity": shortage_quantity,
            "supplier_delivery_date": delivery_date,
        },
        indent=2
    )


@tool
def check_available_cash(request_date: str) -> str:
    """
    Check current cash balance.

    Args:
        request_date: Date to evaluate cash position.

    Returns:
        Cash balance as JSON.
    """
    cash = get_cash_balance(request_date)

    return json.dumps(
        {"cash_balance": cash},
        indent=2
    )


@tool
def get_company_report(request_date: str) -> str:
    """
    Generate complete company report.

    Args:
        request_date: Date used for report generation.

    Returns:
        Financial report as JSON.
    """
    report = generate_financial_report(request_date)

    return json.dumps(
        report,
        default=str,
        indent=2
    )


# Tools for quoting agent
@tool
def search_historical_quotes(search_terms: str) -> str:
    """
    Search historical quote records.

    Args:
        search_terms: Comma-separated search terms.

    Returns:
        Matching quote records as JSON.
    """
    terms = [t.strip() for t in search_terms.split(",") if t.strip()]

    results = search_quote_history(terms)

    return json.dumps(
        results,
        default=str,
        indent=2
    )


@tool
def estimate_quote(item_name: str, quantity: int) -> str:
    """
    Estimate a quote.

    Args:
        item_name: Requested item.
        quantity: Requested quantity.

    Returns:
        Quote estimate as JSON.
    """
    item_name = resolve_item_name(item_name)

    if not item_name:
        return json.dumps({"error": "Item not found"})

    item = CATALOG[item_name]

    discount = get_bulk_discount(quantity)

    total = (
        item["unit_price"]
        * quantity
        * (1 - discount)
    )

    return json.dumps(
        {
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": item["unit_price"],
            "discount": discount,
            "total_price": round(total, 2),
        },
        indent=2,
    )

LAST_ORDER_RESULT = None
CURRENT_REQUEST_DATE = None


def record_order_result(result: dict) -> str:
    """
    Preserve the authoritative process_complete_order result for
    evaluation, then return the same result to the agent as JSON.
    """
    global LAST_ORDER_RESULT

    LAST_ORDER_RESULT = result

    return json.dumps(
        result,
        indent=2,
    )

# Tools for ordering agent

@tool
def process_complete_order(
    items_json: str,
    request_date: str,
    required_delivery_date: str,
) -> str:
    """
    Validate and fulfill an entire customer order.

    Args:
        items_json: JSON list containing customer_item and quantity
            for every requested item.
        request_date: Request date in YYYY-MM-DD format.
        required_delivery_date: Customer delivery deadline in
            YYYY-MM-DD format.

    Returns:
        JSON fulfillment or rejection result.
    """
    global CURRENT_REQUEST_DATE

    if CURRENT_REQUEST_DATE is None:
        return record_order_result({
            "status": "rejected",
            "reason": "The authoritative request date was not provided.",
            "transactions_created": False,
        })

    # The evaluation dataset is authoritative.
    request_date = CURRENT_REQUEST_DATE

    try:
        items = json.loads(items_json)
    except (json.JSONDecodeError, TypeError):
        return record_order_result({
            "status": "rejected",
            "reason": "The submitted item list was invalid.",
            "transactions_created": False,
        })

    if not isinstance(items, list) or not items:
        return record_order_result({
            "status": "rejected",
            "reason": "The order did not contain a valid item list.",
            "transactions_created": False,
        })

    try:
        request_date_dt = datetime.fromisoformat(
            request_date.split("T")[0]
        )
        deadline_dt = datetime.fromisoformat(
            required_delivery_date.split("T")[0]
        )
    except (ValueError, TypeError):
        return record_order_result({
            "status": "rejected",
            "reason": "The request or delivery date was invalid.",
            "transactions_created": False,
        })

    if deadline_dt < request_date_dt:
        return record_order_result({
            "status": "rejected",
            "reason": (
                "The required delivery date is before the request date."
            ),
            "transactions_created": False,
        })

    resolved_items = []
    unsupported_items = []
    subtotal = 0.0
    total_units = 0

    # Resolve and price every item before creating transactions.
    for item in items:
        customer_item = str(
            item.get("customer_item", "")
        ).strip()

        try:
            quantity = int(item.get("quantity", 0))
        except (TypeError, ValueError):
            quantity = 0

        if quantity <= 0:
            unsupported_items.append({
                "customer_item": customer_item,
                "reason": "Quantity must be a positive integer.",
            })
            continue

        exact_item_name = resolve_item_name(customer_item)

        if exact_item_name is None:
            unsupported_items.append({
                "customer_item": customer_item,
                "reason": "No matching catalog item was found.",
            })
            continue

        unit_price = float(
            CATALOG[exact_item_name]["unit_price"]
        )
        line_subtotal = round(
            quantity * unit_price,
            2,
        )

        resolved_items.append({
            "customer_item": customer_item,
            "item_name": exact_item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_subtotal": line_subtotal,
        })

        subtotal += line_subtotal
        total_units += quantity

    # Reject the complete order if any requested product is unsupported.
    if unsupported_items:
        return record_order_result({
            "status": "rejected",
            "reason": (
                "The complete order cannot be fulfilled because "
                "one or more requested products are unsupported."
            ),
            "unsupported_items": unsupported_items,
            "resolved_items": resolved_items,
            "transactions_created": False,
        })

    discount_rate = get_bulk_discount(total_units)
    discount_amount = round(
        subtotal * discount_rate,
        2,
    )
    total_price = round(
        subtotal - discount_amount,
        2,
    )

    # Required inventory-wide health check.
    inventory_snapshot = get_all_inventory(request_date)

    validations = []
    total_restock_cost = 0.0

    # Validate inventory and supplier timing for every item.
    for item in resolved_items:
        stock_result = get_stock_level(
            item["item_name"],
            request_date,
        )

        available_stock = (
            int(stock_result.iloc[0]["current_stock"])
            if not stock_result.empty
            else 0
        )

        shortage = max(
            item["quantity"] - available_stock,
            0,
        )

        supplier_delivery_date = None
        restock_cost = 0.0

        if shortage > 0:
            supplier_delivery_date = get_supplier_delivery_date(
                request_date,
                shortage,
            )

            supplier_date_dt = datetime.fromisoformat(
                supplier_delivery_date
            )

            if supplier_date_dt > deadline_dt:
                return record_order_result({
                    "status": "rejected",
                    "reason": (
                        f"Insufficient inventory for "
                        f"{item['item_name']}. The supplier can "
                        f"deliver the shortage on "
                        f"{supplier_delivery_date}, after the "
                        f"deadline of {required_delivery_date}."
                    ),
                    "item_name": item["item_name"],
                    "requested_quantity": item["quantity"],
                    "available_stock": available_stock,
                    "shortage": shortage,
                    "transactions_created": False,
                })

            restock_cost = round(
                shortage * item["unit_price"],
                2,
            )
            total_restock_cost += restock_cost

        validations.append({
            "item_name": item["item_name"],
            "quantity": item["quantity"],
            "available_stock": available_stock,
            "shortage": shortage,
            "supplier_delivery_date": supplier_delivery_date,
            "restock_cost": restock_cost,
        })

    # Check that required replenishment is affordable.
    available_cash = get_cash_balance(request_date)

    if total_restock_cost > available_cash:
        return record_order_result({
            "status": "rejected",
            "reason": (
                "The required replenishment inventory could not "
                "be obtained for this order."
            ),
            # These fields are retained for internal validation only.
            "available_cash": round(available_cash, 2),
            "required_restock_cost": round(
                total_restock_cost,
                2,
            ),
            "transactions_created": False,
        })

    # Record approved purchases and sales on the request date so
    # each evaluation row captures its own ledger movement.
    restock_transactions = []

    for validation in validations:
        if validation["shortage"] > 0:
            transaction_id = create_transaction(
                item_name=validation["item_name"],
                transaction_type="stock_orders",
                quantity=validation["shortage"],
                price=validation["restock_cost"],
                date=request_date,
            )

            restock_transactions.append({
                "transaction_id": transaction_id,
                "item_name": validation["item_name"],
                "quantity": validation["shortage"],
                "cost": validation["restock_cost"],
                "delivery_date": (
                    validation["supplier_delivery_date"]
                ),
            })

    # Allocate discounted revenue proportionally across sale lines.
    sale_transactions = []
    allocated_revenue = 0.0

    for index, item in enumerate(resolved_items):
        if index == len(resolved_items) - 1:
            line_sale_price = round(
                total_price - allocated_revenue,
                2,
            )
        else:
            line_sale_price = round(
                total_price
                * item["line_subtotal"]
                / subtotal,
                2,
            )

        transaction_id = create_transaction(
            item_name=item["item_name"],
            transaction_type="sales",
            quantity=item["quantity"],
            price=line_sale_price,
            date=request_date,
        )

        allocated_revenue += line_sale_price

        sale_transactions.append({
            "transaction_id": transaction_id,
            "item_name": item["item_name"],
            "quantity": item["quantity"],
            "sale_price": line_sale_price,
            "transaction_date": request_date,
        })

    return record_order_result({
        "status": "fulfilled",
        "items": resolved_items,
        "total_units": total_units,
        "subtotal": round(subtotal, 2),
        "discount_percent": int(discount_rate * 100),
        "discount_amount": discount_amount,
        "total_price": total_price,
        "request_date": request_date,
        "required_delivery_date": required_delivery_date,
        "inventory_items_before_order": len(
            inventory_snapshot
        ),
        "restock_transactions": restock_transactions,
        "sale_transactions": sale_transactions,
        "transactions_created": True,
    })


# Set up your agents and create an orchestration agent that will manage them.
inventory_agent = ToolCallingAgent(
    tools=[
        check_inventory,
        check_all_inventory,
        check_supplier_timeline,
        check_available_cash,
        get_company_report,
    ],
    model=model,
    max_steps=3,
    name="inventory_management_agent",
    description=(
        "Handles inventory validation, supplier lead times, "
        "available cash, and company financial information."
    ),
    instructions="""
You are the Inventory Management Agent.

Your responsibilities are limited to inventory, supplier, cash,
and financial validation.

For each assigned request:

1. Review the complete customer request, quantities, and dates.
2. Use check_all_inventory to obtain the inventory snapshot for the request date.
3. Use check_inventory when the exact catalog item name is available.
4. Use check_supplier_timeline when a shortage may require restocking.
5. Use check_available_cash to evaluate replenishment affordability.
6. Use get_company_report when broader financial information is needed.
7. Return inventory findings only.
8. Report stock levels, shortages, supplier timelines, and cash availability.
9. Do not determine whether an order should be fulfilled or rejected.
10. Do not create sales transactions.
11. Do not calculate final pricing.
12. The Ordering Agent and process_complete_order tool are the authoritative source for fulfillment decisions.
""",
)

quote_agent = ToolCallingAgent(
    tools=[
        search_historical_quotes,
        estimate_quote,
    ],
    model=model,
    max_steps=3,
    name="quote_agent",
    description=(
    "Handles quote analysis, pricing, historical quote retrieval, "
    "and supported product identification."
    ),
    instructions="""
You are the Munder Difflin Quoting Agent.

1. Call search_historical_quotes once.
2. Call estimate_quote once for each requested item.
3. An item is supported when estimate_quote returns an item_name.
4. An item is unsupported only when estimate_quote returns an error.
5. Do not retry an unsupported item using alternate wording.
6. Do not call estimate_quote more than once per item.
7. Report each item estimate as preliminary.
8. Do not calculate or report a combined grand total.
9. Explain that final order-level pricing and discounts are calculated
   by the Ordering Agent.
10. If any estimate fails, identify only the item that failed.
"""
)

ordering_agent = ToolCallingAgent(
    tools=[
        process_complete_order,
    ],
    model=model,
    max_steps=2,
    name="ordering_agent",
    description=(
        "Finalizes complete customer orders using authoritative business "
        "rules for product support, pricing, inventory, supplier timing, "
        "cash, replenishment, and sales transactions."
    ),
    instructions="""
    You are the Ordering Agent.

    1. Extract every requested item and quantity exactly from the complete
    original customer request.
    2. Copy quantities verbatim from the original request.
    3. Never reduce a quantity based on available inventory.
    4. Never replace the requested quantity with available stock.
    5. Never alter quantities based on preliminary agent findings.
    6. Extract the request date exactly as provided in the original request.
    7. Extract the required delivery date exactly as provided.
    8. Never infer, alter, or replace either date.
    9. Ignore preliminary findings that conflict with the original request.
    10. Include every requested item, including unsupported items.
    11. Call process_complete_order exactly once.
    12. After process_complete_order returns, immediately provide its result.
    13. Do not call any other tools.
    14. Never remove unsupported items.
    15. Never process a partial order.
    16. Never call process_complete_order twice.
    17. Preserve the returned fulfilled or rejected status exactly.
    """,
    )


# LLM-based orchestrator that dynamically delegates work
orchestrator_agent = ToolCallingAgent(
    tools=[],
    managed_agents=[
        quote_agent,
        inventory_agent,
        ordering_agent,
    ],
    model=model,
    max_steps=8,
    name="orchestrator_agent",
    description=(
        "Dynamically analyzes requests, selects specialist agents, "
        "delegates tasks, evaluates their outputs, and creates the "
        "final customer-facing response."
    ),
    instructions="""
You are the LLM-based Orchestrator Agent.

You must dynamically decide which specialist agents are needed for each
request. Do not perform specialist work yourself.

Available managed agents:

1. quote_agent
   Use for historical quote research, preliminary pricing,
   product support, and quote analysis.

2. inventory_management_agent
   Use for inventory availability, supplier lead times,
   replenishment feasibility, available cash, and financial reporting.

3. ordering_agent
   Use for authoritative order fulfillment or rejection and
   transaction creation.

Routing rules:

- For a quote-only request, invoke quote_agent.
- For an inventory, supplier, cash, or financial question, invoke
  inventory_management_agent.
- For a complete customer order, invoke quote_agent and
  inventory_management_agent first.
- After receiving their findings, invoke ordering_agent for the
  authoritative fulfillment decision.
- Give each specialist the complete original customer request.
- Give ordering_agent the complete original request plus relevant findings
  from quote_agent and inventory_management_agent.

Authority rules:

- Inventory findings are informational only.
- The inventory_management_agent must not determine fulfillment status.
- The Ordering Agent is the authoritative source for fulfillment or rejection.
- If specialist findings conflict with process_complete_order output,
  use the process_complete_order result.
- Do not fulfill an order without invoking ordering_agent.
- Do not invoke ordering_agent more than once for the same request.
- Do not remove unsupported products.
- Do not authorize partial fulfillment.
- For complete customer orders, base the final response solely on the
  Ordering Agent result.
- Do not modify, reinterpret, or override the Ordering Agent outcome.
- If the Ordering Agent returns fulfilled, report fulfilled.
- If the Ordering Agent returns rejected, report rejected.
- Never describe a partial fulfillment.

Date rules:

- Preserve the original request date and required delivery date exactly.
- Never infer, alter, shorten, or replace a date.
- Include the exact original request date and required delivery date
  when invoking ordering_agent.

Final-response rules:

- For complete customer orders, return the Ordering Agent output verbatim.
- Do not summarize.
- Do not rewrite.
- Do not reinterpret.
- Do not calculate your own totals.
- Do not infer fulfillment status.
- The Ordering Agent output is the single source of truth.
""",
)

def sanitize_customer_response(order_result: dict) -> str:
    """
    Build a customer-safe message from the authoritative order result.
    """
    status = str(
        order_result.get("status", "")
    ).lower()

    if status == "fulfilled":
        total_price = float(
            order_result.get("total_price", 0.0)
        )
        discount_percent = int(
            order_result.get("discount_percent", 0)
        )
        delivery_date = order_result.get(
            "required_delivery_date",
            "the requested delivery date",
        )

        item_summary = ", ".join(
            f"{item.get('quantity', 0)} x "
            f"{item.get('item_name', 'item')}"
            for item in order_result.get("items", [])
        )

        discount_text = (
            f" A {discount_percent}% bulk discount was applied."
            if discount_percent > 0
            else ""
        )

        return (
            f"Order successfully fulfilled for {item_summary}. "
            f"The final total is ${total_price:.2f}."
            f"{discount_text} "
            f"Delivery is scheduled for {delivery_date}."
        ).strip()

    if status == "rejected":
        reason = str(
            order_result.get(
                "reason",
                "The order could not be fulfilled as requested.",
            )
        ).strip()

        return f"Order rejected. {reason}"

    raise ValueError(
        "The authoritative order result has an invalid status."
    )

def call_multi_agent_system(
    request: str,
    authoritative_request_date: str,
) -> dict:
    """
    Run the multi-agent workflow and return the authoritative order
    result plus a separately generated customer-safe response.
    """
    global LAST_ORDER_RESULT
    global CURRENT_REQUEST_DATE

    # Prevent values from a previous request from being reused.
    LAST_ORDER_RESULT = None
    CURRENT_REQUEST_DATE = authoritative_request_date

    orchestrator_agent.run(
        f"""
Analyze and process the following customer request.

Dynamically determine which managed specialist agents are required.
Invoke the relevant agents and use their outputs.

For a complete order:

1. Invoke the Quote Agent for preliminary product and pricing analysis.
2. Invoke the Inventory Management Agent for operational analysis.
3. Invoke the Ordering Agent with the complete original request verbatim.
4. Do not summarize, rewrite, shorten, or modify any requested quantity.
5. The Ordering Agent must call process_complete_order exactly once.
6. Do not create an independent fulfillment determination.
7. You are not finished until the Ordering Agent has been invoked.

The authoritative request date is:
{authoritative_request_date}

Customer request:

{request}
"""
    )

    # Fallback if the orchestrator stops before invoking the
    # Ordering Agent.
    if LAST_ORDER_RESULT is None:
        print(
            "WARN: Orchestrator did not invoke the Ordering Agent. "
            "Invoking the Ordering Agent directly."
        )

        ordering_agent.run(
            f"""
Process the following complete customer order.

The authoritative request date is:
{authoritative_request_date}

Use that exact request date.
Use the original customer request exactly as written.
Copy every requested item and quantity verbatim.
Preserve the required delivery date exactly.
Call process_complete_order exactly once.

Original customer request:

{request}
"""
        )

    if LAST_ORDER_RESULT is None:
        raise ValueError(
            "The Ordering Agent did not call process_complete_order."
        )

    order_result = LAST_ORDER_RESULT.copy()

    # Verify that fulfilled transactions used the dataset date.
    if order_result.get("status") == "fulfilled":
        result_request_date = order_result.get("request_date")

        if result_request_date != authoritative_request_date:
            raise ValueError(
                f"Order used request date {result_request_date}; "
                f"expected {authoritative_request_date}."
            )

    return {
        "customer_response": sanitize_customer_response(
            order_result
        ),
        "order_result": order_result,
    }

def validate_cash_reconciliation(results):
    """Validate each request using its own cash-before and cash-after values."""

    fulfilled_count = 0
    rejected_count = 0
    changed_cash_count = 0

    for row in results:
        request_id = row["request_id"]
        status = row["order_status"]

        cash_before = round(float(row["cash_before"]), 2)
        cash_after = round(float(row["cash_after"]), 2)
        cash_delta = round(cash_after - cash_before, 2)

        row["cash_delta"] = cash_delta

        if cash_delta != 0:
            changed_cash_count += 1

        if status == "rejected":

            rejected_count += 1

            if not str(row.get("rejection_reason", "")).strip():
                raise ValueError(
                    f"Request {request_id} was rejected without a reason."
                )

            if row["transactions_created"]:
                raise ValueError(
                    f"Request {request_id} was rejected but reports "
                    f"that transactions were created."
                )

            if cash_delta != 0:
                raise ValueError(
                    f"Request {request_id} was rejected "
                    f"but cash changed by ${cash_delta:.2f}."
                )

        elif status == "fulfilled":

            fulfilled_count += 1

            if not row["transactions_created"]:
                raise ValueError(
                    f"Request {request_id} was fulfilled but reports "
                    f"that no transactions were created."
                )

            expected_delta = round(
                float(row["total_price"])
                - float(row["restock_cost"]),
                2,
            )

            if cash_delta != expected_delta:
                raise ValueError(
                    f"Request {request_id} was fulfilled, but its "
                    f"cash delta was ${cash_delta:.2f}; expected "
                    f"${expected_delta:.2f}."
                )

        else:
            raise ValueError(
                f"Request {request_id} has an invalid order status: {status}"
            )

    if fulfilled_count < 3:
        raise ValueError(
            f"Only {fulfilled_count} requests were fulfilled; at least 3 are required."
        )

    if rejected_count < 1:
        raise ValueError(
            "At least one request must be rejected with a reason."
        )

    if changed_cash_count < 3:
        raise ValueError(
            f"Only {changed_cash_count} requests changed cash; at least 3 are required."
        )

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = (
            quote_requests_sample
            .sort_values("request_date")
            .reset_index(drop=True)
        )
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"""
        Customer role: {row['job']}
        Order size: {row['need_size']}
        Event: {row['event']}
        Request date: {request_date}

        Customer request:
        {row['request']}
        """.strip()

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # response = call_your_multi_agent_system(request_with_date)
        cash_before = round(
            get_cash_balance(request_date),
            2,
        )

        system_result = call_multi_agent_system(
            request_with_date,
            request_date,
        )

        order_result = system_result["order_result"]
        response = system_result["customer_response"]

        cash_after = round(
            get_cash_balance(request_date),
            2,
        )

        cash_delta = round(
            cash_after - cash_before,
            2,
        )
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        
        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": len(results) + 1,
                "source_row_id": idx + 1,
                "request_date": request_date,
                "job": row["job"],
                "need_size": row["need_size"],
                "event": row["event"],
                "original_request": row["request"],
                "order_status": order_result["status"],
                "transactions_created": bool(order_result.get("transactions_created", False)),
                "rejection_reason": (
                    order_result.get("reason", "")
                    if order_result["status"] == "rejected"
                    else ""
                ),
                "total_price": round(
                    float(order_result.get("total_price", 0.0)),
                    2,
                ),
                "restock_cost": round(
                    sum(
                        float(transaction.get("cost", 0.0))
                        for transaction in order_result.get(
                            "restock_transactions",
                            [],
                        )
                    ),
                    2,
                ),
                "cash_before": cash_before,
                "cash_after": cash_after,
                "cash_delta": cash_delta,
                "cash_balance_as_of_request_date": cash_after,
                "inventory_value_as_of_request_date": round(
                    float(current_inventory),
                    2,
                ),
                "response": response,
            }
        )


        time.sleep(1)

    # Final report
    latest_transaction = pd.read_sql(
    """
    SELECT MAX(transaction_date) AS final_date
    FROM transactions
    """,
    db_engine,)
    final_date = latest_transaction.iloc[0]["final_date"]
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Report Date: {final_date}")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Build and save a diagnostic file before validation.
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "test_results_diagnostic.csv",
        index=False,
    )

    # Stop submission output if any result fails reconciliation.
    validate_cash_reconciliation(results)

    # Rebuild because validation adds or normalizes cash_delta.
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "test_results.csv",
        index=False,
    )

    print("\nCash reconciliation validation passed.")
    print("Submission results written to test_results.csv.")

    return results


if __name__ == "__main__":
    results = run_test_scenarios()
