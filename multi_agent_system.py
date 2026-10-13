"""Multi-Agent Business Operations System.

Coordinates quote analysis, inventory checks, and complete-order processing.
Run from the project directory containing the three source CSV files.
Each simulation run resets the local SQLite database and writes results.

This is a course-developed portfolio prototype. It uses a simplified
request-date ledger and does not provide atomic order writes or durable
duplicate-order protection.
"""

import ast
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Union

import dotenv
import numpy as np
import pandas as pd
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from sqlalchemy import Engine, create_engine
from sqlalchemy.sql import text


# ---------------------------------------------------------------------------
# Database and product catalog
# ---------------------------------------------------------------------------

db_engine = create_engine("sqlite:///munder_difflin.db")

# Paper prices are per sheet unless otherwise specified.
CATALOG_ROWS = [
    ("A4 paper", "paper", 0.05),
    ("Letter-sized paper", "paper", 0.06),
    ("Cardstock", "paper", 0.15),
    ("Colored paper", "paper", 0.10),
    ("Glossy paper", "paper", 0.20),
    ("Matte paper", "paper", 0.18),
    ("Recycled paper", "paper", 0.08),
    ("Eco-friendly paper", "paper", 0.12),
    ("Poster paper", "paper", 0.25),
    ("Banner paper", "paper", 0.30),
    ("Kraft paper", "paper", 0.10),
    ("Construction paper", "paper", 0.07),
    ("Wrapping paper", "paper", 0.15),
    ("Glitter paper", "paper", 0.22),
    ("Decorative paper", "paper", 0.18),
    ("Letterhead paper", "paper", 0.12),
    ("Legal-size paper", "paper", 0.08),
    ("Crepe paper", "paper", 0.05),
    ("Photo paper", "paper", 0.25),
    ("Uncoated paper", "paper", 0.06),
    ("Butcher paper", "paper", 0.10),
    ("Heavyweight paper", "paper", 0.20),
    ("Standard copy paper", "paper", 0.04),
    ("Bright-colored paper", "paper", 0.12),
    ("Patterned paper", "paper", 0.15),
    ("Paper plates", "product", 0.10),
    ("Paper cups", "product", 0.08),
    ("Paper napkins", "product", 0.02),
    ("Disposable cups", "product", 0.10),
    ("Table covers", "product", 1.50),
    ("Envelopes", "product", 0.05),
    ("Sticky notes", "product", 0.03),
    ("Notepads", "product", 2.00),
    ("Invitation cards", "product", 0.50),
    ("Flyers", "product", 0.15),
    ("Party streamers", "product", 0.05),
    ("Decorative adhesive tape (washi tape)", "product", 0.20),
    ("Paper party bags", "product", 0.25),
    ("Name tags with lanyards", "product", 0.75),
    ("Presentation folders", "product", 0.50),
    ("Large poster paper (24x36 inches)", "large_format", 1.00),
    ("Rolls of banner paper (36-inch width)", "large_format", 2.50),
    ("100 lb cover stock", "specialty", 0.50),
    ("80 lb text paper", "specialty", 0.40),
    ("250 gsm cardstock", "specialty", 0.30),
    ("220 gsm poster paper", "specialty", 0.35),
]

paper_supplies = [
    {
        "item_name": name,
        "category": category,
        "unit_price": price,
    }
    for name, category, price in CATALOG_ROWS
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def generate_sample_inventory(
    paper_supplies: list,
    coverage: float = 0.4,
    seed: int = 137,
) -> pd.DataFrame:
    """Generate a seeded subset of the catalog with starting stock quantities."""
    np.random.seed(seed)

    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False,
    )
    selected_items = [paper_supplies[i] for i in selected_indices]

    inventory = []

    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),
                "min_stock_level": np.random.randint(50, 150),
            }
        )

    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """Reset the demo database and load historical requests, quotes, and stock."""
    try:
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],
                "units": [],
                "price": [],
                "transaction_date": [],
            }
        )
        transactions_schema.to_sql(
            "transactions",
            db_engine,
            if_exists="replace",
            index=False,
        )

        initial_date = datetime(2025, 1, 1).isoformat()

        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests",
            db_engine,
            if_exists="replace",
            index=False,
        )

        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda value: (
                    ast.literal_eval(value)
                    if isinstance(value, str)
                    else value
                )
            )

            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda value: value.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda value: value.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda value: value.get("event_type", "")
            )

        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql(
            "quotes",
            db_engine,
            if_exists="replace",
            index=False,
        )

        inventory_df = generate_sample_inventory(
            paper_supplies,
            seed=seed,
        )

        # Opening cash is represented by a sale without an inventory item.
        initial_transactions = [
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        ]

        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        pd.DataFrame(initial_transactions).to_sql(
            "transactions",
            db_engine,
            if_exists="append",
            index=False,
        )

        inventory_df.to_sql(
            "inventory",
            db_engine,
            if_exists="replace",
            index=False,
        )

        return db_engine

    except Exception as exc:
        print(f"Error initializing database: {exc}")
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """Record a purchase or sale in the local transaction ledger."""
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date

        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError(
                "Transaction type must be 'stock_orders' or 'sales'"
            )

        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        transaction.to_sql(
            "transactions",
            db_engine,
            if_exists="append",
            index=False,
        )

        result = pd.read_sql(
            "SELECT last_insert_rowid() AS id",
            db_engine,
        )

        return int(result.iloc[0]["id"])

    except Exception as exc:
        print(f"Error creating transaction: {exc}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """Return positive transaction-derived stock through the supplied date."""
    query = """
        SELECT
            item_name,
            SUM(
                CASE
                    WHEN transaction_type = 'stock_orders' THEN units
                    WHEN transaction_type = 'sales' THEN -units
                    ELSE 0
                END
            ) AS stock
        FROM transactions
        WHERE item_name IS NOT NULL
          AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    result = pd.read_sql(
        query,
        db_engine,
        params={"as_of_date": as_of_date},
    )

    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(
    item_name: str,
    as_of_date: Union[str, datetime],
) -> pd.DataFrame:
    """Return transaction-derived stock for one item through the supplied date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    stock_query = """
        SELECT
            item_name,
            COALESCE(
                SUM(
                    CASE
                        WHEN transaction_type = 'stock_orders' THEN units
                        WHEN transaction_type = 'sales' THEN -units
                        ELSE 0
                    END
                ),
                0
            ) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
          AND transaction_date <= :as_of_date
    """

    return pd.read_sql(
        stock_query,
        db_engine,
        params={
            "item_name": item_name,
            "as_of_date": as_of_date,
        },
    )


def get_supplier_delivery_date(
    input_date_str: str,
    quantity: int,
) -> str:
    """Estimate supplier arrival using quantity-based calendar-day lead times."""
    print(
        "FUNC (get_supplier_delivery_date): "
        f"Calculating for qty {quantity} from '{input_date_str}'"
    )

    try:
        input_date_dt = datetime.fromisoformat(
            input_date_str.split("T")[0]
        )
    except (ValueError, TypeError):
        print(
            "WARN (get_supplier_delivery_date): "
            f"Invalid date '{input_date_str}', using today as base."
        )
        input_date_dt = datetime.now()

    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    delivery_date_dt = input_date_dt + timedelta(days=days)

    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """Calculate recorded sales less purchases through the supplied date."""
    try:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        transactions = pd.read_sql(
            """
            SELECT *
            FROM transactions
            WHERE transaction_date <= :as_of_date
            """,
            db_engine,
            params={"as_of_date": as_of_date},
        )

        if transactions.empty:
            return 0.0

        total_sales = transactions.loc[
            transactions["transaction_type"] == "sales",
            "price",
        ].sum()

        total_purchases = transactions.loc[
            transactions["transaction_type"] == "stock_orders",
            "price",
        ].sum()

        return float(total_sales - total_purchases)

    except Exception as exc:
        print(f"Error getting cash balance: {exc}")
        return 0.0


def generate_financial_report(
    as_of_date: Union[str, datetime],
) -> Dict:
    """Report simulated cash, catalog-valued inventory, and top-selling items."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql(
        "SELECT * FROM inventory",
        db_engine,
    )

    inventory_value = 0.0
    inventory_summary = []

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(
            item["item_name"],
            as_of_date,
        )
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]

        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    top_sales_query = """
        SELECT
            item_name,
            SUM(units) AS total_units,
            SUM(price) AS total_revenue
        FROM transactions
        WHERE transaction_type = 'sales'
          AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """

    top_sales = pd.read_sql(
        top_sales_query,
        db_engine,
        params={"date": as_of_date},
    )

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_sales.to_dict(
            orient="records"
        ),
    }


def search_quote_history(
    search_terms: List[str],
    limit: int = 5,
) -> List[Dict]:
    """Search historical requests and quote explanations for keywords."""
    conditions = []
    params = {}

    for index, term in enumerate(search_terms):
        param_name = f"term_{index}"

        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    where_clause = " OR ".join(conditions) if conditions else "1=1"

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

    with db_engine.connect() as connection:
        result = connection.execute(
            text(query),
            params,
        )

        return [
            dict(row._mapping)
            for row in result
        ]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

dotenv.load_dotenv()

api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("UDACITY_OPENAI_API_KEY")
)

if not api_key:
    raise ValueError(
        "No API key was found. Set OPENAI_API_KEY or "
        "UDACITY_OPENAI_API_KEY in your environment or local .env file."
    )

default_api_base = (
    "https://api.openai.com/v1"
    if os.getenv("OPENAI_API_KEY")
    else "https://openai.vocareum.com/v1"
)

model = OpenAIServerModel(
    model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_base=os.getenv("OPENAI_BASE_URL") or default_api_base,
    api_key=api_key,
)


# ---------------------------------------------------------------------------
# Product resolution and pricing
# ---------------------------------------------------------------------------

CATALOG = {
    item["item_name"]: item
    for item in paper_supplies
}

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


def resolve_item_name(
    customer_item: str,
) -> Union[str, None]:
    """Convert customer wording into an exact catalog item name."""
    normalized = normalize_text(customer_item)

    for exact_name in CATALOG:
        if normalize_text(exact_name) == normalized:
            return exact_name

    if normalized in ITEM_ALIASES:
        return ITEM_ALIASES[normalized]

    for alias in sorted(
        ITEM_ALIASES,
        key=len,
        reverse=True,
    ):
        if alias in normalized:
            return ITEM_ALIASES[alias]

    return None


def get_bulk_discount(total_units: int) -> float:
    """Return the highest qualifying quantity discount."""
    if total_units >= 10000:
        return 0.15
    if total_units >= 5000:
        return 0.12
    if total_units >= 1000:
        return 0.10
    if total_units >= 500:
        return 0.05

    return 0.0


# ---------------------------------------------------------------------------
# Inventory Management Agent tools
# ---------------------------------------------------------------------------

@tool
def check_inventory(
    item_name: str,
    request_date: str,
) -> str:
    """
    Check inventory for a specific item.

    Args:
        item_name: Exact inventory item name.
        request_date: Inventory cutoff date in YYYY-MM-DD format.

    Returns:
        Inventory information as JSON.
    """
    stock = get_stock_level(
        item_name,
        request_date,
    )

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
        Inventory levels as JSON.
    """
    inventory = get_all_inventory(request_date)

    return json.dumps(
        inventory,
        indent=2,
    )


@tool
def check_supplier_timeline(
    request_date: str,
    shortage_quantity: int,
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
        shortage_quantity,
    )

    return json.dumps(
        {
            "request_date": request_date,
            "shortage_quantity": shortage_quantity,
            "supplier_delivery_date": delivery_date,
        },
        indent=2,
    )


@tool
def check_available_cash(request_date: str) -> str:
    """
    Check current cash balance.

    Args:
        request_date: Date used to evaluate cash position.

    Returns:
        Cash balance as JSON.
    """
    cash = get_cash_balance(request_date)

    return json.dumps(
        {"cash_balance": cash},
        indent=2,
    )


@tool
def get_company_report(request_date: str) -> str:
    """
    Generate a company financial report.

    Args:
        request_date: Date used for report generation.

    Returns:
        Financial report as JSON.
    """
    report = generate_financial_report(request_date)

    return json.dumps(
        report,
        default=str,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Quote Agent tools
# ---------------------------------------------------------------------------

@tool
def search_historical_quotes(search_terms: str) -> str:
    """
    Search historical quote records.

    Args:
        search_terms: Comma-separated search terms.

    Returns:
        Matching quote records as JSON.
    """
    terms = [
        term.strip()
        for term in search_terms.split(",")
        if term.strip()
    ]

    results = search_quote_history(terms)

    return json.dumps(
        results,
        default=str,
        indent=2,
    )


@tool
def estimate_quote(
    item_name: str,
    quantity: int,
) -> str:
    """
    Estimate preliminary pricing for one requested item.

    Args:
        item_name: Requested item.
        quantity: Requested quantity.

    Returns:
        Quote estimate as JSON.
    """
    item_name = resolve_item_name(item_name)

    if not item_name:
        return json.dumps(
            {"error": "Item not found"}
        )

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


# ---------------------------------------------------------------------------
# Ordering Agent tool and authoritative result
# ---------------------------------------------------------------------------

LAST_ORDER_RESULT = None
CURRENT_REQUEST_DATE = None


def record_order_result(result: dict) -> str:
    """Preserve the business-tool outcome and return it to the agent as JSON."""
    global LAST_ORDER_RESULT

    LAST_ORDER_RESULT = result

    return json.dumps(
        result,
        indent=2,
    )


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
        return record_order_result(
            {
                "status": "rejected",
                "reason": (
                    "The authoritative request date was not provided."
                ),
                "transactions_created": False,
            }
        )

    # The dataset date overrides dates supplied by an agent.
    request_date = CURRENT_REQUEST_DATE

    try:
        items = json.loads(items_json)
    except (json.JSONDecodeError, TypeError):
        return record_order_result(
            {
                "status": "rejected",
                "reason": "The submitted item list was invalid.",
                "transactions_created": False,
            }
        )

    if not isinstance(items, list) or not items:
        return record_order_result(
            {
                "status": "rejected",
                "reason": (
                    "The order did not contain a valid item list."
                ),
                "transactions_created": False,
            }
        )

    try:
        request_date_dt = datetime.fromisoformat(
            request_date.split("T")[0]
        )
        deadline_dt = datetime.fromisoformat(
            required_delivery_date.split("T")[0]
        )
    except (ValueError, TypeError):
        return record_order_result(
            {
                "status": "rejected",
                "reason": "The request or delivery date was invalid.",
                "transactions_created": False,
            }
        )

    if deadline_dt < request_date_dt:
        return record_order_result(
            {
                "status": "rejected",
                "reason": (
                    "The required delivery date is before the request date."
                ),
                "transactions_created": False,
            }
        )

    resolved_items = []
    unsupported_items = []
    subtotal = 0.0
    total_units = 0

    # Resolve and price every line before creating transactions.
    for item in items:
        customer_item = str(
            item.get("customer_item", "")
        ).strip()

        try:
            quantity = int(item.get("quantity", 0))
        except (TypeError, ValueError):
            quantity = 0

        if quantity <= 0:
            unsupported_items.append(
                {
                    "customer_item": customer_item,
                    "reason": (
                        "Quantity must be a positive integer."
                    ),
                }
            )
            continue

        exact_item_name = resolve_item_name(customer_item)

        if exact_item_name is None:
            unsupported_items.append(
                {
                    "customer_item": customer_item,
                    "reason": (
                        "No matching catalog item was found."
                    ),
                }
            )
            continue

        unit_price = float(
            CATALOG[exact_item_name]["unit_price"]
        )
        line_subtotal = round(
            quantity * unit_price,
            2,
        )

        resolved_items.append(
            {
                "customer_item": customer_item,
                "item_name": exact_item_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_subtotal": line_subtotal,
            }
        )

        subtotal += line_subtotal
        total_units += quantity

    # One unsupported line causes rejection of the complete order.
    if unsupported_items:
        return record_order_result(
            {
                "status": "rejected",
                "reason": (
                    "The complete order cannot be fulfilled because "
                    "one or more requested products are unsupported."
                ),
                "unsupported_items": unsupported_items,
                "resolved_items": resolved_items,
                "transactions_created": False,
            }
        )

    discount_rate = get_bulk_discount(total_units)
    discount_amount = round(
        subtotal * discount_rate,
        2,
    )
    total_price = round(
        subtotal - discount_amount,
        2,
    )

    # This simulation checks committed stock on the request date.
    inventory_snapshot = get_all_inventory(request_date)

    validations = []
    total_restock_cost = 0.0

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
                return record_order_result(
                    {
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
                    }
                )

            restock_cost = round(
                shortage * item["unit_price"],
                2,
            )
            total_restock_cost += restock_cost

        validations.append(
            {
                "item_name": item["item_name"],
                "quantity": item["quantity"],
                "available_stock": available_stock,
                "shortage": shortage,
                "supplier_delivery_date": supplier_delivery_date,
                "restock_cost": restock_cost,
            }
        )

    available_cash = get_cash_balance(request_date)

    if total_restock_cost > available_cash:
        return record_order_result(
            {
                "status": "rejected",
                "reason": (
                    "The required replenishment inventory could not "
                    "be obtained for this order."
                ),
                "available_cash": round(available_cash, 2),
                "required_restock_cost": round(
                    total_restock_cost,
                    2,
                ),
                "transactions_created": False,
            }
        )

    # Book approved commitments on the request date for cash reconciliation.
    # Supplier arrival is checked separately; this is not physical stock tracking.
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

            restock_transactions.append(
                {
                    "transaction_id": transaction_id,
                    "item_name": validation["item_name"],
                    "quantity": validation["shortage"],
                    "cost": validation["restock_cost"],
                    "delivery_date": (
                        validation["supplier_delivery_date"]
                    ),
                }
            )

    # Allocate discounted revenue; the last line receives the rounding remainder.
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

        sale_transactions.append(
            {
                "transaction_id": transaction_id,
                "item_name": item["item_name"],
                "quantity": item["quantity"],
                "sale_price": line_sale_price,
                "transaction_date": request_date,
            }
        )

    return record_order_result(
        {
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
        }
    )


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

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
12. The Ordering Agent and process_complete_order tool are the authoritative
    source for fulfillment decisions.
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
""",
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Workflow execution and customer responses
# ---------------------------------------------------------------------------

def sanitize_customer_response(order_result: dict) -> str:
    """Build a customer-safe message from the authoritative order result."""
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
    """Run the workflow and return its order result and customer message."""
    global LAST_ORDER_RESULT
    global CURRENT_REQUEST_DATE

    # Clear the previous request outcome before running the next workflow.
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


# ---------------------------------------------------------------------------
# Simulation evaluation
# ---------------------------------------------------------------------------

def validate_cash_reconciliation(results):
    """Validate each request against its recorded cash movements."""
    for row in results:
        request_id = row["request_id"]
        status = row["order_status"]

        cash_before = round(
            float(row["cash_before"]),
            2,
        )
        cash_after = round(
            float(row["cash_after"]),
            2,
        )
        cash_delta = round(
            cash_after - cash_before,
            2,
        )

        row["cash_delta"] = cash_delta

        if status == "rejected":
            if not str(
                row.get("rejection_reason", "")
            ).strip():
                raise ValueError(
                    f"Request {request_id} was rejected without a reason."
                )

            if row["transactions_created"]:
                raise ValueError(
                    f"Request {request_id} was rejected but reports "
                    "that transactions were created."
                )

            if cash_delta != 0:
                raise ValueError(
                    f"Request {request_id} was rejected "
                    f"but cash changed by ${cash_delta:.2f}."
                )

        elif status == "fulfilled":
            if not row["transactions_created"]:
                raise ValueError(
                    f"Request {request_id} was fulfilled but reports "
                    "that no transactions were created."
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
                f"Request {request_id} has an invalid order status: "
                f"{status}"
            )


def run_test_scenarios():
    """Reset the demo database, process sample requests, and save results."""
    print("Initializing Database...")
    init_database(db_engine)

    try:
        quote_requests_sample = pd.read_csv(
            "quote_requests_sample.csv"
        )

        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"],
            format="%m/%d/%y",
            errors="coerce",
        )

        quote_requests_sample.dropna(
            subset=["request_date"],
            inplace=True,
        )

        quote_requests_sample = (
            quote_requests_sample
            .sort_values("request_date")
            .reset_index(drop=True)
        )

    except Exception as exc:
        print(f"FATAL: Error loading test data: {exc}")
        return

    initial_date = (
        quote_requests_sample["request_date"]
        .min()
        .strftime("%Y-%m-%d")
    )

    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []

    for index, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime(
            "%Y-%m-%d"
        )

        print(f"\n=== Request {index + 1} ===")
        print(
            f"Context: {row['job']} organizing {row['event']}"
        )
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"""
        Customer role: {row['job']}
        Order size: {row['need_size']}
        Event: {row['event']}
        Request date: {request_date}

        Customer request:
        {row['request']}
        """.strip()

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
        print(
            f"Updated Inventory: ${current_inventory:.2f}"
        )

        results.append(
            {
                "request_id": len(results) + 1,
                "source_row_id": index + 1,
                "request_date": request_date,
                "job": row["job"],
                "need_size": row["need_size"],
                "event": row["event"],
                "original_request": row["request"],
                "order_status": order_result["status"],
                "transactions_created": bool(
                    order_result.get(
                        "transactions_created",
                        False,
                    )
                ),
                "rejection_reason": (
                    order_result.get("reason", "")
                    if order_result["status"] == "rejected"
                    else ""
                ),
                "total_price": round(
                    float(
                        order_result.get(
                            "total_price",
                            0.0,
                        )
                    ),
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

    latest_transaction = pd.read_sql(
        """
        SELECT MAX(transaction_date) AS final_date
        FROM transactions
        """,
        db_engine,
    )

    final_date = latest_transaction.iloc[0]["final_date"]
    final_report = generate_financial_report(final_date)

    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Report Date: {final_date}")
    print(
        f"Final Cash: ${final_report['cash_balance']:.2f}"
    )
    print(
        f"Final Inventory: ${final_report['inventory_value']:.2f}"
    )

    # Preserve diagnostic output even if reconciliation fails.
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "test_results_diagnostic.csv",
        index=False,
    )

    validate_cash_reconciliation(results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "test_results.csv",
        index=False,
    )

    print("\nCash reconciliation validation passed.")
    print("Simulation results written to test_results.csv.")

    return results


if __name__ == "__main__":
    results = run_test_scenarios()
