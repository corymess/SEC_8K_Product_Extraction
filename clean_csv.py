"""
clean_csv.py
Reviews the raw LLM output: fixes text problems and removes rows found wrong by manual review.
Author: Cory Messer
"""

import re

import pandas as pd

RAW_FILE = "new_products_raw.csv"
CSV_FILE = "new_products.csv"
TABLE_FILE = "new_products_table.txt"

# (ticker, product) pairs removed after manual review, grouped by reason
ROWS_TO_REMOVE = {
    "not a product": [("AIZ", "U.S."),
                      ("AXP", "Fixed Rate Reset Noncumulative Preferred Shares, Series E")],
    "wrong details": [("XYZ", "Cash App"), ("XYZ", "Afterpay"), ("AMZN", "Amazon One Medical"),
                      ("AFL", "cancer product")],
    "generic name": [("AFL", "cancer insurance product"), ("AMAT", "six new chipmaking systems"),
                     ("AMAT", "new deposition and etch systems"),
                     ("AMAT", "new deposition, etch and materials modification systems"),
                     ("AMAT", "3D Gate-All-Around transistors"), ("ANET", "AI networking innovations"),
                     ("ANET", "AI-driven enterprise products"), ("ANET", "universal AI spine")],
    "brand family, not a specific product": [("AMZN", "Echo"), ("AMZN", "Kindle"),
                                             ("AMZN", "Amazon Alexa Tablets"), ("AMZN", "Fire TV"),
                                             ("AMZN", "Ring"), ("AMZN", "Blink"), ("AMZN", "Bee")],
    "not new": [("ABT", "Cologuard"), ("ABT", "Cancerguard"), ("AFL", "Miraito"),
                ("AFL", "Anshin Palette"), ("BIIB", "SKYCLARYS"), ("BIIB", "ZURZUVAE"),
                ("AXON", "AI Era Plan")],
    "near-repeat": [("ABNB", "Airbnb app"), ("AMD", "6th Gen EPYC CPUs"), ("AMD", "AMD Instinct MI350P"),
                    ("AMAT", "Nokota VMax 2"), ("ANET", "CloudVision")],
}


def fix_description(description):
    """Remove leftover text that comes before a bullet, along with the bullet itself."""
    return re.split(r"[•◦▪]\s*", description, maxsplit=1)[-1].strip()


if __name__ == "__main__":
    results = pd.read_csv(RAW_FILE)
    print(f"Raw rows: {len(results)}")

    for reason, pairs in ROWS_TO_REMOVE.items():
        keep = [(ticker, product) not in pairs
                for ticker, product in zip(results["stock_name"], results["new_product"])]
        print(f"Removed {len(results) - sum(keep)} rows: {reason}")
        results = results[keep]

    results["new_product"] = results["new_product"].str.replace("\\u00ae", "®", regex=False)
    results["product_description"] = results["product_description"].apply(fix_description)

    results.to_csv(CSV_FILE, index=False)
    results.to_csv(TABLE_FILE, index=False, sep="|")
    print(f"Final rows: {len(results)}, longest description: "
          f"{results['product_description'].str.len().max()}")
