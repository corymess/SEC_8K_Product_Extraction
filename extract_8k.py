# Cory Messer
# NID: co195482
# E-mail: co195482@ucf.edu
# CAP 5619 - Artificial Intelligence for Fintech
# Fall 2026
# Date 10/28/2026

"""
extract_8k.py
Extracts new product announcements from SEC 8-K filings using a local LLM (Ollama).
Author: Cory Messer
"""

import requests
import re
import time
import warnings
import json
import ollama
from io import StringIO
import pandas as pd
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # iXBRL 8-K files are XML-like

from bs4 import BeautifulSoup

# The SEC rejects requests without a User-Agent that includes a name and email
HEADERS = {"User-Agent": "Cory Messer co195482@ucf.edu"}

# Words that suggest a sentence is about a new product (lowercase, partial words on purpose)
PRODUCT_KEYWORDS = ["launch", "introduc", "unveil", "new product", "now available",
                    "next-generation", "fda approv"]
# Only these 8-K items are likely to mention products: earnings, Reg FD, other events
USEFUL_ITEMS = ["2.02", "7.01", "8.01"]

MODEL_NAME = "llama3.2"

GENERIC_NAMES = {"new products", "new product", "products", "product", "none", ""}
MAX_DESCRIPTION = 179  # the assignment requires fewer than 180 characters

TARGET_ROWS = 150  # above the required 100, in case some rows need removing
FILINGS_PER_COMPANY = 10
CSV_FILE = "new_products.csv"
TABLE_FILE = "new_products_table.txt"

# Forces the LLM to answer in exactly this JSON structure
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "product": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["company", "product", "description"],
            },
        }
    },
    "required": ["products"],
}

PROMPT = """You are a named entity recognition (NER) system that reads excerpts from SEC 8-K filings.
List every product the excerpt says was recently launched, introduced, unveiled, released, or approved.

Rules:
- company is the ORGANIZATION that filed the document.
- product is the PRODUCT name, such as "MacBook Pro". Software and medicines count as products.
- Never use a general phrase like "new products" as a product name.
- Skip products described as company history.
- The excerpt may mix in financial comments. Ignore those sentences and still list the products.
- If no product qualifies, return an empty list.

Example excerpt: Acme launched the RoadRunner X scooter and received FDA approval of Zanthrix for migraines. Acme invented the rocket skate in 1950.
Example answer: {{"products": [{{"company": "Acme", "product": "RoadRunner X", "description": "A new scooter."}}, {{"company": "Acme", "product": "Zanthrix", "description": "A medicine for migraines."}}]}}

Company: {company}
Excerpt: {text}"""

def get_cik_map():
    """Download the SEC ticker list and return {ticker: {"cik": ..., "name": ...}}."""
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()  # stop with an error if the request failed
    data = response.json()

    cik_map = {}
    for entry in data.values():
        cik_map[entry["ticker"]] = {
            "cik": str(entry["cik_str"]).zfill(10),  # pad to 10 digits for EDGAR URLs
            "name": entry["title"],
        }
    return cik_map

def get_8k_filings(cik, count=10):
    """Return a list of a company's most recent 8-K filings from the EDGAR Atom feed."""
    search_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                  f"&CIK={cik}&type=8-K&count={count}&output=atom")
    response = requests.get(search_url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")  # Atom feeds are XML

    filings = []
    for entry in soup.find_all("entry"):
        items = entry.find("items-desc")  # some entries leave this out
        filings.append({
            "filing_time": entry.find("updated").text,
            "filing_url": entry.find("filing-href").text,
            "items": items.text if items else "",
        })
    return filings[:count]  # EDGAR returns at least 10, so trim to what was asked for

def get_filing_documents(filing_url):
    """Open a filing's index page and return URLs for the main 8-K and any EX-99 exhibits."""
    time.sleep(0.2)  # stay under the SEC's limit of 10 requests per second
    response = requests.get(filing_url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")

    doc_urls = []
    table = soup.find("table", class_="tableFile")  # first table lists the documents
    if table is None:
        return doc_urls

    for row in table.find_all("tr")[1:]:  # [1:] skips the header row
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].text.strip()  # the Type column
        link = cells[2].find("a")         # the Document column
        if link is None:
            continue
        href = link["href"].replace("/ix?doc=", "")  # remove the online viewer prefix
        if (doc_type.startswith("8-K") or doc_type.startswith("EX-99")) and href.endswith((".htm", ".html")):
            doc_urls.append("https://www.sec.gov" + href)
    return doc_urls


def get_document_text(doc_url):
    """Download one filing document and return it as clean plain text."""
    time.sleep(0.2)
    response = requests.get(doc_url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()  # collapse extra spaces and newlines

def is_useful_filing(items):
    """Return True if the filing covers an 8-K item that may mention products."""
    return any(item in items for item in USEFUL_ITEMS)


def extract_product_sentences(text, max_sentences=15):
    """Return a list of the sentences that contain a product keyword."""
    sentences = re.split(r"(?<=[.!?])\s+", text)  # split after . ! or ?
    kept = []
    for sentence in sentences:
        if len(sentence) < 600 and any(word in sentence.lower() for word in PRODUCT_KEYWORDS):
            kept.append(sentence)
    return kept[:max_sentences]

def extract_products_with_llm(company_name, sentences):
    """Ask the local LLM about each sentence and return all products found as a list of dicts."""
    products = []
    for sentence in sentences:
        prompt = PROMPT.format(company=company_name, text=sentence)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=PRODUCT_SCHEMA,
            options={"temperature": 0},  # 0 = most consistent answers, no creativity
        )
        for item in json.loads(response["message"]["content"])["products"]:
            item["source"] = sentence  # keep the original sentence as a backup description
            products.append(item)
    return products

def clean_products(products):
    """Remove generic names and repeats, and make sure descriptions fit the limit."""
    best = {}
    for item in products:
        key = item["product"].strip().lower()  # so "EBGLYSS" and "Ebglyss" count as the same
        if key in GENERIC_NAMES:
            continue
        if key not in best or len(item["description"]) > len(best[key]["description"]):
            best[key] = item

    cleaned = []
    for item in best.values():
        description = item["description"].strip()
        if len(description.split()) < 5:  # too weak, so use the filing's own sentence
            description = item["source"]
        if len(description) > MAX_DESCRIPTION:
            description = description[:MAX_DESCRIPTION - 3].rstrip() + "..."
        cleaned.append({"product": item["product"].strip(), "description": description})
    return cleaned

def get_sp500_tickers():
    """Return the current S&P 500 tickers from Wikipedia, in the SEC's format."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]  # the first table on the page
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B


def save_results(rows):
    """Write the rows to the CSV file and to a | separated table."""
    results = pd.DataFrame(rows, columns=["company_name", "stock_name", "filing_time",
                                          "new_product", "product_description"])
    results.to_csv(CSV_FILE, index=False)
    results.to_csv(TABLE_FILE, index=False, sep="|")

if __name__ == "__main__":
    cik_map = get_cik_map()
    rows = []
    companies_checked = 0
    filings_checked = 0
    documents_read = 0

    for ticker in get_sp500_tickers():
        if len(rows) >= TARGET_ROWS:
            break
        company = cik_map.get(ticker)  # .get returns None instead of crashing
        if company is None:
            continue

        companies_checked += 1
        seen_products = set()  # products already recorded for this company
        try:
            for filing in get_8k_filings(company["cik"], count=FILINGS_PER_COMPANY):
                filings_checked += 1
                if not is_useful_filing(filing["items"]):
                    continue

                all_text = ""
                for url in get_filing_documents(filing["filing_url"]):
                    all_text += get_document_text(url) + " "
                    documents_read += 1

                product_sentences = extract_product_sentences(all_text)
                if not product_sentences:
                    continue

                products = extract_products_with_llm(company["name"], product_sentences)
                for item in clean_products(products):
                    product_key = item["product"].lower()
                    if product_key in seen_products:
                        continue
                    seen_products.add(product_key)
                    rows.append([company["name"], ticker, filing["filing_time"],
                                 item["product"], item["description"]])
        except Exception as error:  # one bad company should not stop the whole run
            print(f"{ticker} failed: {error}")

        save_results(rows)  # save after every company so a crash loses nothing
        print(f"{ticker}: {len(rows)} rows so far")

    print(f"\nDone. Companies: {companies_checked}, filings: {filings_checked}, "
          f"documents read: {documents_read}, rows: {len(rows)}")
