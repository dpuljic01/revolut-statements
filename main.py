import datetime
import decimal
import io
import os
from multiprocessing import Pool

import camelot
import numpy as np
import pandas as pd
import requests
import simplejson as json  # this one handles decimals correctly
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, Response

app = FastAPI(
    title="Revolut Statement",
    description="Extracting transactions from revolut pdf statements into json/csv/xlsx",
    version="2.5.0",
)


def get_pdf(path, pages="all", **kwargs):
    # for smoother borderless table detection
    kwargs.update({"flavor": "stream"})
    return camelot.read_pdf(path, pages=pages, **kwargs)


def filter_data(trades):
    new_data = []
    keywords = ["BALANCES", "HOLDINGS"]
    data = trades.get("data")
    if not data:
        return trades
    should_write = False
    for trade in data:
        if "ACTIVITY" in trade and not should_write:
            should_write = True
        if any(keywords) in trade:
            should_write = False
        if should_write:
            new_data.append(trade)
    trades["data"] = new_data if new_data else data
    return trades


def filter_tables(tables):
    valid_tables = []
    for i in range(len(tables)):
        if tables[i].df[0].iloc[0] != "ACTIVITY":
            continue
        tables[i].df.columns = tables[i].df.iloc[1]
        tables[i].df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        tables[i].df.dropna(thresh=4, inplace=True)
        tables[i].df = tables[i].df.iloc[1:, :]
        valid_tables.append(tables[i])
    return valid_tables


def to_decimal(field, precision=None):
    if precision:
        decimal.getcontext().prec = precision
    if not field:
        return 0.0
    if not isinstance(field, str):
        return decimal.Decimal(field)
    if "(" in field:
        field = field.replace("(", "-").replace(")", "")
    field = field.replace(",", "")
    return decimal.Decimal(field)


def table_records_to_json(tables):
    trades = {}
    index = 1
    for table in tables:
        if table.df.empty:
            continue
        json_table = json.loads(table.df.to_json(orient="index"))
        json_table_uq = {}
        for _, transaction in json_table.items():
            if transaction["Trade Date"] == "Trade Date":
                continue
            transaction["Symbol"] = (
                transaction.pop("Symbol / Description").split("-")[0].strip()
            )
            transaction["Quantity"] = to_decimal(transaction["Quantity"])
            transaction["Price"] = to_decimal(transaction["Price"])
            transaction["Amount"] = to_decimal(transaction["Amount"])
            commission = 0
            if transaction["Activity Type"] in ["BUY", "SELL"]:
                # statements do not contain commissions, but (quantity * price - amount)
                # should give us commission (at least it looks like it)
                commission = abs(
                    abs(transaction["Quantity"] * transaction["Price"])
                    - abs(transaction["Amount"])
                )
                commission = 0 if commission < 0.01 else commission
            transaction["Commission"] = commission
            transaction["Shares"] = to_decimal(transaction.pop("Quantity"))
            json_table_uq[index] = transaction
            index += 1
        trades.update(json_table_uq)
    return trades


def extract_one_statement(filename):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir_name, "statements", filename)
    tables = get_pdf(file, pages="4-end")  # we don't need first 3 pages
    if not tables.n:
        return

    tables = filter_tables(tables)
    json_trades = table_records_to_json(tables)
    statement = {"filename": filename, "trades": json_trades}
    return statement


def extract_statements():
    files = os.listdir("statements")
    pool = Pool(5)
    statements = pool.map(extract_one_statement, files)
    return statements


@app.get("/exchange")
async def exchange_currency(
    date: str = datetime.datetime.now().date(),
    quantity: decimal.Decimal = 100.0,
    from_: str = "USD",
    to_: str = "HRK",
):
    from_ = from_.upper()
    to_ = to_.upper()
    resp = requests.get(f"https://api.exchangeratesapi.io/{date}?base={from_}")
    if not resp.json().get("rates"):
        return Response(status_code=400)

    result = quantity * decimal.Decimal(resp.json()["rates"][to_])
    resp = {
        "date": date,
        from_.lower(): quantity,
        to_.lower(): result,
    }
    return resp


@app.get("/export")
async def export_revolut_statements(
    format_: str = Query("json", regex="^(json|xlsx|csv)$"),
    filename: str = "report",
):
    statements = extract_statements()
    if format_ == "json":
        return statements
    if format_ == "xlsx":
        writer = pd.ExcelWriter(f"{filename}.xlsx")
        for i in range(len(statements)):
            df = pd.read_json(
                json.dumps(statements[i]["trades"], for_json=True),
                orient="index",
                precise_float=True,
            )
            # max length is 32 so I can't put the whole filename
            sheet_title = f"Sheet_{statements[i]['filename'].split('-')[0]}"
            df.to_excel(writer, sheet_title)
        writer.save()
        return Response("", 204)

    stream = io.StringIO()
    index = 1
    trades = {}
    for statement in statements:
        for _, transaction in statement["trades"].items():
            trades[index] = transaction
            index += 1
    df = pd.read_json(
        json.dumps(trades, for_json=True), orient="index", precise_float=True
    )
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}.csv"
    return response
