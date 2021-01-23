import datetime
import decimal
import io
import os
from functools import partial
from multiprocessing import Pool
import camelot
import numpy as np
import pandas as pd
import requests
import simplejson as json  # this one handles decimals correctly
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, Response
import logging

log = logging.getLogger()
app = FastAPI(
    title="Revolut Statement",
    description="Extracting transactions from revolut pdf statements into `json/csv/xlsx`",
    version="2.5.0",
)


def get_pdf(path, pages="all"):
    params = {"flavor": "stream", "edge_tol": 500}
    return camelot.read_pdf(path, pages=pages, **params)


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
        activity_ix = tables[i].df.index[tables[i].df[0] == "ACTIVITY"].to_list()
        activity_ix = int(activity_ix[0]) if activity_ix else None

        sweep_activity_ix = (
            tables[i].df.index[tables[i].df[0] == "SWEEP ACTIVITY"].to_list()
        )
        sweep_activity_ix = int(sweep_activity_ix[0]) if sweep_activity_ix else None

        # SELECT ITEMS BETWEEN ACTIVITY AND SWEEP ACTIVITY
        # IF SWEEP ACTIVITY IS NULL SELECT ALL FROM ACTIVITY TO THE END
        # IF ACTIVITY IS NULL SKIP
        if activity_ix is None:
            continue

        tables[i].df.columns = tables[i].df.iloc[activity_ix + 1]
        tables[i].df = (
            tables[i].df.iloc[activity_ix:sweep_activity_ix, :]
            if sweep_activity_ix is not None
            else tables[i].df.iloc[activity_ix:, :]
        )
        tables[i].df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        tables[i].df.dropna(thresh=4, inplace=True)
        tables[i].df = tables[i].df.loc[:, tables[i].df.columns.notnull()]
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


def table_records_to_json(tables, buy_sell_only):
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
            if buy_sell_only and transaction["Activity Type"] not in ["BUY", "SELL"]:
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
                commission = 0 if commission < 0.01 else round(commission, 2)
            transaction["Commission"] = commission
            transaction["Shares"] = to_decimal(transaction.pop("Quantity"))
            json_table_uq[index] = transaction
            index += 1
        trades.update(json_table_uq)
    return trades


def extract_one_statement(filename, buy_sell_only):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir_name, "statements", filename)
    tables = get_pdf(file, pages="4-end")  # we don't need first 3 pages
    if not tables.n:
        log.warning(f"{filename} has no tables detected.")
        return

    tables = filter_tables(tables)
    json_trades = table_records_to_json(tables, buy_sell_only)
    date_ = json_trades[1]["Trade Date"].split("/")
    statement = {
        "filename": filename,
        "month": int(date_[0]),
        "year": int(date_[2]),
        "trades": json_trades,
    }
    return statement


def extract_statements(filename, buy_sell_only):
    files = os.listdir("statements")

    statements = []
    if filename == "all" or not filename:
        with Pool(12) as pool:
            statements = pool.map(
                partial(extract_one_statement, buy_sell_only=buy_sell_only), files
            )
            statements = [s for s in statements if s]
        sorted_statements = (
            sorted(statements, key=lambda s: s["month"])
            if len(statements) > 1
            else statements
        )
        return sorted_statements

    if filename in files:
        statements.append(extract_one_statement(filename, buy_sell_only))

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
    format_: str = Query(
        "json", alias="format", regex="^(json|xlsx|csv)$", description="Output format."
    ),
    filename: str = Query(
        "all",
        description="Filename of single statement you want to export. e.g: `statement.pdf`",
    ),
    output_filename: str = Query(
        "report",
        description="Filename of exported document.",
    ),
    buy_sell_only: bool = Query(
        default=False, description="Export only `BUY` and `SELL` orders."
    ),
):
    statements = extract_statements(filename, buy_sell_only)
    if not statements:
        return Response(status_code=404)

    if format_ == "json":
        return statements

    stream = io.BytesIO()
    media_type = "text/csv"
    if format_ == "xlsx":
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        writer = pd.ExcelWriter(stream, engine="xlsxwriter")
        for i in range(len(statements)):
            df = pd.read_json(
                json.dumps(statements[i]["trades"]),
                orient="index",
                precise_float=True,
            )
            # max length is 32 so I can't put the whole filename
            sheet_name = f"Sheet_{statements[i]['month']}_{statements[i]['year']}"
            df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
    else:
        index = 1
        trades = {}
        for statement in statements:
            for transaction in statement["trades"].values():
                trades[index] = transaction
                index += 1
        df = pd.read_json(json.dumps(trades), orient="index", precise_float=True)
        df.to_csv(stream)

    stream.seek(0)
    output = iter([stream.getvalue()])
    response = StreamingResponse(output, media_type=media_type)
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename={output_filename}.{format_}"
    return response
