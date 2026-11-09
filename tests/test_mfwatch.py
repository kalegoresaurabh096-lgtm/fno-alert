import io

import pandas as pd

from fno_alert import mfwatch


def _xlsx(sheets: dict[str, list[list]]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)
    return buf.getvalue()


def _scheme(title, rows):
    return [["HDFC Mutual Fund"], [title], ["Portfolio as on 31-Oct-2026"], [],
            ["Name of the Instrument", "ISIN", "Industry", "Quantity", "Market/Fair Value (Rs. in Lacs.)", "% to NAV"],
            ["Equity & Equity related"]] + rows + [["Total"], ["TREPS", "", "", "", 100.0, 1.0],
                                                   ["Some Bond", "INE001A07ZZ1", "", 10, 50.0, 0.5]]


def test_parse_file_finds_equity_rows_and_scheme():
    data = _xlsx({"HDSC": _scheme("HDFC Small Cap Fund (An open ended equity scheme)", [
        ["Gabriel India Ltd", "INE524A01029", "Auto Components", 1000000, 4500.5, 1.25],
        ["Reliance Industries Ltd", "INE002A01018", "Oil", 10, 25.0, 0.01]])})
    h = mfwatch.to_frame(mfwatch.parse_file(data, "hdfc.xlsx", "HDFC"))
    assert list(h["isin"]) == ["INE524A01029", "INE002A01018"]  # bond and TREPS skipped
    assert h.iloc[0]["scheme"] == "HDFC Small Cap Fund"
    assert h.iloc[0]["qty"] == 1000000 and h.iloc[0]["pct_nav"] == 1.25


def test_new_positions_only_small_caps_new_to_the_fund_house():
    mcap = pd.DataFrame({"isin": ["INE524A01029", "INE002A01018", "INE111B01011"],
                         "rank": [600, 1, 400], "category": ["Small Cap", "Large Cap", "Small Cap"]})
    mcap = pd.concat([mcap, pd.DataFrame({"isin": ["INE777C01011"], "rank": [900], "category": ["Small Cap"]})])
    cols = ["amc", "scheme", "isin", "name", "qty", "value_lakh", "pct_nav"]
    prev = pd.DataFrame([["HDFC", "HDFC Flexi Cap", "INE111B01011", "Old Small", 5, 10, 0.1],
                         ["HDFC", "HDFC Small Cap", "INE999Z01010", "X", 5, 10, 0.1],
                         ["HDFC", "HDFC Nifty Smallcap 250 ETF", "INE999Z01010", "X", 5, 10, 0.1],
                         ["SBI", "SBI Small Cap", "INE999Z01010", "X", 1, 1, 0.1],
                         ["SBI", "SBI Contra", "INE999Z01010", "X", 1, 1, 0.1]], columns=cols)
    cur = pd.DataFrame([["HDFC", "HDFC Small Cap", "INE524A01029", "Gabriel", 10, 4500, 1.2],
                        ["HDFC", "HDFC Small Cap", "INE111B01011", "Old Small", 9, 20, 0.2],  # already held
                        ["HDFC", "HDFC Top 100", "INE002A01018", "Reliance", 9, 20, 0.2],      # large cap
                        ["SBI", "SBI Small Cap", "INE524A01029", "Gabriel", 3, 900, 0.4],
                        ["Quant", "Quant Small Cap", "INE524A01029", "Gabriel", 3, 900, 0.4],  # no prior file
                        ["HDFC", "HDFC Nifty Smallcap 250 ETF", "INE777C01011", "Index add", 3, 900, 0.4],
                        ["HDFC", "HDFC New NFO Fund", "INE777C01011", "Index add", 3, 900, 0.4],  # new scheme
                        ["SBI", "SBI Contra", "INE777C01011", "Token", 3, 50, 0.05]],  # token position
                       columns=cols)
    new = mfwatch.new_positions(prev, cur, mcap)
    assert set(new["amc"]) == {"HDFC", "SBI"}
    assert set(new["isin"]) == {"INE524A01029"}
    assert (new["n_amcs"] == 2).all()
    subject, text, body = mfwatch.render_email(new, "Oct 2026", ["HDFC", "SBI"], ["Kotak"])
    assert "1 stock" in subject and "IPO" not in subject and "Gabriel" in text and "Kotak" in text


def test_equity_isin():
    assert mfwatch.is_equity_isin("INE524A01029")
    assert not mfwatch.is_equity_isin("INE001A07ZZ1")
    assert not mfwatch.is_equity_isin("INF209K01VA1")


def test_month_of_links():
    assert mfwatch.month_of("monthend-portfolio-july-31st-2023.zip") == "2023-07"
    assert mfwatch.month_of("31102026_abslmf_monthlydisclosure.zip") == "2026-10"
    assert mfwatch.month_of("NIMF-MONTHLY-PORTFOLIO-30-Sep-26.xls") == "2026-09"
    assert mfwatch.month_of("Portfolio as on 31st October, 2026") == "2026-10"


def test_xlsx_named_xls_is_read():
    data = _xlsx({"S1": _scheme("PPFAS Flexi Cap Fund", [["Gabriel", "INE524A01029", "Auto", 5, 9.0, 0.4]])})
    assert len(mfwatch.parse_file(data, "PPFAS_Monthly_Portfolio.xls", "PPFAS")) == 1


def test_isin_changed_after_split_matched_by_name():
    cols = ["amc", "scheme", "isin", "name", "qty", "value_lakh", "pct_nav"]
    prev = pd.DataFrame([["SBI", "SBI Contra", "INE999Z01010", "X", 1, 1, 0.1]], columns=cols)
    cur = pd.DataFrame([["SBI", "SBI Contra", "INE0B9K01033", "BANSAL WIRE INDUSTRIES LTD", 3, 900, 0.4]], columns=cols)
    mcap = pd.DataFrame({"isin": ["INE0B9K01025"], "name": ["Bansal Wire Industries Limited"],
                         "rank": [700], "category": ["Small Cap"]})
    new = mfwatch.new_positions(prev, cur, mcap)
    assert new.iloc[0]["mcap_rank"] == 700
