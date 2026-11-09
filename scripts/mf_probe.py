"""One-off: see which fund-house portfolio pages GitHub's runners can reach, and what file links they show."""
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36", "Accept-Language": "en-IN,en;q=0.9"}
PAGES = {
    "amfi_portal": "https://www.amfiindia.com/online-center/portfolio-disclosure",
    "amfi_mcap_xlsx": "https://portal.amfiindia.com/spages/AverageMarketCapitalization30Jun2026.xlsx",
    "amfi_mcap_pdf": "https://portal.amfiindia.com/spages/AverageMarketCapitalization30Jun2026.pdf",
    "hdfc": "https://www.hdfcfund.com/statutory-disclosure/portfolio/monthly-portfolio",
    "sbi": "https://www.sbimf.com/portfolios",
    "nippon": "https://mf.nipponindiaim.com/investor-service/downloads/factsheet-portfolio-and-other-disclosures",
    "axis": "https://www.axismf.com/statutory-disclosures",
    "kotak": "https://www.kotakmf.com/Information/portfolios",
    "icici": "https://www.icicipruamc.com/downloads/others/monthly-portfolio-disclosures",
    "dsp": "https://www.dspim.com/mandatory-disclosures/portfolio-disclosures",
    "tata": "https://www.tatamutualfund.com/schemes-related/portfolio",
    "quant": "https://quantmutual.com/statutory-disclosures",
    "motilal": "https://www.motilaloswalmf.com/download/month-end-portfolio",
    "ppfas": "https://amc.ppfas.com/downloads/portfolio-disclosure/",
    "bandhan": "https://bandhanmutual.com/downloads/disclosures",
    "canara": "https://www.canararobeco.com/statutory-disclosures/scheme-dashboard/scheme-portfolios",
    "invesco": "https://invescomutualfund.com/literature-and-form?tab=Complete",
    "franklin": "https://www.franklintempletonindia.com/reports",
    "absl": "https://mutualfund.adityabirlacapital.com/forms-and-downloads/portfolio",
    "mirae": "https://www.miraeassetmf.co.in/downloads/portfolios",
    "uti": "https://www.utimf.com/downloads/consolidate-all-portfolio-disclosure",
    "edelweiss": "https://www.edelweissmf.com/statutory/portfolio-of-schemes",
    "hsbc": "https://www.assetmanagement.hsbc.co.in/en/mutual-funds/investor-resources/information-library",
    "sundaram": "https://www.sundarammutual.com/Monthly-Portfolio",
    "whiteoak": "https://mf.whiteoakamc.com/regulatory-disclosures/scheme-portfolios",
}
for name, url in PAGES.items():
    try:
        r = requests.get(url, headers=UA, timeout=30, allow_redirects=True)
        body = r.text if "text" in r.headers.get("content-type", "") or "json" in r.headers.get("content-type", "") else ""
        links = sorted(set(re.findall(r'["\'(]([^"\'()\s]+?\.(?:xlsx?|zip)(?:\?[^"\'\s]*)?)["\')]', body, re.I)))
        apis = sorted(set(re.findall(r'["\'](https?://[^"\']*api[^"\']*)["\']', body, re.I)))[:8]
        print(f"=== {name} {r.status_code} {r.headers.get('content-type','')} {len(r.content)}B final={r.url}")
        for l in links[:25]:
            print("   file:", l)
        for a in apis:
            print("   api:", a)
    except Exception as e:  # noqa: BLE001
        print(f"=== {name} ERROR {type(e).__name__}: {e}")
