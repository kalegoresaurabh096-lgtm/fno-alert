"""One-off: open fund-house portfolio pages in a real browser on GitHub's runners and list
recent portfolio file links (pages that build their lists with JavaScript included)."""
import re
from playwright.sync_api import sync_playwright

PAGES = {
    "hdfc": "https://www.hdfcfund.com/statutory-disclosure/portfolio/monthly-portfolio",
    "sbi": "https://www.sbimf.com/portfolios",
    "nippon": "https://mf.nipponindiaim.com/investor-service/downloads/factsheet-portfolio-and-other-disclosures",
    "axis": "https://www.axismf.com/statutory-disclosures",
    "kotak": "https://www.kotakmf.com/Information/portfolios",
    "icici": "https://www.icicipruamc.com/downloads/others/monthly-portfolio-disclosures",
    "dsp": "https://www.dspim.com/mandatory-disclosures/portfolio-disclosures",
    "tata": "https://www.tatamutualfund.com/schemes-related/portfolio",
    "quant": "https://quantmutual.com/statutory-disclosures",
    "motilal": "https://www.motilaloswalmf.com/downloads/scheme-portfolio-details",
    "ppfas": "https://amc.ppfas.com/downloads/portfolio-disclosure/",
    "bandhan": "https://bandhanmutual.com/downloads/disclosures",
    "canara": "https://www.canararobeco.com/statutory-disclosures/scheme-dashboard/scheme-portfolios",
    "invesco": "https://www.invescomutualfund.com/literature-forms/monthly-holdings/equity",
    "franklin": "https://www.franklintempletonindia.com/reports",
    "absl": "https://mutualfund.adityabirlacapital.com/forms-and-downloads/portfolio",
    "mirae": "https://www.miraeassetmf.co.in/downloads/portfolio",
    "uti": "https://www.utimf.com/downloads/consolidate-all-portfolio-disclosure",
    "edelweiss": "https://www.edelweissmf.com/statutory/portfolio-of-schemes",
    "whiteoak": "https://mf.whiteoakamc.com/regulatory-disclosures/scheme-portfolios",
    "lic": "https://www.licmf.com/downloads/monthly-portfolio",
    "union": "https://unionmf.com/downloads/portfolio-disclosure",
    "baroda": "https://www.barodabnpparibasmf.in/downloads/monthly-portfolio-scheme",
}
FILE = re.compile(r"\.(xlsx?|xlsb|zip)(\?|$)", re.I)
RECENT = re.compile(r"(2026|sep|oct|aug)", re.I)

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126 Safari/537.36", locale="en-IN")
    for name, url in PAGES.items():
        page = ctx.new_page()
        xhr = []
        page.on("response", lambda r, xhr=xhr: xhr.append((r.status, r.url))
                if r.request.resource_type in ("xhr", "fetch") else None)
        try:
            resp = page.goto(url, timeout=45000, wait_until="domcontentloaded")
            page.wait_for_timeout(6000)
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => [e.href, e.innerText.trim().slice(0,80)])")
            files = [(h, t) for h, t in hrefs if FILE.search(h)]
            recent = [(h, t) for h, t in files if RECENT.search(h + " " + t)]
            print(f"=== {name} {resp.status if resp else '?'} title={page.title()[:60]!r} links={len(hrefs)} "
                  f"files={len(files)} final={page.url}")
            for h, t in (recent or files)[:30]:
                print(f"   file: {h}  [{t}]")
            for s, u in [x for x in xhr if not re.search(r"google|analytics|facebook|clarity|hotjar|notifyvisitors|"
                                                         r"doubleclick|webengage|moengage|font", x[1])][:15]:
                print(f"   xhr {s}: {u[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"=== {name} ERROR {type(e).__name__}: {str(e)[:200]}")
        page.close()
    b.close()
