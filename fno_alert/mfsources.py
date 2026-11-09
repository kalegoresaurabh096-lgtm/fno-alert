"""Fund houses whose monthly portfolio files the alert reads.

page:    the fund house's portfolio disclosure page
include: regex a file link (URL + link text) must match
exclude: regex it must not match (defaults skip debt-only and fortnightly files)
browser: the page builds its links with JavaScript, so open it in a headless browser
multi:   the fund house publishes one file per scheme; read every file for the month

Checked from GitHub's servers on 2026-11-09. Not readable from there yet: HDFC, Edelweiss
and WhiteOak (Access Denied), Kotak (bot check), and ICICI, Axis, UTI, Quant, Bandhan,
Canara Robeco, Franklin and LIC (no direct file links on the page).
"""

SOURCES = [
    {"amc": "SBI", "page": "https://www.sbimf.com/portfolios", "browser": True,
     "include": r"all-schemes-monthly-portfolio"},
    {"amc": "Nippon India", "page": "https://mf.nipponindiaim.com/investor-service/downloads/"
                                    "factsheet-portfolio-and-other-disclosures",
     "include": r"NIMF-MONTHLY-PORTFOLIO"},
    {"amc": "Aditya Birla Sun Life", "page": "https://mutualfund.adityabirlacapital.com/forms-and-downloads/portfolio",
     "browser": True, "include": r"/monthly-portfolio/20\d\d/"},
    {"amc": "DSP", "page": "https://www.dspim.com/mandatory-disclosures/portfolio-disclosures",
     "include": r"month[-_ ]?end[-_ ]?portfolio"},
    {"amc": "Tata", "page": "https://www.tatamutualfund.com/schemes-related/portfolio", "browser": True,
     "include": r"monthly(%20| |_|-)portfolio"},
    {"amc": "Motilal Oswal", "page": "https://www.motilaloswalmf.com/downloads/scheme-portfolio-details",
     "browser": True, "include": r"scheme(%20| |_|-)portfolio(%20| |_|-)details"},
    {"amc": "PPFAS", "page": "https://amc.ppfas.com/downloads/portfolio-disclosure/",
     "include": r"/(PPFAS_Monthly_Portfolio|Monthly-Portfolio)"},
    {"amc": "Mirae Asset", "page": "https://www.miraeassetmf.co.in/downloads/portfolio", "browser": True,
     "multi": True, "include": r"/portfolios/"},
    {"amc": "Invesco", "page": "https://www.invescomutualfund.com/literature-forms/monthly-holdings/equity",
     "browser": True, "multi": True, "include": r"completes?-monthly-holding"},
    {"amc": "Baroda BNP Paribas", "page": "https://www.barodabnpparibasmf.in/downloads/monthly-portfolio-scheme",
     "browser": True, "include": r"Monthly_Portfolio"},
]
