"""
Shared "find volatile Swiss stocks, excluding foreign companies and SMI
mega-caps" logic, used by both swiss_crash_rebound.py and
swiss_today_screener.py. Free tools only: yfinance (no API key).

Two filters matter for "Swiss stocks with market cap > CHF 500M, excluding
foreign companies":
  1. Market-cap range (MIN/MAX_MARKET_CAP_CHF below) - Yahoo's region='ch'
     screener field means "listed in the CH region", not "domiciled in
     Switzerland", and it includes ETFs/bonds/structured products
     alongside real equities.
  2. `country` == "Switzerland" from each candidate's own info - SIX lists
     several foreign-domiciled companies (confirmed live: Bitcoin Group SE
     is domiciled in Germany, ams-OSRAM AG in Austria, both pass the
     region+market-cap+quoteType==EQUITY filters otherwise). Only this
     per-company `country` field actually distinguishes domestic from
     foreign - the region/exchange filters alone do not.

Single universe as of 2026-08-18 (previously a small-cap default with an
opt-in wider ALL_CAPS_MIN/MAX_MARKET_CAP_CHF band toggled per-request -
removed at the user's explicit request to always include mid/large caps):
market cap > CHF 500M, no upper bound - but always excludes SMI_TICKERS
regardless (see that set's own comment): this scan is looking for
VOLATILE tickers, and the SMI's 20 largest/most-liquid names are the
opposite of that, so they're excluded on purpose rather than just being
an unlikely match.
"""

import concurrent.futures
import datetime
import json
import logging
import os
import time
import yfinance as yf
from yfinance.data import YfData

logger = logging.getLogger(__name__)

# Escape hatch, off by default: seeds yfinance's process-wide crumb/cookie
# jar (YfData is a true singleton - yfinance.data.YfData, one per process,
# metaclass=SingletonMeta) from externally-obtained values instead of
# letting yfinance fetch its own. Confirmed live: this server's outbound IP
# is currently blocked specifically at Yahoo's crumb-fetch endpoint
# (yfinance.data.YfData._get_crumb_csrf -> query2.finance.yahoo.com/v1/
# test/getcrumb), which raises YFRateLimitError - and since a fresh process
# has no cached crumb (never persisted to disk, and yfinance never re-
# fetches once it HAS one - no expiry check in _get_crumb_csrf), every
# scan on a freshly-restarted process re-attempts that same blocked fetch
# and fails immediately, even though the actual data endpoints
# (screener/download/quoteSummary) were never even reached to know if
# THEY'D have worked. This is why the scan worked for a long time, then
# broke right after a run of unrelated redeploys: an old process had
# already cached a working crumb from before the block started and never
# needed to ask again; each restart throws that away.
#
# Seeding a crumb+cookie pair captured from a DIFFERENT, unblocked network
# lets this process skip the blocked fetch entirely (verified: a captured
# crumb+cookies pair authenticates real Yahoo calls with no re-fetch - see
# the PR that added this). NOT guaranteed to work here: Yahoo may bind a
# crumb/cookie pair to the IP that requested it, in which case using it
# from a different IP fails too - that's genuinely unknown without trying
# it live. If the exact same YFRateLimitError keeps happening after
# setting these, that's the answer: IP-bound, and this doesn't help. A
# different error afterward means it worked and something else is going
# on. Remove YF_SEED_CRUMB/YF_SEED_COOKIES once Yahoo's block on this IP's
# own crumb-fetch lifts and a real fetch starts working again - this is a
# stop-gap, not a permanent fix (the seeded crumb/cookies will themselves
# eventually expire on Yahoo's side, at an unknown time).
def _seed_yf_session_from_env():
    seed_crumb = os.environ.get("YF_SEED_CRUMB")
    seed_cookies_json = os.environ.get("YF_SEED_COOKIES")
    if not seed_crumb or not seed_cookies_json:
        return
    data = YfData()
    if data._crumb:
        return  # already seeded or already fetched its own this process - don't clobber either
    for name, value in json.loads(seed_cookies_json).items():
        data._session.cookies.set(name, value)
    data._crumb = seed_crumb
    logger.info("Seeded yfinance crumb/cookies from YF_SEED_CRUMB/YF_SEED_COOKIES (Yahoo crumb-fetch workaround)")


# Market-cap band in CHF - single universe as of 2026-08-18 (see module
# docstring): CHF 500M+ excludes small/micro caps, no upper bound in
# practice - 1 trillion comfortably clears Nestle, the largest SIX-listed
# company by market cap (confirmed live via the snapshot below, ~CHF
# 180-250B depending on the day). Previously two separate bands (a
# small-cap default plus an opt-in wider "all caps" toggle) - collapsed
# into one at the user's explicit request to always include mid/large
# caps, not gate them behind a checkbox.
MIN_MARKET_CAP_CHF = 500_000_000
MAX_MARKET_CAP_CHF = 1_000_000_000_000

# Liquidity floor, applied in filter_domestic below: a ticker trading
# fewer shares than this on average is too thin to trust either the price
# itself (a handful of trades can swing "close" arbitrarily) or the crash-
# rebound/big-loss signals this scan is built to find - a >=5% move on
# ~1,000 shares traded isn't the same finding as the same move on
# 500,000. Measured against the 10-DAY average volume (Ticker.info's
# averageDailyVolume10Day), not a single day's raw volume - a smoother,
# less fluke-prone baseline than "was today specifically thin" (changed
# 2026-08-18 at the user's request; previously compared against
# regularMarketVolume, today's raw figure only).
MIN_AVG_DAILY_VOLUME_10D = 50_000

# Politeness delay between per-ticker yfinance .info calls - this is an
# unofficial/undocumented API, not a documented rate limit to size against
# (unlike the Gemini API elsewhere in this project's sibling repo), so this
# is a conservative default, not a measured cap. Applied PER WORKER (see
# INFO_MAX_WORKERS below), not globally - each worker still paces its own
# sequential requests by this much, concurrency just runs several such
# paced streams at once instead of one.
INFO_REQUEST_DELAY_SECONDS = 0.3

# Bounded concurrency for filter_domestic's per-ticker .info loop - a
# meaningful speedup (roughly INFO_MAX_WORKERS-x) over one ticker at a
# time without going back to the fully-unbounded burst that's already
# bitten this scan once (see DOWNLOAD_CHUNK_SIZE/DOWNLOAD_CHUNK_DELAY_
# SECONDS in swiss_crash_rebound.py - a single fully-concurrent
# yf.download(..., threads=True) call over 100+ tickers is what was
# actually tripping "Too Many Requests" there, confirmed live). YfData is
# a thread-safe singleton (its crumb/cookie fetch is guarded by its own
# lock - see yfinance.data.YfData._cookie_lock), so concurrent
# Ticker.info calls sharing one session are safe with respect to auth
# state; what's NOT yet validated is how many concurrent requests this
# SPECIFIC endpoint (quoteSummary, not the download/screener endpoints
# that already have their own tuned limits) actually tolerates before
# Yahoo starts rate-limiting it - this number is a reasoned starting
# guess, not something stress-tested the way DOWNLOAD_CHUNK_SIZE was.
# Watch Render logs for a rise in "info fetch failed" lines after
# deploying this; lower it if so.
INFO_MAX_WORKERS = 5

# SMI (Swiss Market Index) constituents - the 20 largest, most liquid,
# most heavily analyst-covered names on SIX. Excluded from discovery, not
# because they fail any correctness check (they're all honest,
# Switzerland-domiciled operating companies), but because this scan's
# whole point - crash-then-rebound and big-daily-loss detection - is
# finding VOLATILE tickers, and SMI names are the opposite end of that
# spectrum: heavily traded, closely watched, and structurally the least
# likely names on the exchange to produce the kind of move this scan is
# looking for. Applies regardless of market cap - these names are all
# well within the CHF 500M-1T band (MIN/MAX_MARKET_CAP_CHF), so this
# exclusion is what actually keeps them out, not the cap band itself.
#
# Confirmed live via Wikipedia's SMI constituent table on 2026-08-13
# (https://en.wikipedia.org/wiki/Swiss_Market_Index) - SIX itself reviews
# SMI composition once a year each September, so this WILL drift after
# the next review; re-check that table (or SIX's own six-group.com
# constituent listing) and update this set when it does. Multiple tickers
# per company where a company trades more than one share-class line on
# SIX (e.g. Novartis' NOVN.SW ordinary line and its NOVNEE.SW second
# line, confirmed present as distinct symbols in
# STATIC_DOMESTIC_TICKER_SNAPSHOT below) - both listings are the same
# company, so both are excluded.
SMI_TICKERS = {
    "NOVN.SW", "NOVNEE.SW",  # Novartis
    "RO.SW", "ROP.SW",       # Roche
    "NESN.SW",                # Nestle
    "ABBN.SW", "ABBNE.SW",   # ABB
    "UBSG.SW", "UBSGE.SW",   # UBS
    "CFR.SW",                 # Richemont
    "ZURN.SW",                # Zurich Insurance
    "HOLN.SW",                # Holcim
    "SREN.SW", "SRENE.SW",   # Swiss Re
    "LONN.SW",                # Lonza
    "SCMN.SW",                # Swisscom
    "GIVN.SW",                # Givaudan
    "ALC.SW",                 # Alcon
    "SIKA.SW",                # Sika
    "AMRZ.SW", "AMRZE.SW",   # Amrize
    "SLHN.SW",                # Swiss Life
    "KNIN.SW",                # Kuehne + Nagel
    "GEBN.SW", "GEBNE.SW",   # Geberit
    "PGHN.SW",                # Partners Group
    "LOGN.SW", "LOGNE.SW",   # Logitech
}

# Passes the market-cap band and Switzerland-domicile checks but isn't a
# normal operating company, so it doesn't belong in this scan's universe
# regardless: SNBN.SW is the Swiss National Bank (confirmed live: shows up
# in the market-cap band, mostly canton-held). Add more symbols here as
# other non-operating-company edge cases turn up. Merged with SMI_TICKERS
# (see its own comment) since both are "known good companies that still
# don't belong in this scan's universe," just for different reasons - one
# filtering pass covers both (see _discover_candidates_live and
# discover_candidates' static-fallback branch below).
EXCLUDED_TICKERS = {"SNBN.SW"} | SMI_TICKERS


# Static fallback for when Yahoo's screener endpoint (yf.screen, used by
# _discover_candidates_live below) is unavailable - confirmed live as the
# ONLY endpoint that has ever failed across every "Too Many Requests"
# incident in this scan's history. filter_domestic's own per-ticker .info
# calls and find_crash_then_rebound's yf.download() have never failed in
# any of those incidents - this fallback exists specifically because the
# discovery STEP, not the whole pipeline, is what breaks.
#
# Captured live on 2026-08-13 via discover_candidates() itself (back when
# that meant the wide all-caps band, min=50M/max=1T - now the same as this
# module's single MIN/MAX_MARKET_CAP_CHF band, see that constant's own
# comment for the 2026-08-18 band collapse), from a working (non-blocked)
# network - name/sector only, NOT live price/change%/volume, which the
# real screener response also carries (see discover_candidates' own
# docstring) but a static snapshot fundamentally can't provide.
# discover_candidates below doesn't filter the fallback by the requested
# min/max (no market-cap data in the static entries to filter on) -
# accepted, unchanged tradeoff. Downstream effect: filter_domestic and
# swiss_crash_rebound both work fully off this fallback (neither
# needs the screener's own live quote fields - filter_domestic makes its
# own live .info call per ticker regardless, crash_rebound uses a
# separate yf.download() history call). swiss_today_screener's change%/
# volume_today fields do NOT (still read directly off the screener quote,
# no separate fetch - see that module's own docstring), so it degrades to
# genuinely empty results (not a crash - find_big_loss already fails soft
# on missing fields) whenever this fallback is in use, since there is no
# live "today" data to give it. avg_volume_10d is unaffected either way -
# it comes from filter_domestic's own .info call, same as the other
# fallback-tolerant fields.
#
# Will drift from reality over time (new listings, delistings, M&A) since
# it's not auto-refreshed. Refresh by running discover_candidates()/
# filter_domestic() from a working, non-blocked network and regenerating
# this dict - ticker -> (name, sector).
STATIC_DOMESTIC_TICKER_SNAPSHOT = {
    'ABBN.SW': ('ABB Ltd', 'Industrials'),
    'ABBNE.SW': ('ABB Ltd', 'Industrials'),
    'ACLN.SW': ('Accelleron Industries AG', 'Industrials'),
    'ACLNE.SW': ('Accelleron Industries AG', ''),
    'ADEN.SW': ('Adecco Group AG', 'Industrials'),
    'ADVN.SW': ('Adval Tech Holding AG', 'Industrials'),
    'ADXN.SW': ('Addex Therapeutics Ltd', 'Healthcare'),
    'AERO.SW': ('Montana Aerospace AG', 'Industrials'),
    'AEVS.SW': ('Aevis Victoria SA', 'Healthcare'),
    'ALC.SW': ('Alcon Inc.', 'Healthcare'),
    'ALLN.SW': ('Allreal Holding AG', 'Real Estate'),
    'ALPN.SW': ('Alpine Select AG', 'Financial Services'),
    'ALSN.SW': ('ALSO Holding AG', 'Technology'),
    'AMRZ.SW': ('AMRIZE N', 'Basic Materials'),
    'AMRZE.SW': ('Amrize AG', 'Basic Materials'),
    'APGN.SW': ('APG|SGA SA', 'Communication Services'),
    'ARBN.SW': ('Arbonia AG', 'Industrials'),
    'ARYN.SW': ('ARYZTA AG', 'Consumer Defensive'),
    'ASCN.SW': ('Ascom Holding AG', 'Healthcare'),
    'ASWN.SW': ('Asmallworld AG', 'Consumer Cyclical'),
    'AUTN.SW': ('Autoneum Holding AG', 'Consumer Cyclical'),
    'AVOL.SW': ('Avolta AG', 'Consumer Cyclical'),
    'BAER.SW': ('Julius Bär Gruppe AG', 'Financial Services'),
    'BANB.SW': ('Bachem Holding AG', 'Healthcare'),
    'BARN.SW': ('Barry Callebaut AG', 'Consumer Defensive'),
    'BBN.SW': ('Bellevue Group AG', 'Financial Services'),
    'BCGE.SW': ('Banque Cantonale de Genève SA', 'Financial Services'),
    'BCHN.SW': ('Burckhardt Compression Holding AG', 'Industrials'),
    'BCJ.SW': ('Banque Cantonale du Jura SA', 'Financial Services'),
    'BCVN.SW': ('Banque Cantonale Vaudoise', 'Financial Services'),
    'BEAN.SW': ('BELIMO Holding AG', 'Industrials'),
    'BEKN.SW': ('Berner Kantonalbank AG', 'Financial Services'),
    'BELL.SW': ('Bell Food Group AG', 'Consumer Defensive'),
    'BION.SW': ('BB Biotech AG', 'Healthcare'),
    'BIOV.SW': ('BioVersys AG', 'Healthcare'),
    'BKW.SW': ('BKW AG', 'Utilities'),
    'BLKB.SW': ('Basellandschaftliche Kantonalbank', 'Financial Services'),
    'BOSN.SW': ('Bossard Holding AG', 'Industrials'),
    'BRKN.SW': ('Burkhalter Holding AG', 'Industrials'),
    'BSKP.SW': ('Basler Kantonalbank', 'Financial Services'),
    'BSLN.SW': ('Basilea Pharmaceutica AG', 'Healthcare'),
    'BUCN.SW': ('Bucher Industries AG', 'Industrials'),
    'BVZN.SW': ('BVZ Holding AG', 'Industrials'),
    'BYS.SW': ('Bystronic AG', 'Industrials'),
    'CALN.SW': ('CALIDA Holding AG', 'Consumer Cyclical'),
    'CCAP.SW': ('C Capital Holdings AG', 'Financial Services'),
    'CFR.SW': ('Compagnie Financière Richemont SA', 'Consumer Cyclical'),
    'CFT.SW': ('Compagnie Financière Tradition SA', 'Financial Services'),
    'CFTE.SW': ('Compagnie Financière Tradition SA', 'Financial Services'),
    'CHAM.SW': ('Cham Swiss Properties AG', 'Real Estate'),
    'CICN.SW': ('Cicor Technologies Ltd.', 'Technology'),
    'CLN.SW': ('Clariant AG', 'Basic Materials'),
    'CLTN.SW': ('COLTENE Holding AG', 'Healthcare'),
    'CMBN.SW': ('Cembra Money Bank AG', 'Financial Services'),
    'CMUS.SW': ('CMUS.SW', ''),
    'CNTL.SW': ('Centiel N', 'Industrials'),
    'COTN.SW': ('Comet Holding AG', 'Technology'),
    'CPEN.SW': ('Castle Private Equity AG', 'Financial Services'),
    'CPHN.SW': ('CPH Group AG', 'Basic Materials'),
    'CURN.SW': ('Curatis Holding AG', 'Healthcare'),
    'DAE.SW': ('Dätwyler Holding AG', 'Industrials'),
    'DESN.SW': ('Dottikon ES Holding AG', 'Basic Materials'),
    'DKSH.SW': ('DKSH Holding AG', 'Industrials'),
    'DOCM.SW': ('DocMorris AG', 'Healthcare'),
    'DOKA.SW': ('dormakaba Holding AG', 'Industrials'),
    'DSFIR.SW': ('DSM-Firmenich AG', 'Basic Materials'),
    'EFGN.SW': ('EFG International AG', 'Financial Services'),
    'EMMN.SW': ('Emmi AG', 'Consumer Defensive'),
    'EMSN.SW': ('EMS-CHEMIE HOLDING AG', 'Basic Materials'),
    'EPIC.SW': ('EPIC Suisse AG', 'Real Estate'),
    'ESUN.SW': ('Edisun Power Europe AG', 'Utilities'),
    'EVE.SW': ('EvoNext Holdings SA', 'Financial Services'),
    'FHZN.SW': ('Flughafen Zürich AG', 'Industrials'),
    'FORN.SW': ('Forbo Holding AG', 'Industrials'),
    'FREN.SW': ('Fundamenta Real Estate AG', 'Real Estate'),
    'FTON.SW': ('Feintool International Holding AG', 'Industrials'),
    'GALD.SW': ('GALDERMA GROUP N', 'Healthcare'),
    'GALE.SW': ('Galenica AG', 'Healthcare'),
    'GAM.SW': ('GAM Holding AG', 'Financial Services'),
    'GAV.SW': ('Carlo Gavazzi Holding AG', 'Industrials'),
    'GEBN.SW': ('Geberit AG', 'Industrials'),
    'GEBNE.SW': ('GEBERIT N 2. LINIE', ''),
    'GF.SW': ('Georg Fischer AG', 'Industrials'),
    'GIVN.SW': ('Givaudan SA', 'Basic Materials'),
    'GLKBN.SW': ('Glarner Kantonalbank', 'Financial Services'),
    'GMI.SW': ('Groupe Minoteries SA', 'Consumer Defensive'),
    'GRKP.SW': ('Graubündner Kantonalbank', 'Financial Services'),
    'GURN.SW': ('Gurit Holding AG', 'Basic Materials'),
    'HBAN.SW': ('Helvetia Baloise Holding AG', 'Financial Services'),
    'HBLN.SW': ('Hypothekarbank Lenzburg AG', 'Financial Services'),
    'HBMN.SW': ('HBM Healthcare Investments AG', 'Financial Services'),
    'HBMNE.SW': ('HBM Healthcare Investments AG', 'Financial Services'),
    'HIAG.SW': ('HIAG Immobilien Holding AG', 'Real Estate'),
    'HLEE.SW': ('Highlight Event and Entertainment AG', 'Communication Services'),
    'HOLN.SW': ('Holcim AG', 'Basic Materials'),
    'HUBN.SW': ('Huber+Suhner AG', 'Technology'),
    'IDIA.SW': ('Idorsia Ltd', 'Healthcare'),
    'IFC.SW': ('IFC.SW', ''),
    'IFCN.SW': ('INFICON Holding AG', 'Technology'),
    'IMPN.SW': ('Implenia AG', 'Industrials'),
    'INFRAC.SW': ('INFRACORE N', 'Real Estate'),
    'INRN.SW': ('Interroll Holding AG', 'Industrials'),
    'IREN.SW': ('Investis Holding SA', 'Real Estate'),
    'ISN.SW': ('Intershop Holding AG', 'Real Estate'),
    'JFN.SW': ('Jungfraubahn Holding AG', 'Industrials'),
    'KARN.SW': ('Kardex Holding AG', 'Industrials'),
    'KLIN.SW': ('Klingelnberg AG', 'Industrials'),
    'KNIN.SW': ('Kuehne + Nagel International AG', 'Industrials'),
    'KOMN.SW': ('Komax Holding AG', 'Industrials'),
    'KUD.SW': ('Kudelski SA', 'Technology'),
    'KURN.SW': ('Kuros Biosciences AG', 'Healthcare'),
    'LAND.SW': ('Landis+Gyr Group AG', 'Industrials'),
    'LEHN.SW': ('LEM Holding SA', 'Technology'),
    'LEON.SW': ('Leonteq AG', 'Financial Services'),
    'LISN.SW': ('Chocoladefabriken Lindt & Sprüngli AG', 'Consumer Defensive'),
    'LISNE.SW': ('LINDT N 2.LINIE', 'Consumer Defensive'),
    'LISP.SW': ('Chocoladefabriken Lindt & Sprüngli AG', 'Consumer Defensive'),
    'LISPE.SW': ('LINDT PS 2.LINIE', 'Consumer Defensive'),
    'LOGN.SW': ('Logitech International S.A.', 'Technology'),
    'LOGNE.SW': ('Logitech International S.A.', 'Technology'),
    'LONN.SW': ('Lonza Group AG', 'Healthcare'),
    'LUKN.SW': ('Luzerner Kantonalbank AG', 'Financial Services'),
    'MCHN.SW': ('MCH Group AG', 'Communication Services'),
    'MED.SW': ('Medartis Holding AG', 'Healthcare'),
    'MEDX.SW': ('medmix AG', 'Industrials'),
    'METN.SW': ('Metall Zug AG', 'Healthcare'),
    'MFSP.SW': ('Mobifonds Swiss Property Fund', ''),
    'MIKN.SW': ('Mikron Holding AG', 'Industrials'),
    'MMTX.SW': ('MindMaze Therapeutics Holding SA', 'Healthcare'),
    'MOBN.SW': ('Mobimo Holding AG', 'Real Estate'),
    'MOLN.SW': ('Molecular Partners AG', 'Healthcare'),
    'MOVE.SW': ('Medacta Group SA', 'Healthcare'),
    'MOZN.SW': ('mobilezone holding ag', 'Consumer Cyclical'),
    'MTG.SW': ('Meier Tobler Group AG', 'Industrials'),
    'NBEN.SW': ('nebag ag', 'Financial Services'),
    'NEAG.SW': ('naturenergie holding AG', 'Utilities'),
    'NESN.SW': ('Nestlé S.A.', 'Consumer Defensive'),
    'NOVN.SW': ('Novartis AG', 'Healthcare'),
    'NOVNEE.SW': ('Novartis AG', ''),
    'NREN.SW': ('Novavest Real Estate AG', 'Real Estate'),
    'OERL.SW': ('OC Oerlikon Corporation AG', 'Industrials'),
    'OFN.SW': ('Orell Füssli AG', 'Industrials'),
    'ORON.SW': ('ORIOR AG', 'Consumer Defensive'),
    'PEAN.SW': ('Peach Property Group AG', 'Real Estate'),
    'PEDU.SW': ('Perrot Duval Holding S.A.', 'Industrials'),
    'PEHN.SW': ('Private Equity Holding AG', 'Financial Services'),
    'PGHN.SW': ('Partners Group Holding AG', 'Financial Services'),
    'PLAN.SW': ('Plazza AG', 'Real Estate'),
    'PMN.SW': ('Phoenix Mecano AG', 'Industrials'),
    'PPGN.SW': ('PolyPeptide Group AG', 'Healthcare'),
    'PSPN.SW': ('PSP Swiss Property AG', 'Real Estate'),
    'REHN.SW': ('Romande Energie Holding SA', 'Utilities'),
    'RIEN.SW': ('Rieter Holding AG', 'Industrials'),
    'RO.SW': ('Roche Holding AG', 'Healthcare'),
    'ROP.SW': ('Roche Holding AG', 'Healthcare'),
    'RSGN.SW': ('R&S Group Holding AG', 'Industrials'),
    'SANN.SW': ('Santhera Pharmaceuticals Holding AG', 'Healthcare'),
    'SCHN.SW': ('Schindler Holding AG', 'Industrials'),
    'SCHNE.SW': ('Schindler Holding AG', 'Industrials'),
    'SCHP.SW': ('Schindler Holding AG', 'Industrials'),
    'SCHPE.SW': ('Schindler Holding AG', 'Industrials'),
    'SCMN.SW': ('Swisscom AG', 'Communication Services'),
    'SDZ.SW': ('Sandoz Group AG', 'Healthcare'),
    'SENS.SW': ('Sensirion Holding AG', 'Technology'),
    'SFPN.SW': ('SF Urban Properties AG', 'Real Estate'),
    'SFSN.SW': ('SFS Group AG', 'Industrials'),
    'SFZN.SW': ('Siegfried Holding AG', 'Healthcare'),
    'SGKN.SW': ('St. Galler Kantonalbank AG', 'Financial Services'),
    'SGSN.SW': ('SGS SA', 'Industrials'),
    'SIGN.SW': ('SIG Group AG', 'Consumer Cyclical'),
    'SIKA.SW': ('Sika AG', 'Basic Materials'),
    'SKAN.SW': ('SKAN Group AG', 'Healthcare'),
    'SLDCS.SW': ('Swiss Life REF (CH) ESG Diversified Commercial Switzerland Fund', ''),
    'SLHN.SW': ('Swiss Life Holding AG', 'Financial Services'),
    'SMG.SW': ('SMG Swiss Marketplace Group Holding AG', 'Communication Services'),
    'SOON.SW': ('Sonova Holding AG', 'Healthcare'),
    'SPSN.SW': ('Swiss Prime Site AG', 'Real Estate'),
    'SQN.SW': ('Swissquote Group Holding SA', 'Financial Services'),
    'SRAIL.SW': ('Stadler Rail AG', 'Industrials'),
    'SREN.SW': ('Swiss Re AG', 'Financial Services'),
    'SRENE.SW': ('SWISS RE N 2. LINIE', ''),
    'STGN.SW': ('StarragTornos Group AG', 'Industrials'),
    'STMN.SW': ('Straumann Holding AG', 'Healthcare'),
    'STRN.SW': ('Schlatter Industries AG', 'Industrials'),
    'SUN.SW': ('Sulzer AG', 'Industrials'),
    'SUNN.SW': ('Sunrise Communications AG', 'Communication Services'),
    'SWON.SW': ('SoftwareOne Holding AG', 'Technology'),
    'SWTQ.SW': ('Schweiter Technologies AG', 'Industrials'),
    'TECN.SW': ('Tecan Group AG', 'Healthcare'),
    'TEMN.SW': ('Temenos AG', 'Technology'),
    'TIBN.SW': ('Bergbahnen Engelberg-Trübsee-Titlis AG', 'Consumer Cyclical'),
    'TKBP.SW': ('Thurgauer Kantonalbank', 'Financial Services'),
    'TXGN.SW': ('TX Group AG', 'Communication Services'),
    'TXGNE.SW': ('TX Group AG', 'Communication Services'),
    'UBSG.SW': ('UBS Group AG', 'Financial Services'),
    'UBSGE.SW': ('UBS GROUP N 2. LINIE', 'Financial Services'),
    'UHR.SW': ('The Swatch Group AG', 'Consumer Cyclical'),
    'UHRN.SW': ('The Swatch Group AG', 'Consumer Cyclical'),
    'VACN.SW': ('VAT Group AG', 'Industrials'),
    'VAHN.SW': ('Vaudoise Assurances Holding SA', 'Financial Services'),
    'VARN.SW': ('Varia US Properties AG', 'Real Estate'),
    'VATN.SW': ('Valiant Holding AG', 'Financial Services'),
    'VATNE.SW': ('Valiant Holding AG', 'Financial Services'),
    'VBSN.SW': ('IVF Hartmann Holding AG', 'Healthcare'),
    'VETN.SW': ('Vetropack Holding AG', 'Consumer Cyclical'),
    'VILN.SW': ('Villars Holding S.A.', 'Consumer Defensive'),
    'VLRT.SW': ('Valartis Group AG', 'Financial Services'),
    'VONN.SW': ('Vontobel Holding AG', 'Financial Services'),
    'VZN.SW': ('VZ Holding AG', 'Financial Services'),
    'VZUG.SW': ('V-ZUG Holding AG', 'Consumer Cyclical'),
    'WARN.SW': ('Warteck Invest AG', 'Real Estate'),
    'WIHN.SW': ('WISeKey International Holding AG', 'Technology'),
    'WKBN.SW': ('Walliser Kantonalbank', 'Financial Services'),
    'XLS.SW': ('Xlife Sciences AG', 'Healthcare'),
    'YPSN.SW': ('Ypsomed Holding AG', 'Healthcare'),
    'ZEHN.SW': ('Zehnder Group AG', 'Industrials'),
    'ZUBN.SW': ('Züblin Immobilien Holding AG', 'Real Estate'),
    'ZUGER.SW': ('Zuger Kantonalbank', 'Financial Services'),
    'ZUGN.SW': ('Zug Estates Holding AG', 'Real Estate'),
    'ZURN.SW': ('Zurich Insurance Group AG', 'Financial Services'),
}


def _discover_candidates_live(min_market_cap, max_market_cap):
    """The real screener call - see discover_candidates for the public
    entry point (tries this first, falls back to
    STATIC_DOMESTIC_TICKER_SNAPSHOT on failure). Returns (symbol -> quote
    dict) for quoteType == 'EQUITY' only, excluding EXCLUDED_TICKERS - the
    region+market-cap query alone still returns ETFs/structured products/
    bonds mixed in (confirmed live), so quoteType is filtered here rather
    than trusted from the query. Each quote dict is the FULL raw screener
    response for that symbol (price, live change%, volume, 3-month average
    volume, etc.) - callers needing "today" data (see
    swiss_today_screener.py) can read it straight off this dict,
    no extra fetch needed."""
    _seed_yf_session_from_env()

    query = yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("eq", ["region", "ch"]),
            yf.EquityQuery("btwn", ["intradaymarketcap", min_market_cap, max_market_cap]),
        ],
    )

    candidates = {}
    offset = 0
    page_size = 250  # Yahoo's documented max per page
    while True:
        page = yf.screen(query, offset=offset, size=page_size, sortField="ticker", sortAsc=True)
        quotes = page.get("quotes", [])
        if not quotes:
            break
        for q in quotes:
            symbol = q.get("symbol")
            if q.get("quoteType") == "EQUITY" and symbol and symbol not in EXCLUDED_TICKERS:
                candidates[symbol] = q
        offset += page_size
        if offset >= page.get("total", 0):
            break
    return candidates


def discover_candidates(min_market_cap=MIN_MARKET_CAP_CHF, max_market_cap=MAX_MARKET_CAP_CHF):
    """Tries the live screener first, falls back to
    STATIC_DOMESTIC_TICKER_SNAPSHOT on ANY failure (broad except
    deliberate - this is a resilience fallback, not trying to distinguish
    which specific failure mode occurred) so a Yahoo-side outage degrades
    this scan instead of failing it outright. Self-healing: no redeploy or
    manual toggle needed - the next scan after Yahoo's block lifts just
    uses the live path again automatically, since this always tries live
    first. See STATIC_DOMESTIC_TICKER_SNAPSHOT's own comment for what
    downstream callers lose (today_screener specifically) when running off
    the fallback."""
    try:
        return _discover_candidates_live(min_market_cap, max_market_cap)
    except Exception as e:
        logger.warning(
            "discover_candidates: live screener failed (%r), falling back to "
            "STATIC_DOMESTIC_TICKER_SNAPSHOT (%d tickers, no live quote data)",
            e, len(STATIC_DOMESTIC_TICKER_SNAPSHOT),
        )
        return {
            symbol: {"symbol": symbol, "quoteType": "EQUITY", "longName": name, "sector": sector}
            for symbol, (name, sector) in STATIC_DOMESTIC_TICKER_SNAPSHOT.items()
            if symbol not in EXCLUDED_TICKERS
        }


def _ex_dividend_date(info: dict) -> str | None:
    # yfinance's exDividendDate is a Unix timestamp (seconds), not a date
    # string - confirmed live. None for companies with no dividend history
    # (the key is simply absent from .info), which utcfromtimestamp(None)
    # would raise on, so this checks first rather than catching.
    ts = info.get("exDividendDate")
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date().isoformat()


def _fetch_domestic_entry(symbol, quote, delay_seconds):
    """One ticker's worth of filter_domestic's work - fetch .info, decide
    keep/skip, sleep the politeness delay - factored out so it can run as
    a unit inside a worker thread (see filter_domestic below). Returns
    (symbol, entry_dict_or_None); the caller does the printing/assembly,
    this only computes. Sleeps unconditionally at the end, success or
    failure, same as the original serial loop did - each worker still
    paces its OWN sequential requests by delay_seconds, concurrency just
    runs several such paced streams at once (see INFO_MAX_WORKERS)."""
    try:
        info = yf.Ticker(symbol).info
        country = info.get("country")
        avg_volume_10d = info.get("averageDailyVolume10Day")
        if country != "Switzerland":
            print(f"  skip {symbol}: domiciled in {country!r}, not Switzerland")
            return symbol, None
        if avg_volume_10d is None or avg_volume_10d < MIN_AVG_DAILY_VOLUME_10D:
            print(f"  skip {symbol}: 10-day avg volume {avg_volume_10d!r} below {MIN_AVG_DAILY_VOLUME_10D} floor")
            return symbol, None
        return symbol, {
            "name": quote.get("longName") or quote.get("shortName") or symbol,
            "sector": info.get("sector"),
            "market_cap": quote.get("marketCap"),
            "trailing_eps": info.get("trailingEps"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "ex_dividend_date": _ex_dividend_date(info),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume_10d": avg_volume_10d,
            "quote": quote,
        }
    except Exception as e:
        print(f"  skip {symbol}: info fetch failed ({e!r})")
        return symbol, None
    finally:
        time.sleep(delay_seconds)


def filter_domestic(candidates, delay_seconds=INFO_REQUEST_DELAY_SECONDS, max_workers=INFO_MAX_WORKERS):
    """Keeps only candidates whose own `country` field is Switzerland -
    the one field that actually reflects company domicile rather than
    exchange/listing region (see module docstring) - AND whose 10-day
    average trading volume clears MIN_AVG_DAILY_VOLUME_10D (see that
    constant's own comment for why). Fetched per-ticker via Ticker.info
    since the screener response doesn't include the country field (it
    does carry a 3-month average volume figure too - see
    swiss_today_screener.py's history for why THIS module deliberately
    reads a 10-day figure off .info instead of reusing that - but
    averageDailyVolume10Day is read here regardless, since .info is
    already being fetched for the domicile check at no extra request
    cost, and unlike the screener quote it's present for BOTH discovery
    paths - the static fallback's synthesized quote (see
    STATIC_DOMESTIC_TICKER_SNAPSHOT) has no volume field at all).
    Returns (symbol -> dict) with the original screener `quote` retained
    (for live/"today" fields) alongside domicile-confirmed extras that
    only .info has - sector/trailing_eps (used by the valuation-adjacent
    scripts), avg_volume_10d (used by swiss_crash_rebound.py AND
    swiss_today_screener.py, so both read the SAME already-fetched
    figure rather than two different volume-averaging methodologies)
    plus a handful of extra current-snapshot fields (dividend yield,
    ex-dividend date, trailing/forward P/E, beta, 52-week range) pulled
    from this SAME .info call at no extra request cost, for scripts that
    want a fuller company profile (see swiss_crash_rebound.py's
    run_scan). Fails soft per ticker: a fetch error just excludes that
    ticker with a warning, rather than aborting the whole scan.

    Runs the per-ticker fetches across max_workers threads (see
    INFO_MAX_WORKERS's own comment for the safety reasoning) instead of
    one ticker at a time - roughly a max_workers-x wall-clock speedup for
    this step, which is I/O-bound (waiting on Yahoo's response, not CPU),
    so threads (not processes) are the right tool despite the GIL.
    """
    domestic = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_domestic_entry, symbol, quote, delay_seconds) for symbol, quote in candidates.items()]
        for future in concurrent.futures.as_completed(futures):
            symbol, entry = future.result()
            if entry is not None:
                domestic[symbol] = entry
    return domestic
