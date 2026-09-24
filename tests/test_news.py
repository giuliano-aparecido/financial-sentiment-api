import datetime
import urllib.parse
from types import SimpleNamespace

from app.services import news, news_classifier
from app.services.news_classifier import HeadlineAssessment

_REAL_FETCH_YAHOO_CANDIDATES = news._fetch_yahoo_candidates


class FakeResponse:
    def __init__(self, content=b""):
        self.content = content

    def raise_for_status(self):
        pass


class FakeFeed:
    def __init__(self, entries):
        self.entries = entries


class _StructTime:
    def __init__(self, year, mon, mday):
        self.tm_year, self.tm_mon, self.tm_mday = year, mon, mday


def _entry(title, published="Mon, 22 Jun 2026 00:00", published_parsed=_StructTime(2026, 6, 22)):
    return {"title": title, "published": published, "published_parsed": published_parsed}


def _install(monkeypatch, entries):
    monkeypatch.setattr(news.httpx, "get", lambda url, timeout=None: FakeResponse())
    monkeypatch.setattr(news.feedparser, "parse", lambda content: FakeFeed(entries))


# --- _parse_title_and_publisher ---


def test_parse_title_and_publisher_splits_on_trailing_dash():
    title, publisher = news._parse_title_and_publisher("Oracle cuts guidance - Reuters")
    assert title == "Oracle cuts guidance"
    assert publisher == "Reuters"


def test_parse_title_and_publisher_falls_back_when_no_dash():
    title, publisher = news._parse_title_and_publisher("Oracle cuts guidance")
    assert title == "Oracle cuts guidance"
    assert publisher == news.PUBLISHER_FALLBACK


# --- _is_low_content_headline ---


def test_low_content_headline_catches_trades_up_here_is_why():
    assert news._is_low_content_headline("Intel (INTC) Stock Trades Up, Here Is Why")


def test_low_content_headline_catches_why_stock_dropped_variant():
    assert news._is_low_content_headline("Why Tesla Stock Dropped on Tuesday")
    assert news._is_low_content_headline("Why is Amazon stock rallying today?")
    assert news._is_low_content_headline("Why Adobe (ADBE) Stock Is Falling Today")


def test_low_content_headline_catches_listicle_bait():
    assert news._is_low_content_headline("Should You Buy Microsoft Stock Before April 29?")
    assert news._is_low_content_headline("Is Oracle Stock a Buy at $245?")


def test_low_content_headline_catches_fund_flow_spam():
    assert news._is_low_content_headline("46,643 Shares in Oracle Corporation $ORCL Purchased by Trust Co. of Vermont")


def test_low_content_headline_leaves_real_news_alone():
    assert not news._is_low_content_headline("Oracle cuts guidance, citing softening cloud demand")
    assert not news._is_low_content_headline("Tesla signs major Arizona power deal")


def test_low_content_headline_catches_stock_is_up_today_not_just_shares_are_up_today():
    assert news._is_low_content_headline("Why Netflix (NFLX) Stock Is Up Today")
    assert news._is_low_content_headline("NVDA Stock Is Down Today")


def test_low_content_headline_catches_is_x_still_attractive_after_bait_variant():
    assert news._is_low_content_headline("Is Exxon Mobil (XOM) Still Attractive After A 49% One Year Share Price Surge?")
    assert news._is_low_content_headline("Is Tesla Still a Buy After Its Recent Rally?")


def test_low_content_headline_does_not_catch_genuine_is_x_still_after_analysis_headline():
    # Regression: an early version of this pattern matched ANY "Is X
    # Still Y After Z" shape regardless of what Z was about - the final
    # version requires Z to itself name a price move.
    assert not news._is_low_content_headline(
        "Is the Federal Reserve Still Fighting Inflation After the Latest CPI Report Showed a Surprise Uptick"
    )


# --- 2026-09-18 back-port from portfolio-manager-backend ---


def test_low_content_headline_catches_here_is_why_spelled_out():
    # "here.s why" matched "here's"/"heres" but not "Here Is Why", which
    # is how several outlets write it.
    assert news._is_low_content_headline("IBM Stock Trades Up After Revenue Report, Here Is Why")
    assert news._is_low_content_headline("Here Is Why Nestle Stock Rose Today")
    assert news._is_low_content_headline("Amazon (AMZN): Here Is What Investors Should Know")


def test_low_content_headline_catches_more_fund_filing_shapes():
    # All four results in one live IBM window, 2026-09-18 - shapes the
    # pre-existing branches missed because a share count sits between the
    # verb and "Shares", or the verb isn't in their list.
    assert news._is_low_content_headline(
        "Sequoia Financial Advisors LLC Acquires 14,178 Shares of International Business Machines"
    )
    assert news._is_low_content_headline("Van Hulzen Asset Management LLC Has $31.69 Million Stock Holdings in IBM")
    assert news._is_low_content_headline("Capital Analysts LLC Buys 7,892 Shares of IBM Corporation")


def test_filing_spam_branches_require_a_quantity_not_just_a_financial_word():
    # Every new branch keys on a share count or dollar figure. Gating on a
    # filer keyword instead ("Bank", "Capital", "Financial", "Trust",
    # "Management") would reject these, because those words are also just
    # what financial-sector ISSUERS are called - and a denylist match has
    # nothing downstream to rescue it.
    for title in [
        "Bank of America Buys Stake in Fintech Startup",
        "Deutsche Bank Cuts Position in Troubled Property Unit",
        "IBM Management Raises Stake in Quantum Venture",
        "Berkshire Hathaway Boosts Stake in Occidental Petroleum",
        "Nestle Sells Stake in Its Water Business to Private Equity",
        # A percentage is no better a gate than a keyword: a company
        # raising its own stake reads identically to a fund's 13F delta.
        "Berkshire Hathaway Boosts Stake in Occidental Petroleum by 5%",
        "Vale Reduces Stake in Joint Venture by 30% as Part of Overhaul",
    ]:
        assert not news._is_low_content_headline(title), title


# --- _is_low_quality_publisher ---


def test_denylisted_publisher_matched_however_google_news_spells_it():
    # Confirmed live: exact-string matching let every "marketbeat.com"
    # result through while "MarketBeat" was denied.
    for publisher in [
        "MarketBeat", "marketbeat.com", "Simply Wall St.", "simplywall.st",
        "Zacks", "Zacks.com", "Zacks Investment Research", "TIKR", "TIKR.com",
    ]:
        assert news._is_low_quality_publisher(publisher), publisher


def test_real_publisher_is_not_denylisted():
    for publisher in ["Reuters", "Bloomberg", "Financial Times", "marketscreener.com", "The Motley Fool"]:
        assert not news._is_low_quality_publisher(publisher), publisher


def test_publisher_denylist_matches_whole_tokens_not_raw_prefixes():
    # Short stems are only safe because matching is token-wise: a raw
    # `startswith` on alphanumerics-only keys would let "TIKR" deny
    # "Tikrit Daily" and "Zacks" deny "Zackerman Media".
    assert not news._is_low_quality_publisher("Tikrit Daily")
    assert not news._is_low_quality_publisher("Zackerman Media")
    assert not news._is_low_quality_publisher(news.PUBLISHER_FALLBACK)


# --- company-name folding ---


def test_company_match_name_strips_accents_and_legal_suffixes():
    # `name` is yfinance's registered name; headlines write the plain
    # brand. Both ends are stripped - tail-only leaves "the coca-cola".
    assert news._company_match_name("Nestlé S.A.") == "nestle"
    assert news._company_match_name("Alphabet Inc.") == "alphabet"
    assert news._company_match_name("Mondi plc") == "mondi"
    assert news._company_match_name("The Coca-Cola Company") == "coca-cola"


def test_relevant_headline_matches_registered_name_against_plain_brand():
    assert news._is_relevant_headline("NESN", "Nestlé S.A.", None, "Nestle raises full-year outlook")
    assert news._is_relevant_headline("KO", "The Coca-Cola Company", None, "Coca-Cola lifts full-year guidance")
    # A real 2-character brand still works - the floor is 2, not 3.
    assert news._is_relevant_headline("MMM", "3M Company", None, "3M raises full-year guidance")


def test_relevant_headline_matches_name_on_word_boundary_not_substring():
    # "Nestle" must not match inside another word.
    assert not news._is_relevant_headline("NESN", "Nestlé S.A.", None, "Nestleford Council raises rates")


def test_relevant_headline_requires_the_full_name_when_the_core_is_an_ordinary_word():
    # Suffix stripping goes too far for these: the bare core matches
    # unrelated headlines, and the name tier short-circuits the rest of
    # _is_relevant_headline, so the headline is served as if it were
    # about this company.
    assert not news._is_relevant_headline(
        "TGT", "Target Corporation", None, "Analysts Raise Price Target for Nvidia to $200"
    )
    assert not news._is_relevant_headline("SE", "Sea Limited", None, "Coast Guard Rescues Sailors After Storm at Sea")
    assert not news._is_relevant_headline("BOX", "Box Inc.", None, "Amazon Warns of Cardboard Box Shortage")
    # The full name still matches, which is the pre-stripping behaviour.
    assert news._is_relevant_headline(
        "TGT", "Target Corporation", None, "Target Corporation reports record holiday sales"
    )


def test_low_content_headline_leaves_acquisition_news_alone():
    # A "Shares of X ... Acquired by Y" branch with a gap between the
    # halves also matches ordinary M&A reporting, which states a cause
    # and is exactly what this filter should keep.
    for title in [
        "Shares of Activision Jumped After the Company Was Acquired by Microsoft",
        "Shares of Splunk Surged After It Agreed to Be Acquired by Cisco",
    ]:
        assert not news._is_low_content_headline(title), title


def test_relevant_headline_matches_a_name_whose_legal_suffix_contains_a_slash():
    # Regression test for the Danish "A/S" legal-suffix slash - see
    # _company_match_name's docstring.
    assert news._is_relevant_headline(
        "NVO", "Novo Nordisk A/S", "Healthcare",
        "Novo Nordisk Stock Slides After Unveiling Long Term Pipeline Growth Targets",
    )


def test_slash_in_a_headline_does_not_merge_two_names_into_one_token():
    # Regression test for _normalize_for_match's slash-preservation
    # contract - see its docstring.
    assert news._is_relevant_headline("BIDU", "Baidu", "Technology", "Baidu/Alibaba race for AI dominance")


def test_relevant_headline_ignores_a_single_character_name():
    # routers/analyze.py falls back to the ticker when yfinance has no
    # company name, so `name` can be "V"/"F". The name tier is
    # case-INsensitive, so matching on one letter would make every "v."
    # citation a Visa story; the ticker tier below is the case-sensitive
    # one and still applies.
    assert not news._is_relevant_headline("V", "V", None, "Apple v. Epic Systems ruling lands")


# --- _is_relevant_headline ---


def test_relevant_headline_matches_company_name():
    assert news._is_relevant_headline("ORCL", "Oracle", None, "Oracle cuts cloud guidance")


def test_relevant_headline_matches_ticker_as_standalone_token():
    assert news._is_relevant_headline("NVDA", "Nvidia", None, "NVDA jumps on datacenter demand")


def test_relevant_headline_does_not_match_ticker_as_lowercase_substring():
    # "MA" (Mastercard) must not match inside ordinary words like "market"
    assert not news._is_relevant_headline("MA", "Mastercard", "Financial Services", "market rally continues")


def test_relevant_headline_matches_sector_keyword_without_naming_company():
    assert news._is_relevant_headline("XOM", "Exxon Mobil", "Energy", "OPEC agrees to cut oil output")


def test_relevant_headline_sector_keyword_uses_word_boundaries():
    # "oil" must not match inside "turmoil", "gas" must not match inside "Vegas"
    assert not news._is_relevant_headline("XOM", "Exxon Mobil", "Energy", "Markets face turmoil in Las Vegas")


def test_relevant_headline_rejects_unrelated_macro_news():
    assert not news._is_relevant_headline("ORCL", "Oracle", "Technology", "Fed leaves interest rates unchanged")


def test_relevant_headline_unlisted_sector_falls_back_to_name_ticker_only():
    assert not news._is_relevant_headline("O", "Realty Income", "Real Estate", "Mortgage rates tick higher")


# --- fetch_live_news_rag ---


def test_selects_most_relevant_and_recent_over_irrelevant_newer_headline(monkeypatch):
    _install(monkeypatch, [
        _entry("Wall Street mixed as investors await earnings season - Bloomberg", published_parsed=_StructTime(2026, 6, 23)),
        _entry("Oracle cuts guidance, citing softening cloud demand - Reuters", published_parsed=_StructTime(2026, 6, 22)),
    ])
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert "Oracle cuts guidance" in result
    assert "Wall Street mixed" not in result
    assert published_date == datetime.date(2026, 6, 22)


def test_selects_most_recent_among_multiple_relevant_candidates(monkeypatch):
    _install(monkeypatch, [
        _entry("Oracle signs new cloud deal - Reuters", published_parsed=_StructTime(2026, 6, 20)),
        _entry("Oracle cuts guidance, citing softening cloud demand - Bloomberg", published_parsed=_StructTime(2026, 6, 22)),
    ])
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert "Oracle cuts guidance" in result
    assert published_date == datetime.date(2026, 6, 22)


def test_query_strips_exchange_suffix_before_searching(monkeypatch):
    # Regression: once app.services.ticker started preserving an explicit
    # exchange suffix ("$ARYN.SW" -> "ARYN.SW"), the SAME suffixed value
    # flowed into the Google News query here - confirmed live, "ARYN.SW
    # stock earnings financial news" returns ZERO results (no article's
    # text contains that literal token), while the bare ticker or company
    # name both return real results.
    captured_urls = []
    monkeypatch.setattr(news.httpx, "get", lambda url, timeout=None: (captured_urls.append(url), FakeResponse())[1])
    monkeypatch.setattr(news.feedparser, "parse", lambda content: FakeFeed([
        _entry("ARYZTA (SWX:ARYN) posts wider losses - Reuters"),
    ]))

    news.fetch_live_news_rag("ARYN.SW", "Aryzta", "Consumer Defensive")

    assert len(captured_urls) == 1
    assert "SW" not in urllib.parse.unquote(captured_urls[0]).split("q=")[1].split("&")[0]
    assert "ARYN" in urllib.parse.unquote(captured_urls[0])


def test_relevance_filter_matches_bare_ticker_when_input_has_exchange_suffix(monkeypatch):
    # Same regression, the other direction: a headline that only ever
    # writes the bare ticker ("SWX:ARYN") must still be recognized as
    # relevant even though the ticker passed in has ".SW" attached -
    # re.escape("ARYN.SW") would otherwise require the literal ".SW" text
    # to appear in the headline too, which real articles never write.
    _install(monkeypatch, [
        _entry("ARYZTA (SWX:ARYN) posts wider losses - Reuters"),
    ])
    result, _published_date = news.fetch_live_news_rag("ARYN.SW", "Aryzta", "Consumer Defensive")
    assert "ARYZTA" in result
    assert result != "Data unavailable."


def test_filters_out_low_quality_publisher(monkeypatch):
    _install(monkeypatch, [
        _entry("46,643 Shares in Oracle Corporation $ORCL Purchased by Trust Co. - MarketBeat"),
        _entry("Oracle cuts guidance, citing softening cloud demand - Reuters"),
    ])
    result, _published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert "MarketBeat" not in result
    assert "Oracle cuts guidance" in result


def test_filters_out_low_content_headline_from_a_normal_publisher(monkeypatch):
    _install(monkeypatch, [
        _entry("Why Tesla Stock Dropped on Tuesday - CNBC"),
        _entry("Tesla signs major Arizona power deal - CNBC"),
    ])
    result, _published_date = news.fetch_live_news_rag("TSLA", "Tesla", "Consumer Cyclical")
    assert "Why Tesla Stock Dropped" not in result
    assert "Tesla signs major Arizona power deal" in result


def test_falls_back_to_most_recent_quality_filtered_entry_when_none_are_relevant(monkeypatch):
    _install(monkeypatch, [
        _entry("Wall Street mixed as investors await earnings season - Bloomberg", published_parsed=_StructTime(2026, 6, 20)),
        _entry("Bond markets rally as recession fears ease - Reuters", published_parsed=_StructTime(2026, 6, 22)),
    ])
    result, published_date = news.fetch_live_news_rag("TSLA", "Tesla", "Consumer Cyclical")
    assert result != "Data unavailable."
    assert "Bond markets rally" in result
    assert published_date == datetime.date(2026, 6, 22)


def test_falls_back_to_most_recent_raw_entry_when_every_candidate_is_low_quality(monkeypatch):
    _install(monkeypatch, [
        _entry("46,643 Shares Purchased by Trust Co. - MarketBeat", published_parsed=_StructTime(2026, 6, 20)),
        _entry("TSLA Looks 2.0% Overvalued on GF Value - GuruFocus", published_parsed=_StructTime(2026, 6, 22)),
    ])
    result, published_date = news.fetch_live_news_rag("TSLA", "Tesla", "Consumer Cyclical")
    assert result != "Data unavailable."
    assert "GF Value" in result
    assert published_date == datetime.date(2026, 6, 22)


def test_selects_exactly_one_headline(monkeypatch):
    _install(monkeypatch, [_entry(f"Real headline number {i} about Apple - Reuters") for i in range(10)])
    result, _published_date = news.fetch_live_news_rag("AAPL", "Apple", "Technology")
    assert len(result.splitlines()) == 1


def test_output_line_format_includes_date_title_and_publisher(monkeypatch):
    _install(monkeypatch, [_entry("Oracle cuts guidance, citing softening cloud demand - Reuters", published="Mon, 22 Jun 2026 00:00")])
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert result == "- [Mon, 22 Jun 2026] Oracle cuts guidance, citing softening cloud demand - Reuters"
    assert published_date == datetime.date(2026, 6, 22)


def test_no_entries_returns_data_unavailable(monkeypatch):
    _install(monkeypatch, [])
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert result == "Data unavailable."
    assert published_date is None


def test_fetch_error_returns_data_unavailable(monkeypatch):
    def _boom(url, timeout=None):
        raise Exception("network error")
    monkeypatch.setattr(news.httpx, "get", _boom)
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert result == "Data unavailable."
    assert published_date is None


def test_unparseable_published_date_falls_back_to_rss_order(monkeypatch):
    _install(monkeypatch, [
        _entry("Oracle cuts guidance, citing softening cloud demand - Reuters", published_parsed=None),
        _entry("Oracle signs new cloud deal - Bloomberg", published_parsed=None),
    ])
    result, published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert "Oracle cuts guidance" in result
    assert published_date is None


# --- Gemini screen and Yahoo source ---


def _classifier(verdicts, calls=None):
    def classify(company, ticker, sector, headlines):
        if calls is not None:
            calls.append([title for title, _ in headlines])
        return [HeadlineAssessment(*verdicts.get(title, (False, 1))) for title, _ in headlines]

    return classify


def _yahoo(title, published_date, publisher="Yahoo Finance"):
    display = published_date.strftime("%a, %d %b %Y") if published_date else ""
    return news._Candidate(title, publisher, display, published_date)


def test_classifier_picks_the_newest_important_relevant_headline(monkeypatch):
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({
        "Novo Sets Long-Term Targets": (True, 4),
        "Novo Nordisk lifts outlook": (True, 5),
        "Novo Banco reports record profit": (False, 5),
        "Should you buy Novo Nordisk?": (True, 1),
    }))
    _install(monkeypatch, [
        _entry("Should you buy Novo Nordisk? - Motley Fool", published_parsed=_StructTime(2026, 6, 24)),
        _entry("Novo Banco reports record profit - Reuters", published_parsed=_StructTime(2026, 6, 24)),
        _entry("Novo Sets Long-Term Targets - Reuters", published="Tue, 23 Jun 2026 00:00", published_parsed=_StructTime(2026, 6, 23)),
        _entry("Novo Nordisk lifts outlook - Reuters", published_parsed=_StructTime(2026, 5, 4)),
    ])
    result, published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert result == "- [Tue, 23 Jun 2026] Novo Sets Long-Term Targets - Reuters"
    assert published_date == datetime.date(2026, 6, 23)


def test_within_the_past_week_importance_outranks_recency(monkeypatch):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({
        "BAT names new CEO and cuts guidance": (True, 5),
        "Regulator speeds nicotine pouch approvals": (True, 3),
    }))
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [
        _yahoo("Regulator speeds nicotine pouch approvals", today),
        _yahoo("BAT names new CEO and cuts guidance", today - datetime.timedelta(days=2)),
    ])
    _install(monkeypatch, [])
    result, _published_date = news.fetch_live_news_rag("BATS.L", "British American Tobacco p.l.c.", "Consumer Defensive")
    assert "BAT names new CEO and cuts guidance" in result


def test_a_recent_minor_headline_beats_an_old_important_one(monkeypatch):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({
        "Novo Nordisk lifts outlook": (True, 5),
        "Novo Nordisk shares edge higher on pipeline hopes": (True, 2),
    }))
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [
        _yahoo("Novo Nordisk lifts outlook", today - datetime.timedelta(days=50)),
        _yahoo("Novo Nordisk shares edge higher on pipeline hopes", today - datetime.timedelta(days=1)),
    ])
    _install(monkeypatch, [])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert "Novo Nordisk shares edge higher on pipeline hopes" in result


def test_a_recent_noise_headline_does_not_beat_an_older_important_one(monkeypatch):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({
        "Novo Nordisk cuts guidance": (True, 5),
        "Novo Nordisk shares acquired by Some Advisors LLC": (True, 1),
    }))
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [
        _yahoo("Novo Nordisk cuts guidance", today - datetime.timedelta(days=10)),
        _yahoo("Novo Nordisk shares acquired by Some Advisors LLC", today - datetime.timedelta(days=1)),
    ])
    _install(monkeypatch, [])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert "Novo Nordisk cuts guidance" in result


def test_classifier_falls_back_to_unimportant_relevant_headlines(monkeypatch):
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({"Novo Nordisk shares edge higher": (True, 2)}))
    _install(monkeypatch, [
        _entry("Novo Banco reports record profit - Reuters"),
        _entry("Novo Nordisk shares edge higher - Reuters"),
    ])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert "Novo Nordisk shares edge higher" in result


def test_nothing_relevant_to_the_classifier_falls_back_to_the_regex_selection(monkeypatch):
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({}))
    _install(monkeypatch, [
        _entry("Novo Banco reports record profit - Reuters", published_parsed=_StructTime(2026, 6, 24)),
        _entry("Oracle cuts guidance, citing softening cloud demand - Reuters", published_parsed=_StructTime(2026, 6, 20)),
    ])
    result, _published_date = news.fetch_live_news_rag("ORCL", "Oracle", "Technology")
    assert "Oracle cuts guidance" in result


def test_classifier_breaks_an_importance_tie_by_recency(monkeypatch):
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({
        "Novo cuts prices": (True, 3),
        "Novo opens a plant": (True, 3),
    }))
    _install(monkeypatch, [
        _entry("Novo cuts prices - Reuters", published_parsed=_StructTime(2026, 6, 20)),
        _entry("Novo opens a plant - Reuters", published_parsed=_StructTime(2026, 6, 22)),
    ])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert "Novo opens a plant" in result


def test_a_denylisted_publisher_can_still_be_picked_by_the_classifier(monkeypatch):
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({"Novo Sets Long-Term Targets": (True, 4)}))
    _install(monkeypatch, [
        _entry("Novo Sets Long-Term Targets - TIKR"),
        _entry("Novo Nordisk shares edge higher - Reuters"),
    ])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert result.endswith("Novo Sets Long-Term Targets - TIKR")


def test_yahoo_headlines_join_the_candidates_deduped_against_google(monkeypatch):
    calls = []
    title = "Novo Nordisk Stock Slides After Unveiling Long Term Pipeline Growth Targets"
    monkeypatch.setattr(news_classifier, "classify_headlines", _classifier({title: (True, 4)}, calls))
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [
        _yahoo(title, datetime.date(2026, 9, 23), publisher="Barrons.com"),
        _yahoo("novo banco reports record profit", datetime.date(2026, 9, 23)),
    ])
    _install(monkeypatch, [_entry("Novo Banco reports record profit - Reuters")])
    result, published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert calls == [["Novo Banco reports record profit", title]]
    assert result == f"- [Wed, 23 Sep 2026] {title} - Barrons.com"
    assert published_date == datetime.date(2026, 9, 23)


def test_yahoo_candidates_survive_a_google_fetch_error(monkeypatch):
    def _boom(url, timeout=None):
        raise Exception("network error")

    monkeypatch.setattr(news.httpx, "get", _boom)
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [
        _yahoo("Novo Nordisk wins FDA approval", datetime.date(2026, 9, 23)),
    ])
    result, _published_date = news.fetch_live_news_rag("NVO", "Novo Nordisk A/S", "Healthcare")
    assert "Novo Nordisk wins FDA approval" in result


def test_yahoo_news_is_parsed_from_the_content_payload(monkeypatch):
    payload = [
        {"content": {
            "title": "Novo Sets Long-Term Targets",
            "pubDate": "2026-09-23T14:05:00Z",
            "provider": {"displayName": "Barrons.com"},
        }},
        {"content": {"title": "", "pubDate": "2026-09-23T14:05:00Z"}},
        {"content": {"title": "No provider", "pubDate": "not a date"}},
    ]
    monkeypatch.setattr(news.yf, "Ticker", lambda ticker: SimpleNamespace(news=payload))
    assert _REAL_FETCH_YAHOO_CANDIDATES("NVO") == [
        news._Candidate("Novo Sets Long-Term Targets", "Barrons.com", "Wed, 23 Sep 2026", datetime.date(2026, 9, 23)),
        news._Candidate("No provider", "Yahoo Finance", "", None),
    ]


def test_a_yahoo_failure_yields_no_candidates(monkeypatch):
    def _boom(ticker):
        raise RuntimeError("crumb")

    monkeypatch.setattr(news.yf, "Ticker", _boom)
    assert _REAL_FETCH_YAHOO_CANDIDATES("NVO") == []


def test_a_malformed_yahoo_payload_yields_no_candidates(monkeypatch):
    monkeypatch.setattr(news.yf, "Ticker", lambda ticker: SimpleNamespace(news=["not a dict"]))
    assert _REAL_FETCH_YAHOO_CANDIDATES("NVO") == []
