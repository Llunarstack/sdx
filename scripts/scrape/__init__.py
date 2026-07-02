"""Booru dataset scrapers (danbooru, e621, rule34.xxx) -> SDX JSONL manifests.

Every downloaded post passes through :mod:`scripts.scrape.safety`, a mandatory
filter that drops CSAM-adjacent content. The filter is not optional and is
applied at fetch time so illegal content never lands on disk.
"""
