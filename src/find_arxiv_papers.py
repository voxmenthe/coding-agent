#!/usr/bin/env python3
"""
find_arxiv_papers.py: Find Arxiv paper URLs in cs.* and stat.* categories within a date range and containing specified keywords in title or abstract.

Usage:
  python find_arxiv_papers.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --keywords keyword1 keyword2 [...]
"""
import argparse
import sys
from datetime import datetime

import requests
import feedparser
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find Arxiv paper URLs by date, category, and keywords"
    )
    parser.add_argument(
        '--start-date', required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date', required=True,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--keywords', nargs='+', required=True,
        help='Keywords to search for in title or abstract'
    )
    parser.add_argument(
        '--max-results', type=int, default=100,
        help='Results per request (max 1000)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='show debug info'
    )
    return parser.parse_args()


def build_query(categories, keywords):
    cat_q = ' OR '.join(f'cat:{c}' for c in categories)
    # Add quotes around keywords with spaces for exact phrase matching
    quoted_keywords = []
    for kw in keywords:
        if ' ' in kw:
            quoted_keywords.append(f'all:"{kw}"') # Use f-string directly
        else:
            quoted_keywords.append(f'all:{kw}')
    kw_q = ' OR '.join(quoted_keywords)
    return f'({cat_q}) AND ({kw_q})'


def fetch_entries(query, max_results, verbose, start=0):
    base = 'http://export.arxiv.org/api/query'
    entries = []
    while True:
        url = (
            f"{base}?search_query={query}"
            f"&start={start}&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        if verbose:
            print(f"→ Requesting: {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        if not feed.entries:
            break
        entries.extend(feed.entries)
        if len(feed.entries) < max_results:
            break
        start += max_results
        if verbose:
            print(f"Built query: {query}")
            print(f"Fetched {len(entries)} total entries")
    return entries


def main():
    args = parse_args()
    try:
        start_dt = datetime.fromisoformat(args.start_date)
        end_dt = datetime.fromisoformat(args.end_date)
    except ValueError:
        sys.exit("Error: Dates must be YYYY-MM-DD")

    categories = ['cs.*', 'stat.*']
    query = build_query(categories, args.keywords)
    entries = fetch_entries(query, args.max_results, args.verbose)
    matched_entries = []
    for entry in entries:
        if args.verbose:
            print(f"Entry published: {entry.published[:10]}  Link: {entry.link}")
        pub = datetime.fromisoformat(entry.published[:10])
        if start_dt <= pub <= end_dt:
            text = (entry.title + ' ' + entry.summary).lower()
            # match keyword bounded by start/end, whitespace, parentheses, brackets, or dash
            if any(
                re.search(
                    rf"(?:^|[\s()\[\]\-]){re.escape(kw.lower())}(?:[\s()\[\]\-]|$)",
                    text
                )
                for kw in args.keywords
            ):
                matched_entries.append(entry)

    # write results to file
    date_str = datetime.now().strftime('%Y%m%d')
    keywords_str = '_'.join(args.keywords)
    filename = f"search_{keywords_str}_{date_str}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for e in matched_entries:
            f.write(e.title.strip() + "\n")
            f.write(e.link + "\n")
            f.write(e.summary.strip() + "\n\n")
    print(f"{len(matched_entries)} results found - saved to {filename}")


if __name__ == '__main__':
    main()
