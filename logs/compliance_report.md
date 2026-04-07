# Compliance Report

Generated at (UTC): 2026-04-07T00:34:47+00:00

## Inputs Checked
- robots.txt: https://www.litcharts.com/robots.txt
- Terms: https://www.litcharts.com/terms (Last revised shown on page: August 5, 2024)
- Target URL family: https://www.litcharts.com/shakescleare/shakespeare-translations/henry-iv-part-1

## Findings
- robots.txt does not globally block all crawling for generic user-agent paths, but specific paths are disallowed.
- Terms Section 9 explicitly states automated downloads/scrapes/spiders are not permitted.
- Terms Section 9 also prohibits creating derivative works from protected content.
- Terms indicate copyrighted translation-related content and proprietary rights protections.

## Decision
- Compliance mode: FALLBACK ONLY
- Allowed actions: local processing of public-domain Shakespeare source lines in repository; structural metadata generation; original paraphrase generation.
- Disallowed actions: automated extraction/crawl/scrape/download of LitCharts content for dataset population.
- Reuse constraints: no verbatim or near-verbatim reuse of LitCharts modern paraphrase text.
- Citation/provenance: each row stores reference_source as URL family and line coordinates metadata; no copied external text stored.

## Operational Policy Applied
- Retrieval indexing and alignment against external paraphrase text disabled.
- method field forced to retrieved_paraphrase_guided_original for compliance traceability.
- Low-confidence rows routed to rejection/adjudication queue.
