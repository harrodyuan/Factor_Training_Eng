from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

CSS = r'''
body {
  font-family: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  color: #181818;
  background: #ffffff;
  line-height: 1.58;
}
#notebook-container, .jp-Notebook {
  max-width: 740px;
  margin: 0 auto;
  padding: 0;
}
.lecture-header {
  max-width: 740px;
  margin: 0 auto 24px auto;
  padding-top: 2px;
}
.lecture-header .running-title {
  font-size: 14px;
  color: #333333;
  margin-bottom: 6px;
}
.lecture-header .rule {
  border-top: 1px solid #6f6f6f;
  margin-bottom: 18px;
}
.lecture-header .title {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 6px;
}
.lecture-header .meta {
  font-size: 14px;
  color: #4f4f4f;
}
.jp-Notebook > .jp-Cell.jp-MarkdownCell:first-of-type,
#notebook-container > .cell.text_cell:first-of-type {
  display: none !important;
}
.jp-Cell, .cell {
  margin: 0 0 0.8em 0;
  padding: 0;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.jp-MarkdownCell, .text_cell {
  margin-bottom: 0.9em;
  background: transparent !important;
}
.jp-MarkdownCell .jp-Cell-inputWrapper,
.jp-MarkdownCell .jp-InputArea,
.jp-MarkdownCell .jp-RenderedHTMLCommon,
.text_cell_render {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}
.jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2, .jp-RenderedHTMLCommon h3,
h1, h2, h3 {
  font-family: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  color: #101010;
}
h1 { font-size: 26px; margin: 1.1em 0 0.5em 0; }
h2 { font-size: 20px; margin: 1.0em 0 0.45em 0; }
h3 { font-size: 16px; margin: 0.9em 0 0.35em 0; }
p, li { font-size: 15px; }
ul, ol { padding-left: 1.2em; }
code, pre, .jp-InputArea-editor {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
}
.jp-CodeCell, .code_cell {
  margin: 0.5em 0 1.0em 0;
}
.jp-InputPrompt, .jp-OutputPrompt, .prompt {
  display: none !important;
}
.jp-InputArea, div.input {
  border: 1px solid #d8d8d8;
  border-radius: 2px;
  background: #fafafa !important;
}
.jp-OutputArea, div.output_wrapper {
  margin-top: 0.35em;
}
.jp-OutputArea-output, div.output {
  border-left: 2px solid #d9d9d9;
  padding-left: 10px;
}
.jp-RenderedHTMLCommon table, table {
  border-collapse: collapse;
  font-size: 14px;
}
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td, th, td {
  border: 1px solid #d2d2d2;
  padding: 6px 8px;
}
.lecture-pagebreak {
  height: 0;
  border: 0;
  margin: 0;
}
@media screen {
  .lecture-pagebreak {
    margin: 1.1rem 0;
    border-top: 1px solid #e1e1e1;
  }
}
@media print {
  body { margin: 0; }
  .lecture-pagebreak {
    page-break-before: always;
    break-before: page;
    border: 0;
    margin: 0;
    height: 0;
  }
}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--title', required=True)
    parser.add_argument('--course', default='Financial Data Science II')
    parser.add_argument('--updated', required=True)
    parser.add_argument('--output-stem', type=Path)
    return parser.parse_args()


def build_styled_html(raw_html: str, title: str, course: str, updated: str) -> str:
    header = (
        f'<div class="lecture-header">'
        f'<div class="running-title">{title}</div>'
        f'<div class="rule"></div>'
        f'<div class="title">{title}</div>'
        f'<div class="meta">{course}    Updated {updated}</div>'
        f'</div><main>'
    )
    out = raw_html.replace('</head>', f'<style>{CSS}</style></head>', 1)
    out = re.sub(r'(<body[^>]*>)', r'\1' + header, out, count=1)
    out = out.replace('</body>', '</main></body>', 1)
    return out


async def render_pdf(html_path: Path, pdf_path: Path, title: str, course: str, updated: str) -> None:
    footer = (
        "<div style='width:100%; font-size:9px; color:#555; padding:0 24px; "
        "font-family: Palatino, Georgia, serif;'>"
        f"<span>{course} &nbsp;&nbsp; Updated {updated}</span>"
        "<span style='float:right'><span class='pageNumber'></span> / <span class='totalPages'></span></span>"
        "</div>"
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html_path.as_uri(), wait_until='networkidle')
        await page.emulate_media(media='print')
        await page.pdf(
            path=str(pdf_path),
            format='Letter',
            margin={'top': '0.7in', 'right': '0.7in', 'bottom': '0.7in', 'left': '0.7in'},
            print_background=True,
            display_header_footer=True,
            header_template='<div></div>',
            footer_template=footer,
        )
        await browser.close()


def main() -> None:
    args = parse_args()
    notebook = args.notebook.resolve()
    stem = args.output_stem.resolve() if args.output_stem else notebook.with_suffix('')
    tmp_html = stem.parent / f'{stem.name}__raw.html'
    styled_html = stem.parent / f'{stem.name}_lecture.html'
    styled_pdf = stem.parent / f'{stem.name}_lecture.pdf'

    subprocess.run(
        [
            'jupyter', 'nbconvert', '--to', 'html', '--execute', '--output', tmp_html.name, str(notebook)
        ],
        cwd=str(tmp_html.parent),
        check=True,
    )

    raw_html = tmp_html.read_text(encoding='utf-8')
    styled_html.write_text(build_styled_html(raw_html, args.title, args.course, args.updated), encoding='utf-8')
    tmp_html.unlink(missing_ok=True)
    asyncio.run(render_pdf(styled_html, styled_pdf, args.title, args.course, args.updated))


if __name__ == '__main__':
    main()
