"""CLI rendering — citations must survive being printed.

Rich reads square brackets as style tags, so `console.print("... [c1].")` silently deletes
the marker. For a product whose whole claim is citation integrity, that turns a correctly
cited answer into an apparently uncited one on screen, while every stored artifact looks
fine. It is invisible unless something asserts on the rendered bytes.
"""

import io

from rich.console import Console
from rich.markup import escape

ANSWER = "Profit rose to 1.2 billion euros [c1]. Costs fell across the division [c2][c3]."


def _render(*args, **kwargs) -> str:
    buf = io.StringIO()
    Console(file=buf, width=200, no_color=True).print(*args, **kwargs)
    return buf.getvalue()


def test_rich_markup_would_eat_citation_markers():
    """Documents the trap this module exists to prevent."""
    assert "[c1]" not in _render(ANSWER)


def test_printing_an_answer_preserves_every_citation_marker():
    out = _render(ANSWER, markup=False)
    for marker in ("[c1]", "[c2]", "[c3]"):
        assert marker in out, f"{marker} was dropped when printing the answer"


def test_a_bracketed_doc_id_survives_the_citation_list():
    line = "- [FSTPST0020260801em81000e1] EU to begin enforcing AI Act (FirstPost.com)"
    assert "[FSTPST0020260801em81000e1]" in _render(line, markup=False)


def test_escaped_titles_keep_their_brackets_alongside_styled_output():
    """Headers carry intentional markup, so untrusted parts are escaped rather than disabled."""
    title = "Deutsche Bank [update] cuts costs"
    out = _render(f"[bold]#1[/bold] {escape(title)}")
    assert "[update]" in out


def test_article_text_with_brackets_is_not_mangled():
    text = "The filing [sic] showed a 12% rise."
    assert "[sic]" in _render(f"  {text}…", markup=False)
