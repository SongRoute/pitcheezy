"""Small synthetic checks for document maintenance; no data/model access."""
import pytest

from scripts.check_docs import check, git_files, links, target, graph


def test_paths_length_import_and_preservation(tmp_path):
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    (tmp_path / 'target.md').write_text('# Target\n')
    (tmp_path / 'guide.md').write_text('[good](target.md)\n[bad](missing.md)\n' + '가' * 3000)
    result = check(tmp_path, ['guide.md'], {})
    assert len(result['errors']) == 2
    assert result['local_links'] == 2
    result = check(tmp_path, ['guide.md'], {'guide.md': 'Fixed historical evidence; do not rewrite.'})
    assert result['errors'] == ['guide.md: missing target missing.md']
    (tmp_path / 'CLAUDE.md').write_text('Read AGENTS.md')
    assert 'CLAUDE.md:' in check(tmp_path, [], {})['errors'][0]


def test_fenced_code_references_spaces_and_unicode(tmp_path):
    text = '''```md
[ignored](missing.md)
```
`[also ignored](missing.md)`
[one][ref]
[two][]
[missing][absent]
[space](<한 글.md>)
[ref]: target.md "title"
[two]: target.md
'''
    assert list(links(text)) == [('space', '한 글.md'), ('one', 'target.md'), ('two', 'target.md'), ('missing', None)]
    (tmp_path / '한 글.md').write_text('ok')
    assert target(tmp_path, 'guide.md', '%ED%95%9C%20%EA%B8%80.md#heading') == ('한 글.md', True)
    assert target(tmp_path, 'guide.md', 'https://example.com/a') is None
    assert target(tmp_path, 'guide.md', '../outside.md') is None


def test_git_option_and_markdown_examples_are_not_actions(tmp_path):
    output = tmp_path / 'do-not-write'
    with pytest.raises(ValueError, match='not an option'):
        git_files(tmp_path, '--output=' + str(output))
    assert not output.exists()
    assert list(links(r'\[literal](missing.md)')) == []
    assert list(links('> ```md\n> [example](missing.md)\n> ```\n[actual](exists.md)')) == [('actual', 'exists.md')]


def test_first_reference_and_malformed_url(tmp_path, monkeypatch):
    assert list(links('[x][r]\n[r]: first.md\n[r]: second.md')) == [('x', 'first.md')]
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    (tmp_path / 'guide.md').write_text('[invalid](https://[broken)')
    assert check(tmp_path, ['guide.md'], {})['errors'] == ['guide.md: malformed destination https://[broken']
    monkeypatch.setattr('scripts.check_docs.git_files', lambda root: ['guide.md'])
    result = graph(tmp_path)
    assert result['edges'] == []
    assert result['errors'][0]['error'] == 'malformed destination'
