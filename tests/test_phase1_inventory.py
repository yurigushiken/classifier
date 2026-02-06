import textwrap

from classifier_pipeline.phase1_inventory import (
    count_classifier_tokens,
    extract_zip_url_from_corpus_page,
    parse_chinese_corpora_index,
)

import pylangacq


def test_parse_chinese_corpora_index_extracts_mandarin_entries():
    html = textwrap.dedent(
        """
        <html><body>
        <table>
            <tr><td>Corpus</td><td>Age_Range</td><td>N</td><td>Media</td><td>Comments</td></tr>
            <tr><td>Cantonese</td></tr>
            <tr>
                <td><a href="Cantonese/HKU-70.html">HKU-70</a></td>
                <td>2;6-5;6</td><td>70</td><td>audio</td><td>Cross-sectional</td>
            </tr>
            <tr><td>Mandarin</td></tr>
            <tr>
                <td><a href="Mandarin/Tong.html">Tong</a></td>
                <td>1;11-3;5</td><td>5</td><td>audio</td><td>Longitudinal</td>
            </tr>
            <tr>
                <td><a href="Mandarin/Chang1.html">Chang1</a></td>
                <td>1;7-3;9</td><td>9</td><td>audio</td><td>Longitudinal</td>
            </tr>
            <tr><td>Taiwanese</td></tr>
            <tr>
                <td><a href="Taiwanese/Tsay.html">Tsay</a></td>
                <td>1;8-5;5</td><td>96</td><td>video</td><td>Longitudinal</td>
            </tr>
        </table>
        </body></html>
        """
    ).strip()

    entries = parse_chinese_corpora_index(
        html, base_url="https://talkbank.org/childes/access/Chinese/"
    )
    mandarin = [e for e in entries if e.section == "Mandarin"]

    assert [e.name for e in mandarin] == ["Tong", "Chang1"]
    assert mandarin[0].page_url == "https://talkbank.org/childes/access/Chinese/Mandarin/Tong.html"
    assert mandarin[0].age_range == "1;11-3;5"
    assert mandarin[0].n_files == "5"


def test_extract_zip_url_from_corpus_page():
    html = textwrap.dedent(
        """
        <html><body>
        <a href="https://childes.talkbank.org/data/Chinese/Mandarin/Tong.zip">Download transcripts</a>
        </body></html>
        """
    ).strip()

    zip_url = extract_zip_url_from_corpus_page(html)

    assert zip_url == "https://childes.talkbank.org/data/Chinese/Mandarin/Tong.zip"


def test_count_classifier_tokens_counts_by_participant():
    cha = textwrap.dedent(
        """
        @Begin
        @Participants: CHI Child, MOT Mother
        @ID: chi|Test|CHI|||1;6.|Target_Child||| 
        @ID: mot|Test|MOT|||30;0.|Mother||| 
        *CHI: 我 有 一 个 苹果 .
        *MOT: 给 你 三 只 狗 .
        @End
        """
    ).strip()

    reader = pylangacq.Reader.from_strs([cha], parallel=False)
    classifiers = ["个", "只"]

    counts_all = count_classifier_tokens(reader, classifiers)
    counts_chi = count_classifier_tokens(reader, classifiers, participants={"CHI"})

    assert counts_all["个"] == 1
    assert counts_all["只"] == 1
    assert counts_chi["个"] == 1
    assert counts_chi["只"] == 0
