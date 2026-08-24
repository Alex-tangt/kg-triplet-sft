from dataset.filtering import clean_text, is_boilerplate, is_english


def test_is_english_accepts_plain_english():
    assert is_english("This is a sentence about the telegraph and its history on a wire.")


def test_is_english_rejects_cyrillic():
    assert not is_english(
        "Это исследование посвящено обработке текста. Мы рассматриваем задачи анализа данных."
    )


def test_is_english_rejects_cjk():
    assert not is_english("这是一个用于测试语言过滤器的中文段落。我们希望它被正确识别。")


def test_is_english_rejects_empty_and_short():
    assert not is_english("")
    assert not is_english("xy qz")


def test_is_boilerplate_catches_stub_markers():
    assert is_boilerplate("This article is a stub. You can help Wikipedia by expanding it.")
    assert is_boilerplate("References ^ a b c d e f g h i j k")


def test_is_boilerplate_catches_section_headings():
    assert is_boilerplate("See also the following articles and further reading material.")
    assert is_boilerplate("This biography of a living person needs additional sources.")


def test_is_boilerplate_ignores_normal_prose():
    assert not is_boilerplate(
        "The telegraph connected cities and changed the way news was reported."
    )


def test_clean_text_strips_arxiv_artifacts():
    text = "we show @xmath58 values. citing @xcite work [ cols=\"^,^\",options=\"header \" , ] here."
    cleaned = clean_text(text)
    assert "@xmath" not in cleaned
    assert "@xcite" not in cleaned
    assert "cols=" not in cleaned
    assert "we show values. citing work here." in cleaned


def test_clean_text_is_identity_without_artifacts():
    text = "A clean passage without any artifacts in it."
    assert clean_text(text) == text
