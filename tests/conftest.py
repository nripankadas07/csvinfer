"""Shared fixtures for the csvinfer test-suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def comma_text() -> str:
    return (
        "name,age,active\n"
        "Ada,30,true\n"
        "Grace,40,false\n"
        "Marie,35,true\n"
    )


@pytest.fixture
def tab_text() -> str:
    return "name\tage\tactive\nAda\t30\ttrue\nGrace\t40\tfalse\n"


@pytest.fixture
def semicolon_text() -> str:
    return "name;age;city\nAda;30;London\nGrace;40;Boston\n"


@pytest.fixture
def quoted_text() -> str:
    return (
        'name,bio,year\n'
        '"Ada Lovelace","poet, mathematician",1815\n'
        '"Grace Hopper","admiral, computer scientist",1906\n'
    )


@pytest.fixture
def single_quote_text() -> str:
    return (
        "name,bio\n"
        "'Ada','poet, mathematician'\n"
        "'Grace','computer, scientist'\n"
    )


@pytest.fixture
def crlf_text() -> str:
    return "a,b\r\n1,2\r\n3,4\r\n"


@pytest.fixture
def cr_text() -> str:
    return "a,b\r1,2\r3,4\r"


@pytest.fixture
def bom_text() -> str:
    return "﻿name,age\nAda,30\nGrace,40\n"
