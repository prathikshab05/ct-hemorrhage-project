"""Load two specific public BHSD cases without bundling the CT data."""

import hashlib
import re
import struct
import zlib
from urllib.request import Request, urlopen


ARCHIVE = (
    "https://huggingface.co/datasets/Wendy-Fly/BHSD/"
    "resolve/e4751a31843ad7667b4459eaf3fd9698e6793beb/label_192.zip"
)
CASES = {
    "strong": {
        "description": "The predicted mask overlaps much of the reference hemorrhage.",
        "scan": (27788738, 32461356, "45516b59d5f8c8b76550d46e600a200f155a2fd7b9347ab926bec4ff07357b76"),
        "reference": (20702, 22647, "9368c95b5c6d91db83d63d49eba84e322d78b5fb3b023ec777d33f3b201f1c37"),
    },
    "failure": {
        "description": "The predicted mask barely overlaps the reference hemorrhage.",
        "scan": (710880257, 718922827, "afeb1fd2e72e1897c66b8c11bbb2045af300662439d8e55794d5097cdab20372"),
        "reference": (426543, 427811, "3ac5086febcb408514b5ab87966368ad575fb1aeabac2f6997eb995893a4b6b3"),
    },
}


def fetch_member(start, end, expected_sha256):
    request = Request(ARCHIVE, headers={"Range": f"bytes={start}-{end}"})
    length = end - start + 1
    with urlopen(request, timeout=20) as response:
        expected_range = rf"bytes {start}-{end}/\d+"
        if response.status != 206 or not re.fullmatch(expected_range, response.headers.get("Content-Range", "")):
            raise ValueError("The dataset host did not return the requested archive range.")
        data = response.read(length + 1)
    if len(data) != length or data[:4] != b"PK\x03\x04" or struct.unpack_from("<H", data, 8)[0] != 8:
        raise ValueError("The dataset archive entry was incomplete or unexpected.")
    name_size, extra_size = struct.unpack_from("<HH", data, 26)
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        member = decoder.decompress(data[30 + name_size + extra_size:], 16 * 1024**2)
    except zlib.error as error:
        raise ValueError("The dataset archive entry could not be decompressed.") from error
    if not decoder.eof or hashlib.sha256(member).hexdigest() != expected_sha256:
        raise ValueError("The downloaded archive entry did not match the published example.")
    return member


def fetch_case(case_key):
    case = CASES[case_key]
    return fetch_member(*case["scan"]), fetch_member(*case["reference"])
