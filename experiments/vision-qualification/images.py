#!/usr/bin/env python3
"""Create deterministic orientation/resize probes and fetch the pinned photographic probe."""
import json
import sys
import urllib.request
from pathlib import Path
from PIL import Image

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
source = json.loads(Path(__file__).with_name("image-source.json").read_text())
urllib.request.urlretrieve(source["url"], output / "cats.png")
for name, width, height, pixels in (
    ("wide-gradient", 51, 29, [((x*7+y*3)%256, (y*11)%256, (x*5)%256) for y in range(29) for x in range(51)]),
    ("tall-bands", 31, 57, [((y//9)*40, (x//6)*50, 255-(y//9)*40) for y in range(57) for x in range(31)]),
):
    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    image.save(output / (name + ".png"))
