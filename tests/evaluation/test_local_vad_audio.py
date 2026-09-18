from __future__ import annotations

from array import array
import base64
import math
from pathlib import Path
import random
import zlib

import pytest

from services.conversation.local_vad import LocalVadError, LocalVadModel
from services.conversation.speech_activity_gate import SpeechActivityGate


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "vad" / "silero_vad.onnx"


def linear_to_mulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32635) + 132
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (magnitude & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def mulaw(samples) -> bytes:
    return bytes(linear_to_mulaw(int(max(-1.0, min(1.0, value)) * 32767)) for value in samples)


async def accepted(samples) -> bool:
    model = LocalVadModel(MODEL, max_inference_ms=100.0)
    failures = []
    gate = SpeechActivityGate(model, max_inference_ms=100.0, warning=failures.append)
    payload = mulaw(samples)
    for start in range(0, len(payload), 137):
        if (await gate.analyze(payload[start : start + 137])).triggered:
            assert not failures
            return True
        # A retired/failed detector is not evidence of acoustic rejection.
        assert not failures and not gate.retired
    return False


@pytest.mark.asyncio
async def test_real_model_rejects_generated_silence_tones_mixtures_and_seeded_noise():
    sample_rate = 8000
    count = sample_rate * 2
    timeline = [index / sample_rate for index in range(count)]
    fixtures = [[0.0] * count]
    for frequency in (100, 440, 1000, 2000, 3000):
        for amplitude in (0.03, 0.3, 0.8):
            fixtures.append(
                [amplitude * math.sin(2 * math.pi * frequency * t) for t in timeline]
            )
    fixtures.append(
        [
            0.2 * math.sin(2 * math.pi * 440 * t)
            + 0.2 * math.sin(2 * math.pi * 1000 * t)
            + 0.2 * math.sin(2 * math.pi * 2000 * t)
            for t in timeline
        ]
    )
    rng = random.Random(20260917)
    fixtures.extend([
        [amplitude * rng.gauss(0, 1) for _ in range(count)]
        for amplitude in (0.01, 0.03, 0.1, 0.3)
    ])

    assert [await accepted(fixture) for fixture in fixtures] == [False] * len(fixtures)


# A short mono/8-kHz PCM16 excerpt from the upstream Silero VAD MIT-licensed
# examples/c++/aepyx_8k.wav at immutable revision 7e30209a... . The compressed
# bytes and their source checksum are recorded in models/vad/provenance.json.
SPEECH_PCM16_ZLIB_BASE64 = (
    "eNodmwV0VEm3havqenfHgBAI7oO7++DBIbi7uwxuQ3B398HdMjiDu7uHQFzb+0qdd/63srJmEjp9S498e/dh5YhwhZT0XI"
    "4r9+Vu0hpaINePvHWCoq2oTD11sOc/sVHO1AiWu6x9jtiBxnKekf3t/bcNmW/USXkhYqrSxt8htWv61cAdrUJuV56osHbq"
    "PeEGu272STQ/rfkOXjN0VoFjOYvQxu7RKaeyvkC/UCliZnj1oCfSCeamb33vfl79WjExBV7k9uQ74thqHc+amJjDOYjeCb"
    "0e0STivaOqHCEl0ePuU3H6l3pJ3cWSkQfyD7G10MemTfu9y9VC/JVrXL6KEfu1csIp9g4euuvGl/kxI82uFY6MyjNWyePb"
    "nDYwiXld7HNYvTxhuabb6slB8nfB7m/+898vdZL+FMdEFIy8br8XWJC5KNnhykXvhuXPaw+vpS1ivwUQm5uTEo9+nferDa"
    "+ee3j+T8GHrN3ZUxIjM5ub9+0T8yTkfuvYKoUL1aTVsD11XdzI+Am+KyEL83QM7QeL3LFppzLaBi4o+SNSI83Q+3I6bIY1"
    "pKaT/Dr780jWWfvkfK3Dt0iDXFdS3enUXY0Zodsim0WUsHdmr+laoZTxKanU147xW63TETcLOnPUs7qkZSZPySjob6ityJ"
    "0/38ywK+IiKChMkXL5M+KvfqmdckAulP9JAWfQLZ+RNimtd8Zf/ntyi/AOka/CyiucRcslBd1ZPq7B18XpheyrCjTLv8aW"
    "x/0guXt6Bdcdc6byIkdwnhKhw9RgaZU4j+Z29o+7+L1w1lv70PxbIhcq252tkvNntPKc5QuVq8GNwguHarbt8l3Jy9zOFT"
    "8ffT+X8VWrEWlFWEqye3NqmeyX3lfmBkGwlQv7Ghrv+KC2lsvKZwPnEy/HpSV0h2E55+YpGBKsX0o7l97Sk2AdFgerzYJz"
    "h3RxNNceyWvUTLY8e/vvkLjHmcfUy+HVIl7JG11T08dnb9fbCJ2kwmqKltuewxFmq27LUJP4gvRicXfjuvsyg8dETsu1jl"
    "TLqp763fXeyiXNUJLVPdoD2zjbWFuGdl0erq9Nln/++h1pdg91RQbnWk3LZ3ZI7pEVYV1SHqp9bIflampddaJ2UdunnuH3"
    "MnIlTI0f429tPxpxKTxZyOWemvIr465RRbmullI2sTcSqI9sAxxBtixxgntBcrH48emh9HtIeJ4v9id6kYw36TV8EjujiE"
    "qi+EBcpBaz/XBEa38qea0JWZOTaiU4fIlSk1zVc9QWmnsepGW7HltfpaFSJSGFfpWuawsdQY452iT1Y+Bz5oNf15P2By6r"
    "6bnUYKBbPV8ySnon8Q/iKDmG3aQxUl7bQ3sF22NVl6/ojVxK0oukMO9qYUPYgJABcmH9Q/YSTz8jiU4VzwrR0jw10x5sb6"
    "AOl/ZJMh/kGZAyJamS6xl5FZIWOkHrZBV1TXOBZ7llshvSRLW4dts+xtZPPSE+FQeRbYHe2dtSQrPK6dHyfMchbZKQojt8"
    "93x39d+csRjplVLTttS2UUuQM8S7rLhV3nsgo1l6mquVNUc+ptWWn5P+xtGAZL6wEogiJkpTtDq2peoXsSB9TgbDOuu4f4"
    "WrjzvKP8w6RWXhFd1MmvDLXIPnpJ5wliXIMWpFaSz7B/6Eo+Qt6Up2mx7dbv5lXuC5eReztlnGDOUnIJUsFTuKk4V8wnS2"
    "jJUlHclZGkUb0kmkGTSnw9hBoanwLz9uEmN1YHpgTOCF0Z2/hiNkEDlIlwiqCGJhqaFUWiJCe1aexoOddIRBvArfZb7Stc"
    "B5f4h+RS9h+q2LZKXQWVKlotJ+6QGu5iQxgU1inK4ldlKMjzZv6GMCZ/2d/RsDbQ2P2Q7H3JlUY7WFfqJLei3FyM3lTeJP"
    "YSo1+RMYDtP4AGuBUU0ngcH+ff7zgaHmTOjNbEKQ8JS9YP3EqSKRCso15ILSPnYDjvK11ksz0WxrLjYe6H30NjrXX5jT+T"
    "ceSSJYQGgopYgvxJ7SQumEGM68ZBHJgvswgr+zLlmdrDVmaqC936HbzH8tCgkknX0RRou1RLdQSJwmFhPTWWd2j/YiG+Eb"
    "zwluXt9aY3TTJxnzjVmBg/olqySdKPRiJYXa4jNhmCAyL50jPBCThGzakozFtcpvrTefmO/NyWYt87VZn/fht6yWQGiARN"
    "BadDF7IYwV8gh5hZasP61IBPIXzicHXOSVQODZRmUzySxtteJ9IYWMoUtIFNHhFFlNP7JI8aBwhJbHn/+CalCfdKLFaB+4"
    "Yk20hppO47jl4y/4OB5EerNbtAdZRzaSSeQbrnm8kMg+UUY/8wI8zZIhA25DHWLyZCuKf+IAlUgS7CBzySnSB79DWAztxX"
    "IIfVgvWpOM50fM2lYHvpqPhZ10JD0BMtTljC/mBchMuoA+pE7SDRhMgn6kEvzkIilLftBWbDKNI11gPu/KJ3IvzCCn6Wb2"
    "niyEJlAG7vLd0IJUJDlIU1jIR8AXUpPUI7npULIDNvPefDbO3U6TSVnagXrgJa/LS/D71lK4RrvQSvjO//Hu/AQ/Ci3oFz"
    "qULiVVqE6cuHo7IJkXhx5QAyLwd09IEBkGbTm3ipAZdCAZT3PT6XAI/oAlUI42opcAV56852/5eFKIxtLD9CLs4XmJQgK8"
    "OHkBy6A6fUQicBTryAF+hj/n53lTso7OJAvJdJoJg2AjGQR/8PGY1+qQT+Q48ZAVJBoW8UJAIABRrBc7R0+QZOu8FQMpcJ"
    "rEExNqkO+wkReA/+AbhNDCdBm8gxAiQ2XSkhh8I0SS+aQxPYt3rjMUIHdgNEkhl4GTUvQiv8Lz8j95GLkFJUk0nUH/oSXo"
    "P7wI/5cfsSrBONqEptIL9BIvDF+gNjdgBn1MbpPbEAZNyWiSDJfAx6tDPdqH5iJnYTS/Cxo9wD3QioXguTwAr8z8uBqzoC"
    "ibwn4QgaqEW8/5a/6eN6dz6Qp2TOgGETybD7cqQw9yA4ayjkILeo8IEGS1427+L2TiCZlNxhOKJ+EZRIIFe+l4+pRegxmW"
    "HSLw3E7BHY0FwirTz/AWluH9OkYew3+EkxZYHayECVYs3IGWMJHmoLuInb7hcfwf+GVx3odsIA/oLDqNRJIMPpM34yZ/RP"
    "aQt3CI7CJt4DCMwS8ToqAZmUKPkH3EDyOhOPkb5uPfTSKx8IsvtOL4A8jkqSST/kMi6QA8N1vJWdKe7CU3ccR96CBc9QiS"
    "h98lzzBeNWezSAjkwOcf4bchHTh9R3OSC+QfqMdr45NDSTtK4CEpzAqQV+Q8fOL5SR7YABfIVGDkMd63MuQECSEFyX/Uoh"
    "voIbhE4tguchRH/Ix/Ik1oWzyR4/kKqwaP4RtgIDjgJ5TgW8zXVn34jxLxunBL2M7GwXe+3dzl7xZo5H/o22hsND/x/cID"
    "YZd8WItT3tqeBSmOnsJwf6L7gHNKpi1rnrO5Pl+YL+0Sqku1lHXKbSW3VjjoS+gCRyXbVjUq0NJZPLtg+sYMPfOuqwSpI6"
    "VL15UhSlX7O1Vg9aRf2gz7DVsj+YKWbs7LiM3ukNEmtVvG3czrxlGpIRHEkmqiOkOT2XYq25upN5UEdYL6TF3l3pC2zLMi"
    "81HylsybnrE0p1iXpKoFHQ2U2lIyixUmqYfFi9ruoJmONeplzxtPb/+faf+mnHLW92fzxlCLtQteaYuTn6t7pQ2iJR9W7w"
    "X3dSSot5T+nnj3E3eL1NcZ5bI7+NpbtXmEVDxojBajKrYyWAE0YY/EJPtaraftfNA2XyfvJ+/glMcpj9PC3W0CLc1pYgFH"
    "P7W+csw2T65PVwp7lXT7NuWSetD22e127XFFp2xMe53R0K/pJY32QmH5m+iUz6qP5FniGHmWGmX7Q/2s+bR2Xq/3jLdh2p"
    "b0rekjPOe9V/TZckd1rvrEFqUtls8KV8W2Sl11gTrP0dR+UX+qj/cl4GsHpe3xNPJdCxSQXyjP1HhHCy1c7iH/qbTX1tp6"
    "2/raVqplA9H+aM/C1J0Z+TIzvMP81YzXgsVySKnqDXWqelddZO9m62DfZ++qXlBfGg38Qzx70kh6l7Rc7gwP1S3aQiip9F"
    "fDtRScIbct0ZraDtora5Vs7832vhDP7vSFabtS67sfu/r6YvEc7pEvKDFqLnsVW52gkfYzjvWOXFpLLbc5wvuX62z6Hxk8"
    "pZZrkbuWpymvSEZKsXIBG7en2bKCWzpeBu1xNNNq2jnnHpfrj4zWaUdSCzrHusM9keYPWCGcEUpp5R0JtgGha4KTg9+EPN"
    "Qm2D7TMV6P97DrTMbKtMfZNV3Tsn/73cY/sFpIlGrbsjQ1CEIuBLuDCjoSNUuMttb46nslbzfnJ+d3D3M/9/YwvhsL+RXY"
    "xiYocarh6BK8WgtTy0kp8I1s4ueMLjzGumkkGIONuUZzY6xRkH/mD61rfAOPJYXYVaJifl1N6gsNhTUkH+1EM9lHNp3OE7"
    "oyP9nCakO6UcM8FhipDzBO6pvMGGuRVYPUpVUxz+YUlokd5ShptpwmUXqMxPK1xgbrtGW3ClvLzMoW6Pn1qlZZPh9ctCD7"
    "wILFz8JZ4YhQXCTicSFa8LLWdAOctsKMDwYxzxl7rV3mLz23hd0Tr0QvshviGOkv0RRi2SwaLZwXqrIXtAPpye18jNE9cC"
    "NwP1DWmG9eMh/xFjSc3RByC2/pGWGleEScIa0UxwrvWX5ShQ+y/jYLmPmNvvoavT9+PTeGWuWgPzsldpSay5b4FP/ik3hB"
    "rCmGCnfIW6u8sSjg8dsDawLb9en6d72MtRk6syPiV+mNPFfuJbWRJyifpL+EAfQ895qHjJWBZ/6agU6BpoHb/qeBBWYmML"
    "G/dFoaK/0rDpE7KiWUY/JRMZitgrxWghFsJPvv+Rz+qr4e/hHGAMtBVrABWKHvlafLF5WBymt5gVQV134q0bhk/gps9n/1"
    "qf4X/jF6pNnFKgsTyCT2XIpRkpSlSgXlP+WUfEo8QXPAJuu7HhmI92V53/ve+PvpYWYOKwwENk98K3VViilT5e7KYOWM3E"
    "usSZbw41YDY3vgoK+Or6i/ld8b2GYMsWTaVOBCHL56q9JItakpcg/pGK3Io63vxhOdB3b75vrf+pv5f+orzKV8D41n08Qp"
    "8nl5rbJdtZQf8hPxKDlt/WX+awwy4gOf/Cf9Sb6J/ovGDv6N5GG32E1pglJbmapkSUzsJpYRdpGu/JXZ0DppnA6UCPTEdy"
    "6kX7bCSHG6hxniSvmTnFdpq4yXw6XmwhasiFLM+3ogUClwX69pfNJfGyPM79Yh8ooG2DjxuMSluZJD+iaeFz/THbCJXzC7"
    "mt2MMvpXPVu/oP8y9piZfB6LEOdKz4RwoaxYTzwlRokBVoc84VOs6wY1ZxvFjXfmAbOT+dLqipVkCB2OI5Gxsm4k5pPuig"
    "lCF3Yc0iw/7nmYUdWQzWpWIWsPjyMT6Tlqx0qpIvWS/IwI04WlwjO2nn6CLnj61pmWOcCabhXkffkzXhEWQXniIwHqYW+E"
    "FcJE9oqUxJrmEOyC+lAUipHcEMB6M9lymlWtTXwPmc/60RpsmFAf3/Ea+ZNUIZPJd7KUKLDdeocV8CVrHD8HF8l24iWzsB"
    "KktBJNJSNhBbQi47DKnAOzscqKBQPrhamQCDmhE1yFh1iv7IWx9C3bTYvT92Q9aNAX6sE6nsS34OtH4zNWkVJYe/ogE56Q"
    "J1gXLIG9PAJzeWkSyeZiLRRGDvIyWEVl8iVYG/+ABjiTMVhfnMDMfxYS+E3u5JQMpwNYaXoKtsErvoPvxP6iIRlBWpEE0H"
    "l5qEIK4N7uIKPpJOqge3El8mIF1hduwGmcwVwcVResEwZjJWlnh+hufI+OsI9fABfWTKXJBKhAP9GN5C6ZTu5h0C0GF+AG"
    "ceKrH5KtPIJv5T8gG+tEQotSHebw3jCYj+YmwaRADzOJVoKV1g/rHLh4fyhJg+hO8g/xY601kzwgcTjzklhzHYKaIMFuGA"
    "pAVNKT+PHdQrEGbYL19mxeDiYSkdalpbDeXQQDsJ47wr/hHg8i5VhJuhxW87k8P5yHPNgFWKQT7l0iF3BtdsIG8hKfsASu"
    "w1EwiI12JfegJ1/ED5PCVGVpZC0cgH94CRiHHUcL8jcpTOqScrjLx8hWchl3ZjP0hOpkJ0mE1fAfXw0urM4LsM40lT+xtl"
    "nL+WyCQZkNYxtJEIzj5XhuWEqW0R+kHYmDyvAZZ1WZVCatYB4ZSh5hV+GHpxx4KP7uE6zGn5uT3VAbZ7IL1/8L1uqrcG0A"
    "UiE3WQD3IAs+Yc14hcylaeQIH41RvgbZTgN0GZmGe9AE+mDn4oYn0IR8hinkGb5LBI79Ms6ZkyNwBdqT7fCZl8RTmQUxeG"
    "ZewE04gD3Fd1z7LnjyxkJrrFBfwhac7yaYAvXIdCiIfw/Ewr7ZB4tJKsgEewpug2tQn7Zg0awSmYU1PIP+eGKH03Y0HHvA"
    "g3wK1uWVYBipgyxgDl2AVet8KAHbcOd+kVHY+5wi78gDaAA7IQ1OkmpYX/8JCbh/3UgJ7CDiIR9UglmgklvsEV0MVbBniQ"
    "OJzIVtpDp2NRn8A9wnW6mH3IXe+OzB8BzvcB7yBvvh+/j/MUTCTkzDc2RwGbpCNJwj/UgaNMLu9AlcJg/xrozEW/OSL4aO"
    "eMpyYUel421Yi68aRLKwA/qHd+GPIRn3eg/Uxr6qNd1OGuIdicAaOAnWYefSCE9jJizB3odRRkrS79hhLuXp/DzcwnOcl0"
    "hYWb/Ec5aJPdI1uIV7uJmchgqQg6yAP8gAfHUxOImzuI7PLoanYBj2UUF0D1mHp5bQaf+/RyfhCM6rAd5PgBZwkOeFlXjr"
    "tpMw0h/v6Rbog5GmAOzF9XsPImlO/iOH8PstFMWxDsUzsxJ/9uP8f8NCPLu18EznguYY7x7x6VAX+/fOOOriOJ718JgHeD"
    "F+wjoI00gy7nYOwrHHi+ErYDtkkGc0GkdhJ42hEfghHPu1snQentSzuNI3cGwSxjcDd+0PMgznfBVvRSe6Gu9FJzITxkMq"
    "RqRpAMDoLxxPWzoKfsBlGMjr4BlXMH5fwN1oTR7he1/Ekb7CDj0Zbwwh++EUf8wL8fZ8INylkWwr3ub2xMNH8uLInTLJID"
    "qV9IYiGOOqQCvICxVhAfdbo7iXp+N+X6Sr6C9YweP5JShEt2LsnkqK4M3Pi2OX6RjyEc9aLwiCg1CabsPTUIGUgQ74rxsw"
    "Uj/lOl9NF2G3eB6K45j2Qnuu4BOT6F6qYHdYE3e0PZQEAYC3gn/ITLaSxZLfUAsj5QV4DLmgO67MVLqS5MeouwcjZmt4zx"
    "tBR/oH+4hRMQnS+Gx82huM4Aq5gnH3FPfztRgBnuOeTSThuPNbMG+8gEWYeS7BClIV+dggOAV38f5O4c2gLt76Fvi73uDE"
    "+nE+OQy9+Hhc0WEkCzvaZvQcuYVx/xz8i714LVyH25gXyuDabybfIB079TNWcThAi7HF9A3Mt25j9u1J7pHe9CT5xn/gSO"
    "aQMrQEK0VrkP9lp8n8MGzDuLadtIQJSARm4WnYivniJ78GV/AmpJIw+pMMJvfB5JPhK8an2jSWVMcTkgej5Bm+AD6QJfQ4"
    "ValIDvNHvBI+aSXtQ8pBJ54Ps+hI6kSW1JTntpZbTeApGcQeMELnInerhXH8Ka5JB1Ie382BsakJzcLotIOH8LI8GfLSH7"
    "Q/PY3xsxwoZAj2srE8Ek9BM/4V/mWpbAo9DoV5sBXJVxCTuMhNOIav2423bh3muGs8kQ/iB/gPwnDm7TF/f8dsNQjrinHQ"
    "DqPgfLwHZ/Fu/eRhPJlHw3F8+nz6N+5rXeQqEubn/LQCqYE7egi6k5OkP1KOLOQtsWQtOY2RYzXe5b5kH31G+9JPZDjpAV"
    "Vw9yphdBiPMXkZhJFfyBl2Qjdctb/xvlXFrDmctaXHySvIwjN+nKwhk+EjnMUzX4o/4cOR+a3mwXCQCIwLeaQWwjlc51/Y"
    "yXc1r+mnA5GBX7pkTjAjyAIhQqorueU4LcLW035TeUyREenvfbm9291PPFs97b1tA72N0rSYPNR+Nviuo7b9vtZcu6R42T"
    "NrceCV+1X2p8wmWTEuj2diYIbF2BtxrsLthuOmfa5jqf2jVl4ZJzpJXb2Vb5q7fvbFrM3Z+z1LAj4zieYW8yhf1GraUdtV"
    "Ldr20rZFq68cFrOBmD2NEF9nV01nN+ds927vPh2jNpkpdFRUeaRcRVtsa2RbpM1Qj8rFZZU2turrB5wjs/Jn1c+u687ynr"
    "B20G6CT+qNHX0LfPUI7aj8Sjml7FcENYw0Msf5G2fvyCqb3c21wVMtsBhp3HNmUzK0jipRdsvz5FyyTy6h3JUKqkjKda9x"
    "yPU7O9sV4bJ5Qr266SGxdL58QPkgt5fnKyvkT6Kg/Kl+wDlXlvsamXqSr5XzZ3ZUdpZrsaes7qXthU9itrpO/SVtkbaJOc"
    "XH4t/yNOwlVtlK0hRjl5Hoep59IVNwV/QQ7zsK4gYhXSmgNVIXCER4IXRiVdTzSkF5n1pKPQNu/Y2xzbM6q7WTecu7VwUe"
    "0WvsKSsgVde4+FAsoxYSu0uN1aLKDjZR+KS15/H+cmZx1x/OLq78nov+vIE/ySHhBnkjTVa/SuHKv3I7oQkbiPMcJhaTx9"
    "pr8be+lEBVV91sp3OnZ5Xva6ABbyuMYJKk2nrLRaTzUmXxkRAQh0s3xDLiEnW+sDJw3BT8LbM6OIu70t2W/6u5VmgstBFW"
    "2txaiFRYLIGcd5k4SKouHWB35PpqH/N3gJr7nLOz+rpuuB54C+r3sZePY39Kkx1j5fxSXqWeuEHcJ/7L2rIzdLoaENsalb"
    "ke6J05PSPgXOZu6tetfDJT+ooHlRDbDWknEmFOl7MtUpJ0Vryr5NV81hGfYuz0lMgMzy7ncftLW14iqNeUzoJLniONRiZ8"
    "XfQJGVJlySZ8ERqof8s1zSaBWYHWztcZbdwpnj/8DSFCiJJTpdFCvLxMHYi3Zj8zaEkJhI3MJz7VtonnjVLmp0B9Z67Mfc"
    "4Wvi5mLmgkgPJVzKTDxYJSOGtG5lNB2Ci1EkvjXkc4VqCy19ob74tyV86Kds8JbLe6oCKgahfVGqyAECLdgljTQbOlFcp5"
    "ZO9pYrRtldbGvO8t6S/ufp0ekVUw4CETGCohalV1IeNkieAj3BwGB1kF3AtZmi2216KC9tN1ng6eed7/sptlpLi+manMLW"
    "XJJ9XyynH6Fiu3Vcg7iwolUUkYL7UWuogPHC1t8foi51Z3iGdDVlDWKm8HaC5p6lYlVO4pzRN6kTFWSXM3aSQmiYb4TdCk"
    "jkhQLouRnqZZ6a5vro1ZIc4oo6bUXbumDpPmiXvFEcJW66h/lN6M5pP2yC2Vv2VF2eKYEvQRkjOnpOd35XH2zuzqyg9vtf"
    "VBtbSX7BpWKANpOHluTeQrBb+yUJkoTGB35eTgyiGr+Nu0RqkF3fs9/2Tx7I3mWs0WkmkfLdUnlGh0kznIv0K/Key3XdBu"
    "453ar6ghs0KWwaT0d4mVsgZ5PrmHe7dBdXtkaDt7hBBqreePWCSxAh/8VWm82lTbJxSiN6U7oVIuQ7iUeuJn9fSAZ4Snp3"
    "ukabPF51gemi7brEPGRsxrA4zt/kjSUgu3PxRzwmbmCzqZs5WamjXyV6+U2q5h3sue8nqsdBVbrWxV4DmM8qS6cJz/8Ccb"
    "p6Xt9ipKGVpG+Oz4nmOMsjKrfnzxpBruF4EKvun+CKFoyNWcZx1HoXEgzhotxJLhxmVkfONR186BlHuo3Z1zrtbTGRG/Je"
    "G+a4lRSi+k6+xecLGcT+zboWAg3mzE5pMwc4w1Uqprv6RMoyk0SRsbMlkKyx7062SS7BaN5uY2flz6EHQndLZtCaz37zdO"
    "YL/5wxoHXjnRllv5RNfRKbYOIZXE0VmP4/cnhru6GfstkdySxwcVDNHU1ta5QIpVik0kqvWQv5L22JYrkay30M4xObgt/S"
    "e9TPzS5KGefMZTYyfkVI+HrA/dri4wq/j2GWWxjyoHAeqWc6hDpInMLf8X/Mv21f/s97efyentA/15E/jORttWhTQL+iaU"
    "06f4Y6wzqLI0gCfiCXsje0WxEHkuHXVEyZ9cUkJ4opH1NjDBqkwayCccJCRFew2P/JP1e9AM+9HhdJS0XekmfWAZUo/gP4"
    "Kj+ZLkET+dKTl9ua171mOaTysUXDboilSKNzZc1jkaQUzMUdXVMK235GGN1b0OnzA4O+fvv5I2usroVc0apLNyOLh2aD/t"
    "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"
)


@pytest.mark.asyncio
async def test_real_model_detects_pinned_public_speech_fixture():
    raw = zlib.decompress(base64.b64decode(SPEECH_PCM16_ZLIB_BASE64))
    pcm = array("h")
    pcm.frombytes(raw)
    if __import__("sys").byteorder != "little":
        pcm.byteswap()
    samples = [value / 32768.0 for value in pcm]

    assert await accepted(samples)


def test_corrupt_or_missing_model_is_rejected_before_session_creation(tmp_path):
    corrupt = tmp_path / "silero_vad.onnx"
    corrupt.write_bytes(MODEL.read_bytes()[:-1] + b"x")

    with pytest.raises(LocalVadError, match="integrity"):
        LocalVadModel(corrupt)
    with pytest.raises(LocalVadError, match="unavailable"):
        LocalVadModel(tmp_path / "missing.onnx")
