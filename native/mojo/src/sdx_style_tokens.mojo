# SDX style token utilities — comma merge/dedupe and FNV-1a fingerprint.
# Usage: mojo src/sdx_style_tokens.mojo merge "a, b, B, c"
#        mojo src/sdx_style_tokens.mojo fingerprint "glitch cathedral palette"

from collections import Set


fn fnv1a64_bytes(data: String) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    let prime: UInt64 = 0x00000100000001B3
    for i in range(len(data)):
        h = h ^ UInt64(UInt8(ord(data[i])))
        h = h * prime
    return h


fn merge_comma_dedupe(text: String) -> String:
    var seen = Set[String]()
    var out = List[String]()
    for part in text.split(","):
        let p = part.strip()
        if len(p) == 0:
            continue
        let key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return ", ".join(out)


fn main():
    import sys
    let args = sys.argv()
    if len(args) < 2:
        print("usage: sdx_style_tokens.mojo merge <csv> | fingerprint <text>")
        return
    let cmd = args[1]
    if cmd == "merge":
        if len(args) < 3:
            print("")
            return
        print(merge_comma_dedupe(args[2]))
    elif cmd == "fingerprint":
        if len(args) < 3:
            print("0")
            return
        let h = fnv1a64_bytes(args[2])
        print(h)
    else:
        print("unknown command:", cmd)
