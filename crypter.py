#!/usr/bin/env python3
"""
File Crypter - AES-256-GCM symmetric file/directory encryption tool.

Encryption scheme
-----------------
  KDF   : Argon2id  (time=4, memory=65536 KiB, parallelism=2)
  Cipher: AES-256-GCM streaming (64 KiB chunks)
  Auth  : per-chunk AAD = chunk index (prevents reordering / truncation)
  MAC   : HMAC-SHA256 over all ciphertext (end-to-end integrity sentinel)

File format (CRYPT03)
---------------------
  [ MAGIC   8 B ]  = "CRYPT03\\x00"
  [ FLAGS   1 B ]  bit 0 = lzma-compressed plaintext
  [ SALT   32 B ]
  [ NONCE  12 B ]
  [ CHUNKS …    ]  each chunk: [ length 4 B BE ][ ciphertext+GCM-tag ]
  [ HMAC   32 B ]  HMAC-SHA256(key, all chunk bytes)

Extras added in this version
-----------------------------
  --compress / -z   Compress plaintext with lzma before encrypting
                    (great for text, logs, archives — auto-detected on decrypt)
  --shred  / -s     Securely overwrite source file (3 random passes) then
                    delete it after successful encryption
  verify            Check HMAC + GCM tags of an encrypted file without
                    writing any output — fast "did this survive intact?" test
  Progress bar      Shown automatically for files > 1 MiB
"""

import lzma
import os
import sys
import struct
import hmac
import hashlib
import argparse
import getpass
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC_V3       = b"CRYPT03\x00"
MAGIC_V2       = b"CRYPT02\x00"   # legacy — read-only support
SALT_SIZE      = 32
NONCE_SIZE     = 12
CHUNK_SIZE     = 64 * 1024        # 64 KiB plaintext per chunk
KEY_LEN        = 32               # AES-256
HMAC_SIZE      = 32               # SHA-256

FLAG_COMPRESSED = 0x01            # bit 0 of FLAGS byte

SHRED_PASSES   = 3

# Argon2id (OWASP recommended minimum for file encryption)
ARGON2_TIME    = 4
ARGON2_MEMORY  = 65_536           # KiB = 64 MiB
ARGON2_PARA    = 2

# Header sizes
_V3_HEADER     = len(MAGIC_V3) + 1 + SALT_SIZE + NONCE_SIZE   # magic+flags+salt+nonce
_V2_HEADER     = len(MAGIC_V2)     + SALT_SIZE + NONCE_SIZE   # magic+salt+nonce


# ---------------------------------------------------------------------------
# Key derivation — Argon2id
# ---------------------------------------------------------------------------

def derive_key(password: str, salt: bytes) -> bytes:
    return hash_secret_raw(
        secret=password.encode(),
        salt=salt,
        time_cost=ARGON2_TIME,
        memory_cost=ARGON2_MEMORY,
        parallelism=ARGON2_PARA,
        hash_len=KEY_LEN,
        type=Type.ID,
    )


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

_PROGRESS_THRESHOLD = 1 * 1024 * 1024   # show bar for files > 1 MiB

def _make_progress(total: int):
    """Return a (update(n), done()) pair or no-ops when file is small."""
    if total <= _PROGRESS_THRESHOLD:
        return (lambda n: None), (lambda: None)

    done_bytes = [0]
    width = 36

    def update(n: int) -> None:
        done_bytes[0] += n
        pct   = min(done_bytes[0] / total, 1.0)
        filled = int(width * pct)
        bar   = "█" * filled + "░" * (width - filled)
        mb_done  = done_bytes[0] / 1_048_576
        mb_total = total         / 1_048_576
        print(f"\r  [{bar}] {pct:5.1%}  {mb_done:.1f}/{mb_total:.1f} MiB",
              end="", flush=True)

    def finish() -> None:
        print()   # newline after the bar

    return update, finish


# ---------------------------------------------------------------------------
# Streaming encrypt / decrypt helpers
# ---------------------------------------------------------------------------

def _encrypt_stream(fin, fout, key: bytes, nonce: bytes,
                    *, progress=None) -> bytes:
    """Encrypt *fin* → *fout* in chunks; return HMAC over written bytes."""
    aesgcm    = AESGCM(key)
    mac       = hmac.new(key, digestmod=hashlib.sha256)
    chunk_idx = 0

    while True:
        plaintext = fin.read(CHUNK_SIZE)
        if not plaintext:
            break
        aad    = struct.pack(">Q", chunk_idx)
        ct     = aesgcm.encrypt(nonce, plaintext, aad)
        prefix = struct.pack(">I", len(ct))
        fout.write(prefix)
        fout.write(ct)
        mac.update(prefix)
        mac.update(ct)
        if progress:
            progress(len(plaintext))
        chunk_idx += 1

    return mac.digest()


def _decrypt_stream(fin, fout, key: bytes, nonce: bytes,
                    expected_hmac: bytes, *, progress=None) -> None:
    """Decrypt chunks from *fin* → *fout*; raise on auth failure."""
    aesgcm    = AESGCM(key)
    mac       = hmac.new(key, digestmod=hashlib.sha256)
    chunk_idx = 0

    while True:
        length_bytes = fin.read(4)
        if not length_bytes:
            break
        if len(length_bytes) < 4:
            raise ValueError("Truncated file — missing chunk length.")
        ct_len = struct.unpack(">I", length_bytes)[0]
        ct     = fin.read(ct_len)
        if len(ct) < ct_len:
            raise ValueError("Truncated file — incomplete chunk.")

        mac.update(length_bytes)
        mac.update(ct)

        aad = struct.pack(">Q", chunk_idx)
        try:
            plaintext = aesgcm.decrypt(nonce, ct, aad)
        except Exception:
            raise ValueError("Decryption failed — wrong password or corrupted file.")

        fout.write(plaintext)
        if progress:
            progress(len(plaintext))
        chunk_idx += 1

    if not hmac.compare_digest(mac.digest(), expected_hmac):
        raise ValueError("HMAC mismatch — file has been tampered with.")


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _compress_to_tmp(src: Path) -> Path:
    """lzma-compress *src* into a sibling temp file; return its path."""
    tmp = src.with_suffix(src.suffix + ".lzmatmp")
    with src.open("rb") as fin, lzma.open(tmp, "wb", preset=6) as fout:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            fout.write(chunk)
    return tmp


def _decompress_tmp(src: Path) -> Path:
    """lzma-decompress *src* into a sibling temp file; return its path."""
    tmp = src.with_suffix(src.suffix + ".lzmatmp")
    with lzma.open(src, "rb") as fin, tmp.open("wb") as fout:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            fout.write(chunk)
    return tmp


# ---------------------------------------------------------------------------
# Secure shred
# ---------------------------------------------------------------------------

def shred_file(path: Path) -> None:
    """Overwrite *path* with random bytes (SHRED_PASSES times) then delete."""
    size = path.stat().st_size
    with path.open("r+b") as f:
        for _ in range(SHRED_PASSES):
            f.seek(0)
            # Write in chunks to avoid large allocations
            remaining = size
            while remaining:
                n = min(remaining, CHUNK_SIZE)
                f.write(os.urandom(n))
                remaining -= n
            f.flush()
            os.fsync(f.fileno())
    path.unlink()
    print(f"[+] Shredded: {path}  ({SHRED_PASSES} passes)")
    print("    Note: on SSDs wear-levelling may retain copies in flash cells.")


# ---------------------------------------------------------------------------
# Single-file encrypt / decrypt
# ---------------------------------------------------------------------------

def encrypt_file(src: Path, dst: Path, password: str,
                 *, compress: bool = False, shred: bool = False) -> None:
    _check_no_overwrite(dst)

    salt  = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key   = derive_key(password, salt)
    flags = FLAG_COMPRESSED if compress else 0x00

    # Optional: compress plaintext to a temp file first
    compress_tmp: Path | None = None
    enc_src = src
    if compress:
        print(f"[*] Compressing {src} …")
        compress_tmp = _compress_to_tmp(src)
        enc_src = compress_tmp
        ratio = compress_tmp.stat().st_size / max(src.stat().st_size, 1)
        print(f"    Compression ratio: {ratio:.1%}")

    file_size   = enc_src.stat().st_size
    upd, finish = _make_progress(file_size)

    dst_tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with enc_src.open("rb") as fin, dst_tmp.open("wb") as fout:
            header = MAGIC_V3 + bytes([flags]) + salt + nonce
            fout.write(header)
            digest = _encrypt_stream(fin, fout, key, nonce, progress=upd)
            fout.write(digest)
        finish()
        dst_tmp.replace(dst)
    except Exception:
        dst_tmp.unlink(missing_ok=True)
        raise
    finally:
        if compress_tmp:
            compress_tmp.unlink(missing_ok=True)

    print(f"[+] Encrypted: {src}  →  {dst}")

    if shred:
        shred_file(src)


def decrypt_file(src: Path, dst: Path, password: str) -> None:
    _check_no_overwrite(dst)

    with src.open("rb") as f:
        magic = f.read(len(MAGIC_V3))

        if magic == MAGIC_V3:
            flags_byte = f.read(1)
            flags      = flags_byte[0] if flags_byte else 0
            header_len = _V3_HEADER
        elif magic == MAGIC_V2:
            flags      = 0
            header_len = _V2_HEADER
        else:
            raise ValueError(
                "Not a valid crypter file (bad magic). "
                "Was it encrypted with a different tool?"
            )

        salt  = f.read(SALT_SIZE)
        nonce = f.read(NONCE_SIZE)

        f.seek(-HMAC_SIZE, 2)
        expected_hmac = f.read(HMAC_SIZE)

        f.seek(header_len)

        key         = derive_key(password, salt)
        file_size   = src.stat().st_size - header_len - HMAC_SIZE
        upd, finish = _make_progress(max(file_size, 0))

        compressed   = bool(flags & FLAG_COMPRESSED)
        dec_tmp_path = dst.with_suffix(dst.suffix + ".dectmp") if compressed else None
        out_path     = dec_tmp_path if compressed else dst
        dst_tmp      = out_path.with_suffix(out_path.suffix + ".tmp")

        try:
            with dst_tmp.open("wb") as fout:
                _decrypt_stream(
                    _BoundedReader(f, -HMAC_SIZE), fout, key, nonce,
                    expected_hmac, progress=upd,
                )
            finish()
            dst_tmp.replace(out_path)

            if compressed:
                print(f"[*] Decompressing …")
                decomp_tmp = _decompress_tmp(out_path)
                dec_tmp_path.unlink(missing_ok=True)
                decomp_tmp.replace(dst)
        except Exception:
            dst_tmp.unlink(missing_ok=True)
            if dec_tmp_path:
                dec_tmp_path.unlink(missing_ok=True)
            raise

    print(f"[+] Decrypted: {src}  →  {dst}")


# ---------------------------------------------------------------------------
# Verify — integrity check without writing output
# ---------------------------------------------------------------------------

def verify_file(src: Path, password: str) -> None:
    """Re-derive key, re-MAC all chunks, check GCM tags — no plaintext written."""
    with src.open("rb") as f:
        magic = f.read(len(MAGIC_V3))

        if magic == MAGIC_V3:
            flags_byte = f.read(1)
            flags      = flags_byte[0] if flags_byte else 0
            header_len = _V3_HEADER
        elif magic == MAGIC_V2:
            flags      = 0
            header_len = _V2_HEADER
        else:
            raise ValueError("Not a valid crypter file (bad magic).")

        salt  = f.read(SALT_SIZE)
        nonce = f.read(NONCE_SIZE)

        f.seek(-HMAC_SIZE, 2)
        expected_hmac = f.read(HMAC_SIZE)

        f.seek(header_len)

        print(f"[*] Deriving key …")
        key = derive_key(password, salt)

        aesgcm    = AESGCM(key)
        mac       = hmac.new(key, digestmod=hashlib.sha256)
        chunk_idx = 0
        total_pt  = 0

        reader = _BoundedReader(f, -HMAC_SIZE)
        file_size   = src.stat().st_size - header_len - HMAC_SIZE
        upd, finish = _make_progress(max(file_size, 0))

        while True:
            length_bytes = reader.read(4)
            if not length_bytes:
                break
            if len(length_bytes) < 4:
                raise ValueError("Truncated file — missing chunk length.")
            ct_len = struct.unpack(">I", length_bytes)[0]
            ct     = reader.read(ct_len)
            if len(ct) < ct_len:
                raise ValueError("Truncated file — incomplete chunk.")

            mac.update(length_bytes)
            mac.update(ct)

            aad = struct.pack(">Q", chunk_idx)
            try:
                pt = aesgcm.decrypt(nonce, ct, aad)
            except Exception:
                raise ValueError(f"GCM authentication failed at chunk {chunk_idx}.")

            total_pt += len(pt)
            upd(len(pt))
            chunk_idx += 1

        finish()

        if not hmac.compare_digest(mac.digest(), expected_hmac):
            raise ValueError("HMAC mismatch — file has been tampered with or corrupted.")

    compressed_str = " (lzma-compressed plaintext)" if flags & FLAG_COMPRESSED else ""
    print(f"[+] OK — {src}")
    print(f"    Chunks : {chunk_idx}")
    print(f"    Plaintext size : {total_pt:,} B{compressed_str}")
    print(f"    HMAC   : verified")
    print(f"    GCM tags: all {chunk_idx} passed")


# ---------------------------------------------------------------------------
# Directory encrypt / decrypt
# ---------------------------------------------------------------------------

def encrypt_dir(src: Path, dst: Path, password: str,
                *, compress: bool = False, shred: bool = False) -> None:
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    dst.mkdir(parents=True)

    files = [p for p in src.rglob("*") if p.is_file()]
    print(f"[*] Encrypting {len(files)} file(s) from {src}")

    errors = 0
    for src_file in files:
        rel      = src_file.relative_to(src)
        dst_file = (dst / rel).with_suffix((dst / rel).suffix + ".enc")
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            encrypt_file(src_file, dst_file, password, compress=compress, shred=shred)
        except Exception as exc:
            print(f"[-] Failed {src_file}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"[!] {errors} file(s) failed.", file=sys.stderr)
    else:
        print(f"[+] Directory encrypted → {dst}")


def decrypt_dir(src: Path, dst: Path, password: str) -> None:
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    dst.mkdir(parents=True)

    files = [p for p in src.rglob("*.enc") if p.is_file()]
    print(f"[*] Decrypting {len(files)} file(s) from {src}")

    errors = 0
    for src_file in files:
        rel      = src_file.relative_to(src)
        dst_name = rel.with_suffix("") if rel.suffix == ".enc" else rel
        dst_file = dst / dst_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            decrypt_file(src_file, dst_file, password)
        except Exception as exc:
            print(f"[-] Failed {src_file}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"[!] {errors} file(s) failed.", file=sys.stderr)
    else:
        print(f"[+] Directory decrypted → {dst}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _BoundedReader:
    """Wraps a seekable file and hides the last *tail* bytes from reads."""

    def __init__(self, f, tail: int):
        self._f   = f
        pos       = f.tell()
        f.seek(0, 2)
        self._end = f.tell() + tail   # tail is negative
        f.seek(pos)

    def read(self, n=-1):
        remaining = max(0, self._end - self._f.tell())
        if n < 0 or n > remaining:
            n = remaining
        return self._f.read(n) if n > 0 else b""


def _check_no_overwrite(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            "Remove it manually or choose a different output name."
        )


def get_password(confirm: bool = False) -> str:
    pw = getpass.getpass("Password: ")
    if not pw:
        print("[-] Password cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if confirm:
        pw2 = getpass.getpass("Confirm password: ")
        if pw != pw2:
            print("[-] Passwords do not match.", file=sys.stderr)
            sys.exit(1)
    return pw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AES-256-GCM file/directory crypter  (Argon2id · streaming · lzma)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  crypter.py enc secret.txt secret.txt.enc
  crypter.py enc -z secret.txt secret.txt.enc        # compress then encrypt
  crypter.py enc -s secret.txt secret.txt.enc        # shred source after encrypt
  crypter.py dec secret.txt.enc secret.txt
  crypter.py verify secret.txt.enc                   # check integrity (no output written)
  crypter.py encdir photos/ photos.enc/ -z -s
  crypter.py decdir photos.enc/ photos/
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # enc
    enc_p = sub.add_parser("enc", help="Encrypt a single file")
    enc_p.add_argument("input",  type=Path)
    enc_p.add_argument("output", type=Path)
    enc_p.add_argument("-p", "--password")
    enc_p.add_argument("-z", "--compress", action="store_true",
                       help="lzma-compress plaintext before encrypting")
    enc_p.add_argument("-s", "--shred", action="store_true",
                       help="Securely overwrite + delete source after encryption")

    # dec
    dec_p = sub.add_parser("dec", help="Decrypt a single file")
    dec_p.add_argument("input",  type=Path)
    dec_p.add_argument("output", type=Path)
    dec_p.add_argument("-p", "--password")

    # verify
    ver_p = sub.add_parser("verify", help="Verify HMAC + GCM integrity without decrypting")
    ver_p.add_argument("input", type=Path)
    ver_p.add_argument("-p", "--password")

    # encdir
    ed_p = sub.add_parser("encdir", help="Encrypt a directory tree")
    ed_p.add_argument("input",  type=Path)
    ed_p.add_argument("output", type=Path)
    ed_p.add_argument("-p", "--password")
    ed_p.add_argument("-z", "--compress", action="store_true")
    ed_p.add_argument("-s", "--shred",    action="store_true")

    # decdir
    dd_p = sub.add_parser("decdir", help="Decrypt a directory tree")
    dd_p.add_argument("input",  type=Path)
    dd_p.add_argument("output", type=Path)
    dd_p.add_argument("-p", "--password")

    return parser


def main() -> None:
    args    = build_parser().parse_args()
    confirm = args.cmd in ("enc", "encdir")
    password = getattr(args, "password", None) or get_password(confirm=confirm)

    try:
        if args.cmd == "enc":
            encrypt_file(args.input, args.output, password,
                         compress=args.compress, shred=args.shred)
        elif args.cmd == "dec":
            decrypt_file(args.input, args.output, password)
        elif args.cmd == "verify":
            verify_file(args.input, password)
        elif args.cmd == "encdir":
            encrypt_dir(args.input, args.output, password,
                        compress=args.compress, shred=args.shred)
        elif args.cmd == "decdir":
            decrypt_dir(args.input, args.output, password)
    except (ValueError, FileExistsError) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
