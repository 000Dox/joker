#!/usr/bin/env python3
"""
File Crypter - AES-256-GCM symmetric file/directory encryption tool.

Encryption scheme
-----------------
  KDF   : Argon2id  (time=4, memory=65536 KiB, parallelism=2)
  Cipher: AES-256-GCM streaming (64 KiB chunks)
  Auth  : per-chunk AAD = chunk index (prevents reordering / truncation)

File format
-----------
  [ MAGIC  8 B ]
  [ SALT  32 B ]
  [ NONCE 12 B ]
  [ CHUNKS … ]
      each chunk: [ length 4 B BE ][ ciphertext+tag (len bytes) ]
  [ HMAC-SHA256 of header+ciphertext 32 B ]   (integrity sentinel)
"""

import os
import sys
import struct
import hmac
import hashlib
import argparse
import getpass
import shutil
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC          = b"CRYPT02\x00"   # bumped version marker
SALT_SIZE      = 32               # bytes
NONCE_SIZE     = 12               # bytes
CHUNK_SIZE     = 64 * 1024        # 64 KiB plaintext per chunk
KEY_LEN        = 32               # AES-256
HMAC_SIZE      = 32               # SHA-256 digest

# Argon2id parameters (OWASP recommended minimum for file encryption)
ARGON2_TIME    = 4
ARGON2_MEMORY  = 65_536           # KiB = 64 MiB
ARGON2_PARA    = 2


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
# Streaming encrypt/decrypt helpers
# ---------------------------------------------------------------------------

def _encrypt_stream(fin, fout, key: bytes, nonce: bytes) -> bytes:
    """Encrypt *fin* → *fout* in chunks; return HMAC over written bytes."""
    aesgcm = AESGCM(key)
    mac = hmac.new(key, digestmod=hashlib.sha256)
    chunk_idx = 0

    while True:
        plaintext = fin.read(CHUNK_SIZE)
        if not plaintext:
            break
        aad = struct.pack(">Q", chunk_idx)          # chunk index as AAD
        ct  = aesgcm.encrypt(nonce, plaintext, aad)
        length_prefix = struct.pack(">I", len(ct))
        fout.write(length_prefix)
        fout.write(ct)
        mac.update(length_prefix)
        mac.update(ct)
        chunk_idx += 1

    return mac.digest()


def _decrypt_stream(fin, fout, key: bytes, nonce: bytes, expected_hmac: bytes) -> None:
    """Decrypt chunks from *fin* → *fout*; raise on auth failure."""
    aesgcm = AESGCM(key)
    mac = hmac.new(key, digestmod=hashlib.sha256)
    chunk_idx = 0

    while True:
        length_bytes = fin.read(4)
        if not length_bytes:
            break
        if len(length_bytes) < 4:
            raise ValueError("Truncated file — missing chunk length.")
        ct_len = struct.unpack(">I", length_bytes)[0]
        ct = fin.read(ct_len)
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
        chunk_idx += 1

    if not hmac.compare_digest(mac.digest(), expected_hmac):
        raise ValueError("HMAC mismatch — file has been tampered with.")


# ---------------------------------------------------------------------------
# Single-file encrypt / decrypt
# ---------------------------------------------------------------------------

def encrypt_file(src: Path, dst: Path, password: str) -> None:
    _check_no_overwrite(dst)

    salt  = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key   = derive_key(password, salt)

    # Write to a temp file first; move only on success (atomic-ish)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with src.open("rb") as fin, tmp.open("wb") as fout:
            header = MAGIC + salt + nonce
            fout.write(header)
            digest = _encrypt_stream(fin, fout, key, nonce)
            fout.write(digest)
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(f"[+] Encrypted: {src}  →  {dst}")


def decrypt_file(src: Path, dst: Path, password: str) -> None:
    _check_no_overwrite(dst)

    with src.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(
                "Not a valid crypter v2 file (bad magic). "
                "Was it encrypted with an older version?"
            )
        salt  = f.read(SALT_SIZE)
        nonce = f.read(NONCE_SIZE)

        # Read everything except the trailing HMAC to a temp buffer position
        # We need to seek backwards from EOF to get the HMAC.
        f.seek(-HMAC_SIZE, 2)
        expected_hmac = f.read(HMAC_SIZE)

        # Rewind to start of chunk data
        f.seek(len(MAGIC) + SALT_SIZE + NONCE_SIZE)

        key = derive_key(password, salt)

        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            with tmp.open("wb") as fout:
                # Wrap fin so _decrypt_stream stops before the trailing HMAC
                _decrypt_stream(_BoundedReader(f, -HMAC_SIZE), fout, key, nonce, expected_hmac)
            tmp.replace(dst)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    print(f"[+] Decrypted: {src}  →  {dst}")


# ---------------------------------------------------------------------------
# Directory encrypt / decrypt
# ---------------------------------------------------------------------------

def encrypt_dir(src: Path, dst: Path, password: str) -> None:
    """Encrypt all files under *src* into a mirrored tree at *dst*."""
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    dst.mkdir(parents=True)

    files = [p for p in src.rglob("*") if p.is_file()]
    print(f"[*] Encrypting {len(files)} file(s) from {src}")

    errors = 0
    for src_file in files:
        rel  = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            encrypt_file(src_file, dst_file.with_suffix(dst_file.suffix + ".enc"), password)
        except Exception as exc:
            print(f"[-] Failed {src_file}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"[!] {errors} file(s) failed.", file=sys.stderr)
    else:
        print(f"[+] Directory encrypted → {dst}")


def decrypt_dir(src: Path, dst: Path, password: str) -> None:
    """Decrypt all *.enc files under *src* into a mirrored tree at *dst*."""
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    dst.mkdir(parents=True)

    files = [p for p in src.rglob("*.enc") if p.is_file()]
    print(f"[*] Decrypting {len(files)} file(s) from {src}")

    errors = 0
    for src_file in files:
        rel      = src_file.relative_to(src)
        # Strip the trailing .enc
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
        self._f    = f
        pos        = f.tell()
        f.seek(0, 2)
        self._end  = f.tell() + tail   # tail is negative
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
        description="AES-256-GCM file/directory crypter (Argon2id + streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  crypter.py enc secret.txt secret.txt.enc
  crypter.py dec secret.txt.enc secret.txt
  crypter.py encdir photos/ photos.enc/
  crypter.py decdir photos.enc/ photos/
  crypter.py enc -p mysecret secret.txt secret.txt.enc
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in [
        ("enc",    "Encrypt a single file"),
        ("dec",    "Decrypt a single file"),
        ("encdir", "Encrypt a directory tree"),
        ("decdir", "Decrypt a directory tree"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("input",  type=Path, help="Input file or directory")
        sp.add_argument("output", type=Path, help="Output file or directory")
        sp.add_argument("-p", "--password", help="Password (omit for interactive prompt)")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    confirm = args.cmd in ("enc", "encdir")
    password = args.password or get_password(confirm=confirm)

    try:
        if args.cmd == "enc":
            encrypt_file(args.input, args.output, password)
        elif args.cmd == "dec":
            decrypt_file(args.input, args.output, password)
        elif args.cmd == "encdir":
            encrypt_dir(args.input, args.output, password)
        elif args.cmd == "decdir":
            decrypt_dir(args.input, args.output, password)
    except (ValueError, FileExistsError) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
