#!/usr/bin/env python3
"""
File Crypter - AES-256-GCM symmetric file encryption tool.
Encrypts and decrypts files using a password-derived key (PBKDF2 + AES-256-GCM).
"""

import os
import sys
import struct
import argparse
import getpass
from pathlib import Path

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"CRYPT01\x00"   # 8-byte file header
SALT_SIZE = 32            # bytes
NONCE_SIZE = 12           # bytes (96-bit, standard for GCM)
ITERATIONS = 200_000
KEY_LEN = 32              # bytes (AES-256)


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=salt,
        iterations=ITERATIONS,
    )
    return kdf.derive(password.encode())


# ---------------------------------------------------------------------------
# Encrypt
# ---------------------------------------------------------------------------

def encrypt_file(src: Path, dst: Path, password: str) -> None:
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = derive_key(password, salt)

    plaintext = src.read_bytes()

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # GCM tag appended

    with dst.open("wb") as f:
        f.write(MAGIC)
        f.write(salt)
        f.write(nonce)
        f.write(struct.pack(">Q", len(ciphertext)))
        f.write(ciphertext)

    print(f"[+] Encrypted: {src} -> {dst}")


# ---------------------------------------------------------------------------
# Decrypt
# ---------------------------------------------------------------------------

def decrypt_file(src: Path, dst: Path, password: str) -> None:
    with src.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Not a valid crypter file (bad magic bytes).")

        salt = f.read(SALT_SIZE)
        nonce = f.read(NONCE_SIZE)
        ct_len = struct.unpack(">Q", f.read(8))[0]
        ciphertext = f.read(ct_len)

    key = derive_key(password, salt)
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception:
        raise ValueError("Decryption failed — wrong password or corrupted file.")

    dst.write_bytes(plaintext)
    print(f"[+] Decrypted: {src} -> {dst}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_password(confirm: bool = False) -> str:
    pw = getpass.getpass("Password: ")
    if confirm:
        pw2 = getpass.getpass("Confirm password: ")
        if pw != pw2:
            print("[-] Passwords do not match.", file=sys.stderr)
            sys.exit(1)
    return pw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AES-256-GCM file crypter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  crypter.py enc secret.txt secret.txt.enc
  crypter.py dec secret.txt.enc secret.txt
  crypter.py enc -p mysecret secret.txt secret.txt.enc
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in [("enc", "Encrypt a file"), ("dec", "Decrypt a file")]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("input", type=Path, help="Input file")
        sp.add_argument("output", type=Path, help="Output file")
        sp.add_argument("-p", "--password", help="Password (omit for interactive prompt)")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "enc":
        password = args.password or get_password(confirm=True)
        encrypt_file(args.input, args.output, password)

    elif args.cmd == "dec":
        password = args.password or get_password(confirm=False)
        try:
            decrypt_file(args.input, args.output, password)
        except ValueError as exc:
            print(f"[-] {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
