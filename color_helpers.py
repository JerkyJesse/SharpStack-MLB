"""Color and formatting helpers using colorama."""

import sys
import os

# Force UTF-8 on Windows consoles (needed when double-clicking .py files)
if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")   # switch console code page to UTF-8
    os.system("")                          # enable ANSI escape processing
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from colorama import init as colorama_init, Fore, Back, Style

colorama_init(autoreset=True)

# Test whether stdout can actually print Unicode after all the above
_USE_UNICODE = False
try:
    sys.stdout.write("\r")  # harmless write to test the stream
    "─".encode(sys.stdout.encoding or "ascii")
    _USE_UNICODE = True
except Exception:
    _USE_UNICODE = False

_DIV_CHAR = "─" if _USE_UNICODE else "-"


def _c(text, *codes):
    return "".join(codes) + str(text) + Style.RESET_ALL

def cok(t):   return _c(t, Fore.GREEN,   Style.BRIGHT)
def cerr(t):  return _c(t, Fore.RED,     Style.BRIGHT)
def cwarn(t): return _c(t, Fore.YELLOW)
def chi(t):   return _c(t, Fore.CYAN,    Style.BRIGHT)
def cdim(t):  return _c(t, Style.DIM)
def cbold(t): return _c(t, Style.BRIGHT)
def cgrn(t):  return _c(t, Fore.GREEN)
def cred(t):  return _c(t, Fore.RED)
def cmag(t):  return _c(t, Fore.MAGENTA, Style.BRIGHT)
def cyel(t):  return _c(t, Fore.YELLOW,  Style.BRIGHT)
def cblu(t):  return _c(t, Fore.BLUE,    Style.BRIGHT)

def div(n=80):
    print(cdim(_DIV_CHAR * n))

def hdr(t):
    print("\n" + Back.GREEN + Fore.BLACK + Style.BRIGHT + "  " + t + "  " + Style.RESET_ALL)
