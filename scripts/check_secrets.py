import re
import subprocess
import sys


PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{20,}", "疑似 API key（sk-…）"),
    (r"AKIA[0-9A-Z]{16}", "疑似 AWS Access Key"),
    (r"ghp_[A-Za-z0-9]{30,}", "疑似 GitHub Token"),
    (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "疑似私钥"),
]


ALLOW_FILES = {"config.example.txt", "scripts/check_secrets.py"}

ALLOW_VALUES = re.compile(r"(REPLACE|EXAMPLE|YOUR_OWN|xxxx|placeholder)", re.I)


def git_files() -> list[str]:
    out = subprocess.run(
        [
            "git", "-c", "core.quotepath=false", "ls-files",
            "--cached", "--others", "--exclude-standard", "-z",
        ],
        capture_output=True,
        check=True,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in out.stdout.split(b"\0")
        if item
    ]


def main() -> int:
    try:
        files = git_files()
    except Exception as e:
        print(f"读不到 git 文件列表（是否在仓库根目录？）：{e}")
        return 1

    problems = 0
    for path in files:
        if path in ALLOW_FILES:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, label in PATTERNS:
                for m in re.finditer(pat, line):
                    if ALLOW_VALUES.search(m.group(0)):
                        continue
                    problems += 1
                    print(f"  ✘ {path}:{lineno}  {label}")

    print(f"\n检查了 {len(files)} 个仓库文件，发现问题 {problems} 处。")
    if problems:
        print("提示：把真实 key 从代码里删掉，改用 config.txt（已 gitignore）或环境变量。")
        return 1
    print("Ok：仓库里没有发现真实密钥。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
