import sys


def main() -> None:
    sys.stderr.write(
        "recall: use one of the installed entrypoints.\n"
        "  recall --help        # CLI (index, eval, scrub-test, serve)\n"
        "  recall-mcp           # stdio MCP server\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
