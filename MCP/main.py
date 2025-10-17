# main.py
import asyncio
from mcp_client import client


async def main():
    print("🚀 Launching MCP client...")
    await client()


if __name__ == "__main__":
    asyncio.run(main())
