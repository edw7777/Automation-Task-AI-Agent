import os
import asyncio
from datetime import datetime, timedelta

import yfinance as yf
import mplfinance as mpf

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from llama_index.core.memory import ChatMemoryBuffer

api_key = os.getenv("GOOGLE_API_KEY")


def get_candlestick_chart(ticker: str, days: int = 14) -> str:
    """
    Fetches the last `days` of OHLCV data for `ticker` using yfinance,
    plots a candlestick chart with 5- and 10-day moving averages and volume,
    saves it to '<ticker>_candlestick.png', and returns the filepath.
    """
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.Ticker(ticker).history(start=start, end=end)
    print(df)
    chart_path = f"{ticker}_candlestick.png"
    mpf.plot(
        df,
        type='candle',
        mav=(5, 10),
        volume=True,
        title=f"{ticker} Candlestick ({days}d)",
        savefig=chart_path
    )
    return chart_path

# print(get_candlestick_chart("PLTR", 14))

async def get_tools():
    return [
        FunctionTool.from_defaults(fn=get_candlestick_chart),
        *CodeInterpreterToolSpec().to_tool_list(),
    ]

async def main():
    # Initialize Gemini LLM
    chart_path = get_candlestick_chart(ticker="PLTR", days=14)

    llm = GoogleGenAI(model="models/gemini-2.0-flash")

    memory = ChatMemoryBuffer(token_limit=4096)
    system_prompt = (
        "You are an intelligent trading agent.  \n"
        "1) Before doing anything else, you *must* call the FunctionTool:\n"
        "   get_candlestick_chart(ticker=\"PLTR\", days=14)\n"
        "   —exactly that line, with no markdown fences or comments.  \n"
        "2) Wait for the tool to return the filepath.  \n"
        "3) Once you have the filepath, immediately run Python code via the code_interpreter tool to:\n"
        "   • Load the file and extract the last 14 days of OHLCV data\n"
        "   • Compute 5-day & 10-day moving averages\n"
        "   • Identify candlestick patterns (hammer, doji, engulfing, etc.)\n"
        "   • Evaluate MA momentum, crossovers, and volume trends\n"
        "   • Determine support & resistance levels\n"
        "   • Print average, std dev, and a one-sentence summary of up/down/stable\n"
        "   • Recommend buy price, profit target, stop-loss if R:R ≥ 2:1, or explain why not\n"
        "4) Do not output any code or analysis until you have done step 1 and step 2.  \n"
        "5) When running code, do _not_ wrap it in backticks—just output the Python.\n"
    )


    # Build the trading agent
    agent = FunctionAgent(
        llm=llm,
        retrieve_tools=get_tools,
        memory = memory,
        system_prompt=system_prompt,
        verbose=True,
    )

    # User prompt to trigger the analysis
    prompt = (
    # f-string to inject the actual chart_path you pre-ran in your driver code
    f"Here is the candlestick chart path: {chart_path}\n"
    "Return you analysis from the python code in json format eg {High: 100, Low: 10, etc...}"
)
    prompt= "Say hi fn"

    # Run the agent
    response = await agent.run(prompt)
    print(response.response.content)

if __name__ == "__main__":
    asyncio.run(main())