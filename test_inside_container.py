# Test script to be executed inside Docker container
import asyncio

async def simplified_add_execute_code_cell(cell_content: str) -> str:
    """Simplified version of add_execute_code_cell function"""
    print("!!! DEBUG: Entering add_execute_code_cell, returning immediately.")
    await asyncio.sleep(0.1)  # Simulate tiny bit of work
    return "DEBUG: Tool called, did nothing."

async def test_function():
    print("Starting test of simplified add_execute_code_cell...")
    result = await simplified_add_execute_code_cell("print('Test code')")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_function())
