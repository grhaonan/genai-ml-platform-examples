from strands import tool
import sys
import os
import json


@tool
def user_prompt(
   prompt: str
) -> str:
    """
    ## Prompt the user for input during tool execution

This utility function enables **interactive tools** to collect **real-time input from users** during execution. It is especially useful in workflows where certain parameters, configurations, or clarifications are not predetermined and must be provided manually.

The function displays a user-friendly prompt, waits for the user to respond, and returns the input as a string. It ensures smooth progression by actively involving users when automation alone is insufficient.

### Typical Use Cases:
- Requesting missing or ambiguous inputs (e.g., architecture component names, service types)
- Asking the user to select or confirm options (e.g., instance types, regions)
- Gathering custom configuration parameters (e.g., number of nodes, storage size)
- Guiding users through multi-step wizards or setup flows

### Args:
- `prompt (str)`: A clearly worded message or question that describes what information is required from the user.

### Returns:
- `str`: The user’s input as a string.

### Example:
```python
    region = user_prompt("Please enter the AWS region for deployment (e.g., us-west-2):")
"""
    # Display the prompt to the user
    print("--" * 20)
    print("User Prompt")
    print("--" * 20)
    print(prompt)
    # Get user input
    user_input = input("Please enter your response: ")
    print("--" * 20)
    print("End of User Prompt")
    print("--" * 20)
    return user_input