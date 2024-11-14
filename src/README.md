
# Thought Generation Pipeline

This project provides a comprehensive framework for generating and organizing thoughts (sub-topics) based on a main concept using Large Language Models (LLMs). It supports both OpenAI's GPT models and Anthropic's Claude models, allowing for both horizontal (parallel) and vertical (hierarchical) thought generation to create structured sub-topics for knowledge organization and brainstorming.

---

## Project Structure

The thought generation pipeline is organized into various levels—**idea**, **thought**, and **sub_thought**—which allow for a structured hierarchy to break down concepts.

### Hierarchical Levels

1. **Idea Level**
   - **Definition**: Represents the main overarching topic.
   - **Components**: Contains one or more **thoughts**.
   - **Model**: `IdeaJSONModel`
     - Fields:
       - `idea`: str - The primary topic or theme.
       - `thoughts`: Optional[List[ThoughtJSONModel]] - Main thoughts for the idea.
       - `clusters`: Optional[List[ClusterJSONModel]] - Applied when clustering is necessary.

2. **Thought Level**
   - **Definition**: Represents major themes under the main **idea**.
   - **Components**: Includes a name, description, and related **sub_thoughts**.
   - **Model**: `ThoughtJSONModel`
     - Fields:
       - `thought`: str - Name of the thought.
       - `description`: Optional[str] - Brief summary of the thought.
       - `sub_thoughts`: Optional[List[SubThoughtJSONModel]] - Sub-thoughts providing further details.

3. **Sub-Thought Level**
   - **Definition**: Detailed components related to each **thought**.
   - **Model**: `SubThoughtJSONModel`
     - Fields:
       - `name`: str - Name of the sub_thought.
       - `description`: str - Explanation of the sub_thought.
       - `importance`: Optional[str] - Relative importance (optional).
       - `connection_to_next`: Optional[str] - Connection to subsequent sub-thoughts (optional).

---

## Data Input/Output Pipeline

The pipeline includes specific workflows and JSON structures for handling idea, thought, and sub-thought generations using LLMs.

### Pipeline Workflow

1. **Horizontal Thought Generation**
   - Generates **thoughts** based on the main **idea**.
   - **Input**: High-level **idea** JSON file.
   - **Output**: JSON files containing generated **thoughts** for the **idea**.

2. **Vertical Sub-Thought Generation**
   - Expands each **thought** into **sub_thoughts**.
   - **Input**: JSON files from horizontal thought generation.
   - **Output**: JSON files with detailed **sub_thoughts** for each **thought**.

### Directory and File Path Configurations

Configuration paths are specified in `config.py` to ensure consistent file management. Key directories include:

- **BASE_DIR**: Root directory of the project.
- **INPUT_OUTPUT_DIR**: Stores all input/output data.
  - `input_output/thought_generation`
    - **THOUGHT_GENERATION_OPENAI_OUTPUT_DIR**: Stores OpenAI-generated output files.
    - **THOUGHT_GENERATION_CLAUDE_OUTPUT_DIR**: Stores Claude-generated output files.
- **Memory Directory**: `MEMORY_DIR` for intermediate or memory-based processing.

### JSON File Structures

- **Horizontal Thought JSON Structure**:
  ```json
  {
    "idea": "embedded systems",
    "thoughts": [
      {
        "thought": "Real-time Operating Systems (RTOS)",
        "description": "Explanation about RTOS and its importance...",
        "sub_thoughts": null
      },
      ...
    ],
    "clusters": null
  }
  ```
- **Vertical Thought JSON Structure**:
  ```json
  {
    "thought": "System Software Development",
    "sub_thoughts": [
      {
        "name": "Kernel programming",
        "description": "Low-level code for managing resources...",
        ...
      },
      ...
    ]
  }
  ```

---

## LLM Models

Models configured in `config.py` include:

### Anthropic (Claude):
- **CLAUDE_OPUS**: "claude-3-opus-20240229"
- **CLAUDE_SONNET**: "claude-3-5-sonnet-20241022"
- **CLAUDE_HAIKU**: "claude-3-haiku-20240307"

### OpenAI:
- **GPT_35_TURBO**: "gpt-3.5-turbo"
- **GPT_4_TURBO**: "gpt-4-turbo"
- **GPT_4_TURBO_32K**: "gpt-4-turbo-32k"

---

## Running the Pipelines

### OpenAI GPT Pipeline (1a)

To generate thoughts using OpenAI's models:
1. Set the main concept and output paths in `config.py` for:
   - `rank_of_sub_thoughts_output_0_json`: File path for parallel thoughts.
   - `array_of_sub_thoughts_output_0_json`: File path for vertical thoughts.

2. Run pipeline 1a:
   ```bash
   python main.py
   ```

### Claude Pipeline (1b)

To generate thoughts using Claude models:
1. Set the main concept and output paths in `config.py` for:
   - `rank_of_sub_thoughts_output_1_json`: File path for parallel thoughts.
   - `array_of_sub_thoughts_output_1_json`: File path for vertical thoughts.

2. Run pipeline 1b:
   ```bash
   python main.py
   ```

---

## Configuration

Ensure `config.py` has unique file paths for OpenAI and Claude outputs to prevent overwriting. Example:
```python
rank_of_sub_thoughts_output_0_json = "output/openai_rank_of_thoughts.json"
array_of_sub_thoughts_output_0_json = "output/openai_array_of_thoughts.json"
rank_of_sub_thoughts_output_1_json = "output/claude_rank_of_thoughts.json"
array_of_sub_thoughts_output_1_json = "output/claude_array_of_thoughts.json"
```

## Logging

Logs are saved in the `logs` directory to capture information on pipeline execution and error handling.

---

## Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set up API keys in `.env`:
  ```bash
  OPENAI_API_KEY=your_openai_key
  ANTHROPIC_API_KEY=your_anthropic_key
  ```

## Pipeline Details

- **Horizontal Thought Generation**: Generates main thoughts under a primary idea.
- **Vertical Thought Generation**: Expands each main thought into detailed sub-thoughts.

This configuration ensures consistency in handling ideas, thoughts, and sub-thoughts, allowing for scalable content generation across applications.


## ThoughtReader: How to Access Thoughts Usage Guide

The `ThoughtReader` class provides a convenient way to load, validate, and interact with structured JSON data related to a main idea and its associated thoughts and sub-thoughts. This can be useful for applications that require sequential processing of complex, hierarchical data.

### Requirements

Ensure you have the following Python libraries installed:

- `pydantic`
- `logging`
- `json`

To install `pydantic`, run:

```bash
pip install pydantic
```

### Setting Up

1. **Prepare the JSON File**: The JSON file should contain a top-level structure with an `"idea"` key (a string representing the main theme) and a `"thoughts"` key (a list of thought objects).
   - Each thought object should contain `"thought"` (name), `"description"`, and a `"sub_thoughts"` list.
   - Each sub-thought should contain `"name"`, `"description"`, `"importance"`, and an optional `"connection_to_next"` field.

2. **Initialize the ThoughtReader**: Use the path to the JSON file as an argument.

### Basic Usage

Below are examples of how to use each method of the `ThoughtReader` class.

#### 1. Initializing ThoughtReader

```python
from thought_reader import ThoughtReader

# Initialize ThoughtReader with your JSON file
thought_reader = ThoughtReader("path/to/your_json_file.json")
```

#### 2. Getting the Main Idea

```python
# Fetch the main idea or theme of the thoughts data
main_idea = thought_reader.get_idea()
print("Main Idea:", main_idea)
# Example Output: Main Idea: embedded software development
```

#### 3. Listing Thought Names

```python
# Get a list of thought names only (list[str])
thought_names = thought_reader.get_thoughts()
print("Thought Names:")
for thought_name in thought_names:
    print(thought)
# Example Output: 
# Thought Names:
# Definition and Scope
# Hardware-Software Integration
```

#### 4. Getting Thoughts with Descriptions

```python
# Retrieve thoughts along with their descriptions
thoughts_with_descriptions = thought_reader.get_thoughts_and_descriptions()
print("\nThoughts and Descriptions:")
for thought in thoughts_with_descriptions:
    print(f"{thought['thought']}: {thought['description']}")
# Example Output:
# Definition and Scope: This cluster focuses on basic concepts and definitions...
# Hardware-Software Integration: This cluster focuses on...
```

#### 5. Retrieving Sub-Thoughts for a Specific Thought
> **<span style="color:white">This method generates the actual content for LLMs to generate the question queues for the conversational agent (facilitator).</span>**

Example: Human friendly version
```python
# Get sub-thoughts for a specific thought
thought_name = "Definition and Scope"   # one of the "thoughts" in the idea "embedded software development"
sub_thoughts = thought_reader.get_sub_thoughts_for_thought(thought_name)
print(f"\nSub-Thoughts for '{thought_name}':")
for sub in sub_thoughts:
    print(f"{sub['name']}: {sub['description']} (Importance: {sub['importance']})")
# Example Output:
# Project Requirements Analysis: Identify and document the requirements...
# System Architecture Design: Design the overall architecture...
```

Example: LLM friendly version
```python
# Get sub-thoughts for a specific thought
thought_name = "Definition and Scope"   # one of the "thoughts" in the idea "embedded software development"
sub_thoughts = thought_reader.get_sub_thoughts_for_thought(thought_name)
print(f"\nSub-Thoughts for '{thought_name}':")
for sub in sub_thoughts:
    print(f"{sub['name']}: {sub['description']} (Importance: {sub['importance']})")
# Example Output:
# Project Requirements Analysis: Identify and document the requirements...
# System Architecture Design: Design the overall architecture...
```
The "raw output" from the ThoughtReader (a list of dictionaries) would generally be easier to load into prompts for an LLM, especially when generating specific prompts for each sub_thought. Here’s why:

Advantages of Format A
Structured Data: Format A maintains a structured data format where each sub_thought is encapsulated in a dictionary with keys for name, description, importance, and connection_to_next. This makes it easy to access each piece of information programmatically, allowing you to insert specific values into a prompt.

Simpler Prompt Generation: With structured data, you can dynamically create prompts by looping through each dictionary and accessing fields like name and description without additional parsing or text manipulation.

Consistent Fields: Since each sub_thought in Format A has a consistent set of keys, you avoid parsing issues or potential inconsistencies in how data is organized (which could happen with Format B’s text-based structure).

Example Usage with Format A
Given Format A, here’s how you could use it to create prompts programmatically:

python
Copy code
for sub_thought in sub_thoughts:
    prompt = f"As a facilitator, generate a question for {sub_thought['name']}, defined as {sub_thought['description']}."
    # Send prompt to LLM or store for further use
    print(prompt)

  
### Error Handling

- If the JSON file structure is incorrect or missing required fields, `ThoughtReader` will raise a `ValueError`.
- If an invalid thought name is provided in `get_sub_thoughts_for_thought`, a warning will be logged, and an empty list will be returned.

### Sample JSON Structure

Here’s an example JSON structure compatible with `ThoughtReader`:

```json
{
    "idea": "embedded software development",
    "thoughts": [
        {
            "thought": "Definition and Scope",
            "description": "Basic concepts and definitions crucial for understanding embedded software.",
            "sub_thoughts": [
                {
                    "name": "Project Requirements Analysis",
                    "description": "Identify and document requirements for the embedded system.",
                    "importance": "High",
                    "connection_to_next": "Defines system architecture."
                }
                // Additional sub-thoughts...
            ]
        }
        // Additional thoughts...
    ]
}
