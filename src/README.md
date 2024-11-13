
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
rank_of_sub_thoughts_output_0_json = "output/openai_ranked_thoughts.json"
array_of_sub_thoughts_output_0_json = "output/openai_array_of_thoughts.json"
rank_of_sub_thoughts_output_1_json = "output/claude_ranked_thoughts.json"
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
```