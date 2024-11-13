""" Utilities function to load Q&A from a JSON file """

import json


def load_questions_answers(file_path: str) -> list:
    """
    Load questions & answers from a JSON file w/t error handling.

    Parameters:
    -----------
    file_path (str): The path to the JSON file.

    Returns:
    --------
    list: A list of dictionaries, each containing a question and an answer.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

            # Check if the "qa_pairs" key exists and is a list
            if "qa_pairs" not in data or not isinstance(data["qa_pairs"], list):
                raise ValueError(
                    "JSON file is missing 'qa_pairs' key or it is not a list."
                )

            qa_dict = {}
            for pair in data["qa_pairs"]:
                # Check if each item in qa_pairs has "question" and "answer" keys
                if (
                    not isinstance(pair, dict)
                    or "question" not in pair
                    or "answer" not in pair
                ):
                    raise ValueError(
                        "Each item in 'qa_pairs' must be a dictionary containing 'question' and 'answer' keys."
                    )

                # Add the q&a pair to the dictionary
                qa_dict[pair["question"]] = pair["answer"]

            return qa_dict

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return []
    except ValueError as ve:
        print(f"Error: {ve}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def main():
    file_path = "questions_answers.json"
    qa_dict = load_questions_answers(file_path)
    if qa_dict:
        print("Loaded QA pairs successfully.")
        for question, answer in qa_dict.items():
            print(f"Q: {question}")
            print(f"A: {answer}")
    else:
        print("Failed to load QA pairs.")


if __name__ == "__main__":
    main()
